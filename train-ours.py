# ============================================================
# SBERT + GCN Training Script (Colab-friendly)
# - TQDM shows per-batch: Loss, F1, Acc
# - At the end: epoch-level plots for Loss/F1/Acc/Precision/Recall
# - Saves test confusion matrix
# ============================================================

# !pip install torch-geometric  # Uncomment if you really need PyG for your gnn2.py

import os
from datetime import datetime
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from scipy import sparse
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

# Local Imports
from dataset import TokenizedDataset
from sbert import SiameseBERT
from gnn3 import SBERTGCN, normalize_edge_index


# ---------------------------
# Device
# ---------------------------
def _device(device_str: str) -> torch.device:
    if device_str == "gpu":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("GPU not available, falling back to CPU.")
        return torch.device("cpu")
    return torch.device("cpu")


# ---------------------------
# Graph loading
# ---------------------------
def _load_graph(dataset_dir: str, device: torch.device):
    adj = sparse.load_npz(os.path.join(dataset_dir, "graph_adj.npz")).tocsr()
    node_ids = np.load(os.path.join(dataset_dir, "graph_adj_node_ids.npy"))
    coo = adj.tocoo()

    edge_index = torch.from_numpy(
        np.vstack([coo.row, coo.col]).astype(np.int64)
    ).to(device)

    # normalize_edge_index is assumed to return (edge_index, edge_weight)
    edge_index, edge_weight = normalize_edge_index(edge_index, len(node_ids))
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    return (edge_index, edge_weight), node_ids


# ---------------------------
# Batch pair -> graph indices
# ---------------------------
def _pairs_to_indices(batch, id_to_idx, device):
    idx1, idx2, labs = [], [], []

    for a, b, y in zip(batch["issue_id1"], batch["issue_id2"], batch["label"]):
        a_id, b_id = int(a), int(b)
        if a_id in id_to_idx and b_id in id_to_idx:
            idx1.append(id_to_idx[a_id])
            idx2.append(id_to_idx[b_id])
            labs.append(float(y))

    if not idx1:
        return None, None, None

    return (
        torch.tensor(idx1, device=device),
        torch.tensor(idx2, device=device),
        torch.tensor(labs, device=device),
    )


# ---------------------------
# Threshold selection on VAL
# ---------------------------
def _get_best_threshold(sims: np.ndarray, labs: np.ndarray) -> float:
    thresholds = np.linspace(0.1, 0.9, 17)
    best_f1, best_th = 0.0, 0.5
    for th in thresholds:
        preds = (sims >= th).astype(int)
        f1 = f1_score(labs, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_th = f1, th
    return float(best_th)


# ---------------------------
# Epoch runner
# - tqdm: per-batch Loss/F1/Acc only
# - returns epoch metrics: loss/acc/f1/precision/recall + sims/labs
# ---------------------------
def _run_epoch(
    loader,
    node_emb,
    model,
    edge_index,
    edge_weight,
    id_to_idx,
    device,
    criterion,
    optimizer=None,
    thresh=0.5,
    desc="Run",
):
    is_train = optimizer is not None
    model.train(is_train)
    node_emb.train(is_train)

    total_loss = 0.0
    seen_batches = 0
    all_sims, all_labs = [], []

    pbar = tqdm(loader, desc=desc, leave=False)

    for batch in pbar:
        idx, idy, labs_batch = _pairs_to_indices(batch, id_to_idx, device)
        if idx is None:
            continue

        ids1 = batch["input_ids1"].to(device)
        m1 = batch["attention_mask1"].to(device)
        ids2 = batch["input_ids2"].to(device)
        m2 = batch["attention_mask2"].to(device)

        with torch.set_grad_enabled(is_train):
            out1, out2 = model(
                node_emb,
                idx,
                idy,
                ids1,
                m1,
                ids2,
                m2,
                edge_index,
                edge_weight,
            )
            out1 = F.normalize(out1, dim=1)
            out2 = F.normalize(out2, dim=1)

            loss = criterion(out1, out2, (2 * labs_batch - 1.0))

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        sims = F.cosine_similarity(out1, out2).detach().cpu().numpy()
        labs = labs_batch.detach().cpu().numpy().astype(int)
        preds = (sims >= thresh).astype(int)

        b_loss = float(loss.item())
        b_f1 = f1_score(labs, preds, zero_division=0)
        b_acc = float((labs == preds).mean())

        pbar.set_postfix({"L": f"{b_loss:.3f}", "F1": f"{b_f1:.3f}", "Acc": f"{b_acc:.3f}"})

        total_loss += b_loss
        seen_batches += 1
        all_sims.extend(sims.tolist())
        all_labs.extend(labs.tolist())

    all_sims = np.asarray(all_sims, dtype=np.float32)
    all_labs = np.asarray(all_labs, dtype=np.int32)

    if all_labs.size == 0:
        return {
            "loss": 0.0,
            "acc": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "sims": all_sims,
            "labs": all_labs,
        }

    preds_all = (all_sims >= thresh).astype(int)

    return {
        "loss": total_loss / max(1, seen_batches),
        "acc": float((preds_all == all_labs).mean()),
        "f1": f1_score(all_labs, preds_all, zero_division=0),
        "precision": precision_score(all_labs, preds_all, zero_division=0),
        "recall": recall_score(all_labs, preds_all, zero_division=0),
        "sims": all_sims,
        "labs": all_labs,
    }


# ---------------------------
# Plot epoch-level metrics (end only)
# ---------------------------
def _save_epoch_plots(epoch_hist, ckpt_dir, dataset_name=""):
    metrics = ["loss", "f1", "acc", "precision", "recall"]
    epochs = np.arange(1, len(epoch_hist["train"]["loss"]) + 1)

    for m in metrics:
        plt.figure(figsize=(10, 4))
        plt.plot(epochs, epoch_hist["train"][m], label=f"Train {m.upper()}")
        plt.plot(epochs, epoch_hist["val"][m], label=f"Val {m.upper()}")
        plt.title(f"Epoch-level {m.upper()} - {dataset_name}")
        plt.xlabel("Epoch")
        plt.ylabel(m.upper())
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(ckpt_dir, f"epoch_{m}.png"))
        plt.close()


# ---------------------------
# Confusion matrix plot (end only)
# ---------------------------
def _save_confusion_matrix(ts_m, ckpt_dir, test_th=0.5):
    preds = (ts_m["sims"] >= test_th).astype(int)
    cm = confusion_matrix(ts_m["labs"], preds)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title(f"Test Confusion Matrix (th={test_th:.2f})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["Unique", "Duplicate"])
    plt.yticks([0, 1], ["Unique", "Duplicate"])

    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(os.path.join(ckpt_dir, "test_confusion_matrix.png"))
    plt.close()


# ---------------------------
# Main training pipeline
# ---------------------------
def train_bertgnn(csv_path, dataset_dir, batch_size=16, epochs=5, lr=2e-5, model_name="sbert-gnn", device="cpu"):
    device = _device(device)
    dataset_name = os.path.basename(os.path.normpath(dataset_dir))
    run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ckpt_dir = os.path.join("checkpoints", f"{model_name}_{dataset_name}", run_ts)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Model: {model_name} | Dataset: {dataset_name} | Artifacts: {ckpt_dir}")

    (edge_index, edge_weight), node_ids = _load_graph(dataset_dir, device)
    id_to_idx = {int(nid): i for i, nid in enumerate(node_ids)}

    dataset = TokenizedDataset(csv_path)
    n = len(dataset)

    n_train = int(0.6 * n)
    n_val = int(0.2 * n)
    n_test = n - n_train - n_val

    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    }

    node_emb = nn.Embedding(len(node_ids), 128).to(device)
    model = SBERTGCN(SiameseBERT().to(device), len(node_ids), 128, 128, l=0.5).to(device)

    optimizer = AdamW(list(model.parameters()) + list(node_emb.parameters()), lr=lr)
    criterion = nn.CosineEmbeddingLoss(margin=0.5)

    best_val_f1 = 0.0
    locked_th = 0.5

    epoch_hist = {
        "train": {k: [] for k in ["loss", "f1", "acc", "precision", "recall"]},
        "val": {k: [] for k in ["loss", "f1", "acc", "precision", "recall"]},
    }

    for epoch in range(epochs):
        tr_m = _run_epoch(
            loaders["train"],
            node_emb,
            model,
            edge_index,
            edge_weight,
            id_to_idx,
            device,
            criterion,
            optimizer=optimizer,
            thresh=locked_th,
            desc=f"Ep {epoch+1} Train",
        )

        val_m = _run_epoch(
            loaders["val"],
            node_emb,
            model,
            edge_index,
            edge_weight,
            id_to_idx,
            device,
            criterion,
            optimizer=None,
            thresh=locked_th,
            desc=f"Ep {epoch+1} Val",
        )

        for k in ["loss", "f1", "acc", "precision", "recall"]:
            epoch_hist["train"][k].append(tr_m[k])
            epoch_hist["val"][k].append(val_m[k])

        epoch_best_th = _get_best_threshold(val_m["sims"], val_m["labs"])

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Val F1: {val_m['f1']:.4f} | Val Prec: {val_m['precision']:.4f} | Val Rec: {val_m['recall']:.4f} | "
            f"Next Th: {epoch_best_th:.2f}"
        )

        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            locked_th = epoch_best_th
            torch.save(
                {"model": model.state_dict(), "emb": node_emb.state_dict(), "thresh": locked_th},
                os.path.join(ckpt_dir, "best_model.pth"),
            )

    # -----------------------
    # Final test (best ckpt)
    # -----------------------
    print("\n" + "=" * 20 + " FINAL TEST " + "=" * 20)
    best_ckpt = torch.load(os.path.join(ckpt_dir, "best_model.pth"), weights_only=False)
    model.load_state_dict(best_ckpt["model"])
    node_emb.load_state_dict(best_ckpt["emb"])
    test_th = float(best_ckpt["thresh"])

    ts_m = _run_epoch(
        loaders["test"],
        node_emb,
        model,
        edge_index,
        edge_weight,
        id_to_idx,
        device,
        criterion,
        optimizer=None,
        thresh=test_th,
        desc="Test Set",
    )

    _save_epoch_plots(epoch_hist, ckpt_dir, dataset_name=dataset_name)
    _save_confusion_matrix(ts_m, ckpt_dir, test_th=test_th)

    print(
        f"Final Test | F1: {ts_m['f1']:.4f} | Prec: {ts_m['precision']:.4f} | "
        f"Rec: {ts_m['recall']:.4f} | Th: {test_th:.2f}"
    )
    print(f"Saved plots in: {ckpt_dir}")


# ---------------------------
# Entry
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Train a Siamese BERT GCN model.")
    parser.add_argument("--dataset_name", type=str, default="eclipse", help="The name of the dataset.")
    parser.add_argument("--model_name", type=str, default="sbert-gnn", help="The name of the model to use.")
    parser.add_argument("--batch_size", type=int, default=64, help="The batch size for training.")
    parser.add_argument("--lr", type=float, default=2e-5, help="The learning rate.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"], help="The device to use for training.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")

    args = parser.parse_args()

    dataset_dir = f"datasets/{args.dataset_name}"
    csv_path = os.path.join(dataset_dir, f"tokenized_pairs_train_bert-base-uncased_50000.csv")

    if not os.path.exists(csv_path):
        print(f"Dataset not found at {csv_path}")
        return

    train_bertgnn(
        csv_path,
        dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_name=args.model_name,
        lr=args.lr,
        device=args.device
    )

if __name__ == "__main__":
    main()
