# -*- coding: utf-8 -*-
"""
SiameseBERT & GCN Training Script
Reproducibility Package
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
from datetime import datetime
from scipy import sparse

# === PROJE İMPORTLARI (Bu dosyaların repoda olduğundan emin olun) ===
from dataset import TokenizedDataset
from sbert import SiameseBERT
try:
    from gnn3 import SBERTGCN, normalize_edge_index
    GNN_AVAILABLE = True
except ImportError:
    print("UYARI: gnn3.py bulunamadı veya import edilemedi. GCN eğitimi çalışmayabilir.")
    GNN_AVAILABLE = False

# ==============================================================================
#  KONFİGÜRASYON (Burayı değiştirerek denemeler yapabilirsiniz)
# ==============================================================================

# 1. TEMEL PARAMETRELER
DATASET_NAME = "eclipse"          # 'eclipse', 'thunderbird' vb. klasör ismi
MODEL_NAME = "microsoft/codebert-base"  # 'bert-base-uncased', 'roberta-base' vb.

# 2. EĞİTİM MODU
# True: Sadece SiameseBERT eğitir.
# False: SBERT + GCN (Graph Convolutional Network) eğitir.
TRAIN_ONLY_SIAMESE = True 

# 3. HİPERPARAMETRELER
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 2e-5

# 4. PATH AYARLARI
# Colab'de veya lokalde çalışırken datasetlerin nerede olduğu.
# Varsayılan: kodun olduğu yerdeki 'datasets' klasörü.
BASE_DATA_DIR = "datasets" 
CHECKPOINT_DIR = "checkpoints"

# ==============================================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def train_siamese(
    csv_path: str,
    model_name: str = "bert-base-uncased",
    batch_size: int = 1024,
    epochs: int = 3,
    lr: float = 2e-5,
    dataset_name: str = "thunderbird",
):
    device = get_device()
    print(f"Using device: {device}")
    
    # 1. Load Dataset
    print(f"Loading dataset from {csv_path}...")
    dataset = TokenizedDataset(csv_path)
    print(f"Dataset size: {len(dataset)}")

    # Split into Train (60%), Validation (20%), Test (20%)
    total_len = len(dataset)
    train_len = int(0.6 * total_len)
    val_len = int(0.2 * total_len)
    test_len = total_len - train_len - val_len

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_len, val_len, test_len])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 2. Initialize Model
    print(f"Initializing SiameseBERT with {model_name}...")
    model = SiameseBERT(model_name=model_name)
    model.to(device)

    # 3. Setup Training
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.CosineEmbeddingLoss(margin=0.5)

    model.train()
    
    # Metrics storage
    epoch_losses, epoch_accuracies = [], []
    epoch_f1s, epoch_recalls, epoch_precisions = [], [], []
    val_losses, val_accuracies = [], []
    val_f1s, val_recalls, val_precisions = [], []
    
    step_losses = []
    global_step = 0
    embedding_dict = {}

    # Save directories
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_root = os.path.join(CHECKPOINT_DIR, dataset_name)
    save_dir = os.path.join(save_root, run_timestamp)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Checkpoints will be saved under: {save_dir}")

    iter_log_path = os.path.join(save_dir, "train_step_losses.csv")
    
    with open(iter_log_path, "w", newline="") as f_log:
        writer = csv.writer(f_log)
        writer.writerow(["global_step", "epoch", "batch_idx", "loss"])
        threshold = 0.5 

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            total_loss = 0.0
            total_correct = 0
            total_examples = 0
            all_preds, all_labels = [], []

            model.train()
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")

            for batch_idx, batch in progress_bar:
                global_step += 1
                
                input_ids1 = batch['input_ids1'].to(device)
                attention_mask1 = batch['attention_mask1'].to(device)
                input_ids2 = batch['input_ids2'].to(device)
                attention_mask2 = batch['attention_mask2'].to(device)
                issue_id1 = batch.get("issue_id1")
                issue_id2 = batch.get("issue_id2")
                labels01 = batch['label'].to(device).float()
                
                targets = 2 * labels01 - 1.0
                optimizer.zero_grad()
                
                embeddings1, embeddings2 = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
                
                # Embedding Dictionary Update
                if issue_id1 is not None and issue_id2 is not None:
                     # Re-compute CLS for dictionary (as per original code logic)
                    cls1 = model.create_embedding(input_ids1, attention_mask1)
                    cls2 = model.create_embedding(input_ids2, attention_mask2)
                    for i in range(len(issue_id1)):
                        if hasattr(issue_id1[i], 'item'):
                            embedding_dict[issue_id1[i].item()] = cls1[i].detach().cpu()
                        if hasattr(issue_id2[i], 'item'):
                            embedding_dict[issue_id2[i].item()] = cls2[i].detach().cpu()

                loss = criterion(embeddings1, embeddings2, targets)
                loss.backward()
                optimizer.step()

                loss_value = loss.item()
                total_loss += loss_value

                with torch.no_grad():
                    cos_sim = F.cosine_similarity(embeddings1, embeddings2)
                    preds = (cos_sim > threshold).float()
                    total_correct += (preds == labels01).sum().item()
                    total_examples += labels01.numel()
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels01.cpu().numpy())

                step_losses.append((global_step, loss_value))
                writer.writerow([global_step, epoch + 1, batch_idx, loss_value])
                
                progress_bar.set_postfix({'loss': f"{loss_value:.4f}"})

            # Epoch Metrics
            avg_loss = total_loss / len(train_loader)
            avg_acc = total_correct / total_examples
            epoch_f1 = f1_score(all_labels, all_preds)
            epoch_rec = recall_score(all_labels, all_preds)
            epoch_prec = precision_score(all_labels, all_preds)

            print(f"Train | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f} | F1: {epoch_f1:.4f}")
            
            epoch_losses.append(avg_loss)
            epoch_accuracies.append(avg_acc)
            epoch_f1s.append(epoch_f1)
            epoch_recalls.append(epoch_rec)
            epoch_precisions.append(epoch_prec)

            # Validation Loop
            model.eval()
            val_loss = 0.0
            val_all_sims = []
            val_all_labels = []

            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    input_ids1 = batch['input_ids1'].to(device)
                    attention_mask1 = batch['attention_mask1'].to(device)
                    input_ids2 = batch['input_ids2'].to(device)
                    attention_mask2 = batch['attention_mask2'].to(device)
                    labels01 = batch['label'].to(device).float()
                    targets = 2 * labels01 - 1.0

                    embeddings1, embeddings2 = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
                    loss = criterion(embeddings1, embeddings2, targets)
                    val_loss += loss.item()
                    
                    cos_sim = F.cosine_similarity(embeddings1, embeddings2)
                    val_all_sims.extend(cos_sim.cpu().numpy())
                    val_all_labels.extend(labels01.cpu().numpy())

            avg_val_loss = val_loss / len(val_loader)
            
            # Adaptive Thresholding logic
            thresholds = np.arange(0.3, 0.95, 0.05)
            best_f1 = -1.0
            best_metrics = {}
            val_sims_np = np.array(val_all_sims)
            val_labels_np = np.array(val_all_labels)

            for th in thresholds:
                preds = (val_sims_np >= th).astype(int)
                f1 = f1_score(val_labels_np, preds, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_metrics = {
                        'acc': (preds == val_labels_np).mean(),
                        'rec': recall_score(val_labels_np, preds, zero_division=0),
                        'prec': precision_score(val_labels_np, preds, zero_division=0)
                    }

            print(f"Val   | Loss: {avg_val_loss:.4f} | F1: {best_f1:.4f} (Best Thresh)")

            val_losses.append(avg_val_loss)
            val_f1s.append(best_f1)
            val_accuracies.append(best_metrics.get('acc', 0))
            val_recalls.append(best_metrics.get('rec', 0))
            val_precisions.append(best_metrics.get('prec', 0))

            # Save Checkpoint
            ckpt_path = os.path.join(save_dir, f"sbert_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), ckpt_path)

    # Save Embeddings
    emb_path = os.path.join(save_dir, "embedding_dict.pt")
    torch.save(embedding_dict, emb_path)
    print(f"Saved embeddings to {emb_path}")

    # Plotting code simplified for brevity (can be kept from original)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs+1), epoch_losses, label='Train Loss')
    plt.plot(range(1, epochs+1), val_losses, label='Val Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()

# --- GCN Helper Functions (Original logic preserved) ---

def _load_graph(dataset_dir: str, device: torch.device):
    adj_path = os.path.join(dataset_dir, "graph_adj.npz")
    node_ids_path = os.path.join(dataset_dir, "graph_adj_node_ids.npy")
    
    if not os.path.exists(adj_path):
        raise FileNotFoundError(f"Graph files not found at {dataset_dir}")

    adj = sparse.load_npz(adj_path).tocsr()
    node_ids = np.load(node_ids_path)
    coo = adj.tocoo()

    edge_index = torch.from_numpy(
        np.vstack([coo.row, coo.col]).astype(np.int64)
    ).to(device)

    edge_index, edge_weight = normalize_edge_index(edge_index, len(node_ids))
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    return (edge_index, edge_weight), node_ids

def _pairs_to_indices(batch, id_to_idx, device):
    idx1, idx2, labs = [], [], []
    for a, b, y in zip(batch["issue_id1"], batch["issue_id2"], batch["label"]):
        a_id, b_id = int(a), int(b)
        if a_id in id_to_idx and b_id in id_to_idx:
            idx1.append(id_to_idx[a_id])
            idx2.append(id_to_idx[b_id])
            labs.append(float(y))
            
    if not idx1: return None, None, None
    return torch.tensor(idx1, device=device), torch.tensor(idx2, device=device), torch.tensor(labs, device=device)

def _run_epoch_gcn(loader, node_emb, model, edge_index, edge_weight, id_to_idx, device, criterion, optimizer=None, thresh=0.5):
    is_train = optimizer is not None
    model.train(is_train)
    node_emb.train(is_train)
    
    total_loss = 0.0
    all_sims, all_labs = [], []
    
    for batch in tqdm(loader, leave=False):
        idx, idy, labs_batch = _pairs_to_indices(batch, id_to_idx, device)
        if idx is None: continue

        ids1 = batch["input_ids1"].to(device)
        m1 = batch["attention_mask1"].to(device)
        ids2 = batch["input_ids2"].to(device)
        m2 = batch["attention_mask2"].to(device)

        with torch.set_grad_enabled(is_train):
            out1, out2 = model(node_emb, idx, idy, ids1, m1, ids2, m2, edge_index, edge_weight)
            out1 = F.normalize(out1, dim=1)
            out2 = F.normalize(out2, dim=1)
            loss = criterion(out1, out2, (2 * labs_batch - 1.0))

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        sims = F.cosine_similarity(out1, out2).detach().cpu().numpy()
        all_sims.extend(sims)
        all_labs.extend(labs_batch.cpu().numpy())
        total_loss += loss.item()

    all_sims = np.array(all_sims)
    all_labs = np.array(all_labs)
    preds = (all_sims >= thresh).astype(int)
    
    return {
        "loss": total_loss / len(loader),
        "f1": f1_score(all_labs, preds, zero_division=0),
        "sims": all_sims,
        "labs": all_labs
    }

def train_bertgnn(csv_path, dataset_dir, batch_size=16, epochs=5, lr=2e-5, model_name="bert-base-uncased"):
    if not GNN_AVAILABLE:
        print("GNN modülleri eksik, işlem iptal ediliyor.")
        return

    device = get_device()
    print(f"Training SBERT+GCN on {dataset_dir}")
    
    # Load Graph
    (edge_index, edge_weight), node_ids = _load_graph(dataset_dir, device)
    id_to_idx = {int(nid): i for i, nid in enumerate(node_ids)}
    
    # Load Dataset
    dataset = TokenizedDataset(csv_path)
    n = len(dataset)
    n_train, n_val = int(0.6 * n), int(0.2 * n)
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n - n_train - n_val])
    
    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    }

    # Model Setup
    node_emb = nn.Embedding(len(node_ids), 128).to(device)
    # Note: Using SBERTGCN wrapping SiameseBERT
    sbert_model = SiameseBERT(model_name=model_name).to(device)
    model = SBERTGCN(sbert_model, len(node_ids), 128, 128, l=0.5).to(device)
    
    optimizer = AdamW(list(model.parameters()) + list(node_emb.parameters()), lr=lr)
    criterion = nn.CosineEmbeddingLoss(margin=0.5)

    save_dir = os.path.join(CHECKPOINT_DIR, f"{os.path.basename(dataset_dir)}_gcn_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        tr_m = _run_epoch_gcn(loaders["train"], node_emb, model, edge_index, edge_weight, id_to_idx, device, criterion, optimizer=optimizer)
        val_m = _run_epoch_gcn(loaders["val"], node_emb, model, edge_index, edge_weight, id_to_idx, device, criterion)
        
        print(f"Train Loss: {tr_m['loss']:.4f} | Val F1: {val_m['f1']:.4f}")
        
        # Simple checkpointing
        torch.save(model.state_dict(), os.path.join(save_dir, f"model_ep{epoch+1}.pth"))

# ==============================================================================
#  MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    # Pathleri oluştur
    dataset_dir_path = os.path.join(BASE_DATA_DIR, DATASET_NAME)
    
    # CSV dosyasını bul (genelleştirilmiş isim arama veya sabit isim)
    # Burada kullanıcı kodundaki sabit isme sadık kalıyoruz ama kontrol ediyoruz
    possible_csvs = [
        f"tokenized_pairs_train_{MODEL_NAME.split('/')[-1]}_50000.csv",
        f"tokenized_pairs_train_{MODEL_NAME}_50000.csv",
        "tokenized_pairs_train.csv" # Fallback
    ]
    
    csv_path = None
    for fname in possible_csvs:
        p = os.path.join(dataset_dir_path, fname)
        if os.path.exists(p):
            csv_path = p
            break
            
    if not csv_path:
        print(f"HATA: Dataset CSV bulunamadı! Aranan yer: {dataset_dir_path}")
        print("Lütfen DATASET_NAME parametresini kontrol edin veya dosya ismini düzeltin.")
    else:
        print(f"Dataset bulundu: {csv_path}")
        
        if TRAIN_ONLY_SIAMESE:
            print(">>> Mod: Sadece SiameseBERT Eğitimi")
            train_siamese(
                csv_path=csv_path,
                model_name=MODEL_NAME,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                lr=LEARNING_RATE,
                dataset_name=DATASET_NAME
            )
        else:
            print(">>> Mod: SBERT + GCN Eğitimi")
            train_bertgnn(
                csv_path=csv_path,
                dataset_dir=dataset_dir_path,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                lr=LEARNING_RATE,
                model_name=MODEL_NAME
            )