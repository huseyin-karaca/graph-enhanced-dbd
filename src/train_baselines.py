import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
import csv
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
from datetime import datetime
import numpy as np
import argparse

from dataset import TokenizedDataset
from sbert import SiameseBERT



def train(
    csv_path: str,
    model_name: str = "bert-base-uncased",
    batch_size: int = 1024,
    epochs: int = 3,
    lr: float = 2e-5,
    dataset_name: str = "thunderbird",
    device: str = "cpu",

):
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

    print(f"Train size: {len(train_dataset)} | Validation size: {len(val_dataset)} | Test size: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")

    # 2. Initialize Model
    print(f"Initializing SiameseBERT with {model_name}...")
    model = SiameseBERT(model_name=model_name)
    model.to(device)

    # 3. Setup Training
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.CosineEmbeddingLoss(margin=0.5)

    model.train()

    epoch_losses = []
    epoch_accuracies = []
    epoch_f1s = []
    epoch_recalls = []
    epoch_precisions = []

    val_losses = []
    val_accuracies = []
    val_f1s = []
    val_recalls = []
    val_precisions = []

    # === store per-iteration loss here ===
    step_losses = []             # list of (global_step, loss_value)
    global_step = 0

    embedding_dict = {}

    # 0. Create save directory with date (and time) ONCE
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # e.g. 2025-12-09_13-45-02
    # Extract model name for better organization (e.g., "codebert-base" from "microsoft/codebert-base")
    model_short_name = model_name.split("/")[-1] if "/" in model_name else model_name
    save_root = os.path.join("checkpoints", f"{model_short_name}_{dataset_name}")
    save_dir = os.path.join(save_root, run_timestamp)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Checkpoints and figures will be saved under: {save_dir}")

    # optional: CSV log for each iteration - save in save_dir
    iter_log_path = os.path.join(save_dir, "train_step_losses.csv")
    last_all_labels = None
    last_all_preds = None

    with open(iter_log_path, "w", newline="") as f_log:
        writer = csv.writer(f_log)
        writer.writerow(["global_step", "epoch", "batch_idx", "loss"])

        threshold = 0.5  # for accuracy from cosine similarity

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            total_loss = 0.0
            total_correct = 0
            total_examples = 0

            all_preds = []
            all_labels = []

            # Training Loop
            model.train()
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")

            for batch_idx, batch in progress_bar:
                global_step += 1

                # Move batch to device
                input_ids1 = batch['input_ids1'].to(device)
                attention_mask1 = batch['attention_mask1'].to(device)
                input_ids2 = batch['input_ids2'].to(device)
                attention_mask2 = batch['attention_mask2'].to(device)
                issue_id1 = batch["issue_id1"]
                issue_id2 = batch["issue_id2"]

                labels01 = batch['label'].to(device).float()  # {0,1}

                # For CosineEmbeddingLoss: targets in {1, -1}
                targets = 2 * labels01 - 1.0  # 0 -> -1, 1 -> +1

                optimizer.zero_grad()

                # Forward pass
                embeddings1, embeddings2 = model(
                    input_ids1, attention_mask1,
                    input_ids2, attention_mask2
                )

                if issue_id1 is not None and issue_id2 is not None:
                    # Optimize: Batch processing for CLS tokens
                    # Note: This runs BERT a second time, but in batch mode (faster than loop)
                    # Ideally, modify model.forward to return CLS tokens to avoid re-computation
                    cls1 = model.create_embedding(input_ids1, attention_mask1)
                    cls2 = model.create_embedding(input_ids2, attention_mask2)

                    for i in range(len(issue_id1)):
                        embedding_dict[issue_id1[i].item()] = cls1[i].detach().cpu()
                        embedding_dict[issue_id2[i].item()] = cls2[i].detach().cpu()

                loss = criterion(embeddings1, embeddings2, targets)

                loss.backward()
                optimizer.step()

                loss_value = loss.item()
                total_loss += loss_value

                # ---- Accuracy computation ----
                with torch.no_grad():
                    cos_sim = F.cosine_similarity(embeddings1, embeddings2)  # [B]
                    preds = (cos_sim > threshold).float()                   # {0,1}
                    correct = (preds == labels01).sum().item()
                    total_correct += correct
                    total_examples += labels01.numel()
                    batch_acc = correct / labels01.numel()

                    preds_np = preds.cpu().numpy()
                    labels_np = labels01.cpu().numpy()

                    all_preds.extend(preds_np)
                    all_labels.extend(labels_np)

                    batch_f1 = f1_score(labels_np, preds_np, zero_division=0)

                # log per-iteration loss
                step_losses.append((global_step, loss_value))
                writer.writerow([global_step, epoch + 1, batch_idx, loss_value])

                progress_bar.set_postfix({
                    'loss': f"{loss_value:.4f}",
                    'acc': f"{batch_acc:.3f}",
                    'f1': f"{batch_f1:.3f}"
                })

            avg_loss = total_loss / len(train_loader)
            avg_acc = total_correct / total_examples

            epoch_f1 = f1_score(all_labels, all_preds)
            epoch_recall = recall_score(all_labels, all_preds)
            epoch_precision = precision_score(all_labels, all_preds)

            print(f"Train | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f} | F1: {epoch_f1:.4f} | Rec: {epoch_recall:.4f} | Prec: {epoch_precision:.4f}")

            epoch_losses.append(avg_loss)
            epoch_accuracies.append(avg_acc)
            epoch_f1s.append(epoch_f1)
            epoch_recalls.append(epoch_recall)
            epoch_precisions.append(epoch_precision)

            # Validation Loop
            model.eval()
            val_loss = 0.0
            val_all_sims = []
            val_all_labels = []

            # Open a file to log sample comparisons for this epoch
            sample_log_path = os.path.join(save_dir, f"val_samples_epoch_{epoch+1}.txt")
            with open(sample_log_path, "w") as f_sample:
                f_sample.write("Issue1_ID\tIssue2_ID\tLabel\tSimilarity\n")

                with torch.no_grad():
                    for batch in tqdm(val_loader, desc="Validation"):
                        input_ids1 = batch['input_ids1'].to(device)
                        attention_mask1 = batch['attention_mask1'].to(device)
                        input_ids2 = batch['input_ids2'].to(device)
                        attention_mask2 = batch['attention_mask2'].to(device)
                        labels01 = batch['label'].to(device).float()
                        targets = 2 * labels01 - 1.0

                        # Get issue IDs for logging
                        val_issue_id1 = batch.get("issue_id1")
                        val_issue_id2 = batch.get("issue_id2")

                        embeddings1, embeddings2 = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
                        loss = criterion(embeddings1, embeddings2, targets)
                        val_loss += loss.item()


                        if val_issue_id1 is not None and val_issue_id2 is not None:
                        # Optimize: Batch processing for CLS tokens
                                # Note: This runs BERT a second time, but in batch mode (faster than loop)
                                # Ideally, modify model.forward to return CLS tokens to avoid re-computation
                                cls1 = model.create_embedding(input_ids1, attention_mask1)
                                cls2 = model.create_embedding(input_ids2, attention_mask2)

                                for i in range(len(val_issue_id1)):
                                    embedding_dict[val_issue_id1[i].item()] = cls1[i].detach().cpu()
                                    embedding_dict[val_issue_id2[i].item()] = cls2[i].detach().cpu()


                        cos_sim = F.cosine_similarity(embeddings1, embeddings2)

                        val_all_sims.extend(cos_sim.cpu().numpy())
                        val_all_labels.extend(labels01.cpu().numpy())

                        # Log samples
                        if val_issue_id1 is not None and val_issue_id2 is not None:
                            sims_np = cos_sim.cpu().numpy()
                            labels_np = labels01.cpu().numpy()
                            ids1_np = val_issue_id1.numpy() if isinstance(val_issue_id1, torch.Tensor) else val_issue_id1
                            ids2_np = val_issue_id2.numpy() if isinstance(val_issue_id2, torch.Tensor) else val_issue_id2

                            for i in range(len(sims_np)):
                                f_sample.write(f"{ids1_np[i]}\t{ids2_np[i]}\t{labels_np[i]}\t{sims_np[i]:.4f}\n")

            print(f"Validation samples logged to {sample_log_path}")
            avg_val_loss = val_loss / len(val_loader)

            # Adaptive Thresholding
            thresholds = np.arange(0.3, 0.95, 0.05)
            best_f1 = -1.0
            best_thresh = 0.5
            best_metrics = {}

            val_sims_np = np.array(val_all_sims)
            val_labels_np = np.array(val_all_labels)

            for th in thresholds:
                preds = (val_sims_np >= th).astype(int)
                f1 = f1_score(val_labels_np, preds, zero_division=0)

                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = th
                    best_metrics = {
                        'acc': (preds == val_labels_np).mean(),
                        'rec': recall_score(val_labels_np, preds, zero_division=0),
                        'prec': precision_score(val_labels_np, preds, zero_division=0)
                    }

            print(f"Val   | Best Thresh: {best_thresh:.2f} | Loss: {avg_val_loss:.4f} | Acc: {best_metrics['acc']:.4f} | F1: {best_f1:.4f} | Rec: {best_metrics['rec']:.4f} | Prec: {best_metrics['prec']:.4f}")

            val_losses.append(avg_val_loss)
            val_accuracies.append(best_metrics['acc'])
            val_f1s.append(best_f1)
            val_recalls.append(best_metrics['rec'])
            val_precisions.append(best_metrics['prec'])

            # keep last epoch's preds/labels (using best threshold)
            last_all_labels = np.array(all_labels)
            last_all_preds = np.array(all_preds)

            # Save checkpoint
            ckpt_path = os.path.join(save_dir, f"sbert_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Model saved to {ckpt_path}")

    print(f"Per-iteration losses saved to {iter_log_path}")

    # === Confusion matrix for LAST epoch only ===
    if last_all_labels is not None and last_all_preds is not None:
        cm = confusion_matrix(last_all_labels, last_all_preds, labels=[0.0, 1.0])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])

        fig, ax = plt.subplots(figsize=(5, 5))
        disp.plot(ax=ax, values_format="d")
        ax.set_title(f"Confusion Matrix - Last Epoch (Epoch {epochs})")

        cm_path = os.path.join(save_dir, "confusion_matrix_last_epoch.png")
        fig.savefig(cm_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved confusion matrix for last epoch to {cm_path}")
    else:
        print("Warning: No labels/preds collected; confusion matrix not generated.")

    # Save embeddings
    emb_path = os.path.join(save_dir, "embedding_dict.pt")
    print(f"Saving {len(embedding_dict)} embeddings to {emb_path}...")
    torch.save(embedding_dict, emb_path)
    print("Done saving embeddings.")

    # Plotting epoch-level loss, accuracy, F1, precision, recall
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(12, 10))

    # --- Loss & Accuracy ---
    plt.subplot(2, 1, 1)
    plt.plot(epochs_range, epoch_losses, marker='o', linestyle='-', label='Train Loss')
    plt.plot(epochs_range, val_losses, marker='x', linestyle='-', label='Val Loss')
    plt.plot(epochs_range, epoch_accuracies, marker='s', linestyle='--', label='Train Accuracy')
    plt.plot(epochs_range, val_accuracies, marker='d', linestyle='--', label='Val Accuracy')
    plt.title('Training & Validation Loss & Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()

    # --- F1, Precision, Recall ---
    plt.subplot(2, 1, 2)
    # F1
    plt.plot(epochs_range, epoch_f1s, marker='^', linestyle='-', label='Train F1')
    plt.plot(epochs_range, val_f1s, marker='v', linestyle='-', label='Val F1')
    # Recall
    plt.plot(epochs_range, epoch_recalls, marker='o', linestyle='-', label='Train Recall')
    plt.plot(epochs_range, val_recalls, marker='o', linestyle='--', label='Val Recall')
    # Precision
    plt.plot(epochs_range, epoch_precisions, marker='s', linestyle='-', label='Train Precision')
    plt.plot(epochs_range, val_precisions, marker='s', linestyle='--', label='Val Precision')

    plt.title('Training & Validation F1 / Precision / Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    loss_curve_path = os.path.join(save_dir, "loss_acc_pr_rc_f1_curve.png")
    plt.savefig(loss_curve_path)
    plt.close()
    print(f"Loss, accuracy, precision, recall, F1 curves saved to {loss_curve_path}")

    # === Test Evaluation ===
    print("\n" + "="*30)
    print(f"TEST EVALUATION (Threshold={best_thresh:.2f})")
    print("="*30)

    model.eval()
    test_loss = 0.0
    test_all_preds = []
    test_all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids1 = batch['input_ids1'].to(device)
            attention_mask1 = batch['attention_mask1'].to(device)
            input_ids2 = batch['input_ids2'].to(device)
            attention_mask2 = batch['attention_mask2'].to(device)
            labels01 = batch['label'].to(device).float()
            targets = 2 * labels01 - 1.0

            embeddings1, embeddings2 = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
            loss = criterion(embeddings1, embeddings2, targets)
            test_loss += loss.item()

            cos_sim = F.cosine_similarity(embeddings1, embeddings2)
            preds = (cos_sim >= best_thresh).float()

            test_all_preds.extend(preds.cpu().numpy())
            test_all_labels.extend(labels01.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    test_acc = (np.array(test_all_preds) == np.array(test_all_labels)).mean()
    test_f1 = f1_score(test_all_labels, test_all_preds, zero_division=0)
    test_rec = recall_score(test_all_labels, test_all_preds, zero_division=0)
    test_prec = precision_score(test_all_labels, test_all_preds, zero_division=0)

    print(f"Test  | Loss: {avg_test_loss:.4f} | Acc: {test_acc:.4f} | F1: {test_f1:.4f} | Rec: {test_rec:.4f} | Prec: {test_prec:.4f}")

    # Test Confusion Matrix
    cm_test = confusion_matrix(test_all_labels, test_all_preds, labels=[0.0, 1.0])
    print("\nTest Confusion Matrix:")
    print(cm_test)

    # Plot Test Confusion Matrix
    disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=[0, 1])
    fig_test, ax_test = plt.subplots(figsize=(5, 5))
    disp_test.plot(ax=ax_test, values_format="d", cmap='Blues')
    ax_test.set_title(f"Test Confusion Matrix (Threshold={best_thresh:.2f})")

    cm_test_path = os.path.join(save_dir, "confusion_matrix_test.png")
    fig_test.savefig(cm_test_path, bbox_inches="tight")
    plt.close(fig_test)
    print(f"Saved test confusion matrix to {cm_test_path}")

    # Save Test Results
    test_results_path = os.path.join(save_dir, "test_results.txt")
    with open(test_results_path, "w") as f:
        f.write(f"Model Name: {model_name}\n")
        f.write(f"Test Evaluation (Threshold={best_thresh:.2f})\n")
        f.write(f"Loss: {avg_test_loss:.4f}\n")
        f.write(f"Accuracy: {test_acc:.4f}\n")
        f.write(f"F1 Score: {test_f1:.4f}\n")
        f.write(f"Recall: {test_rec:.4f}\n")
        f.write(f"Precision: {test_prec:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm_test))
    print(f"Test results saved to {test_results_path}")


def main():
    parser = argparse.ArgumentParser(description="Train a Siamese BERT model.")
    parser.add_argument("--dataset_name", type=str, default="eclipse", help="The name of the dataset.")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="The name of the model to use.")
    parser.add_argument("--batch_size", type=int, default=2, help="The batch size for training.")
    parser.add_argument("--lr", type=float, default=2e-5, help="The learning rate.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"], help="The device to use for training.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")

    args = parser.parse_args()

    if args.device == "gpu":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            print("GPU not available, falling back to CPU.")
            device = "cpu"
    else:
        device = "cpu"

    dataset_dir = f"datasets/{args.dataset_name}/"
    cur_dir = os.getcwd()
    csv_path = os.path.join(cur_dir, dataset_dir, f"tokenized_pairs_train_{args.model_name.split('/')[-1]}_50000.csv")

    print("CSV path:", csv_path)

    if os.path.exists(csv_path):
        train(
            csv_path,
            batch_size=args.batch_size,
            epochs=args.epochs,
            dataset_name=args.dataset_name,
            model_name=args.model_name,
            lr=args.lr,
            device=device
        )
    else:
        print(f"Dataset not found at {csv_path}")


if __name__ == "__main__":
    main()

