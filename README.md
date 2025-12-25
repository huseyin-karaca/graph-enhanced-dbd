# CS 588 Term Project: Duplicate Bug Report Detection

This repository contains the reproducibility package for our CS 588 term project on duplicate bug report detection using Siamese BERT and Graph Neural Networks.

## 📋 Project Overview

This project implements and compares two approaches for detecting duplicate bug reports:

1. **Baseline Models** (`train-baselines.py`): Siamese BERT architecture with various pretrained models:
   - CodeBERT (`microsoft/codebert-base`)
   - BERT (`bert-base-uncased`)
   - RoBERTa (`roberta-base`)

2. **Proposed Model** (`train-ours.py`): Siamese BERT + Graph Convolutional Network (GCN)
   - Combines semantic similarity from BERT with structural information from bug report graphs
   - Uses GCN to propagate information through the bug report network

## 🚀 Getting Started (Google Colab)

> **⚡ Quick Start**: For step-by-step Colab instructions with copy-paste cells, see [COLAB_QUICKSTART.md](COLAB_QUICKSTART.md)

### 1. Clone the Repository

```python
!git clone https://github.com/YOUR_USERNAME/cs588-codes.git
%cd cs588-codes
```

### 2. Install Dependencies

```python
!pip install -r requirements.txt
```

### 3. Download and Extract Dataset

The dataset is hosted on Google Drive and needs to be downloaded and extracted.

**Option A: Automated Setup (Recommended)**

```python
!python colab_setup.py
```

This script will:
- Install all dependencies
- Download the dataset from Google Drive
- Extract it to the `datasets/` directory
- Verify the installation
- Clean up temporary files

**Option B: Manual Setup**

```python
# Download dataset using gdown
!gdown https://drive.google.com/uc?id=1ef68tggrWtdrUv-Bb2l-E2yUVlo8yKgA

# Create datasets directory if it doesn't exist
!mkdir -p datasets

# Extract the dataset
!tar -xzf dataset.tar.gz -C datasets/

# Verify extraction
!ls -la datasets/

# Clean up
!rm dataset.tar.gz
```

### 4. Verify Dataset (Optional but Recommended)

```python
!python verify_dataset.py
```

This script will:
- Check if all required files exist
- Verify file sizes
- Confirm graph files are present for GNN training
- Provide a summary of dataset completeness

### 5. Run Training

#### Option A: Train Baseline Models (Siamese BERT)

```python
!python train-baselines.py
```

You can modify the baseline model and dataset in `train-baselines.py` (lines 444-456):

```python
baseline = "codebert-base"  # Options: "codebert-base", "bert-base-uncased", "roberta-base"
relative_path = "eclipse"    # Dataset name: "eclipse", "thunderbird", etc.
```

#### Option B: Train GNN-Enhanced Model (SBERT + GCN)

```python
!python train-ours.py
```

Modify parameters in `train-ours.py` (lines 373-376):

```python
D_DIR = "datasets/eclipse"  # Dataset directory
CSV = os.path.join(D_DIR, "tokenized_pairs_train_bert-base-uncased_50000.csv")
train_bertgnn(CSV, D_DIR, epochs=10, batch_size=64)
```

## 📁 Project Structure

```
cs588-codes/
├── README.md                          # This file
├── COLAB_QUICKSTART.md               # Quick start guide for Google Colab
├── requirements.txt                   # Python dependencies
├── colab_setup.py                    # Automated Colab setup script
├── verify_dataset.py                 # Dataset verification script
├── dataset.py                         # Dataset loading utilities
├── sbert.py                          # Siamese BERT model
├── gnn3.py                           # GCN model architecture
├── train-baselines.py                # Training script for baseline models
├── train-ours.py                     # Training script for GNN-enhanced model
├── datasets/                         # Dataset directory (created after extraction)
│   ├── eclipse/
│   │   ├── tokenized_pairs_train_*.csv
│   │   ├── graph_adj.npz
│   │   └── graph_adj_node_ids.npy
│   └── [other datasets]/
├── checkpoints/                      # Model checkpoints and results (auto-created)
│   └── {model_name}_{dataset_name}/
│       └── {YYYY-MM-DD_HH-MM-SS}/
│           ├── sbert_epoch_*.pth     # Model checkpoints per epoch
│           ├── best_model.pth        # Best model (GNN only)
│           ├── embedding_dict.pt     # Node embeddings
│           ├── test_results.txt      # Final test metrics
│           ├── train_step_losses.csv # Per-iteration training losses
│           ├── confusion_matrix_*.png # Confusion matrices
│           ├── val_samples_*.txt     # Validation samples
│           └── *.png                 # Training curves and plots
└── colab_setup.py                    # Automated Colab setup script
```

## 📊 Output Structure

Both training scripts automatically organize outputs in a structured manner:

### Baseline Models (`train-baselines.py`)

Outputs are saved to: `checkpoints/{model_name}_{dataset_name}/{YYYY-MM-DD_HH-MM-SS}/`

Example: `checkpoints/codebert-base_eclipse/2025-12-25_14-30-00/`

- **Model Checkpoints**: `sbert_epoch_{epoch}.pth` - PyTorch model state for each epoch
- **Embeddings**: `embedding_dict.pt` - Issue ID to embedding mapping
- **Metrics**:
  - `test_results.txt` - Final test set performance metrics
  - `train_step_losses.csv` - Per-iteration training losses
- **Visualizations**:
  - `loss_acc_pr_rc_f1_curve.png` - Training curves for all metrics
  - `confusion_matrix_last_epoch.png` - Training confusion matrix
  - `confusion_matrix_test.png` - Test set confusion matrix
- **Validation Logs**: `val_samples_epoch_{epoch}.txt` - Similarity scores for validation pairs

### GNN Model (`train-ours.py`)

Outputs are saved to: `checkpoints/{model_name}_{dataset_name}/{YYYY-MM-DD_HH-MM-SS}/`

Example: `checkpoints/sbert-gnn_eclipse/2025-12-25_14-30-00/`

- **Model Checkpoints**: `best_model.pth` - Best model based on validation F1
- **Visualizations**:
  - `epoch_loss.png`, `epoch_f1.png`, `epoch_acc.png` - Training curves
  - `epoch_precision.png`, `epoch_recall.png` - Precision/recall curves
  - `test_confusion_matrix.png` - Test set confusion matrix

## 🔧 Configuration

### Hyperparameters

#### Baseline Models (`train-baselines.py`)

```python
train(
    csv_path="path/to/data.csv",
    model_name="bert-base-uncased",  # Pretrained model
    batch_size=1024,                 # Batch size
    epochs=3,                        # Number of epochs
    lr=2e-5,                         # Learning rate
    dataset_name="thunderbird"       # Dataset identifier
)
```

#### GNN Model (`train-ours.py`)

```python
train_bertgnn(
    csv_path="path/to/data.csv",
    dataset_dir="datasets/eclipse",  # Must contain graph files
    batch_size=16,                   # Batch size
    epochs=5,                        # Number of epochs
    lr=2e-5                          # Learning rate
)
```

## 📈 Evaluation Metrics

Both models are evaluated using:

- **Accuracy**: Overall classification accuracy
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: Ratio of true positives to predicted positives
- **Recall**: Ratio of true positives to actual positives
- **Confusion Matrix**: Visual representation of classification results

The models use **adaptive thresholding** on the validation set to find the optimal similarity threshold for classification.

## 💾 Dataset Format

### Required Files

1. **Tokenized Pairs CSV**: `tokenized_pairs_train_{model}_{size}.csv`
   - Columns: `issue_id1`, `issue_id2`, `input_ids1`, `attention_mask1`, `input_ids2`, `attention_mask2`, `label`
   - Labels: 0 (not duplicate), 1 (duplicate)

2. **Graph Files** (for GNN model only):
   - `graph_adj.npz`: Sparse adjacency matrix
   - `graph_adj_node_ids.npy`: Node ID mapping

## 🔍 Model Architecture

### Siamese BERT

- Shared BERT encoder for both bug reports
- Projection layer to 128-dimensional embeddings
- Cosine similarity for duplicate detection
- Contrastive loss function (CosineEmbeddingLoss)

### SBERT + GCN

- Siamese BERT for semantic understanding
- 2-layer GCN for structural information
- Weighted fusion: `λ * BERT_emb + (1-λ) * GCN_emb`
- Persistent embedding table updated during training

## 📝 Requirements

See `requirements.txt` for the complete list. Main dependencies:

- `torch` - PyTorch framework
- `transformers` - Hugging Face transformers (BERT, RoBERTa, CodeBERT)
- `torch-geometric` - Graph neural network library
- `scikit-learn` - Metrics and evaluation
- `gdown` - Google Drive downloader
- `tqdm` - Progress bars
- `matplotlib` - Visualization
- `scipy` - Sparse matrix operations

## 🎯 Results

Training progress is displayed in real-time with progress bars showing:
- Batch-level: Loss, Accuracy, F1 Score
- Epoch-level: Complete metrics summary
- Validation: Adaptive threshold selection
- Test: Final evaluation with best model

Example output:
```
Epoch 1/5 | Val F1: 0.8234 | Val Prec: 0.8156 | Val Rec: 0.8313 | Next Th: 0.65
Test | F1: 0.8456 | Prec: 0.8389 | Rec: 0.8524 | Th: 0.65
```

## 🐛 Troubleshooting

### Out of Memory Errors

If you encounter CUDA OOM errors, try:
- Reducing `batch_size` in the training function
- Using gradient accumulation
- Clearing GPU cache: `torch.cuda.empty_cache()`

### Missing Dataset Files

Ensure you've:
1. Downloaded the dataset using `gdown`
2. Extracted to the correct location (`datasets/`)
3. Verified the CSV and graph files exist

### Import Errors

Make sure all dependencies are installed:
```python
!pip install -r requirements.txt --upgrade
```

## 📚 Citation

If you use this code in your research, please cite our work:

```bibtex
@misc{cs588-duplicate-detection,
  title={Duplicate Bug Report Detection using Siamese BERT and Graph Neural Networks},
  author={Your Name},
  year={2025},
  howpublished={CS 588 Term Project}
}
```

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the authors.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
