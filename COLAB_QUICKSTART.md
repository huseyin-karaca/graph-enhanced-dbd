# 🚀 Google Colab Quick Start Guide

This guide will help you quickly set up and run the duplicate bug report detection models on Google Colab.

## 📝 One-Click Setup

Copy and paste the following cells into a new Google Colab notebook:

### Cell 1: Clone Repository and Install Dependencies

```python
# Clone the repository
!git clone https://github.com/YOUR_USERNAME/cs588-codes.git
%cd cs588-codes

# Install dependencies
!pip install -q -r requirements.txt

print("✅ Setup complete!")
```

### Cell 2: Download and Extract Dataset

```python
# Download dataset from Google Drive
!gdown https://drive.google.com/uc?id=1ef68tggrWtdrUv-Bb2l-E2yUVlo8yKgA

# Create datasets directory and extract
!mkdir -p datasets
!tar -xzf dataset.tar.gz -C datasets/

# Verify extraction
print("\n📁 Dataset structure:")
!ls -la datasets/

# Clean up
!rm dataset.tar.gz

print("\n✅ Dataset ready!")
```

### Cell 3: Verify Dataset

```python
import os

# Check dataset contents
for dataset in os.listdir("datasets"):
    dataset_path = os.path.join("datasets", dataset)
    if os.path.isdir(dataset_path):
        print(f"\n📁 {dataset}/")
        files = os.listdir(dataset_path)
        
        # Check for required files
        has_csv = any(f.endswith('.csv') for f in files)
        has_graph = 'graph_adj.npz' in files
        has_nodes = 'graph_adj_node_ids.npy' in files
        
        print(f"  {'✅' if has_csv else '❌'} CSV files")
        print(f"  {'✅' if has_graph else '❌'} Graph adjacency")
        print(f"  {'✅' if has_nodes else '❌'} Node IDs")
        print(f"  Total files: {len(files)}")
```

---

## 🏋️ Option A: Train Baseline Models

### Cell 4A: Configure Baseline Training

```python
# Configuration
BASELINE_MODEL = "codebert-base"  # Options: "codebert-base", "bert-base-uncased", "roberta-base"
DATASET_NAME = "eclipse"          # Your dataset name
BATCH_SIZE = 64                   # Adjust based on GPU memory
EPOCHS = 5
LEARNING_RATE = 2e-5

print(f"""
Configuration:
  Model: {BASELINE_MODEL}
  Dataset: {DATASET_NAME}
  Batch Size: {BATCH_SIZE}
  Epochs: {EPOCHS}
  Learning Rate: {LEARNING_RATE}
""")
```

### Cell 5A: Run Baseline Training

```python
import os
from train_baselines import train

# Model mapping
model_mapping = {
    "codebert-base": "microsoft/codebert-base",
    "bert-base-uncased": "bert-base-uncased",
    "roberta-base": "roberta-base"
}

model_name = model_mapping[BASELINE_MODEL]
dataset_dir = f"datasets/{DATASET_NAME}/"
csv_path = os.path.join(dataset_dir, f"tokenized_pairs_train_{BASELINE_MODEL}_50000.csv")

if os.path.exists(csv_path):
    print(f"✅ Starting training with {model_name}...\n")
    train(
        csv_path=csv_path,
        model_name=model_name,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        dataset_name=DATASET_NAME
    )
else:
    print(f"❌ Dataset not found: {csv_path}")
    print(f"Available files:")
    !ls -la {dataset_dir}
```

---

## 🏋️ Option B: Train GNN-Enhanced Model

### Cell 4B: Configure GNN Training

```python
# Configuration
DATASET_NAME = "eclipse"          # Your dataset name
BATCH_SIZE = 64                   # Adjust based on GPU memory
EPOCHS = 10
LEARNING_RATE = 2e-5

print(f"""
Configuration (SBERT + GCN):
  Model: Siamese BERT + GCN
  Dataset: {DATASET_NAME}
  Batch Size: {BATCH_SIZE}
  Epochs: {EPOCHS}
  Learning Rate: {LEARNING_RATE}
""")
```

### Cell 5B: Run GNN Training

```python
import os
from train_ours import train_bertgnn

# Paths
dataset_dir = f"datasets/{DATASET_NAME}"
csv_path = os.path.join(dataset_dir, "tokenized_pairs_train_bert-base-uncased_50000.csv")

# Verify required files
required_files = [
    csv_path,
    os.path.join(dataset_dir, "graph_adj.npz"),
    os.path.join(dataset_dir, "graph_adj_node_ids.npy")
]

all_exist = all(os.path.exists(f) for f in required_files)

if all_exist:
    print("✅ All required files found. Starting training...\n")
    train_bertgnn(
        csv_path=csv_path,
        dataset_dir=dataset_dir,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        model_name="sbert-gnn"
    )
else:
    print("❌ Missing required files:")
    for f in required_files:
        status = "✅" if os.path.exists(f) else "❌"
        print(f"  {status} {f}")
```

---

## 📊 View Results

### Cell 6: List Training Runs

```python
import os
from glob import glob

checkpoints_dir = "checkpoints"
if os.path.exists(checkpoints_dir):
    print("📊 Training Results\n" + "="*60)
    
    for model_dataset in sorted(os.listdir(checkpoints_dir)):
        model_dataset_path = os.path.join(checkpoints_dir, model_dataset)
        if os.path.isdir(model_dataset_path):
            print(f"\n📁 {model_dataset}/")
            runs = sorted(os.listdir(model_dataset_path), reverse=True)
            
            for run in runs[:3]:  # Show last 3 runs
                run_path = os.path.join(model_dataset_path, run)
                if os.path.isdir(run_path):
                    print(f"  ⏰ {run}")
                    
                    # Show test results if available
                    test_results = os.path.join(run_path, "test_results.txt")
                    if os.path.exists(test_results):
                        with open(test_results, 'r') as f:
                            content = f.read()
                            # Print first few lines
                            lines = content.split('\n')[:7]
                            for line in lines:
                                print(f"     {line}")
else:
    print("❌ No results yet. Train a model first!")
```

### Cell 7: Visualize Training Curves

```python
from IPython.display import Image, display
import os
from glob import glob

# Find the most recent run
def get_latest_run(checkpoints_dir):
    all_runs = []
    for model_dataset in os.listdir(checkpoints_dir):
        model_dataset_path = os.path.join(checkpoints_dir, model_dataset)
        if os.path.isdir(model_dataset_path):
            for run in os.listdir(model_dataset_path):
                run_path = os.path.join(model_dataset_path, run)
                if os.path.isdir(run_path):
                    all_runs.append((run_path, os.path.getmtime(run_path)))
    
    if all_runs:
        return sorted(all_runs, key=lambda x: x[1], reverse=True)[0][0]
    return None

latest_run = get_latest_run("checkpoints")

if latest_run:
    print(f"📊 Results from: {latest_run}\n")
    
    # Display all PNG files
    png_files = sorted(glob(os.path.join(latest_run, "*.png")))
    
    for png_file in png_files:
        print(f"\n{'='*60}")
        print(f"📈 {os.path.basename(png_file)}")
        print('='*60)
        display(Image(filename=png_file))
else:
    print("❌ No training runs found")
```

### Cell 8: Download Results (Optional)

```python
from google.colab import files
import shutil

# Create a zip of all checkpoints
if os.path.exists("checkpoints"):
    print("📦 Creating zip file...")
    shutil.make_archive("training_results", 'zip', "checkpoints")
    
    print("⬇️ Downloading...")
    files.download("training_results.zip")
    print("✅ Download complete!")
else:
    print("❌ No checkpoints found")
```

---

## 🎯 Quick Tips

### 🔧 Reduce Memory Usage

If you get Out of Memory (OOM) errors:

```python
import torch

# Clear GPU cache
torch.cuda.empty_cache()

# Use smaller batch size
BATCH_SIZE = 16  # or even 8
```

### 📊 Monitor GPU Usage

```python
# Check GPU status
!nvidia-smi

# Monitor during training
!watch -n 1 nvidia-smi  # Updates every second
```

### 💾 Save to Google Drive

Mount your Google Drive to save results permanently:

```python
from google.colab import drive
drive.mount('/content/drive')

# Copy results to Drive
!cp -r checkpoints /content/drive/MyDrive/cs588-results/
```

### ⚡ Use Better GPU

In Colab, go to: **Runtime → Change runtime type → GPU → T4 or A100**

---

## 🐛 Common Issues

### Issue: "Dataset not found"
**Solution**: Check the CSV filename matches exactly. List files with:
```python
!ls -la datasets/eclipse/
```

### Issue: "Module not found"
**Solution**: Reinstall dependencies:
```python
!pip install -r requirements.txt --upgrade --force-reinstall
```

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size and clear cache:
```python
torch.cuda.empty_cache()
BATCH_SIZE = 16
```

---

## 📧 Need Help?

- Check the main [README.md](README.md) for detailed documentation
- Open an issue on GitHub
- Review the training script configuration parameters

**Happy Training! 🚀**

