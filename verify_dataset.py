"""
Dataset Verification Script
Checks if the dataset is properly downloaded and structured.
"""

import os
import sys
from pathlib import Path

def check_file(filepath, description):
    """Check if a file exists and print its status."""
    exists = os.path.exists(filepath)
    status = "✅" if exists else "❌"
    
    if exists:
        size = os.path.getsize(filepath)
        if size < 1024:
            size_str = f"{size} bytes"
        elif size < 1024 * 1024:
            size_str = f"{size/1024:.2f} KB"
        else:
            size_str = f"{size/(1024*1024):.2f} MB"
        print(f"{status} {description}: {filepath} ({size_str})")
    else:
        print(f"{status} {description}: {filepath} (NOT FOUND)")
    
    return exists

def verify_dataset(dataset_dir):
    """Verify a single dataset directory."""
    print(f"\n{'='*60}")
    print(f"📁 Checking dataset: {os.path.basename(dataset_dir)}")
    print('='*60)
    
    if not os.path.exists(dataset_dir):
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return False
    
    all_good = True
    
    # Check for CSV files
    csv_files = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]
    if csv_files:
        print(f"\n📊 Found {len(csv_files)} CSV file(s):")
        for csv_file in sorted(csv_files)[:5]:  # Show first 5
            check_file(os.path.join(dataset_dir, csv_file), f"  CSV")
        if len(csv_files) > 5:
            print(f"  ... and {len(csv_files) - 5} more CSV files")
    else:
        print("❌ No CSV files found!")
        all_good = False
    
    # Check for graph files (required for GNN model)
    print("\n🕸️  Graph files (required for GNN):")
    has_graph = check_file(
        os.path.join(dataset_dir, "graph_adj.npz"),
        "  Graph adjacency matrix"
    )
    has_nodes = check_file(
        os.path.join(dataset_dir, "graph_adj_node_ids.npy"),
        "  Node ID mapping"
    )
    
    if not has_graph or not has_nodes:
        print("⚠️  Warning: Graph files missing. GNN model (train-ours.py) won't work.")
        print("   Baseline models (train-baselines.py) will still work.")
    
    # Count total files
    all_files = os.listdir(dataset_dir)
    print(f"\n📦 Total files in dataset: {len(all_files)}")
    
    return all_good and has_graph and has_nodes

def main():
    """Main verification function."""
    print("\n" + "="*60)
    print("🔍 CS588 Dataset Verification")
    print("="*60)
    
    datasets_dir = "datasets"
    
    # Check if datasets directory exists
    if not os.path.exists(datasets_dir):
        print(f"\n❌ Datasets directory not found: {datasets_dir}")
        print("\n💡 Run the setup first:")
        print("   1. Download: gdown https://drive.google.com/uc?id=1ef68tggrWtdrUv-Bb2l-E2yUVlo8yKgA")
        print("   2. Extract: tar -xzf dataset.tar.gz -C datasets/")
        print("   OR run: python colab_setup.py")
        return False
    
    # Find all dataset subdirectories
    dataset_dirs = [
        os.path.join(datasets_dir, d)
        for d in os.listdir(datasets_dir)
        if os.path.isdir(os.path.join(datasets_dir, d))
    ]
    
    if not dataset_dirs:
        print(f"\n❌ No datasets found in {datasets_dir}/")
        return False
    
    print(f"\n✅ Found {len(dataset_dirs)} dataset(s)")
    
    # Verify each dataset
    results = []
    for dataset_dir in sorted(dataset_dirs):
        result = verify_dataset(dataset_dir)
        results.append((os.path.basename(dataset_dir), result))
    
    # Summary
    print("\n" + "="*60)
    print("📋 Summary")
    print("="*60)
    
    all_verified = True
    for dataset_name, is_complete in results:
        status = "✅ COMPLETE" if is_complete else "⚠️  PARTIAL"
        print(f"{status} - {dataset_name}")
        if not is_complete:
            all_verified = False
    
    print("\n" + "="*60)
    
    if all_verified:
        print("✅ All datasets are properly configured!")
        print("\n🚀 You can now run:")
        print("   - Baseline models: python train-baselines.py")
        print("   - GNN model: python train-ours.py")
    else:
        print("⚠️  Some datasets are incomplete")
        print("\n💡 You can still:")
        print("   - Run baseline models if CSV files exist")
        print("   - Run GNN models only on complete datasets")
    
    print("="*60 + "\n")
    
    return all_verified

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

