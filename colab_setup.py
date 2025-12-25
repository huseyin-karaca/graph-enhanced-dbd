"""
Google Colab Setup Script for CS588 Duplicate Bug Report Detection
Run this script in a Colab cell to set up the environment and download the dataset.
"""

import os
import sys
import subprocess

def run_command(cmd, description=""):
    """Run a shell command and print its output."""
    if description:
        print(f"\n{'='*60}")
        print(f"📦 {description}")
        print('='*60)
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.returncode

def setup_environment():
    """Complete setup for Google Colab environment."""
    
    print("\n" + "="*60)
    print("🚀 CS588 Duplicate Bug Report Detection - Setup")
    print("="*60)
    
    # 1. Install dependencies
    if run_command("pip install -q -r requirements.txt", 
                   "Installing dependencies...") != 0:
        print("❌ Failed to install dependencies")
        return False
    print("✅ Dependencies installed successfully")
    
    # 2. Download dataset
    dataset_url = "https://drive.google.com/uc?id=1ef68tggrWtdrUv-Bb2l-E2yUVlo8yKgA"
    if run_command(f"gdown {dataset_url}", 
                   "Downloading dataset from Google Drive...") != 0:
        print("❌ Failed to download dataset")
        return False
    print("✅ Dataset downloaded successfully")
    
    # 3. Create datasets directory
    os.makedirs("datasets", exist_ok=True)
    print("✅ Created datasets directory")
    
    # 4. Extract dataset
    if run_command("tar -xzf dataset.tar.gz -C datasets/", 
                   "Extracting dataset...") != 0:
        print("❌ Failed to extract dataset")
        return False
    print("✅ Dataset extracted successfully")
    
    # 5. Verify extraction
    print("\n" + "="*60)
    print("📂 Verifying dataset structure...")
    print("="*60)
    
    datasets_dir = "datasets"
    if os.path.exists(datasets_dir):
        for item in os.listdir(datasets_dir):
            item_path = os.path.join(datasets_dir, item)
            if os.path.isdir(item_path):
                print(f"\n📁 {item}/")
                files = os.listdir(item_path)
                for f in sorted(files)[:10]:  # Show first 10 files
                    file_path = os.path.join(item_path, f)
                    size = os.path.getsize(file_path)
                    size_str = f"{size:,} bytes" if size < 1024*1024 else f"{size/(1024*1024):.2f} MB"
                    print(f"   - {f} ({size_str})")
                if len(files) > 10:
                    print(f"   ... and {len(files)-10} more files")
    
    # 6. Clean up
    if os.path.exists("dataset.tar.gz"):
        os.remove("dataset.tar.gz")
        print("\n✅ Cleaned up temporary files")
    
    print("\n" + "="*60)
    print("✨ Setup completed successfully!")
    print("="*60)
    print("\n📚 Next steps:")
    print("   1. To train baseline models: !python train-baselines.py")
    print("   2. To train GNN model: !python train-ours.py")
    print("   3. Check 'checkpoints/' directory for results")
    print("\n💡 Tip: Modify hyperparameters directly in the training scripts")
    print("="*60 + "\n")
    
    return True

if __name__ == "__main__":
    success = setup_environment()
    sys.exit(0 if success else 1)

