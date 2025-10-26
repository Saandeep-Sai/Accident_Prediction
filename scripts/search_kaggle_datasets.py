"""
Search and download Kaggle datasets for accident detection.

This script helps you:
1. Check Kaggle API authentication
2. Search for relevant datasets
3. Download datasets automatically
"""

import os
import sys
import json
from pathlib import Path

def check_kaggle_auth():
    """Check if Kaggle API credentials are configured."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if not kaggle_json.exists():
        print("\n" + "="*70)
        print("⚠️  KAGGLE API NOT CONFIGURED")
        print("="*70)
        print("\nTo use Kaggle datasets, you need to set up authentication:\n")
        print("1. Create a Kaggle account at: https://www.kaggle.com/account")
        print("2. Go to Account Settings → API section")
        print("3. Click 'Create New Token' to download kaggle.json")
        print("4. Place kaggle.json in:")
        print(f"   {kaggle_dir}")
        print("\nSteps:")
        print(f"   mkdir {kaggle_dir}")
        print(f"   move Downloads\\kaggle.json {kaggle_json}")
        print("\n" + "="*70)
        return False
    
    print("✓ Kaggle API credentials found!")
    return True

def search_datasets(keyword):
    """Search Kaggle for datasets."""
    import subprocess
    
    print(f"\n🔍 Searching Kaggle for: '{keyword}'")
    print("="*70)
    
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "list", "-s", keyword, "--max-size", "5GB"],
            capture_output=True,
            text=True,
            check=True
        )
        
        print(result.stdout)
        
        if "No datasets found" in result.stdout:
            print(f"\n⚠️  No datasets found for '{keyword}'")
            return []
        
        return result.stdout
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error searching: {e}")
        print(e.stderr)
        return []

def download_dataset(dataset_name, output_dir="data/raw_videos"):
    """Download a specific Kaggle dataset."""
    import subprocess
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📥 Downloading dataset: {dataset_name}")
    print(f"📁 Output directory: {output_path.absolute()}")
    print("="*70)
    
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_name, "-p", str(output_path), "--unzip"],
            capture_output=True,
            text=True,
            check=True
        )
        
        print(result.stdout)
        print("\n✓ Download complete!")
        print(f"\nFiles saved to: {output_path.absolute()}")
        
        # List downloaded files
        print("\nDownloaded files:")
        for file in output_path.rglob("*.*"):
            if file.is_file():
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  - {file.name} ({size_mb:.2f} MB)")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed: {e}")
        print(e.stderr)
        return False

def main():
    """Main search and download workflow."""
    
    print("\n" + "="*70)
    print("KAGGLE DATASET SEARCH & DOWNLOAD")
    print("="*70)
    
    # Check authentication
    if not check_kaggle_auth():
        sys.exit(1)
    
    # Search keywords
    keywords = [
        "car accident",
        "traffic accident",
        "road accident video",
        "cctv accident",
        "vehicle crash",
        "traffic collision"
    ]
    
    print("\n" + "="*70)
    print("SEARCH OPTIONS")
    print("="*70)
    
    for i, keyword in enumerate(keywords, 1):
        print(f"[{i}] Search for: {keyword}")
    print(f"[{len(keywords) + 1}] Custom search")
    print(f"[{len(keywords) + 2}] Download specific dataset")
    print(f"[0] Exit")
    
    choice = input("\nEnter your choice: ").strip()
    
    if choice == "0":
        print("\nExiting...")
        return
    
    if choice == str(len(keywords) + 1):
        # Custom search
        keyword = input("\nEnter search keyword: ").strip()
        if keyword:
            search_datasets(keyword)
    
    elif choice == str(len(keywords) + 2):
        # Direct download
        print("\nEnter dataset in format: USERNAME/DATASET-NAME")
        print("Example: awsaf49/accident-detection-dataset")
        dataset = input("\nDataset: ").strip()
        
        if dataset:
            download_dataset(dataset)
    
    elif choice.isdigit() and 1 <= int(choice) <= len(keywords):
        # Search predefined keyword
        keyword = keywords[int(choice) - 1]
        results = search_datasets(keyword)
        
        if results:
            print("\n" + "="*70)
            print("To download a dataset, copy the name (USER/DATASET) and run:")
            print("python scripts/search_kaggle_datasets.py")
            print("Then choose option for direct download.")
            print("="*70)
            
            download_choice = input("\nDo you want to download one now? (y/n): ").strip().lower()
            if download_choice == 'y':
                dataset = input("\nEnter dataset name (USER/DATASET): ").strip()
                if dataset:
                    download_dataset(dataset)
    
    else:
        print("\n❌ Invalid choice!")

if __name__ == "__main__":
    main()
