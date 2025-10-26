"""
Download public accident detection datasets.

This script helps download publicly available traffic accident datasets
for testing the video accident detection system.
"""

import subprocess
import os
from pathlib import Path
import urllib.request
import zipfile
import json

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CLIPS_DIR = DATA_DIR / "clips"
RAW_DIR = DATA_DIR / "raw_videos"

# Ensure directories exist
CLIPS_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Public datasets available
DATASETS = {
    "1": {
        "name": "Sample CCTV Accidents (GitHub)",
        "description": "Small collection from GitHub repositories",
        "source": "github",
        "urls": [
            # These are example URLs - you should replace with actual dataset repos
            "https://github.com/example/accident-dataset/sample1.mp4",
            "https://github.com/example/accident-dataset/sample2.mp4",
        ],
        "size": "~50MB"
    },
    "2": {
        "name": "Kaggle - Car Crash Dataset",
        "description": "Download via Kaggle API (requires Kaggle account)",
        "source": "kaggle",
        "dataset_id": "USERNAME/car-crash-dataset",  # Replace with actual dataset
        "size": "~500MB",
        "requires_api": True
    },
    "3": {
        "name": "Sample Videos from Open Sources",
        "description": "Curated sample clips from various open sources",
        "source": "direct",
        "urls": [
            # Add direct download links here
        ],
        "size": "~100MB"
    }
}

def download_file(url, output_path):
    """Download a file from URL."""
    print(f"  Downloading: {url}")
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"  ✓ Downloaded: {output_path.name}")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False

def download_github_dataset(urls):
    """Download videos from GitHub URLs."""
    print("\nDownloading from GitHub...")
    success_count = 0
    
    for i, url in enumerate(urls, 1):
        filename = f"github_sample_{i}.mp4"
        output_path = CLIPS_DIR / filename
        
        if download_file(url, output_path):
            success_count += 1
    
    return success_count

def download_kaggle_dataset(dataset_id):
    """Download dataset from Kaggle using API."""
    print("\nDownloading from Kaggle...")
    print("This requires:")
    print("  1. Kaggle account")
    print("  2. Kaggle API token (kaggle.json)")
    print("  3. kaggle package installed")
    
    # Check if kaggle is installed
    try:
        import kaggle
    except ImportError:
        print("\n✗ Kaggle package not installed.")
        print("Install with: pip install kaggle")
        return False
    
    # Check for API credentials
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("\n✗ Kaggle API credentials not found.")
        print("Get your API token from: https://www.kaggle.com/account")
        print("Place kaggle.json in: ~/.kaggle/")
        return False
    
    try:
        # Download dataset
        output_dir = RAW_DIR / "kaggle_dataset"
        output_dir.mkdir(exist_ok=True)
        
        print(f"Downloading {dataset_id}...")
        subprocess.run([
            "kaggle", "datasets", "download",
            "-d", dataset_id,
            "-p", str(output_dir),
            "--unzip"
        ], check=True)
        
        print("✓ Dataset downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Failed to download: {e}")
        return False

def setup_kaggle_instructions():
    """Show instructions for setting up Kaggle."""
    print("\n" + "="*60)
    print("HOW TO DOWNLOAD FROM KAGGLE")
    print("="*60)
    print("\n1. Create Kaggle Account:")
    print("   Visit: https://www.kaggle.com/account/register")
    
    print("\n2. Get API Token:")
    print("   a. Go to: https://www.kaggle.com/account")
    print("   b. Scroll to 'API' section")
    print("   c. Click 'Create New Token'")
    print("   d. Save the downloaded kaggle.json file")
    
    print("\n3. Install Kaggle Package:")
    print("   pip install kaggle")
    
    print("\n4. Place API Token:")
    kaggle_dir = Path.home() / ".kaggle"
    print(f"   Create folder: {kaggle_dir}")
    print(f"   Move kaggle.json to: {kaggle_dir / 'kaggle.json'}")
    
    print("\n5. Find a Dataset:")
    print("   Visit: https://www.kaggle.com/datasets")
    print("   Search: 'car accident video' or 'traffic accident'")
    print("   Copy the dataset ID (username/dataset-name)")
    
    print("\n6. Download:")
    print("   kaggle datasets download -d USERNAME/DATASET-NAME")
    print("="*60)

def manual_youtube_download():
    """Guide for manual YouTube download."""
    print("\n" + "="*60)
    print("QUICK MANUAL DOWNLOAD VIA YOUTUBE")
    print("="*60)
    
    print("\nStep 1: Search YouTube")
    print("  Go to: https://www.youtube.com/results?search_query=CCTV+car+accident+night")
    
    print("\nStep 2: Pick a video and copy the URL")
    
    print("\nStep 3: Download with yt-dlp:")
    print('  python -m yt_dlp -f "best[height<=720]" -o "data/clips/%(title)s.%(ext)s" "YOUR_URL"')
    
    print("\nStep 4: Run detector:")
    print("  python scripts/video_accident_detector.py --input data/clips/YOUR_VIDEO.mp4")
    
    print("="*60)

def show_dataset_options():
    """Display available dataset options."""
    print("\n" + "="*60)
    print("PUBLIC ACCIDENT DETECTION DATASETS")
    print("="*60)
    
    for key, dataset in DATASETS.items():
        print(f"\n[{key}] {dataset['name']}")
        print(f"    Description: {dataset['description']}")
        print(f"    Source: {dataset['source']}")
        print(f"    Size: {dataset['size']}")
        if dataset.get('requires_api'):
            print(f"    ⚠️  Requires Kaggle API setup")
    
    print("\n[M] Manual YouTube download (guided)")
    print("[K] Setup Kaggle API (instructions)")
    print("[Q] Quit")
    print("="*60)

def main():
    print("="*60)
    print("PUBLIC DATASET DOWNLOADER")
    print("="*60)
    
    print("\nℹ️  RECOMMENDED APPROACH:")
    print("   Use YouTube for quick testing (fastest)")
    print("   Use Kaggle for larger labeled datasets")
    
    while True:
        show_dataset_options()
        
        choice = input("\nSelect option: ").strip().upper()
        
        if choice == "Q":
            print("Exiting...")
            break
        
        elif choice == "M":
            manual_youtube_download()
            input("\nPress Enter to continue...")
        
        elif choice == "K":
            setup_kaggle_instructions()
            input("\nPress Enter to continue...")
        
        elif choice in DATASETS:
            dataset = DATASETS[choice]
            
            if dataset['source'] == 'github':
                success = download_github_dataset(dataset['urls'])
                print(f"\n✓ Downloaded {success} files")
                
            elif dataset['source'] == 'kaggle':
                if dataset.get('requires_api'):
                    print("\nThis requires Kaggle API setup.")
                    setup_now = input("Show setup instructions? (y/n): ").strip().lower()
                    if setup_now == 'y':
                        setup_kaggle_instructions()
                    else:
                        download_kaggle_dataset(dataset['dataset_id'])
                        
            elif dataset['source'] == 'direct':
                success = download_github_dataset(dataset['urls'])
                print(f"\n✓ Downloaded {success} files")
            
            input("\nPress Enter to continue...")
        
        else:
            print("Invalid choice!")
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Check downloaded files:")
    print(f"   ls {CLIPS_DIR}")
    print("\n2. Run detector on a video:")
    print("   python scripts/video_accident_detector.py --input data/clips/VIDEO.mp4")
    print("\n3. Check results:")
    print("   ls results/")
    print("="*60)

if __name__ == "__main__":
    main()
