"""
One-click dataset download for testing.

This script downloads sample videos from public sources for testing
the accident detection system.
"""

import subprocess
import sys
from pathlib import Path

# Ensure data/clips exists
clips_dir = Path("data/clips")
clips_dir.mkdir(parents=True, exist_ok=True)

print("="*60)
print("QUICK DATASET DOWNLOAD")
print("="*60)

print("\n🎯 SIMPLEST METHOD: YouTube")
print("\nStep 1: Go to YouTube and search:")
print("  https://www.youtube.com/results?search_query=CCTV+car+accident+night")

print("\nStep 2: Copy a video URL (e.g., https://www.youtube.com/watch?v=ABC123)")

print("\nStep 3: Run this command:")
print('  python -m yt_dlp -f "best[height<=720]" -o "data/clips/test.mp4" "YOUR_URL"')

print("\n" + "="*60)
print("ALTERNATIVE: I'll show you how to download a specific dataset")
print("="*60)

choice = input("\nDo you want to:\n[1] See Kaggle setup instructions\n[2] See direct YouTube command\n[3] Exit\n\nChoice: ").strip()

if choice == "1":
    print("\n" + "="*60)
    print("KAGGLE DATASET DOWNLOAD")
    print("="*60)
    
    print("\n1. Install Kaggle:")
    print("   pip install kaggle")
    
    print("\n2. Get API Token:")
    print("   - Go to: https://www.kaggle.com/account")
    print("   - Scroll to API section")
    print("   - Click 'Create New Token'")
    print("   - Download kaggle.json")
    
    print("\n3. Place Token:")
    kaggle_dir = Path.home() / ".kaggle"
    print(f"   - Create: {kaggle_dir}")
    print(f"   - Move kaggle.json to: {kaggle_dir}")
    
    print("\n4. Search for datasets:")
    print("   https://www.kaggle.com/datasets")
    print("   Search: 'car accident' or 'traffic collision'")
    
    print("\n5. Download (example):")
    print("   kaggle datasets download -d USERNAME/DATASET -p data/raw_videos --unzip")

elif choice == "2":
    print("\n" + "="*60)
    print("YOUTUBE DOWNLOAD COMMAND")
    print("="*60)
    
    print("\n1. Open this URL in your browser:")
    print("   https://www.youtube.com/results?search_query=CCTV+car+accident")
    
    print("\n2. Pick a video and copy the URL")
    
    print("\n3. Run this command (replace YOUR_URL):")
    print('   python -m yt_dlp -f "best[height<=720]" -o "data/clips/test.mp4" "YOUR_URL"')
    
    print("\n4. Then test:")
    print("   python scripts/video_accident_detector.py --input data/clips/test.mp4")
    
    print("\nEXAMPLE:")
    url = input("\nPaste a YouTube URL here (or press Enter to skip): ").strip()
    
    if url:
        print(f"\nYour download command:")
        print(f'python -m yt_dlp -f "best[height<=720]" -o "data/clips/test.mp4" "{url}"')
        
        run_now = input("\nRun download now? (y/n): ").strip().lower()
        if run_now == 'y':
            try:
                subprocess.run([
                    sys.executable, "-m", "yt_dlp",
                    "-f", "best[height<=720]",
                    "-o", "data/clips/test.mp4",
                    url
                ], check=True)
                
                print("\n✓ Download complete!")
                print("\nRun detector:")
                print("  python scripts/video_accident_detector.py --input data/clips/test.mp4")
                
            except Exception as e:
                print(f"\n✗ Download failed: {e}")
                print("\nMake sure yt-dlp is installed:")
                print("  pip install yt-dlp")

else:
    print("\nExiting. See DIRECT_DOWNLOAD.md for detailed instructions.")

print("\n" + "="*60)
