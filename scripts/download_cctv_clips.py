"""
Download and trim CCTV night accident clips for testing.

This script downloads YouTube videos containing CCTV footage of traffic accidents
(especially in low-light/night conditions) and trims them to short segments.

Usage:
    python scripts/download_cctv_clips.py
"""

import subprocess
import os
import csv
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw_videos"
CLIPS_DIR = BASE_DIR / "data" / "clips"
MANIFEST_PATH = BASE_DIR / "data" / "manifest.csv"

# Ensure directories exist
RAW_DIR.mkdir(parents=True, exist_ok=True)
CLIPS_DIR.mkdir(parents=True, exist_ok=True)

# Curated list of CCTV accident videos
# Format: (video_id, start_time, end_time, clip_name, description, lighting)
CLIPS_TO_DOWNLOAD = [
    # Note: These are example entries. Replace with actual YouTube video IDs
    # Format: (youtube_id, start_sec, end_sec, output_name, description, lighting_condition)
    # For demonstration, I'm using placeholder IDs - you should replace with real ones
    
    # Example structure (replace with real video IDs from your search):
    # ("dQw4w9WgXcQ", 10, 25, "intersection_night_01", "Intersection collision at night", "night"),
    # ("VIDEO_ID_2", 15, 35, "highway_night_01", "Highway collision low light", "night"),
]

# Since I cannot search YouTube directly, I'll create a template structure
# You should fill in actual video IDs from your YouTube searches

SEARCH_QUERIES = [
    "CCTV car accident night",
    "CCTV traffic accident night", 
    "surveillance camera car crash night",
    "intersection crash CCTV night",
    "traffic collision CCTV surveillance"
]

def download_video(video_id, output_path):
    """Download a YouTube video using yt-dlp."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = [
        "yt-dlp",
        "-f", "best[height<=720]",  # Max 720p for efficiency
        "-o", str(output_path),
        url
    ]
    
    print(f"Downloading {video_id}...")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  ✓ Downloaded successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed to download: {e.stderr}")
        return False

def trim_video(input_path, output_path, start_sec, end_sec):
    """Trim video using ffmpeg."""
    # Convert to H.264 720p for consistency
    cmd = [
        "ffmpeg",
        "-i", str(input_path),
        "-ss", str(start_sec),
        "-to", str(end_sec),
        "-vf", "scale=1280:720",
        "-r", "25",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-y",  # Overwrite without asking
        str(output_path)
    ]
    
    print(f"  Trimming {start_sec}s to {end_sec}s...")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  ✓ Trimmed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed to trim: {e.stderr}")
        return False

def create_manifest(clips_info):
    """Create a CSV manifest of downloaded clips."""
    with open(MANIFEST_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'start_sec', 'end_sec', 'label', 'description', 'lighting'])
        
        for clip in clips_info:
            writer.writerow([
                clip['filename'],
                clip['start_sec'],
                clip['end_sec'],
                'accident',
                clip['description'],
                clip['lighting']
            ])
    
    print(f"\n✓ Manifest created at {MANIFEST_PATH}")

def main():
    """Main download and processing pipeline."""
    print("=" * 60)
    print("CCTV Clip Downloader & Trimmer")
    print("=" * 60)
    
    if not CLIPS_TO_DOWNLOAD:
        print("\n⚠ WARNING: No clips configured!")
        print("\nTo use this script:")
        print("1. Search YouTube using these queries:")
        for query in SEARCH_QUERIES:
            print(f"   - {query}")
        print("\n2. Find suitable CCTV night accident videos")
        print("3. Add video IDs and timestamps to CLIPS_TO_DOWNLOAD list")
        print("4. Run this script again")
        print("\nExample entry:")
        print('("VIDEO_ID", 15, 35, "clip_name", "Description", "night"),')
        return
    
    clips_info = []
    success_count = 0
    
    for video_id, start_sec, end_sec, clip_name, description, lighting in CLIPS_TO_DOWNLOAD:
        print(f"\n[{len(clips_info) + 1}/{len(CLIPS_TO_DOWNLOAD)}] Processing: {clip_name}")
        
        # Download full video
        raw_video = RAW_DIR / f"{video_id}.mp4"
        if not raw_video.exists():
            if not download_video(video_id, raw_video):
                continue
        else:
            print(f"  ℹ Already downloaded")
        
        # Trim to clip
        clip_output = CLIPS_DIR / f"{clip_name}.mp4"
        if trim_video(raw_video, clip_output, start_sec, end_sec):
            clips_info.append({
                'filename': f"{clip_name}.mp4",
                'start_sec': start_sec,
                'end_sec': end_sec,
                'description': description,
                'lighting': lighting
            })
            success_count += 1
    
    # Create manifest
    if clips_info:
        create_manifest(clips_info)
    
    print("\n" + "=" * 60)
    print(f"✓ Successfully processed {success_count}/{len(CLIPS_TO_DOWNLOAD)} clips")
    print(f"✓ Clips saved to: {CLIPS_DIR}")
    print(f"✓ Manifest: {MANIFEST_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    main()
