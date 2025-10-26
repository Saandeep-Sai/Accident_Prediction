"""
Convert image sequences from Kaggle datasets into test videos.

This script creates videos from the image frame datasets we downloaded,
allowing us to test the video_accident_detector.py pipeline.
"""

import cv2
import os
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm


def natural_sort_key(s):
    """Sort strings containing numbers in natural order."""
    import re
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', str(s))]


def create_video_from_images(image_folder, output_path, fps=25, max_frames=None, 
                            frame_size=None, include_pattern='*.jpg'):
    """
    Create a video from a sequence of images.
    
    Args:
        image_folder: Path to folder containing images
        output_path: Path to save output video
        fps: Frames per second for output video
        max_frames: Maximum number of frames to include (None = all)
        frame_size: Tuple (width, height) to resize frames, or None to use original size
        include_pattern: Pattern to match image files
    """
    image_folder = Path(image_folder)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get list of images
    image_files = sorted(image_folder.glob(include_pattern), key=natural_sort_key)
    
    if max_frames:
        image_files = image_files[:max_frames]
    
    if not image_files:
        print(f"No images found in {image_folder} matching pattern {include_pattern}")
        return False
    
    print(f"Found {len(image_files)} images in {image_folder}")
    
    # Read first image to get dimensions
    first_frame = cv2.imread(str(image_files[0]))
    if first_frame is None:
        print(f"Could not read first image: {image_files[0]}")
        return False
    
    # Determine frame size
    if frame_size:
        height, width = frame_size
    else:
        height, width = first_frame.shape[:2]
    
    print(f"Creating video: {output_path}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {len(image_files)}")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Failed to create video writer for {output_path}")
        return False
    
    # Write frames
    for img_path in tqdm(image_files, desc="Creating video"):
        frame = cv2.imread(str(img_path))
        
        if frame is None:
            print(f"Warning: Could not read {img_path}, skipping")
            continue
        
        # Resize if needed
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        
        out.write(frame)
    
    out.release()
    
    # Get file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    duration = len(image_files) / fps
    
    print(f"✓ Video created successfully!")
    print(f"  Output: {output_path}")
    print(f"  Size: {file_size_mb:.2f} MB")
    print(f"  Duration: {duration:.2f} seconds")
    
    return True


def create_sample_videos(base_dir='data'):
    """Create sample videos from both datasets."""
    base_path = Path(base_dir)
    
    samples = [
        {
            'name': 'accident_sample_1',
            'folder': base_path / 'videos' / 'Accident' / 'Accident',
            'frames': 300,  # ~12 seconds at 25 fps
            'start_idx': 0
        },
        {
            'name': 'accident_sample_2',
            'folder': base_path / 'videos' / 'Accident' / 'Accident',
            'frames': 300,
            'start_idx': 1000
        },
        {
            'name': 'accident_sample_3',
            'folder': base_path / 'videos' / 'Accident' / 'Accident',
            'frames': 300,
            'start_idx': 3000
        },
    ]
    
    created_videos = []
    
    for sample in samples:
        folder = sample['folder']
        if not folder.exists():
            print(f"Skipping {sample['name']}: folder not found: {folder}")
            continue
        
        # Get subset of images
        all_images = sorted(folder.glob('*.jpg'), key=natural_sort_key)
        start = sample['start_idx']
        end = start + sample['frames']
        
        if start >= len(all_images):
            print(f"Skipping {sample['name']}: start index {start} >= total images {len(all_images)}")
            continue
        
        # Create temporary folder with subset
        temp_folder = base_path / 'temp_frames' / sample['name']
        temp_folder.mkdir(parents=True, exist_ok=True)
        
        print(f"\nPreparing {sample['name']}...")
        for i, img_idx in enumerate(range(start, min(end, len(all_images)))):
            src = all_images[img_idx]
            dst = temp_folder / f"{i:04d}.jpg"
            # Create symlink or copy
            try:
                if not dst.exists():
                    import shutil
                    shutil.copy2(src, dst)
            except Exception as e:
                print(f"Error copying {src}: {e}")
                continue
        
        output_path = base_path / 'clips' / f"{sample['name']}.mp4"
        
        if create_video_from_images(temp_folder, output_path, fps=25, frame_size=(1280, 720)):
            created_videos.append(output_path)
        
        # Cleanup temp folder
        import shutil
        try:
            shutil.rmtree(temp_folder)
        except:
            pass
    
    return created_videos


def main():
    parser = argparse.ArgumentParser(description='Convert image sequences to videos')
    parser.add_argument('--input', '-i', help='Input folder containing images')
    parser.add_argument('--output', '-o', help='Output video path')
    parser.add_argument('--fps', type=int, default=25, help='Frames per second (default: 25)')
    parser.add_argument('--max-frames', type=int, help='Maximum number of frames to include')
    parser.add_argument('--width', type=int, help='Output width (default: original)')
    parser.add_argument('--height', type=int, help='Output height (default: original)')
    parser.add_argument('--create-samples', action='store_true', 
                       help='Create sample videos from downloaded datasets')
    
    args = parser.parse_args()
    
    if args.create_samples:
        print("Creating sample videos from datasets...")
        videos = create_sample_videos()
        print(f"\n{'='*60}")
        print(f"Created {len(videos)} sample videos:")
        for video in videos:
            print(f"  - {video}")
        print("\nYou can now test these with:")
        print("  python scripts/video_accident_detector.py --input data/clips/accident_sample_1.mp4")
    
    elif args.input and args.output:
        frame_size = None
        if args.width and args.height:
            frame_size = (args.height, args.width)
        
        create_video_from_images(
            args.input, 
            args.output, 
            fps=args.fps,
            max_frames=args.max_frames,
            frame_size=frame_size
        )
    
    else:
        parser.print_help()
        print("\nExamples:")
        print("  # Create sample videos from datasets:")
        print("  python scripts/images_to_video.py --create-samples")
        print()
        print("  # Convert specific folder:")
        print("  python scripts/images_to_video.py -i data/videos/Accident/Accident -o data/clips/test.mp4")
        print("  python scripts/images_to_video.py -i data/videos/Accident/Accident -o data/clips/test.mp4 --max-frames 250 --fps 25")


if __name__ == '__main__':
    main()
