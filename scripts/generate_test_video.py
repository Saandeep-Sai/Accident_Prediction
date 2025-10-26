"""
Generate a simple synthetic test video for testing the accident detector.

This creates a short video with moving rectangles to simulate vehicles,
including a simulated collision scenario.
"""

import cv2
import numpy as np
from pathlib import Path

def create_test_video(output_path, duration_sec=10, fps=25):
    """
    Create a synthetic test video with moving objects.
    
    Args:
        output_path: Path to save the video
        duration_sec: Duration in seconds
        fps: Frames per second
    """
    width, height = 1280, 720
    total_frames = duration_sec * fps
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    print(f"Generating test video: {output_path}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Duration: {duration_sec}s @ {fps} FPS")
    print(f"  Total frames: {total_frames}")
    
    # Simulate two vehicles
    # Vehicle 1: Moving from left to right
    vehicle1_start_x = 100
    vehicle1_y = 300
    vehicle1_speed = 8
    
    # Vehicle 2: Moving from right to left (collision course)
    vehicle2_start_x = width - 100
    vehicle2_y = 320
    vehicle2_speed = 6
    
    for frame_num in range(total_frames):
        # Create dark background (simulate night)
        frame = np.ones((height, width, 3), dtype=np.uint8) * 30
        
        # Add some noise to simulate low-light
        noise = np.random.randint(-10, 10, (height, width, 3), dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Draw road
        cv2.rectangle(frame, (0, 250), (width, 450), (40, 40, 40), -1)
        cv2.line(frame, (0, 350), (width, 350), (60, 60, 60), 2)
        
        # Calculate vehicle positions
        vehicle1_x = vehicle1_start_x + (frame_num * vehicle1_speed)
        vehicle2_x = vehicle2_start_x - (frame_num * vehicle2_speed)
        
        # Vehicle 1 (red car)
        v1_rect = (int(vehicle1_x), vehicle1_y, 80, 50)
        cv2.rectangle(frame, v1_rect[:2], 
                     (v1_rect[0] + v1_rect[2], v1_rect[1] + v1_rect[3]),
                     (40, 40, 180), -1)  # Dark red
        cv2.rectangle(frame, v1_rect[:2], 
                     (v1_rect[0] + v1_rect[2], v1_rect[1] + v1_rect[3]),
                     (60, 60, 220), 2)
        
        # Vehicle 2 (blue car)
        v2_rect = (int(vehicle2_x), vehicle2_y, 80, 50)
        cv2.rectangle(frame, v2_rect[:2], 
                     (v2_rect[0] + v2_rect[2], v2_rect[1] + v2_rect[3]),
                     (180, 40, 40), -1)  # Dark blue
        cv2.rectangle(frame, v2_rect[:2], 
                     (v2_rect[0] + v2_rect[2], v2_rect[1] + v2_rect[3]),
                     (220, 60, 60), 2)
        
        # Simulate collision (vehicles overlap in the middle)
        collision_frame = int(total_frames * 0.5)
        if frame_num >= collision_frame and frame_num < collision_frame + fps:
            # Add collision effect
            cv2.circle(frame, (int((vehicle1_x + vehicle2_x) / 2), 
                              int((vehicle1_y + vehicle2_y) / 2) + 25),
                      30, (255, 255, 255), -1)
            cv2.putText(frame, "COLLISION!", (width//2 - 100, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        # Add frame number
        cv2.putText(frame, f"Frame: {frame_num}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        out.write(frame)
        
        if (frame_num + 1) % 25 == 0:
            print(f"  Generated {frame_num + 1}/{total_frames} frames...")
    
    out.release()
    print(f"✓ Test video created: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")

def main():
    # Create test videos directory
    test_dir = Path("data/clips")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate test video
    output_path = test_dir / "synthetic_test.mp4"
    create_test_video(output_path, duration_sec=10, fps=25)
    
    print("\n" + "="*60)
    print("Test video ready!")
    print("="*60)
    print("Run the detector with:")
    print(f"  python scripts/video_accident_detector.py --input {output_path}")
    print("="*60)

if __name__ == "__main__":
    main()
