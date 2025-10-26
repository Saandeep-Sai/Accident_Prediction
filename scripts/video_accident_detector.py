"""
Video-based Accident Detection System with Low-Light Enhancement

This script processes CCTV footage (especially night/low-light conditions) to detect
traffic accidents using:
1. Image enhancement (CLAHE + gamma correction)
2. Object detection (YOLOv8)
3. Object tracking (Norfair)
4. Accident heuristics (collision detection + sudden stops)

Usage:
    python scripts/video_accident_detector.py --input path/to/video.mp4 [options]

Examples:
    # Basic usage
    python scripts/video_accident_detector.py --input data/clips/test.mp4
    
    # With custom output
    python scripts/video_accident_detector.py --input data/clips/test.mp4 --output results/output.mp4
    
    # Adjust sensitivity
    python scripts/video_accident_detector.py --input data/clips/test.mp4 --collision-threshold 0.3
"""

import argparse
import cv2
import numpy as np
import json
from pathlib import Path
from ultralytics import YOLO
from norfair import Detection, Tracker, Video
from norfair.distances import mean_euclidean
import warnings
warnings.filterwarnings('ignore')

# Configuration
class Config:
    # Enhancement parameters
    CLAHE_CLIP_LIMIT = 2.0
    CLAHE_TILE_SIZE = (8, 8)
    GAMMA = 1.5  # Gamma correction for low-light boost
    
    # Detection parameters
    YOLO_MODEL = "yolov8n.pt"  # Nano model for speed (use yolov8m.pt for better accuracy)
    CONFIDENCE_THRESHOLD = 0.3  # Lower for low-light
    IOU_THRESHOLD = 0.5
    
    # Classes to detect (COCO dataset)
    TARGET_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
    
    # Tracking parameters
    DISTANCE_THRESHOLD = 100  # pixels
    HIT_COUNTER_MAX = 10
    INITIALIZATION_DELAY = 3
    
    # Accident heuristics
    COLLISION_IOU_THRESHOLD = 0.15  # Bounding box overlap
    SUDDEN_STOP_THRESHOLD = 5.0  # pixels/frame velocity drop
    MIN_VELOCITY_FOR_ACCIDENT = 3.0  # minimum speed to consider
    ACCIDENT_CONFIRMATION_FRAMES = 5  # frames to confirm accident

class VideoEnhancer:
    """Low-light video enhancement using CLAHE and gamma correction."""
    
    def __init__(self, clahe_clip=2.0, clahe_tile_size=(8, 8), gamma=1.5):
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile_size)
        self.gamma = gamma
        self.gamma_table = self._build_gamma_table(gamma)
    
    def _build_gamma_table(self, gamma):
        """Build lookup table for gamma correction."""
        return np.array([((i / 255.0) ** (1.0 / gamma)) * 255 
                        for i in range(256)]).astype("uint8")
    
    def enhance(self, frame):
        """Apply enhancement to a single frame."""
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        l_enhanced = self.clahe.apply(l)
        
        # Apply gamma correction
        l_enhanced = cv2.LUT(l_enhanced, self.gamma_table)
        
        # Merge back
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Optional: slight denoising
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 3, 3, 7, 21)
        
        return enhanced

class AccidentDetector:
    """Detect accidents using collision and sudden-stop heuristics."""
    
    def __init__(self, config):
        self.config = config
        self.accident_events = []
        self.track_history = {}  # track_id -> list of (x, y, frame_num)
        self.collision_candidates = {}  # (id1, id2) -> frame_count
        
    def update_track_history(self, tracked_objects, frame_num):
        """Update position history for each tracked object."""
        for obj in tracked_objects:
            track_id = obj.id
            centroid = obj.estimate[0]  # (x, y)
            
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            
            self.track_history[track_id].append({
                'centroid': centroid,
                'bbox': obj.data['bbox'],
                'frame': frame_num
            })
            
            # Keep only recent history (last 30 frames)
            if len(self.track_history[track_id]) > 30:
                self.track_history[track_id].pop(0)
    
    def calculate_velocity(self, track_id):
        """Calculate velocity (pixels/frame) for a track."""
        if track_id not in self.track_history or len(self.track_history[track_id]) < 2:
            return 0.0
        
        history = self.track_history[track_id]
        recent = history[-5:]  # Last 5 frames
        
        if len(recent) < 2:
            return 0.0
        
        velocities = []
        for i in range(1, len(recent)):
            p1 = recent[i-1]['centroid']
            p2 = recent[i]['centroid']
            dist = np.linalg.norm(np.array(p2) - np.array(p1))
            velocities.append(dist)
        
        return np.mean(velocities) if velocities else 0.0
    
    def bbox_iou(self, bbox1, bbox2):
        """Calculate IoU between two bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Intersection
        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)
        
        inter_area = max(0, xi_max - xi_min) * max(0, yi_max - yi_min)
        
        # Union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def detect_accidents(self, tracked_objects, frame_num):
        """Detect accidents using heuristics."""
        accidents = []
        
        # Check all pairs of tracked objects
        for i, obj1 in enumerate(tracked_objects):
            for obj2 in tracked_objects[i+1:]:
                id1, id2 = obj1.id, obj2.id
                bbox1 = obj1.data.get('bbox')
                bbox2 = obj2.data.get('bbox')
                
                if bbox1 is None or bbox2 is None:
                    continue
                
                # Check collision (high IoU)
                iou = self.bbox_iou(bbox1, bbox2)
                
                if iou > self.config.COLLISION_IOU_THRESHOLD:
                    vel1 = self.calculate_velocity(id1)
                    vel2 = self.calculate_velocity(id2)
                    
                    # Check if at least one was moving
                    if vel1 > self.config.MIN_VELOCITY_FOR_ACCIDENT or vel2 > self.config.MIN_VELOCITY_FOR_ACCIDENT:
                        pair = tuple(sorted([id1, id2]))
                        
                        if pair not in self.collision_candidates:
                            self.collision_candidates[pair] = 0
                        
                        self.collision_candidates[pair] += 1
                        
                        # Confirm accident if collision persists
                        if self.collision_candidates[pair] >= self.config.ACCIDENT_CONFIRMATION_FRAMES:
                            accidents.append({
                                'frame': frame_num,
                                'type': 'collision',
                                'objects': [id1, id2],
                                'iou': iou,
                                'velocities': [vel1, vel2]
                            })
                            # Reset to avoid duplicate detections
                            self.collision_candidates[pair] = 0
        
        # Check sudden stops (for single vehicles)
        for obj in tracked_objects:
            track_id = obj.id
            if track_id not in self.track_history or len(self.track_history[track_id]) < 10:
                continue
            
            history = self.track_history[track_id]
            
            # Compare velocity 10 frames ago vs now
            if len(history) >= 10:
                old_frames = history[-10:-5]
                recent_frames = history[-5:]
                
                old_vels = []
                for i in range(1, len(old_frames)):
                    p1 = old_frames[i-1]['centroid']
                    p2 = old_frames[i]['centroid']
                    old_vels.append(np.linalg.norm(np.array(p2) - np.array(p1)))
                
                recent_vels = []
                for i in range(1, len(recent_frames)):
                    p1 = recent_frames[i-1]['centroid']
                    p2 = recent_frames[i]['centroid']
                    recent_vels.append(np.linalg.norm(np.array(p2) - np.array(p1)))
                
                if old_vels and recent_vels:
                    old_vel = np.mean(old_vels)
                    recent_vel = np.mean(recent_vels)
                    velocity_drop = old_vel - recent_vel
                    
                    if old_vel > self.config.MIN_VELOCITY_FOR_ACCIDENT and velocity_drop > self.config.SUDDEN_STOP_THRESHOLD:
                        accidents.append({
                            'frame': frame_num,
                            'type': 'sudden_stop',
                            'object': track_id,
                            'velocity_drop': velocity_drop
                        })
        
        if accidents:
            self.accident_events.extend(accidents)
        
        return len(accidents) > 0

def yolo_detections_to_norfair(detections_yolo, track_points="centroid"):
    """Convert YOLO detections to Norfair format."""
    norfair_detections = []
    
    for det in detections_yolo:
        bbox = det.boxes.xyxy[0].cpu().numpy()
        confidence = float(det.boxes.conf[0])
        class_id = int(det.boxes.cls[0])
        
        # Calculate centroid
        centroid = np.array([
            [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        ])
        
        norfair_detections.append(
            Detection(
                points=centroid,
                scores=np.array([confidence]),
                data={'bbox': bbox, 'class_id': class_id}
            )
        )
    
    return norfair_detections

def process_video(input_path, output_path, config):
    """Main video processing pipeline."""
    print(f"Processing video: {input_path}")
    
    # Initialize components
    enhancer = VideoEnhancer(
        clahe_clip=config.CLAHE_CLIP_LIMIT,
        clahe_tile_size=config.CLAHE_TILE_SIZE,
        gamma=config.GAMMA
    )
    
    print("Loading YOLO model...")
    model = YOLO(config.YOLO_MODEL)
    
    tracker = Tracker(
        distance_function=mean_euclidean,
        distance_threshold=config.DISTANCE_THRESHOLD,
        hit_counter_max=config.HIT_COUNTER_MAX,
        initialization_delay=config.INITIALIZATION_DELAY,
    )
    
    accident_detector = AccidentDetector(config)
    
    # Open video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    frame_num = 0
    accident_detected = False
    accident_frames = []
    
    print(f"Processing {total_frames} frames at {fps} FPS...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        
        # Enhance frame for low-light
        enhanced_frame = enhancer.enhance(frame)
        
        # Detect objects
        results = model(enhanced_frame, conf=config.CONFIDENCE_THRESHOLD, 
                       iou=config.IOU_THRESHOLD, classes=config.TARGET_CLASSES, 
                       verbose=False)
        
        # Convert to Norfair detections
        detections = yolo_detections_to_norfair(results[0])
        
        # Track objects
        tracked_objects = tracker.update(detections=detections)
        
        # Update tracking history
        accident_detector.update_track_history(tracked_objects, frame_num)
        
        # Detect accidents
        has_accident = accident_detector.detect_accidents(tracked_objects, frame_num)
        
        if has_accident:
            accident_detected = True
            accident_frames.append(frame_num)
        
        # Draw on frame
        output_frame = enhanced_frame.copy()
        
        # Draw tracked objects
        for obj in tracked_objects:
            bbox = obj.data.get('bbox')
            if bbox is not None:
                x1, y1, x2, y2 = map(int, bbox)
                color = (0, 255, 0) if not has_accident else (0, 0, 255)
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(output_frame, f"ID:{obj.id}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw accident indicator
        if has_accident:
            cv2.putText(output_frame, "ACCIDENT DETECTED!", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        # Draw frame number and status
        cv2.putText(output_frame, f"Frame: {frame_num}/{total_frames}", (10, height-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        out.write(output_frame)
        
        if frame_num % 30 == 0:
            print(f"  Processed {frame_num}/{total_frames} frames...")
    
    cap.release()
    out.release()
    
    print(f"\n✓ Processing complete!")
    print(f"  Output video: {output_path}")
    print(f"  Accident detected: {accident_detected}")
    print(f"  Accident events: {len(accident_detector.accident_events)}")
    
    return {
        'accident_detected': accident_detected,
        'accident_frames': accident_frames,
        'accident_events': accident_detector.accident_events,
        'total_frames': total_frames,
        'fps': fps
    }

def save_results(results, output_json):
    """Save detection results to JSON."""
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to: {output_json}")

def main():
    parser = argparse.ArgumentParser(description="Video-based Accident Detection System")
    parser.add_argument('--input', '-i', required=True, help="Input video path")
    parser.add_argument('--output', '-o', help="Output video path (default: results/output.mp4)")
    parser.add_argument('--collision-threshold', type=float, default=0.15, 
                       help="IoU threshold for collision detection (default: 0.15)")
    parser.add_argument('--gamma', type=float, default=1.5,
                       help="Gamma correction value for low-light enhancement (default: 1.5)")
    parser.add_argument('--confidence', type=float, default=0.3,
                       help="YOLO confidence threshold (default: 0.3)")
    
    args = parser.parse_args()
    
    # Setup paths
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{input_path.stem}_detected.mp4"
    
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_json = output_path.with_suffix('.json')
    
    # Update config from args
    config = Config()
    config.COLLISION_IOU_THRESHOLD = args.collision_threshold
    config.GAMMA = args.gamma
    config.CONFIDENCE_THRESHOLD = args.confidence
    
    # Process video
    try:
        results = process_video(input_path, output_path, config)
        save_results(results, output_json)
        
        print("\n" + "="*60)
        print("DETECTION SUMMARY")
        print("="*60)
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        print(f"Accident Detected: {'YES' if results['accident_detected'] else 'NO'}")
        print(f"Total Events: {len(results['accident_events'])}")
        if results['accident_frames']:
            print(f"Accident Frames: {results['accident_frames'][:10]}...")  # First 10
        print("="*60)
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
