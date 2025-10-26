"""
Enhanced video accident prediction with object detection.

This version detects vehicles and pedestrians, then classifies if accidents are occurring,
drawing bounding boxes around detected entities.
"""

import os
import sys

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    if sys.stderr.encoding != 'utf-8':
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm

try:
    from tensorflow import keras
except ImportError:
    print("TensorFlow not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow"])
    from tensorflow import keras

try:
    from ultralytics import YOLO
except ImportError:
    print("Ultralytics YOLO not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
    from ultralytics import YOLO


class VideoAccidentDetector:
    """Detect accidents in videos with object detection and bounding boxes."""
    
    def __init__(self, classifier_path, yolo_model='yolov8n.pt', img_size=224, threshold=0.5):
        """
        Initialize detector with both object detector and accident classifier.
        
        Args:
            classifier_path: Path to accident classifier model
            yolo_model: YOLOv8 model size (n/s/m/l/x)
            img_size: Image size for classifier
            threshold: Classification threshold
        """
        self.img_size = img_size
        self.threshold = threshold
        
        # Load accident classifier
        print(f"Loading accident classifier from: {classifier_path}")
        self.classifier = keras.models.load_model(classifier_path)
        print("[OK] Classifier loaded")
        
        # Load YOLO object detector
        print(f"Loading YOLO detector: {yolo_model}")
        self.detector = YOLO(yolo_model)
        print("[OK] YOLO detector loaded")
        
        # Classes of interest (vehicles and people)
        self.target_classes = [
            0,   # person
            1,   # bicycle
            2,   # car
            3,   # motorcycle
            5,   # bus
            7,   # truck
        ]
        
        self.class_names = {
            0: 'person',
            1: 'bicycle', 
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }
    
    def classify_frame(self, frame):
        """Classify if frame contains accident."""
        # Preprocess frame for classifier
        img = cv2.resize(frame, (self.img_size, self.img_size))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Predict
        pred = self.classifier.predict(img, verbose=0)[0][0]
        
        # Determine class
        is_accident = pred >= self.threshold
        
        return {
            'prediction': 'Accident' if is_accident else 'Safe',
            'probability': float(pred) if is_accident else float(1 - pred),
            'raw_score': float(pred)
        }
    
    def detect_objects(self, frame):
        """Detect vehicles and people in frame."""
        results = self.detector(frame, verbose=False)[0]
        
        detections = []
        for box in results.boxes:
            cls = int(box.cls[0])
            if cls in self.target_classes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                
                detections.append({
                    'class': cls,
                    'class_name': self.class_names.get(cls, 'unknown'),
                    'confidence': conf,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                })
        
        return detections
    
    def draw_detections(self, frame, detections, is_accident, accident_prob):
        """Draw bounding boxes and labels on frame."""
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        
        # Draw bounding boxes for detected objects
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cls_name = det['class_name']
            conf = det['confidence']
            
            # Color: Red if accident detected, Green if safe
            color = (0, 0, 255) if is_accident else (0, 255, 0)
            
            # Draw bounding box
            thickness = 3 if is_accident else 2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label background
            label = f"{cls_name}: {conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)
            
            # Draw label text
            cv2.putText(annotated, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Draw status banner at top
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
        annotated = cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0)
        
        # Status text
        status = "ACCIDENT DETECTED" if is_accident else "SAFE"
        status_color = (0, 0, 255) if is_accident else (0, 255, 0)
        cv2.putText(annotated, status, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3, cv2.LINE_AA)
        
        # Confidence
        conf_text = f"Confidence: {accident_prob:.1%}"
        cv2.putText(annotated, conf_text, (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Object count
        obj_text = f"Objects: {len(detections)}"
        cv2.putText(annotated, obj_text, (w - 250, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Warning banner at bottom if accident
        if is_accident:
            warning_text = "!!! ACCIDENT DETECTED !!!"
            (text_w, text_h), _ = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            text_x = (w - text_w) // 2
            
            cv2.rectangle(annotated, (0, h - 80), (w, h), (0, 0, 255), -1)
            cv2.putText(annotated, warning_text, (text_x, h - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
        
        return annotated
    
    def process_video(self, video_path, output_path=None, json_path=None, 
                     sample_rate=1, aggregate_method='max'):
        """
        Process video with object detection and accident classification.
        
        Args:
            video_path: Path to input video
            output_path: Path to save annotated video (optional)
            json_path: Path to save results JSON (optional)
            sample_rate: Process every Nth frame
            aggregate_method: 'max', 'mean', or 'vote'
        """
        print(f"\n{'='*60}")
        print(f"Processing video: {video_path}")
        print(f"{'='*60}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Total frames: {total_frames}")
        print(f"  Sample rate: every {sample_rate} frame(s)")
        
        # Setup output video
        out = None
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Try H.264 codec for better compatibility
            codecs_to_try = [
                ('avc1', 'H.264 (avc1)'),
                ('H264', 'H.264 (H264)'),
                ('X264', 'H.264 (X264)'),
                ('mp4v', 'MPEG-4 (mp4v)')
            ]
            
            for fourcc_str, codec_name in codecs_to_try:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                    test_out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                    if test_out.isOpened():
                        out = test_out
                        print(f"  Using codec: {codec_name}")
                        break
                    else:
                        test_out.release()
                except:
                    pass
            
            if out:
                print(f"  Output video: {output_path}")
        
        # Process frames
        frame_results = []
        accident_frames = []
        frame_idx = 0
        
        pbar = tqdm(total=total_frames, desc="Processing frames", unit="frame")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every Nth frame
            if frame_idx % sample_rate == 0:
                # Detect objects
                detections = self.detect_objects(frame)
                
                # Classify accident
                classification = self.classify_frame(frame)
                is_accident = classification['prediction'] == 'Accident'
                
                # Store results
                frame_results.append({
                    'frame': frame_idx,
                    'time': frame_idx / fps if fps > 0 else 0,
                    'classification': classification,
                    'detections': detections,
                    'object_count': len(detections)
                })
                
                if is_accident:
                    accident_frames.append(frame_idx)
                
                # Annotate frame
                if out and out.isOpened():
                    annotated = self.draw_detections(
                        frame, detections, is_accident, classification['probability']
                    )
                    out.write(annotated)
            elif out and out.isOpened():
                # Write unprocessed frames to maintain timing
                out.write(frame)
            
            frame_idx += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        if out:
            out.release()
        
        # Aggregate predictions
        predictions = [r['classification']['raw_score'] for r in frame_results]
        
        if aggregate_method == 'max':
            final_prob = max(predictions)
        elif aggregate_method == 'mean':
            final_prob = sum(predictions) / len(predictions)
        else:  # vote
            votes = sum(1 for p in predictions if p >= self.threshold)
            final_prob = votes / len(predictions)
        
        is_accident = final_prob >= self.threshold
        
        # Compile results
        results = {
            'video_info': {
                'path': str(video_path),
                'width': width,
                'height': height,
                'fps': fps,
                'total_frames': total_frames,
                'duration_seconds': total_frames / fps if fps > 0 else 0
            },
            'prediction': {
                'prediction': 'Accident' if is_accident else 'Safe',
                'confidence': final_prob * 100,
                'accident_probability': final_prob * 100,
                'threshold': self.threshold
            },
            'statistics': {
                'total_frames': len(frame_results),
                'accident_frames': len(accident_frames),
                'accident_percentage': (len(accident_frames) / len(frame_results) * 100) if frame_results else 0,
                'mean_probability': sum(predictions) / len(predictions) * 100 if predictions else 0,
                'min_probability': min(predictions) * 100 if predictions else 0,
                'max_probability': max(predictions) * 100 if predictions else 0,
                'std_probability': np.std(predictions) * 100 if predictions else 0,
                'total_objects_detected': sum(r['object_count'] for r in frame_results),
                'avg_objects_per_frame': sum(r['object_count'] for r in frame_results) / len(frame_results) if frame_results else 0
            },
            'processing': {
                'sample_rate': sample_rate,
                'aggregate_method': aggregate_method
            },
            'frame_details': frame_results
        }
        
        # Save JSON
        if json_path:
            json_path = Path(json_path)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n[OK] Results saved to: {json_path}")
        
        print(f"\n{'='*60}")
        print("[OK] Processing complete!")
        print(f"{'='*60}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Video accident detection with object detection')
    parser.add_argument('--model', '-m', required=True, help='Path to accident classifier model')
    parser.add_argument('--video', '-v', required=True, help='Path to input video')
    parser.add_argument('--output', '-o', help='Path to save annotated video')
    parser.add_argument('--json', '-j', help='Path to save results JSON')
    parser.add_argument('--yolo', default='yolov8n.pt', help='YOLO model (n/s/m/l/x)')
    parser.add_argument('--sample-rate', type=int, default=3, help='Process every Nth frame')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold')
    parser.add_argument('--aggregate', choices=['max', 'mean', 'vote'], default='max',
                       help='Prediction aggregation method')
    
    args = parser.parse_args()
    
    # Create detector
    detector = VideoAccidentDetector(
        classifier_path=args.model,
        yolo_model=args.yolo,
        threshold=args.threshold
    )
    
    # Process video
    results = detector.process_video(
        video_path=args.video,
        output_path=args.output,
        json_path=args.json,
        sample_rate=args.sample_rate,
        aggregate_method=args.aggregate
    )
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Prediction: {results['prediction']['prediction']}")
    print(f"Confidence: {results['prediction']['confidence']:.2f}%")
    print(f"Frames analyzed: {results['statistics']['total_frames']}")
    print(f"Accident frames: {results['statistics']['accident_frames']}")
    print(f"Detection rate: {results['statistics']['accident_percentage']:.1f}%")
    print(f"Objects detected: {results['statistics']['total_objects_detected']}")
    print(f"Avg objects/frame: {results['statistics']['avg_objects_per_frame']:.1f}")
    print("="*60)


if __name__ == '__main__':
    main()
