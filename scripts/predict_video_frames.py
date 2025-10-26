"""
Predict accidents in videos using trained frame classifier.

This script processes videos frame-by-frame using the trained CNN model
and aggregates predictions to determine if accidents occurred.
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


class VideoAccidentPredictor:
    """Predict accidents in videos using trained classifier."""
    
    def __init__(self, model_path, img_size=(224, 224), threshold=0.5):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained model (.h5 file)
            img_size: Image size expected by model
            threshold: Classification threshold
        """
        self.img_size = img_size
        self.threshold = threshold
        
        print(f"Loading model from: {model_path}")
        self.model = keras.models.load_model(model_path)
        print("[OK] Model loaded successfully")
        
        # Load class indices if available
        class_indices_path = Path(model_path).parent / 'class_indices.json'
        if class_indices_path.exists():
            with open(class_indices_path) as f:
                self.class_indices = json.load(f)
            print(f"[OK] Class indices loaded: {self.class_indices}")
        else:
            self.class_indices = {'Accident': 1, 'Non Accident': 0}
    
    def predict_frame(self, frame):
        """
        Predict accident probability for a single frame.
        
        Args:
            frame: BGR image (numpy array)
            
        Returns:
            dict: Prediction results
        """
        # Preprocess
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, self.img_size)
        frame_array = frame_resized / 255.0
        frame_array = np.expand_dims(frame_array, axis=0)
        
        # Predict
        prob = self.model.predict(frame_array, verbose=0)[0][0]
        pred = 'Accident' if prob > self.threshold else 'Non-Accident'
        
        return {
            'prediction': pred,
            'probability': float(prob),
            'confidence': float(prob if prob > 0.5 else 1 - prob)
        }
    
    def process_video(self, video_path, output_path=None, sample_rate=1, 
                     show_progress=True, aggregate_method='max'):
        """
        Process entire video and predict accidents.
        
        Args:
            video_path: Path to input video
            output_path: Path to save annotated video (None = no output video)
            sample_rate: Process every Nth frame (1 = all frames)
            show_progress: Show progress bar
            aggregate_method: How to aggregate predictions ('max', 'mean', 'vote')
            
        Returns:
            dict: Prediction results
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        print(f"\nProcessing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Frames: {frame_count}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Sample rate: Every {sample_rate} frame(s)")
        
        # Setup output video if requested
        out = None
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Try different codecs for better browser compatibility
            # H.264 is the most compatible codec for web browsers
            codecs_to_try = [
                ('avc1', 'H.264 (avc1)'),  # H.264 codec - best browser support
                ('H264', 'H.264 (H264)'),  # Alternative H.264 fourcc
                ('X264', 'H.264 (X264)'),  # Another H.264 variant
                ('mp4v', 'MPEG-4 (mp4v)')  # Fallback
            ]
            
            out = None
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
            
            if out is None:
                print("  WARNING: Could not initialize video writer with any codec")
                print("  Video will not be saved")
            else:
                print(f"  Output video: {output_path}")
        
        # Process frames
        frame_predictions = []
        accident_frames = []
        frame_idx = 0
        
        pbar = tqdm(total=frame_count, disable=not show_progress, desc="Processing")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frame
            if frame_idx % sample_rate == 0:
                # Predict
                result = self.predict_frame(frame)
                frame_predictions.append({
                    'frame': frame_idx,
                    'time': frame_idx / fps if fps > 0 else 0,
                    **result
                })
                
                if result['prediction'] == 'Accident':
                    accident_frames.append(frame_idx)
                
                # Annotate frame
                if out and out.isOpened():
                    h, w = frame.shape[:2]
                    
                    # Create a copy for annotation
                    annotated_frame = frame.copy()
                    
                    # Colors in BGR format (OpenCV uses BGR, not RGB)
                    # Red for accident, Green for safe
                    if result['prediction'] == 'Accident':
                        color = (0, 0, 255)  # BGR: Blue=0, Green=0, Red=255
                    else:
                        color = (0, 255, 0)  # BGR: Blue=0, Green=255, Red=0
                    
                    # Draw semi-transparent banner at top
                    overlay = annotated_frame.copy()
                    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
                    annotated_frame = cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0)
                    
                    # Add prediction text
                    label = f"{result['prediction']}: {result['probability']:.2%}"
                    cv2.putText(annotated_frame, label, (20, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3, cv2.LINE_AA)
                    
                    # Add frame counter
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    frame_text = f"Frame: {frame_idx}/{total_frames}"
                    cv2.putText(annotated_frame, frame_text, (w - 300, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # If accident detected, add border highlight and warning
                    if result['prediction'] == 'Accident':
                        border_thickness = 8
                        # Draw red border
                        cv2.rectangle(annotated_frame, (border_thickness, border_thickness), 
                                    (w - border_thickness, h - border_thickness), 
                                    (0, 0, 255), border_thickness)
                        
                        # Add warning banner at bottom
                        warning_text = "!!! ACCIDENT DETECTED !!!"
                        text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                        text_x = (w - text_size[0]) // 2
                        
                        # Draw warning banner background
                        cv2.rectangle(annotated_frame, (0, h - 80), (w, h), (0, 0, 255), -1)
                        # Draw warning text
                        cv2.putText(annotated_frame, warning_text, (text_x, h - 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
                    
                    # Write the annotated frame to output video
                    out.write(annotated_frame)
            elif out and out.isOpened():
                # Write frame without processing (to maintain video timing)
                out.write(frame)
            
            frame_idx += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        if out:
            out.release()
        
        # Aggregate predictions
        if not frame_predictions:
            final_prediction = {
                'prediction': 'Unknown',
                'confidence': 0.0,
                'accident_probability': 0.0
            }
        else:
            probs = [p['probability'] for p in frame_predictions]
            
            if aggregate_method == 'max':
                # Use maximum probability across all frames
                max_prob = max(probs)
                final_prediction = {
                    'prediction': 'Accident' if max_prob > self.threshold else 'Non-Accident',
                    'confidence': max_prob if max_prob > 0.5 else 1 - max_prob,
                    'accident_probability': max_prob
                }
            elif aggregate_method == 'mean':
                # Use mean probability
                mean_prob = np.mean(probs)
                final_prediction = {
                    'prediction': 'Accident' if mean_prob > self.threshold else 'Non-Accident',
                    'confidence': mean_prob if mean_prob > 0.5 else 1 - mean_prob,
                    'accident_probability': mean_prob
                }
            else:  # vote
                # Majority vote
                accident_votes = sum(1 for p in frame_predictions if p['prediction'] == 'Accident')
                total_votes = len(frame_predictions)
                vote_ratio = accident_votes / total_votes
                
                final_prediction = {
                    'prediction': 'Accident' if vote_ratio > 0.5 else 'Non-Accident',
                    'confidence': vote_ratio if vote_ratio > 0.5 else 1 - vote_ratio,
                    'accident_probability': vote_ratio
                }
        
        # Compile results
        results = {
            'video_path': str(video_path),
            'video_info': {
                'width': width,
                'height': height,
                'fps': fps,
                'frame_count': frame_count,
                'duration': duration
            },
            'processing': {
                'sample_rate': sample_rate,
                'frames_processed': len(frame_predictions),
                'aggregate_method': aggregate_method
            },
            'prediction': final_prediction,
            'accident_frames': accident_frames,
            'frame_predictions': frame_predictions if len(frame_predictions) < 1000 else [],  # Limit output size
            'statistics': {
                'total_frames': len(frame_predictions),
                'accident_frames': len(accident_frames),
                'accident_percentage': len(accident_frames) / len(frame_predictions) * 100 if frame_predictions else 0,
                'min_probability': min(probs) if probs else 0,
                'max_probability': max(probs) if probs else 0,
                'mean_probability': np.mean(probs) if probs else 0,
                'std_probability': np.std(probs) if probs else 0
            }
        }
        
        # Print summary
        print(f"\n{'='*60}")
        print("PREDICTION RESULTS")
        print(f"{'='*60}")
        print(f"Final Prediction: {final_prediction['prediction']}")
        print(f"Confidence: {final_prediction['confidence']:.2%}")
        print(f"Accident Probability: {final_prediction['accident_probability']:.2%}")
        print(f"\nStatistics:")
        print(f"  Frames processed: {results['statistics']['total_frames']}")
        print(f"  Accident frames: {results['statistics']['accident_frames']} ({results['statistics']['accident_percentage']:.1f}%)")
        print(f"  Probability range: {results['statistics']['min_probability']:.2%} - {results['statistics']['max_probability']:.2%}")
        print(f"  Mean probability: {results['statistics']['mean_probability']:.2%} ± {results['statistics']['std_probability']:.2%}")
        
        if accident_frames:
            print(f"\nAccident detected at frames: {accident_frames[:10]}" + 
                  (" ..." if len(accident_frames) > 10 else ""))
            print(f"Time stamps: {[f'{f/fps:.2f}s' for f in accident_frames[:5]]}" +
                  (" ..." if len(accident_frames) > 5 else ""))
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Predict accidents in videos')
    parser.add_argument('--model', '-m', required=True, help='Path to trained model (.h5)')
    parser.add_argument('--video', '-v', required=True, help='Path to input video')
    parser.add_argument('--output', '-o', help='Path to save annotated video')
    parser.add_argument('--json', '-j', help='Path to save JSON results')
    parser.add_argument('--threshold', type=float, default=0.5, 
                       help='Classification threshold (default: 0.5)')
    parser.add_argument('--sample-rate', type=int, default=1,
                       help='Process every Nth frame (default: 1 = all frames)')
    parser.add_argument('--aggregate', choices=['max', 'mean', 'vote'], default='max',
                       help='Aggregation method for predictions')
    parser.add_argument('--img-size', type=int, default=224,
                       help='Image size (default: 224)')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = VideoAccidentPredictor(
        model_path=args.model,
        img_size=(args.img_size, args.img_size),
        threshold=args.threshold
    )
    
    # Process video
    results = predictor.process_video(
        video_path=args.video,
        output_path=args.output,
        sample_rate=args.sample_rate,
        aggregate_method=args.aggregate
    )
    
    # Save JSON results
    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[OK] Results saved to: {json_path}")
    else:
        # Auto-save next to video
        json_path = Path(args.video).parent / f"{Path(args.video).stem}_prediction.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[OK] Results saved to: {json_path}")
    
    print(f"\n{'='*60}")
    print("[OK] Processing complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
