"""
Quick demo script to show model prediction on sample images.
"""

import cv2
import numpy as np
from pathlib import Path
import sys

try:
    from tensorflow import keras
except ImportError:
    print("Installing TensorFlow...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow"])
    from tensorflow import keras


def predict_image(model_path, image_path, img_size=(224, 224)):
    """Quick image prediction demo."""
    # Load model
    print(f"Loading model: {model_path}")
    model = keras.models.load_model(model_path)
    
    # Load and preprocess image
    print(f"Loading image: {image_path}")
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, img_size)
    img_array = img_resized / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    print("Predicting...")
    prob = model.predict(img_array, verbose=0)[0][0]
    pred = 'Accident' if prob > 0.5 else 'Non-Accident'
    confidence = prob if prob > 0.5 else 1 - prob
    
    # Display results
    print(f"\n{'='*60}")
    print(f"PREDICTION RESULTS")
    print(f"{'='*60}")
    print(f"Image: {Path(image_path).name}")
    print(f"Prediction: {pred}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Accident Probability: {prob:.2%}")
    print(f"{'='*60}\n")
    
    return {
        'prediction': pred,
        'probability': prob,
        'confidence': confidence
    }


def test_on_dataset_samples(model_path, dataset_dir='data/prepared_dataset/val'):
    """Test model on a few validation samples."""
    dataset_path = Path(dataset_dir)
    
    print(f"\n{'='*60}")
    print(f"Testing on Validation Dataset Samples")
    print(f"{'='*60}\n")
    
    # Test accident samples
    accident_dir = dataset_path / 'Accident'
    if accident_dir.exists():
        accident_imgs = list(accident_dir.glob('*.jpg'))[:3]
        print(f"Testing {len(accident_imgs)} ACCIDENT samples:")
        for img in accident_imgs:
            result = predict_image(model_path, img)
            status = "✅" if result['prediction'] == 'Accident' else "❌"
            print(f"{status} Correct: {result['prediction']}, Confidence: {result['confidence']:.2%}")
        print()
    
    # Test non-accident samples
    non_accident_dir = dataset_path / 'Non Accident'
    if non_accident_dir.exists():
        non_accident_imgs = list(non_accident_dir.glob('*.jpg'))[:3]
        print(f"Testing {len(non_accident_imgs)} NON-ACCIDENT samples:")
        for img in non_accident_imgs:
            result = predict_image(model_path, img)
            status = "✅" if result['prediction'] == 'Non-Accident' else "❌"
            print(f"{status} Correct: {result['prediction']}, Confidence: {result['confidence']:.2%}")
        print()


if __name__ == '__main__':
    model_path = 'models/accident_classifier_mobilenet.h5'
    
    if len(sys.argv) > 1:
        # Test specific image
        image_path = sys.argv[1]
        predict_image(model_path, image_path)
    else:
        # Test on validation samples
        test_on_dataset_samples(model_path)
