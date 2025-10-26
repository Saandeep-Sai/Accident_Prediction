"""
Train a CNN classifier on accident/non-accident image frames.

This script trains a deep learning model on the image datasets from Kaggle,
which can then be used for frame-by-frame video prediction.
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import cv2
from tqdm import tqdm
import json
import argparse

# Deep learning imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import MobileNetV2, ResNet50V2
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
except ImportError:
    print("TensorFlow not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow"])
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import MobileNetV2, ResNet50V2
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


class AccidentClassifier:
    """CNN classifier for accident detection from images."""
    
    def __init__(self, img_size=(224, 224), model_type='mobilenet'):
        """
        Initialize classifier.
        
        Args:
            img_size: Tuple (height, width) for input images
            model_type: 'mobilenet', 'resnet', or 'simple_cnn'
        """
        self.img_size = img_size
        self.model_type = model_type
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build the classification model."""
        if self.model_type == 'mobilenet':
            # Transfer learning with MobileNetV2 (lightweight, good for deployment)
            base_model = MobileNetV2(
                input_shape=(*self.img_size, 3),
                include_top=False,
                weights='imagenet'
            )
            base_model.trainable = False  # Freeze base model initially
            
            model = models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dropout(0.3),
                layers.Dense(128, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(1, activation='sigmoid')  # Binary classification
            ])
            
        elif self.model_type == 'resnet':
            # Transfer learning with ResNet50V2 (more accurate, heavier)
            base_model = ResNet50V2(
                input_shape=(*self.img_size, 3),
                include_top=False,
                weights='imagenet'
            )
            base_model.trainable = False
            
            model = models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dropout(0.3),
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(1, activation='sigmoid')
            ])
            
        else:  # simple_cnn
            # Simple CNN from scratch (fast training, decent accuracy)
            model = models.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(1, activation='sigmoid')
            ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        self.model = model
        return model
    
    def train(self, train_dir, val_dir=None, epochs=20, batch_size=32, 
              augment=True, output_dir='models'):
        """
        Train the model on image data.
        
        Args:
            train_dir: Directory containing training images (with Accident/Non-Accident subdirs)
            val_dir: Directory containing validation images (or None to split from train)
            epochs: Number of training epochs
            batch_size: Batch size for training
            augment: Whether to use data augmentation
            output_dir: Directory to save model and results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.model is None:
            self.build_model()
        
        print(f"\n{'='*60}")
        print(f"Training {self.model_type.upper()} model")
        print(f"{'='*60}")
        print(f"Image size: {self.img_size}")
        print(f"Batch size: {batch_size}")
        print(f"Epochs: {epochs}")
        print(f"Data augmentation: {augment}")
        print(f"Output directory: {output_dir}")
        
        # Data generators
        if augment:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True,
                fill_mode='nearest',
                validation_split=0.2 if val_dir is None else 0.0
            )
        else:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=0.2 if val_dir is None else 0.0
            )
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Load training data
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='binary',
            subset='training' if val_dir is None else None,
            shuffle=True
        )
        
        # Load validation data
        if val_dir:
            val_generator = val_datagen.flow_from_directory(
                val_dir,
                target_size=self.img_size,
                batch_size=batch_size,
                class_mode='binary',
                shuffle=False
            )
        else:
            val_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=self.img_size,
                batch_size=batch_size,
                class_mode='binary',
                subset='validation',
                shuffle=False
            )
        
        print(f"\nTraining samples: {train_generator.samples}")
        print(f"Validation samples: {val_generator.samples}")
        print(f"Classes: {train_generator.class_indices}")
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                str(output_dir / f'best_model_{self.model_type}.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train
        print(f"\n{'='*60}")
        print("Starting training...")
        print(f"{'='*60}\n")
        
        self.history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        final_model_path = output_dir / f'accident_classifier_{self.model_type}.h5'
        self.model.save(str(final_model_path))
        print(f"\n✓ Model saved to: {final_model_path}")
        
        # Save training history
        history_path = output_dir / f'training_history_{self.model_type}.json'
        with open(history_path, 'w') as f:
            json.dump({k: [float(v) for v in vals] for k, vals in self.history.history.items()}, f, indent=2)
        print(f"✓ Training history saved to: {history_path}")
        
        # Save class indices
        class_indices_path = output_dir / 'class_indices.json'
        with open(class_indices_path, 'w') as f:
            json.dump(train_generator.class_indices, f, indent=2)
        print(f"✓ Class indices saved to: {class_indices_path}")
        
        # Plot training history
        self.plot_training_history(output_dir / f'training_plots_{self.model_type}.png')
        
        # Evaluate on validation set
        print(f"\n{'='*60}")
        print("Evaluating on validation set...")
        print(f"{'='*60}")
        
        val_loss, val_acc, val_prec, val_recall = self.model.evaluate(val_generator, verbose=1)
        print(f"\nValidation Results:")
        print(f"  Loss: {val_loss:.4f}")
        print(f"  Accuracy: {val_acc:.4f}")
        print(f"  Precision: {val_prec:.4f}")
        print(f"  Recall: {val_recall:.4f}")
        
        # Generate predictions for confusion matrix
        val_generator.reset()
        y_pred_probs = self.model.predict(val_generator, verbose=1)
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()
        y_true = val_generator.classes
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['Non-Accident', 'Accident']))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        self.plot_confusion_matrix(cm, output_dir / f'confusion_matrix_{self.model_type}.png')
        
        return self.history
    
    def plot_training_history(self, save_path):
        """Plot and save training history."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Train')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Train')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Training plots saved to: {save_path}")
    
    def plot_confusion_matrix(self, cm, save_path):
        """Plot and save confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Non-Accident', 'Accident'],
                   yticklabels=['Non-Accident', 'Accident'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Confusion matrix saved to: {save_path}")
    
    def load_model(self, model_path):
        """Load a trained model."""
        self.model = keras.models.load_model(model_path)
        print(f"✓ Model loaded from: {model_path}")
        return self.model
    
    def predict_image(self, image_path, threshold=0.5):
        """
        Predict accident probability for a single image.
        
        Args:
            image_path: Path to image file or numpy array
            threshold: Classification threshold
            
        Returns:
            dict: Prediction results
        """
        if isinstance(image_path, (str, Path)):
            img = cv2.imread(str(image_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image_path
        
        # Preprocess
        img_resized = cv2.resize(img, self.img_size)
        img_array = img_resized / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        prob = self.model.predict(img_array, verbose=0)[0][0]
        pred = 'Accident' if prob > threshold else 'Non-Accident'
        
        return {
            'prediction': pred,
            'probability': float(prob),
            'confidence': float(prob if prob > 0.5 else 1 - prob)
        }


def prepare_dataset_structure(base_dir='data'):
    """
    Organize downloaded images into train/val structure.
    
    Expected structure after running:
    data/prepared_dataset/
        train/
            Accident/
            Non-Accident/
        val/
            Accident/
            Non-Accident/
    """
    base_path = Path(base_dir)
    
    # Check available datasets
    dataset1 = base_path / 'raw_videos' / 'data'  # First Kaggle dataset
    dataset2 = base_path / 'videos' / 'Accident' / 'Accident'  # Second Kaggle dataset
    
    if not dataset1.exists() and not dataset2.exists():
        print("Error: No datasets found!")
        print(f"  Looking for: {dataset1}")
        print(f"           or: {dataset2}")
        return None
    
    # Use first dataset if available (already has train/val split)
    if dataset1.exists():
        print(f"✓ Using dataset: {dataset1}")
        print("  This dataset already has train/test/val splits")
        
        # Create symlinks or use existing structure
        prepared_dir = base_path / 'prepared_dataset'
        if prepared_dir.exists():
            import shutil
            shutil.rmtree(prepared_dir)
        
        prepared_dir.mkdir(parents=True, exist_ok=True)
        
        # Map existing structure
        train_dir = prepared_dir / 'train'
        val_dir = prepared_dir / 'val'
        
        import shutil
        shutil.copytree(dataset1 / 'train', train_dir)
        shutil.copytree(dataset1 / 'val', val_dir)
        
        print(f"✓ Dataset prepared at: {prepared_dir}")
        return prepared_dir
    
    else:
        print(f"Dataset 1 not found, would use dataset 2")
        return None


def main():
    parser = argparse.ArgumentParser(description='Train accident classifier')
    parser.add_argument('--model-type', choices=['mobilenet', 'resnet', 'simple_cnn'],
                       default='mobilenet', help='Model architecture')
    parser.add_argument('--img-size', type=int, default=224, help='Image size (square)')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--no-augment', action='store_true', help='Disable data augmentation')
    parser.add_argument('--prepare-only', action='store_true', help='Only prepare dataset')
    parser.add_argument('--train-dir', help='Custom training directory')
    parser.add_argument('--val-dir', help='Custom validation directory')
    
    args = parser.parse_args()
    
    # Prepare dataset
    print("Preparing dataset...")
    if args.train_dir:
        train_dir = Path(args.train_dir)
        val_dir = Path(args.val_dir) if args.val_dir else None
    else:
        dataset_root = prepare_dataset_structure()
        if dataset_root is None:
            print("Failed to prepare dataset!")
            return
        train_dir = dataset_root / 'train'
        val_dir = dataset_root / 'val'
    
    if args.prepare_only:
        print("Dataset preparation complete!")
        return
    
    # Train model
    classifier = AccidentClassifier(
        img_size=(args.img_size, args.img_size),
        model_type=args.model_type
    )
    
    classifier.train(
        train_dir=train_dir,
        val_dir=val_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        augment=not args.no_augment,
        output_dir='models'
    )
    
    print(f"\n{'='*60}")
    print("✓ Training complete!")
    print(f"{'='*60}")
    print("\nNext steps:")
    print("  1. Check training plots in models/ directory")
    print("  2. Use the model for video prediction:")
    print(f"     python scripts/predict_video_frames.py --model models/accident_classifier_{args.model_type}.h5 --video data/clips/test.mp4")


if __name__ == '__main__':
    main()
