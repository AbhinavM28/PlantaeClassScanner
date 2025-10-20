
"""
MobileNetV3 Transfer Learning for PlantaeClassScanner
Optimized for Raspberry Pi 4B deployment

Author: Abhinav M
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    ReduceLROnPlateau,
    TensorBoard,
    CSVLogger
)

try:
    from src.utils.logger import setup_logger
    from src.utils.config_loader import get_config
    logger = setup_logger("ModelTraining")
    config = get_config()
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("ModelTraining")
    config = None


class PlantClassifierTrainer:
    """Train MobileNetV3 for plant classification"""
    
    def __init__(self, data_dir="data", output_dir="models"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Training parameters (from config or defaults)
        if config:
            self.img_size = tuple(config.get('model.input_size', [224, 224]))
            self.batch_size = config.get('training.batch_size', 32)
            self.epochs = config.get('training.epochs', 50)
            self.learning_rate = config.get('training.learning_rate', 0.001)
        else:
            self.img_size = (224, 224)
            self.batch_size = 32
            self.epochs = 50
            self.learning_rate = 0.001
        
        # Create training session directory
        self.session_name = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_dir = self.output_dir / "training" / self.session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Paths
        self.train_dir = self.data_dir / "train"
        self.val_dir = self.data_dir / "val"
        self.test_dir = self.data_dir / "test"
        
        # Model
        self.model = None
        self.history = None
        self.num_classes = None
        self.class_names = None
        
        logger.info("=" * 70)
        logger.info("🌿 PlantaeClassScanner Model Training")
        logger.info("=" * 70)
        logger.info(f"Session: {self.session_name}")
        logger.info(f"Output directory: {self.session_dir}")
        logger.info(f"Image size: {self.img_size}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Epochs: {self.epochs}")
        logger.info(f"Learning rate: {self.learning_rate}")
    
    def create_data_generators(self):
        """Create data generators with augmentation"""
        logger.info("\n📊 Creating data generators...")
        
        # Training data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        # Validation/Test data (no augmentation, only rescaling)
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        self.train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=42
        )
        
        self.val_generator = val_test_datagen.flow_from_directory(
            self.val_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        self.test_generator = val_test_datagen.flow_from_directory(
            self.test_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        # Store class information
        self.num_classes = self.train_generator.num_classes
        self.class_names = list(self.train_generator.class_indices.keys())
        
        logger.info(f"✅ Train samples: {self.train_generator.samples}")
        logger.info(f"✅ Val samples: {self.val_generator.samples}")
        logger.info(f"✅ Test samples: {self.test_generator.samples}")
        logger.info(f"✅ Number of classes: {self.num_classes}")
        logger.info(f"✅ Classes: {', '.join(self.class_names)}")
        
        # Save class mapping
        class_mapping = {
            "class_names": self.class_names,
            "class_indices": self.train_generator.class_indices
        }
        with open(self.session_dir / "class_mapping.json", "w") as f:
            json.dump(class_mapping, f, indent=2)
    
    def build_model(self):
        """Build MobileNetV3-Small with custom head"""
        logger.info("\n🏗️  Building model architecture...")
        
        # Load pre-trained MobileNetV3-Small (ImageNet weights)
        base_model = keras.applications.MobileNetV3Small(
            input_shape=(*self.img_size, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        logger.info(f"✅ Base model: MobileNetV3-Small")
        logger.info(f"✅ Base model layers: {len(base_model.layers)}")
        logger.info(f"✅ Base model trainable: {base_model.trainable}")
        
        # Build custom classification head
        inputs = keras.Input(shape=(*self.img_size, 3))
        x = base_model(inputs, training=False)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = keras.Model(inputs, outputs)
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')]
        )
        
        logger.info(f"✅ Model compiled")
        logger.info(f"✅ Total parameters: {self.model.count_params():,}")
        
        # Print model summary to log file
        summary_path = self.session_dir / "model_summary.txt"
        with open(summary_path, "w") as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
        logger.info(f"✅ Model summary saved: {summary_path}")
    
    def setup_callbacks(self):
        """Setup training callbacks"""
        logger.info("\n⚙️  Setting up callbacks...")
        
        callbacks = [
            # Model checkpointing (save best model)
            ModelCheckpoint(
                filepath=str(self.session_dir / "best_model.h5"),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            
            # Early stopping
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Learning rate reduction
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # TensorBoard logging
            TensorBoard(
                log_dir=str(self.session_dir / "tensorboard_logs"),
                histogram_freq=1,
                write_graph=True,
                write_images=False,
                update_freq='epoch'
            ),
            
            # CSV logging
            CSVLogger(
                filename=str(self.session_dir / "training_log.csv"),
                append=False
            )
        ]
        
        logger.info("✅ Callbacks configured:")
        logger.info("   - ModelCheckpoint (save best model)")
        logger.info("   - EarlyStopping (patience=10)")
        logger.info("   - ReduceLROnPlateau (patience=5)")
        logger.info("   - TensorBoard logging")
        logger.info("   - CSV logging")
        
        return callbacks
    
    def train(self):
        """Execute training"""
        logger.info("\n🚀 Starting training...")
        logger.info("=" * 70)
        
        callbacks = self.setup_callbacks()
        
        # Calculate steps
        steps_per_epoch = self.train_generator.samples // self.batch_size
        validation_steps = self.val_generator.samples // self.batch_size
        
        # Train model
        start_time = datetime.now()
        
        self.history = self.model.fit(
            self.train_generator,
            epochs=self.epochs,
            validation_data=self.val_generator,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        logger.info("=" * 70)
        logger.info(f"✅ Training complete!")
        logger.info(f"⏱️  Training time: {training_time/60:.2f} minutes")
        
        # Save training history
        history_path = self.session_dir / "training_history.json"
        with open(history_path, "w") as f:
            # Convert numpy types to Python types for JSON
            history_dict = {k: [float(v) for v in vals] 
                          for k, vals in self.history.history.items()}
            json.dump(history_dict, f, indent=2)
        
        logger.info(f"✅ Training history saved: {history_path}")
    
    def evaluate(self):
        """Evaluate model on test set"""
        logger.info("\n📈 Evaluating on test set...")
        
        test_loss, test_acc, test_top3 = self.model.evaluate(
            self.test_generator,
            verbose=1
        )
        
        logger.info("=" * 70)
        logger.info("TEST RESULTS")
        logger.info("=" * 70)
        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.info(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        logger.info(f"Test Top-3 Accuracy: {test_top3:.4f} ({test_top3*100:.2f}%)")
        logger.info("=" * 70)
        
        # Save evaluation results
        eval_results = {
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "test_top3_accuracy": float(test_top3),
            "num_test_samples": self.test_generator.samples
        }
        
        with open(self.session_dir / "evaluation_results.json", "w") as f:
            json.dump(eval_results, f, indent=2)
        
        return eval_results
    
    def save_final_model(self):
        """Save final model in multiple formats"""
        logger.info("\n💾 Saving final model...")
        
        # Save as .h5
        h5_path = self.session_dir / "final_model.h5"
        self.model.save(h5_path)
        logger.info(f"✅ Saved H5 model: {h5_path}")
        
        # Save as SavedModel format
        savedmodel_path = self.session_dir / "saved_model"
        self.model.save(savedmodel_path)
        logger.info(f"✅ Saved SavedModel: {savedmodel_path}")
        
        return h5_path, savedmodel_path
    
    def run_full_training(self):
        """Execute complete training pipeline"""
        try:
            # Create data generators
            self.create_data_generators()
            
            # Build model
            self.build_model()
            
            # Train
            self.train()
            
            # Evaluate
            eval_results = self.evaluate()
            
            # Save model
            self.save_final_model()
            
            logger.info("\n" + "=" * 70)
            logger.info("🎉 TRAINING PIPELINE COMPLETE!")
            logger.info("=" * 70)
            logger.info(f"Session directory: {self.session_dir}")
            logger.info(f"Final test accuracy: {eval_results['test_accuracy']*100:.2f}%")
            logger.info("=" * 70)
            
            # Create summary file
            summary = {
                "session_name": self.session_name,
                "timestamp": datetime.now().isoformat(),
                "model_architecture": "MobileNetV3-Small",
                "num_classes": self.num_classes,
                "class_names": self.class_names,
                "training_params": {
                    "epochs_run": len(self.history.history['loss']),
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                    "image_size": self.img_size
                },
                "results": eval_results
            }
            
            with open(self.session_dir / "training_summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}", exc_info=True)
            return False


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train plant classifier")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--output-dir", default="models", help="Output directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = PlantClassifierTrainer(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    # Override epochs if specified
    if args.epochs:
        trainer.epochs = args.epochs
    if args.batch_size:
        trainer.batch_size = args.batch_size
    
    # Run training
    success = trainer.run_full_training()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()