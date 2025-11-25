"""
train.py

Stage 1 training script for iSeg 2017 (T1+T2 -> 4-class segmentation).

Usage:
    python -m src.train
    or
    python train.py (from root)

Make sure:
    - iSeg-2017-Training/ is extracted in the data folder.
    - All source files are installed from src package.
"""

import os
import sys
import time
import traceback

# Add parent directory to path if running as script
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# -------------------------------------------
# CONFIG
# -------------------------------------------
MODEL_NAME = "iSeg4ClassUNet"
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "iSeg-4Class-Segmentation"

DATA_DIR = "/app/data/iSeg-2017-Training"
OUTPUT_MODEL = "/app/models/unet_stage1_4class.keras"
EPOCHS = 10
BATCH_SIZE = 4


def main():
    global TRACKING_URI
    try:
        # Import MLflow first to check connectivity
        import mlflow
        import mlflow.keras
        
        print("=" * 60)
        print("Starting iSeg Model Training")
        print("=" * 60)
        print(f"MLflow tracking URI: {TRACKING_URI}")
        print(f"Data directory: {DATA_DIR}")
        print(f"Output model: {OUTPUT_MODEL}")
        print(f"Epochs: {EPOCHS}")
        print(f"Batch size: {BATCH_SIZE}")
        print("=" * 60)
        
        # Test MLflow connection
        try:
            mlflow.set_tracking_uri(TRACKING_URI)
            print(f"✓ Connected to MLflow at {TRACKING_URI}")
        except Exception as e:
            print(f"⚠ Warning: Could not connect to MLflow: {e}")
            print("Continuing without MLflow tracking...")
            TRACKING_URI = None
        
        # Import from src package
        from tensorflow.keras.callbacks import Callback, ModelCheckpoint
        from src.preprocess import load_subjects, create_slice_dataset
        from src.models import build_unet_stage1
        
        # MLflow callback
        class MLflowMetrics(Callback):
            def on_epoch_end(self, epoch, logs=None):
                if TRACKING_URI:
                    logs = logs or {}
                    for k, v in logs.items():
                        try:
                            mlflow.log_metric(k, float(v), step=epoch)
                        except:
                            pass
        
        # Start MLflow run
        run_context = None
        if TRACKING_URI:
            try:
                mlflow.set_experiment(EXPERIMENT_NAME)
                run_context = mlflow.start_run(run_name="TrainingRun-4Class")
                run_context.__enter__()
                print("✓ Started MLflow run")
            except Exception as e:
                print(f"⚠ Warning: Could not start MLflow run: {e}")
                run_context = None
        
        try:
            # Log parameters
            if run_context:
                mlflow.log_param("epochs", EPOCHS)
                mlflow.log_param("batch_size", BATCH_SIZE)
                mlflow.log_param("architecture", "U-Net 4-class")
                mlflow.log_param("data_dir", DATA_DIR)
            
            # Load data
            print("\nLoading subjects from:", DATA_DIR)
            subjects = load_subjects(DATA_DIR)
            print(f"✓ Loaded {len(subjects)} subjects")
            
            print("\nCreating dataset...")
            X_train, X_val, Y_train, Y_val = create_slice_dataset(subjects)
            
            print(f"✓ Training data: X={X_train.shape}  Y={Y_train.shape}")
            print(f"✓ Validation:    X={X_val.shape}  Y={Y_val.shape}")
            
            if run_context:
                mlflow.log_param("input_shape", X_train.shape[1:])
                mlflow.log_param("train_samples", X_train.shape[0])
                mlflow.log_param("val_samples", X_val.shape[0])
            
            # Build model
            print("\nBuilding model...")
            model = build_unet_stage1(input_shape=X_train.shape[1:])
            print("✓ Model built successfully")
            
            # Create output directory
            os.makedirs(os.path.dirname(OUTPUT_MODEL), exist_ok=True)
            
            # Setup callbacks
            callbacks = []
            
            checkpoint = ModelCheckpoint(
                OUTPUT_MODEL,
                monitor="val_loss",
                save_best_only=True,
                verbose=1
            )
            callbacks.append(checkpoint)
            
            if run_context:
                callbacks.append(MLflowMetrics())
            
            # Train
            print("\n" + "=" * 60)
            print("Starting training...")
            print("=" * 60)
            
            start_time = time.time()
            
            history = model.fit(
                X_train, Y_train,
                validation_data=(X_val, Y_val),
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                callbacks=callbacks,
                verbose=1
            )
            
            training_time = time.time() - start_time
            
            print("=" * 60)
            print("✓ Training completed!")
            print(f"Training time: {training_time:.2f} seconds")
            print("=" * 60)
            
            # Log artifacts and metrics
            if run_context:
                try:
                    mlflow.log_artifact(OUTPUT_MODEL)
                    mlflow.log_metric("training_time_sec", training_time)
                    
                    # Log final metrics
                    final_val_loss = history.history['val_loss'][-1]
                    final_val_acc = history.history['val_accuracy'][-1]
                    mlflow.log_metric("final_val_loss", final_val_loss)
                    mlflow.log_metric("final_val_accuracy", final_val_acc)
                    
                    # Log model
                    print("\nLogging model to MLflow...")
                    try:
                        mlflow.keras.log_model(
                            model,
                            artifact_path="model"
                        )
                        print("✓ Model logged successfully")
                    except Exception as e:
                        print(f"⚠ Warning: Could not log model: {e}")
                    
                except Exception as e:
                    print(f"⚠ Warning: Could not log to MLflow: {e}")
            
            print(f"\n✓ Model saved to: {OUTPUT_MODEL}")
            print("\nFinal Metrics:")
            print(f"  - Validation Loss: {history.history['val_loss'][-1]:.4f}")
            print(f"  - Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
            
        finally:
            # Close MLflow run
            if run_context:
                try:
                    run_context.__exit__(None, None, None)
                except:
                    pass
        
        print("\n" + "=" * 60)
        print("Training pipeline completed successfully!")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("ERROR: Training failed!")
        print("=" * 60)
        print(f"Error: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        print("=" * 60)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
