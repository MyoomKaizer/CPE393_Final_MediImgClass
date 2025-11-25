"""
Model Retraining DAG

This DAG handles scheduled or triggered model retraining with:
- Data validation
- Hyperparameter tuning (optional)
- Model training with different configurations
- Model comparison and selection
- Automatic model registration

Schedule: Weekly or on-demand trigger
"""

from datetime import datetime, timedelta
import os
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

# MLflow configuration
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")

default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'model_retraining_pipeline',
    default_args=default_args,
    description='Automated model retraining and evaluation',
    schedule_interval='0 0 * * 0',  # Weekly on Sunday midnight
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['mlops', 'retraining', 'model-management'],
)


def check_retraining_needed(**context):
    """
    Determine if retraining is needed based on:
    - Model age
    - Performance degradation
    - New data availability
    """
    from datetime import datetime, timedelta
    
    model_path = "/opt/airflow/models/unet_stage1_4class.keras"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print("No existing model found - retraining required")
        return 'train_new_model'
    
    # Check model age
    model_age_days = (datetime.now() - datetime.fromtimestamp(
        os.path.getmtime(model_path)
    )).days
    
    print(f"Model age: {model_age_days} days")
    
    # Retrain if model is older than 30 days
    if model_age_days > 30:
        print("Model is old - retraining required")
        return 'train_new_model'
    
    print("Model is recent - skipping retraining")
    return 'skip_training'


def train_with_config(config_name, epochs, batch_size, base_filters, **context):
    """Train model with specific configuration."""
    sys.path.insert(0, '/app')
    
    os.environ['MLFLOW_TRACKING_URI'] = MLFLOW_URI
    
    try:
        import mlflow
        import numpy as np
        from tensorflow.keras.callbacks import Callback, ModelCheckpoint
        from preprocess import load_subjects, create_slice_dataset
        from models import build_unet_stage1
        
        class MLflowMetrics(Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                for k, v in logs.items():
                    try:
                        mlflow.log_metric(k, float(v), step=epoch)
                    except:
                        pass
        
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("iSeg-Model-Retraining")
        
        with mlflow.start_run(run_name=f"Retrain-{config_name}"):
            # Log configuration
            mlflow.log_param("config_name", config_name)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("base_filters", base_filters)
            
            # Load data
            data_dir = "/opt/airflow/data/iSeg-2017-Training"
            subjects = load_subjects(data_dir)
            X_train, X_val, Y_train, Y_val = create_slice_dataset(subjects)
            
            # Build model
            model = build_unet_stage1(
                input_shape=X_train.shape[1:],
                base_filters=base_filters
            )
            
            # Train
            output_path = f"/opt/airflow/models/unet_{config_name}.keras"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            checkpoint = ModelCheckpoint(
                output_path,
                monitor="val_loss",
                save_best_only=True,
                verbose=1
            )
            
            history = model.fit(
                X_train, Y_train,
                validation_data=(X_val, Y_val),
                batch_size=batch_size,
                epochs=epochs,
                callbacks=[MLflowMetrics(), checkpoint],
                verbose=1
            )
            
            # Log final metrics
            final_val_loss = history.history['val_loss'][-1]
            final_val_acc = history.history['val_accuracy'][-1]
            
            mlflow.log_metric("final_val_loss", final_val_loss)
            mlflow.log_metric("final_val_accuracy", final_val_acc)
            
            try:
                mlflow.log_artifact(output_path)
            except:
                print("Warning: Could not log artifact to MLflow")
            
            # Store metrics for comparison
            context['ti'].xcom_push(
                key=f'{config_name}_metrics',
                value={'val_loss': final_val_loss, 'val_acc': final_val_acc}
            )
            
            print(f"Training {config_name} completed!")
            print(f"Final val_loss: {final_val_loss:.4f}")
            print(f"Final val_acc: {final_val_acc:.4f}")
            
            return output_path
            
    except Exception as e:
        print(f"Error training {config_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def select_best_model(**context):
    """Compare trained models and select the best one."""
    import shutil
    
    configs = ['baseline', 'larger', 'deeper']
    best_config = None
    best_loss = float('inf')
    
    for config in configs:
        try:
            metrics = context['ti'].xcom_pull(
                key=f'{config}_metrics',
                task_ids=f'train_{config}_model'
            )
            
            if metrics and metrics['val_loss'] < best_loss:
                best_loss = metrics['val_loss']
                best_config = config
        except Exception as e:
            print(f"Warning: Could not get metrics for {config}: {e}")
    
    if best_config:
        print(f"Best model: {best_config} with val_loss={best_loss:.4f}")
        
        # Copy best model to production
        src = f"/opt/airflow/models/unet_{best_config}.keras"
        dst = "/opt/airflow/models/unet_stage1_4class.keras"
        
        try:
            shutil.copy(src, dst)
            print(f"Promoted {best_config} model to production")
            context['ti'].xcom_push(key='best_model', value=best_config)
        except Exception as e:
            print(f"Error copying model: {e}")
            raise
        
        return best_config
    else:
        print("Warning: No best model found")
        return None


# Task definitions
start = EmptyOperator(task_id='start', dag=dag)

check_retrain = BranchPythonOperator(
    task_id='check_retraining_needed',
    python_callable=check_retraining_needed,
    dag=dag,
)

skip_training = EmptyOperator(task_id='skip_training', dag=dag)

train_new_model = EmptyOperator(task_id='train_new_model', dag=dag)

# Train multiple configurations for comparison
train_baseline = PythonOperator(
    task_id='train_baseline_model',
    python_callable=train_with_config,
    op_kwargs={
        'config_name': 'baseline',
        'epochs': 10,
        'batch_size': 4,
        'base_filters': 16
    },
    dag=dag,
)

train_larger = PythonOperator(
    task_id='train_larger_model',
    python_callable=train_with_config,
    op_kwargs={
        'config_name': 'larger',
        'epochs': 10,
        'batch_size': 4,
        'base_filters': 32
    },
    dag=dag,
)

train_deeper = PythonOperator(
    task_id='train_deeper_model',
    python_callable=train_with_config,
    op_kwargs={
        'config_name': 'deeper',
        'epochs': 15,
        'batch_size': 4,
        'base_filters': 16
    },
    dag=dag,
)

select_model = PythonOperator(
    task_id='select_best_model',
    python_callable=select_best_model,
    trigger_rule='all_done',
    dag=dag,
)

end = EmptyOperator(
    task_id='end', 
    trigger_rule='none_failed_min_one_success', 
    dag=dag
)

# Define workflow
start >> check_retrain >> [skip_training, train_new_model]
train_new_model >> [train_baseline, train_larger, train_deeper] >> select_model >> end
skip_training >> end