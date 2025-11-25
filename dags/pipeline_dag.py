"""
iSeg 2017 Brain Segmentation ML Pipeline DAG

This DAG orchestrates the complete ML workflow:
1. Data validation
2. Model training
3. Model inference on test subjects
4. Visualization and reporting
5. Model registration

Schedule: Daily at 2 AM (can be adjusted)
"""

from datetime import datetime, timedelta
import os
import json

try:
    from airflow import DAG
    from airflow.providers.docker.operators.docker import DockerOperator
    from airflow.operators.python import PythonOperator, BranchPythonOperator
    from airflow.operators.empty import EmptyOperator
    from airflow.utils.task_group import TaskGroup
    from docker.types import Mount
except ImportError as e:
    print(f"Import error: {e}")
    raise

# Environment configuration
HOST_DATA = os.environ.get("HOST_DATA", "/opt/airflow/data")
HOST_MODELS = os.environ.get("HOST_MODELS", "/opt/airflow/models")
HOST_OUTPUTS = os.environ.get("HOST_OUTPUTS", "/opt/airflow/outputs")
HOST_MLFLOW = os.environ.get("HOST_MLFLOW", "/opt/airflow/mlflow_data")

# Network mode - use compose network
# Docker Compose creates network as: <project_dir>_<network_name>
# Get the actual network name
import subprocess
try:
    result = subprocess.run(
        ["docker", "network", "ls", "--filter", "name=iseg", "--format", "{{.Name}}"],
        capture_output=True,
        text=True,
        timeout=5
    )
    available_networks = [n.strip() for n in result.stdout.strip().split('\n') if n.strip()]
    TRAINER_NETWORK_MODE = available_networks[0] if available_networks else os.environ.get("TRAINER_NETWORK_MODE", "cpe393_final_mediimgclass_iseg_network")
    print(f"Using Docker network: {TRAINER_NETWORK_MODE}")
except Exception as e:
    print(f"Could not auto-detect network, using env var: {e}")
    TRAINER_NETWORK_MODE = os.environ.get("TRAINER_NETWORK_MODE", "cpe393_final_mediimgclass_iseg_network")

# MLflow URI
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# Default arguments
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=3),
    'execution_timeout': timedelta(hours=2),
}

# Create DAG
dag = DAG(
    'iseg_brain_segmentation_pipeline',
    default_args=default_args,
    description='Complete MLOps pipeline for iSeg 2017 brain segmentation',
    schedule_interval=None,  # Manual trigger
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['mlops', 'medical-imaging', 'segmentation', 'brain-mri'],
    is_paused_upon_creation=False,
)


def validate_data(**context):
    """Validate that required data files exist."""
    print("Starting data validation...")
    
    try:
        from glob import glob
        
        data_dir = "/opt/airflow/data/iSeg-2017-Training"
        print(f"Checking data directory: {data_dir}")
        
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Check for T1, T2, and label files
        t1_files = glob(os.path.join(data_dir, "subject-*-T1.hdr"))
        t2_files = glob(os.path.join(data_dir, "subject-*-T2.hdr"))
        label_files = glob(os.path.join(data_dir, "subject-*-label.hdr"))
        
        if not t1_files:
            raise FileNotFoundError("No T1 files found")
        if not t2_files:
            raise FileNotFoundError("No T2 files found")
        if not label_files:
            raise FileNotFoundError("No label files found")
        
        num_subjects = len(t1_files)
        
        print(f"✓ Data validation passed!")
        print(f"  - Found {num_subjects} subjects")
        print(f"  - T1 files: {len(t1_files)}")
        print(f"  - T2 files: {len(t2_files)}")
        print(f"  - Label files: {len(label_files)}")
        
        # Push to XCom
        context['ti'].xcom_push(key='num_subjects', value=num_subjects)
        context['ti'].xcom_push(key='data_dir', value=data_dir)
        
        return num_subjects
        
    except Exception as e:
        print(f"Error in validate_data: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def check_model_exists(**context):
    """Check if a trained model already exists."""
    print("Checking if model exists...")
    
    try:
        model_path = "/opt/airflow/models/unet_stage1_4class.keras"
        exists = os.path.isfile(model_path)
        
        print(f"Model path: {model_path}")
        print(f"Model exists: {exists}")
        
        context['ti'].xcom_push(key='model_exists', value=exists)
        context['ti'].xcom_push(key='model_path', value=model_path)
        
        return exists
        
    except Exception as e:
        print(f"Error in check_model_exists: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def decide_to_train(**context):
    """Branch: return 'train_model' if model doesn't exist, otherwise 'skip_train'."""
    print("Deciding whether to train...")
    
    try:
        ti = context['ti']
        model_exists = ti.xcom_pull(key='model_exists', task_ids='check_model')
        
        if model_exists:
            print("Model exists - skipping training")
            return 'skip_train'
        
        print("Model doesn't exist - will train")
        return 'train_model'
        
    except Exception as e:
        print(f"Error in decide_to_train: {str(e)}")
        import traceback
        traceback.print_exc()
        # Default to skip if error
        return 'skip_train'


def visualize_predictions(**context):
    """Generate and log visualization slices for all predictions."""
    print("Starting visualization...")
    
    try:
        import sys
        sys.path.insert(0, '/app')
        
        os.environ['MLFLOW_TRACKING_URI'] = MLFLOW_URI
        
        from glob import glob
        from view_predict_slice import view_and_log_slices
        
        pred_files = glob("/opt/airflow/outputs/subject-*-pred.nii.gz")
        
        if not pred_files:
            print("⚠ No prediction files found to visualize")
            return
        
        print(f"Visualizing {len(pred_files)} prediction files...")
        
        for pred_file in pred_files:
            print(f"Processing: {pred_file}")
            try:
                view_and_log_slices(pred_file, mlflow_uri=MLFLOW_URI)
                print(f"✓ Visualized: {pred_file}")
            except Exception as e:
                print(f"⚠ Could not visualize {pred_file}: {e}")
        
        print("✓ Visualization complete!")
        
    except Exception as e:
        print(f"Error during visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        # Don't fail pipeline on visualization errors
        print("Continuing despite visualization error...")


def generate_pipeline_report(**context):
    """Generate a summary report of the pipeline run."""
    print("Generating pipeline report...")
    
    try:
        num_subjects = context['ti'].xcom_pull(key='num_subjects', task_ids='validate_data')
        model_exists = context['ti'].xcom_pull(key='model_exists', task_ids='check_model')
        
        report = {
            'pipeline_run_date': datetime.now().isoformat(),
            'execution_date': str(context['execution_date']),
            'dag_run_id': context['dag_run'].run_id,
            'num_subjects': num_subjects,
            'model_retrained': not model_exists,
            'mlflow_tracking_uri': MLFLOW_URI,
            'status': 'SUCCESS'
        }
        
        os.makedirs("/opt/airflow/outputs", exist_ok=True)
        report_path = f"/opt/airflow/outputs/pipeline_report_{context['ds_nodash']}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print("=" * 60)
        print("Pipeline Report:")
        print(json.dumps(report, indent=2))
        print("=" * 60)
        
        return report_path
        
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        import traceback
        traceback.print_exc()
        # Don't fail pipeline on report errors
        return None


# =============================================================================
# Task Definitions
# =============================================================================

with dag:
    # Task 1: Validate data
    task_validate_data = PythonOperator(
        task_id='validate_data',
        python_callable=validate_data,
        provide_context=True,
    )
    
    # Task 2: Check if model exists
    task_check_model = PythonOperator(
        task_id='check_model',
        python_callable=check_model_exists,
        provide_context=True,
    )
    
    # Task 3: Decide whether to train
    task_decide_train = BranchPythonOperator(
        task_id='decide_train',
        python_callable=decide_to_train,
        provide_context=True,
    )
    
    # Task 4: Skip training (dummy task)
    skip_train = EmptyOperator(
        task_id='skip_train',
    )
    
    # Task 5: Train model using Docker
    task_train_model = DockerOperator(
        task_id='train_model',
        image='iseg_trainer:latest',
        api_version='auto',
        auto_remove=True,
        docker_url='unix://var/run/docker.sock',
        network_mode=TRAINER_NETWORK_MODE,
        mount_tmp_dir=False,
        mounts=[
            Mount(source=HOST_DATA, target='/app/data', type='bind'),
            Mount(source=HOST_MODELS, target='/app/models', type='bind'),
            Mount(source=HOST_OUTPUTS, target='/app/outputs', type='bind'),
            Mount(source=HOST_MLFLOW, target='/mlflow', type='bind'),
        ],
        command='python /app/train.py',
        environment={
            'MLFLOW_TRACKING_URI': MLFLOW_URI,
            'PYTHONUNBUFFERED': '1',
        },
    )
    
    # Task 6: Join point after training decision
    train_done = EmptyOperator(
        task_id='train_done',
        trigger_rule='none_failed_min_one_success',
    )
    
    # Task Group: Run inference on multiple subjects
    with TaskGroup('inference_tasks') as inference_group:
        for subject_num in range(1, 4):  # Subjects 1-3
            DockerOperator(
                task_id=f'inference_subject_{subject_num}',
                image='iseg_trainer:latest',
                api_version='auto',
                auto_remove=True,
                docker_url='unix://var/run/docker.sock',
                network_mode=TRAINER_NETWORK_MODE,
                mount_tmp_dir=False,
                mounts=[
                    Mount(source=HOST_DATA, target='/app/data', type='bind'),
                    Mount(source=HOST_MODELS, target='/app/models', type='bind'),
                    Mount(source=HOST_OUTPUTS, target='/app/outputs', type='bind'),
                ],
                command=f'python /app/inference.py --subject-id {subject_num}',
                environment={
                    'MLFLOW_TRACKING_URI': MLFLOW_URI,
                    'PYTHONUNBUFFERED': '1',
                },
            )
    
    # Task 7: Visualize predictions
    task_visualize = PythonOperator(
        task_id='visualize_predictions',
        python_callable=visualize_predictions,
        provide_context=True,
    )
    
    # Task 8: Generate report
    task_report = PythonOperator(
        task_id='generate_report',
        python_callable=generate_pipeline_report,
        provide_context=True,
    )
    
    # Define workflow dependencies
    task_validate_data >> task_check_model >> task_decide_train
    task_decide_train >> [task_train_model, skip_train]
    [task_train_model, skip_train] >> train_done
    train_done >> inference_group >> task_visualize >> task_report