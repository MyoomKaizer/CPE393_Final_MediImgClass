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
HOST_DATA = os.environ.get("HOST_DATA", "/opt/airflow/data")
HOST_MODELS = os.environ.get("HOST_MODELS", "/opt/airflow/models")
HOST_OUTPUTS = os.environ.get("HOST_OUTPUTS", "/opt/airflow/outputs")
HOST_MLFLOW = os.environ.get("HOST_MLFLOW", "/opt/airflow/mlflow_data")
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
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=3),
    'execution_timeout': timedelta(hours=2),
}
dag = DAG(
    'iseg_brain_segmentation_pipeline',
    default_args=default_args,
    description='Complete MLOps pipeline for iSeg 2017 brain segmentation',
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['mlops', 'medical-imaging', 'segmentation', 'brain-mri'],
    is_paused_upon_creation=False,
)


def validate_data(**context):
    print("Starting data validation...")
    
    try:
        from glob import glob
        
        data_dir = "/opt/airflow/data/iSeg-2017-Training"
        print(f"Checking data directory: {data_dir}")
        
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

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

        context['ti'].xcom_push(key='num_subjects', value=num_subjects)
        context['ti'].xcom_push(key='data_dir', value=data_dir)
        
        return num_subjects
        
    except Exception as e:
        print(f"Error in validate_data: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def check_model_exists(**context):
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
        return 'skip_train'


def visualize_predictions(**context):
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
        print("Continuing despite visualization error...")


def generate_pipeline_report(**context):
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
        return None

with dag:
    task_validate_data = PythonOperator(
        task_id='validate_data',
        python_callable=validate_data,
        provide_context=True,
    )

    task_check_model = PythonOperator(
        task_id='check_model',
        python_callable=check_model_exists,
        provide_context=True,
    )

    task_decide_train = BranchPythonOperator(
        task_id='decide_train',
        python_callable=decide_to_train,
        provide_context=True,
    )

    skip_train = EmptyOperator(
        task_id='skip_train',
    )

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
        command='python -m src.train',
        environment={
            'MLFLOW_TRACKING_URI': MLFLOW_URI,
            'PYTHONUNBUFFERED': '1',
        },
    )

    train_done = EmptyOperator(
        task_id='train_done',
        trigger_rule='none_failed_min_one_success',
    )

    with TaskGroup('inference_tasks') as inference_group:
        for subject_num in range(1, 4):
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
                command=f'python -m src.inference --subject-id {subject_num}',
                environment={
                    'MLFLOW_TRACKING_URI': MLFLOW_URI,
                    'PYTHONUNBUFFERED': '1',
                },
            )

    task_visualize = PythonOperator(
        task_id='visualize_predictions',
        python_callable=visualize_predictions,
        provide_context=True,
    )

    task_report = PythonOperator(
        task_id='generate_report',
        python_callable=generate_pipeline_report,
        provide_context=True,
    )

    task_monitoring = DockerOperator(
        task_id='monitor_drift_degradation',
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
        command='python -m src.monitoring_evidently',
        environment={
            'MLFLOW_TRACKING_URI': MLFLOW_URI,
            'PYTHONUNBUFFERED': '1',
        },
    )

    task_validate_data >> task_check_model >> task_decide_train
    task_decide_train >> [task_train_model, skip_train]
    [task_train_model, skip_train] >> train_done
    train_done >> inference_group >> task_monitoring >> task_visualize >> task_report