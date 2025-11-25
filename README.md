# iSeg 2017 Brain Segmentation - MLOps Pipeline

A complete MLOps pipeline for brain MRI segmentation using the iSeg 2017 dataset. This project demonstrates medical imaging, deep learning, workflow orchestration, and production-ready ML practices.

## 🎯 Project Overview

This pipeline performs automatic 4-class brain tissue segmentation from multimodal MRI (T1 and T2 weighted images):
- **Classes**: Cerebrospinal Fluid (CSF), Gray Matter (GM), White Matter (WM), Background
- **Architecture**: U-Net with encoder-decoder structure
- **Input**: T1 and T2 weighted MRI volumes (iSeg 2017 dataset)
- **Output**: 3D segmentation predictions in NIfTI format

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Airflow Orchestration                  │
├─────────────────────────────────────────────────────────┤
│  - Data Validation                                      │
│  - Model Training (U-Net, TensorFlow/Keras)             │
│  - Inference on Multiple Subjects                       │
│  - Visualization & Reporting                            │
│  - MLflow Experiment Tracking                           │
└─────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────┐
│              Docker Containerization                    │
├─────────────────────────────────────────────────────────┤
│  • iseg_trainer: Training & Inference Container         │
│  • airflow_webserver: Orchestration UI                  │
│  • mlflow_server: Experiment Tracking & Model Registry  │
│  • postgres: Airflow & MLflow Metadata                  │
└─────────────────────────────────────────────────────────┘
```

## 📋 Prerequisites

- **Docker** (with Docker Compose)
- **Windows PowerShell 5.1+** (or Bash/Linux equivalent)
- **~10GB disk space** (for data + models)
- **GPU optional** (CPU-based inference is supported)

## 🚀 Quick Start

### 1. Clone and Setup
```powershell
cd C:\Users\<username>\Repo\ML\CPE393_Final_MediImgClass
```

### 2. Start the Stack
```powershell
docker-compose up -d
```

This starts:
- **Airflow WebServer**: http://localhost:8080
- **MLflow UI**: http://localhost:5000
- **PostgreSQL**: Internal database

### 3. Verify Services
```powershell
docker-compose ps
```

All containers should show `Up` status and healthy health checks.

### 4. Run the Pipeline

**Via Airflow UI:**
1. Navigate to http://localhost:8080
2. Login with default credentials (admin/admin)
3. Find `iseg_brain_segmentation_pipeline` DAG
4. Click the **play button** to trigger manually

**Via CLI:**
```powershell
docker exec airflow_scheduler airflow dags trigger iseg_brain_segmentation_pipeline
docker logs -f airflow_scheduler
```

## 📁 Project Structure

```
.
├── src/                             # Main Python package (all production code)
│   ├── __init__.py                  # Package exports (build_unet, load_subjects, etc.)
│   ├── models.py                    # U-Net architecture definition
│   ├── preprocess.py                # Data loading & preprocessing utilities
│   ├── train.py                     # Training pipeline with MLflow integration
│   ├── inference.py                 # Inference with predictions & logging
│   ├── view_hdr.py                  # HDR volume visualization
│   └── view_predict_slice.py        # Prediction slice visualization & logging
├── dags/
│   └── pipeline_dag.py              # Airflow DAG orchestration
├── data/
│   └── iSeg-2017-Training/          # MRI dataset (not in repo)
├── models/
│   └── unet_stage1_4class.keras     # Trained model output
├── outputs/
│   ├── subject-*-pred.nii.gz        # Inference predictions
│   └── pipeline_report_*.json       # Pipeline execution reports
├── logs/
│   └── [Airflow & Docker logs]
├── Dockerfile                       # Training container
├── Dockerfile.airflow               # Airflow extensions
├── docker-compose.yml               # Multi-service orchestration
├── requirements.txt                 # Python dependencies
└── README.md                        # Documentation
```

### Key Files

The project contains only essential files:
- **src/** - All production-ready code in Python package format
- **dags/** - Airflow orchestration definitions
- **data/**, **models/**, **outputs/** - Data directories (mounted in Docker)
- **Dockerfile**, **docker-compose.yml** - Container configuration
- **requirements.txt** - Python dependencies

## 📦 Package Architecture

All code is organized in the `src/` Python package for clean separation between production code and configuration files.

### Execution Flow

**Docker Training:**
```
Docker container runs: python -m src.train
  ↓
src/train.py main() function executes
  ├─ imports from src.models
  ├─ imports from src.preprocess
  └─ logs to MLflow at http://mlflow:5000
```

**Docker Inference:**
```
Airflow DAG runs: python -m src.inference --subject-id N
  ↓
src/inference.py parse_args() and run_inference() execute
  ├─ imports from src.preprocess
  ├─ loads model from /app/models/
  └─ logs to MLflow
```

**Airflow Orchestration:**
```
Airflow DAG (dags/pipeline_dag.py) executes:
  ├─ Data validation (Python in Airflow container)
  ├─ Training: DockerOperator → python -m src.train
  ├─ Inference: DockerOperator → python -m src.inference --subject-id {1,2,3}
  └─ Visualization & reporting (Python in Airflow container)
```

### Module Dependencies

```
src/__init__.py
  └─ exports: build_unet, build_unet_stage1, load_subjects, create_slice_dataset, normalize_volume

src/models.py (dependencies: tensorflow, keras)
  └─ Defines: conv_block(), build_unet(), build_unet_stage1()

src/preprocess.py (dependencies: nibabel, scikit-learn, scikit-image, numpy)
  └─ Defines: load_subjects(), create_slice_dataset(), normalize_volume()

src/train.py (dependencies: src.models, src.preprocess, mlflow, tensorflow)
  └─ Defines: main() - full training pipeline with MLflow tracking

src/inference.py (dependencies: src.preprocess, nibabel, mlflow, tensorflow)
  └─ Defines: parse_args(), run_inference(), _load_volume()

src/view_hdr.py (dependencies: nibabel, matplotlib)
  └─ Defines: view_hdr_volume() - 3D volume visualization

src/view_predict_slice.py (dependencies: nibabel, mlflow, matplotlib)
  └─ Defines: view_and_log_slices(), parse_args() - slice visualization & MLflow logging
```

## 🔧 Configuration

### Environment Variables (docker-compose.yml)

```yaml
MLFLOW_TRACKING_URI: "http://mlflow:5000"
AIRFLOW_UID: 50000
PYTHONUNBUFFERED: "1"
```

### Pipeline Parameters (dags/pipeline_dag.py)

```python
EPOCHS = 10
BATCH_SIZE = 4
MODEL_NAME = "iSeg4ClassUNet"
DATA_DIR = "/app/data/iSeg-2017-Training"
```

### Dataset

The iSeg 2017 dataset must be placed in `./data/iSeg-2017-Training/` with structure:
```
iSeg-2017-Training/
├── subject-1-T1.hdr
├── subject-1-T1.img
├── subject-1-T2.hdr
├── subject-1-T2.img
├── subject-1-label.hdr
├── subject-1-label.img
├── subject-2-T1.hdr
└── [... more subjects ...]
```

## 📊 Pipeline Tasks

### 1. Data Validation (`validate_data`)
- Checks for required data files (T1, T2, labels)
- Counts available subjects
- Validates directory structure

### 2. Model Check (`check_model`)
- Checks if trained model exists
- Determines if retraining is needed

### 3. Training Decision (`decide_train`)
- **Branch Task**: Routes to training or skipping
- If model exists: Skip training
- If not: Proceed to training

### 4. Model Training (`train_model`)
- Runs in Docker container with TensorFlow/Keras
- Trains U-Net for 10 epochs
- Logs metrics to MLflow
- Saves best model checkpoint
- Logs model artifacts

### 5. Inference Tasks (`inference_tasks`)
- **Parallel execution** on subjects 1-3
- Runs in Docker containers
- Generates predictions in NIfTI format
- Logs inference metrics to MLflow

### 6. Visualization (`visualize_predictions`)
- Creates 2D slices from 3D predictions
- Generates comparison images
- Logs visualizations as artifacts

### 7. Reporting (`generate_report`)
- Creates JSON summary report
- Logs pipeline execution details
- Records training/inference metrics

## 📈 MLflow Integration

All experiments and model artifacts are tracked in MLflow:

**Access MLflow UI:** http://localhost:5000

**Features:**
- ✓ Experiment tracking for training runs
- ✓ Metrics logging (loss, accuracy, inference time)
- ✓ Artifacts storage (models, predictions, visualizations)
- ✓ Model versioning and comparison

**Example metrics tracked:**
```
Training:
- loss, accuracy per epoch
- val_loss, val_accuracy
- training_time_sec
- model artifacts

Inference:
- inference_time_sec per subject
- prediction outputs
- visualization images
```

## 🐳 Docker Containers

### iseg_trainer:latest
- **Base**: python:3.9-slim
- **Purpose**: Training & Inference execution
- **Packages**: TensorFlow 2.20, Keras, MLflow 2.16.0, nibabel, scikit-image
- **Volumes**: `/app/data`, `/app/models`, `/app/outputs`

### airflow_webserver & airflow_scheduler
- **Base**: apache/airflow:2.7.3-python3.9
- **Purpose**: Workflow orchestration and UI
- **Features**: Docker provider, MLflow integration

### mlflow_server
- **Image**: ghcr.io/mlflow/mlflow:v2.16.0
- **Purpose**: Experiment tracking and model registry
- **Storage**: SQLite database + artifact store

### postgres
- **Image**: postgres:14
- **Purpose**: Metadata storage for Airflow & MLflow

## 🔄 Workflow

```
START
  ↓
[validate_data] → Checks if data exists
  ↓
[check_model] → Checks if model exists
  ↓
[decide_train] → Branch decision
  ├─ YES → [train_model] → Trains U-Net (MLflow logged)
  └─ NO  → [skip_train] → Dummy task
  ↓
[train_done] → Join point
  ↓
[inference_tasks] → Parallel inference on subjects 1-3
  ├─ [inference_subject_1]
  ├─ [inference_subject_2]
  └─ [inference_subject_3]
  ↓
[visualize_predictions] → Generate visualizations
  ↓
[generate_report] → Create summary report
  ↓
END (Success/Failure)
```

## 📊 Expected Output

After successful pipeline execution:

```
outputs/
├── subject-1-pred.nii.gz          # 3D prediction volume
├── subject-2-pred.nii.gz          # 3D prediction volume
├── subject-3-pred.nii.gz          # 3D prediction volume
├── pipeline_report_20251125.json  # Execution summary
└── [visualization artifacts in MLflow]

models/
└── unet_stage1_4class.keras       # Trained model

MLflow UI (http://localhost:5000):
├── Experiment: iSeg-4Class-Segmentation
│   ├── Training Run
│   │   ├── Parameters: epochs=10, batch_size=4
│   │   ├── Metrics: final_val_loss, final_val_accuracy
│   │   └── Artifacts: trained model
│   └── Multiple Inference Runs
│       ├── Metrics: inference_time_sec
│       └── Artifacts: predictions, visualizations
```

## 🛠️ Development & Debugging

### View Logs

**Airflow Scheduler:**
```powershell
docker logs -f airflow_scheduler
```

**Airflow Webserver:**
```powershell
docker logs -f airflow_webserver
```

**MLflow Server:**
```powershell
docker logs -f mlflow_server
```

**Training Container (during execution):**
```powershell
docker logs <container_id>
```

### Rebuild Containers

If dependencies change:
```powershell
docker build --no-cache -t iseg_trainer:latest .
docker-compose up -d --build
```

### Reset Everything

```powershell
docker-compose down
docker volume rm cpe393_final_mediimgclass_postgres_data cpe393_final_mediimgclass_mlflow_data
docker-compose up -d
```

### Manual Testing

Test training locally:
```powershell
docker run --rm `
  --network iseg_network `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/models:/app/models `
  -e MLFLOW_TRACKING_URI=http://mlflow:5000 `
  iseg_trainer:latest python train.py
```

Test inference:
```powershell
docker run --rm `
  --network iseg_network `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/models:/app/models `
  -v ${PWD}/outputs:/app/outputs `
  -e MLFLOW_TRACKING_URI=http://mlflow:5000 `
  iseg_trainer:latest python inference.py --subject-id 1
```

## ⚠️ Known Issues & Troubleshooting

### MLflow Server Not Healthy
**Symptom**: MLflow container keeps restarting

**Solution**:
```powershell
docker-compose down
Remove-Item -Recurse -Force .\mlflow_data
docker-compose up -d
```

### Training Takes Too Long
**Solution**: Reduce EPOCHS in pipeline_dag.py (default: 10)

### Inference Fails - "ModuleNotFoundError: nibabel"
**Cause**: Airflow container doesn't have ML dependencies

**Solution**: Ensure inference runs in DockerOperator (✓ Fixed in current version)

### Out of Memory
**Solution**: Reduce BATCH_SIZE in train.py (default: 4 → try 2)

## 📚 References

- **iSeg 2017 Dataset**: http://iseg2017.web.unc.edu/
- **Airflow Documentation**: https://airflow.apache.org/
- **MLflow Documentation**: https://mlflow.org/
- **TensorFlow/Keras**: https://www.tensorflow.org/
- **Docker Compose**: https://docs.docker.com/compose/

## 📝 License

Project for CPE393 - Machine Learning in Production

## 👥 Author

Poonnawat Nontanakcheevin 65070503424
Pongpong Prakobnoppakao 65070503426
Pataraphol Pholngam 65070503432
Garice Denoncin 68540460043
Enzhuo Cao 68540470003
Created for CPE393 Final Project - Medical Image Segmentation MLOps Pipeline

---

**Last Updated**: November 25, 2025
**Status**: ✅ Fully Functional - All Components Tested
