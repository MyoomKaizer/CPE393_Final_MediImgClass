"""
iSeg Brain Segmentation MLOps Pipeline

A complete MLOps pipeline for 4-class brain MRI segmentation using
Airflow, MLflow, Docker, and TensorFlow/Keras.
"""

__version__ = "1.0.0"
__author__ = "CPE393 Team"

from .models import build_unet, build_unet_stage1
from .preprocess import load_subjects, create_slice_dataset, normalize_volume

__all__ = [
    "build_unet",
    "build_unet_stage1",
    "load_subjects",
    "create_slice_dataset",
    "normalize_volume",
]
