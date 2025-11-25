r"""
inference.py

Run 4-class segmentation with trained Stage 1 U-Net.

Usage (PowerShell):

    (venv) PS> python inference.py `
        --t1_path .\iSeg-2017-Training\subject-1-T1.hdr `
        --t2_path .\iSeg-2017-Training\subject-1-T2.hdr `
        --model_path .\unet_stage1_4class.keras `
        --out_path subject-1-pred.nii.gz

Requirements:
    - A trained 4-class model (.keras), e.g. unet_stage1_4class.keras
"""

import argparse
import os
import time
import mlflow
import nibabel as nib
import numpy as np
from tensorflow.keras.models import load_model

from preprocess import normalize_volume

LABEL_VALUES = [0, 10, 150, 250]
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")

def parse_args():
    parser = argparse.ArgumentParser(description="Run 4-class U-Net inference")
    
    # Allow either specific paths OR subject-id for automated pipelines
    parser.add_argument("--t1_path", type=str, default=None, help="Path to T1 volume")
    parser.add_argument("--t2_path", type=str, default=None, help="Path to T2 volume")
    parser.add_argument("--model_path", type=str, default="/app/models/unet_stage1_4class.keras", help="Path to .keras model")
    parser.add_argument("--out_path", type=str, default=None, help="Output NIfTI path")
    parser.add_argument("--subject-id", type=int, default=None, help="Subject ID for automated inference")
    parser.add_argument("--data-dir", type=str, default="/app/data/iSeg-2017-Training", help="Data directory")
    parser.add_argument("--output-dir", type=str, default="/app/outputs", help="Output directory")
    
    args = parser.parse_args()
    
    # If subject-id is provided, construct paths automatically
    if args.subject_id is not None:
        data_dir = args.data_dir
        output_dir = args.output_dir
        args.t1_path = f"{data_dir}/subject-{args.subject_id}-T1.hdr"
        args.t2_path = f"{data_dir}/subject-{args.subject_id}-T2.hdr"
        args.out_path = f"{output_dir}/subject-{args.subject_id}-pred.nii.gz"
    
    # Validate required paths
    if args.t1_path is None or args.t2_path is None or args.out_path is None:
        parser.error("Either provide --subject-id OR all of --t1_path, --t2_path, --out_path")
    
    return args

def _load_volume(path: str):
    img = nib.load(path)
    vol = img.get_fdata()
    if vol.ndim == 4 and vol.shape[-1] == 1:
        vol = vol[..., 0]
    return img, vol


def run_inference(t1_path, t2_path, model_path, out_path):

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("iSeg-4Class-Inference")
    print("setting tracking uri to http://")
    with mlflow.start_run(run_name="InferenceRun"):

        mlflow.log_param("t1_path", t1_path)
        mlflow.log_param("t2_path", t2_path)
        mlflow.log_param("model_path", model_path)

        start = time.time()

        # Load model
        model = load_model(model_path, compile=False)

        t1_img, t1_vol = _load_volume(t1_path)
        _, t2_vol = _load_volume(t2_path)

        t1_norm = normalize_volume(t1_vol)
        t2_norm = normalize_volume(t2_vol)
        sx, sy, sz = t1_norm.shape

        pred_volume = np.zeros((sx, sy, sz), dtype=np.uint16)

        for z in range(sz):
            x = np.stack(
                [t1_norm[:, :, z], t2_norm[:, :, z]],
                axis=-1
            )[None, ...]

            probs = model.predict(x, verbose=0)[0]
            class_idx = np.argmax(probs, axis=-1)

            out_slice = np.zeros_like(class_idx)
            for i, v in enumerate(LABEL_VALUES):
                out_slice[class_idx == i] = v

            pred_volume[:, :, z] = out_slice

        pred_img = nib.Nifti1Image(pred_volume, affine=t1_img.affine)
        nib.save(pred_img, out_path)

        mlflow.log_artifact(out_path)

        mlflow.log_metric("inference_time_sec", time.time() - start)

        print("Inference complete!")
        print("Saved:", out_path)

if __name__ == "__main__":
    args = parse_args()
    run_inference(
        t1_path=args.t1_path,
        t2_path=args.t2_path,
        model_path=args.model_path,
        out_path=args.out_path,
    )
