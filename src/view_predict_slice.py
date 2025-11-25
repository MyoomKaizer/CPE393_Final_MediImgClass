"""
view_predict_slice.py

Visualize prediction slices from a NIfTI segmentation output
and log slice PNGs to MLflow as artifacts.

Usage:
    python view_predict_slice.py --pred_path subject-01-pred.nii.gz
"""

import argparse
import os
import sys
import mlflow
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")


def view_and_log_slices(pred_path, mlflow_uri=MLFLOW_URI):    

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("iSeg-4Class-Visualization")

    with mlflow.start_run(run_name="SliceViewer"):

        mlflow.log_param("prediction_file", pred_path)

        img = nib.load(pred_path)
        vol = img.get_fdata()

        label_values = np.array([0, 10, 150, 250])
        colors = [
            "#000000",
            "#A7D8FF",
            "#B8A19C",
            "#F4F1DE",
        ]
        cmap = ListedColormap(colors)

        val_to_idx = {v: i for i, v in enumerate(label_values)}
        vol_idx = np.vectorize(val_to_idx.get)(vol)

        step = 10
        slice_indices = list(range(0, vol_idx.shape[2], step))

        out_dir = "slice_previews"
        os.makedirs(out_dir, exist_ok=True)

        for z in slice_indices:
            plt.figure(figsize=(4, 4))
            plt.imshow(vol_idx[:, :, z], cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
            plt.title(f"Slice {z}")
            plt.axis("off")

            save_path = os.path.join(out_dir, f"slice_{z}.png")
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()

            mlflow.log_artifact(save_path)

        print("All preview slices logged to MLflow!")
        print("Run URL:", mlflow.get_artifact_uri())


def parse_args():
    p = argparse.ArgumentParser(description="Visualize predicted segmentation slices.")
    p.add_argument("--pred_path", required=True, help="Path to .nii or .nii.gz prediction file")
    return p.parse_args()


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    args = parse_args()
    view_and_log_slices(args.pred_path)
