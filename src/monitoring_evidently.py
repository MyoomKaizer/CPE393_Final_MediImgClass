import os
import json
import glob
import sys
import numpy as np
import nibabel as nib
import pandas as pd
import mlflow
from scipy.ndimage import label as cc_label
from skimage.util import img_as_ubyte
from skimage.filters import threshold_otsu
from skimage.measure import regionprops
from scipy.stats import entropy
from evidently import Report, Dataset, DataDefinition, MulticlassClassification
from evidently.presets import DataDriftPreset, ClassificationPreset

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import load_subjects, normalize_volume, VALUE_TO_INDEX

DATA_DIR = "/app/data/iSeg-2017-Training"
OUTPUTS_DIR = "/app/outputs"
MONITORING_OUTPUT = os.path.join(OUTPUTS_DIR, "monitoring")
os.makedirs(MONITORING_OUTPUT, exist_ok=True)

LABEL_VALUES = [0, 10, 150, 250]
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MAX_ROWS = 200_000


def compute_segmentation_features(seg_slice):
    seg_slice = np.asarray(seg_slice).flatten()

    hist = np.bincount(seg_slice, minlength=4)
    total = hist.sum() if hist.sum() > 0 else 1

    features = {
        "seg_fg_ratio": 1.0 - (hist[0] / total),
        "seg_class1_ratio": hist[1] / total,
        "seg_class2_ratio": hist[2] / total,
        "seg_class3_ratio": hist[3] / total,
        "seg_entropy": float(entropy(hist / total)),
    }

    side = int(np.sqrt(seg_slice.size))
    if side * side == seg_slice.size:
        mask = (seg_slice.reshape(side, side) > 0).astype(np.uint8)
        labeled, num_components = cc_label(mask)
        features["seg_num_components"] = int(num_components)

        if num_components > 0:
            props = regionprops(labeled)
            largest = max(props, key=lambda p: p.area)
            features["seg_largest_area"] = float(largest.area)
            features["seg_bbox_area"] = float(largest.bbox_area)
        else:
            features["seg_largest_area"] = 0.0
            features["seg_bbox_area"] = 0.0
    else:
        features["seg_num_components"] = 0
        features["seg_largest_area"] = 0.0
        features["seg_bbox_area"] = 0.0

    return features


def load_reference_dataset():
    print("Loading reference dataset using src.preprocess.load_subjects...")
    subjects = load_subjects(DATA_DIR)

    slices_list = []
    for t1_vol, t2_vol, label_vol in subjects:
        t1_norm = normalize_volume(t1_vol)
        t2_norm = normalize_volume(t2_vol)
        for z in range(t1_norm.shape[2]):
            t1_slice = t1_norm[:, :, z].flatten()
            t2_slice = t2_norm[:, :, z].flatten()
            features = np.stack([t1_slice, t2_slice], axis=1)
            label_slice = label_vol[:, :, z].flatten()
            label_indices = np.array([VALUE_TO_INDEX.get(v, 0) for v in label_slice])
            df_slice = pd.DataFrame(features, columns=["T1", "T2"])
            df_slice["label"] = label_indices
            df_slice["prediction"] = label_indices

            seg_features = compute_segmentation_features(label_indices)
            for k, v in seg_features.items():
                df_slice[k] = v

            slices_list.append(df_slice)

    df_ref = pd.concat(slices_list, ignore_index=True)
    if len(df_ref) > MAX_ROWS:
        df_ref = df_ref.sample(MAX_ROWS, random_state=42)
    print(f"Reference dataset shape: {df_ref.shape}")
    return df_ref

def _load_volume(path: str):
    img = nib.load(path)
    vol = img.get_fdata()
    if vol.ndim == 4 and vol.shape[-1] == 1:
        vol = vol[..., 0]
    return vol

def load_current_dataset():
    print("Loading current dataset from outputs and data dir...")
    pred_files = sorted(glob.glob(os.path.join(OUTPUTS_DIR, "subject-*-pred.nii.gz")))

    slices_list = []
    for pred_path in pred_files:
        base_name = os.path.basename(pred_path).replace("-pred.nii.gz", "")
        subject_id = base_name.split("-")[1]

        t1_path = os.path.join(DATA_DIR, f"subject-{subject_id}-T1.hdr")
        t2_path = os.path.join(DATA_DIR, f"subject-{subject_id}-T2.hdr")
        label_path = os.path.join(DATA_DIR, f"subject-{subject_id}-label.hdr")

        if not (os.path.isfile(t1_path) and os.path.isfile(t2_path) and os.path.isfile(label_path)):
            print(f"Missing data for subject {subject_id}, skipping...")
            continue

        t1_vol = normalize_volume(_load_volume(t1_path))
        t2_vol = normalize_volume(_load_volume(t2_path))
        label_vol = _load_volume(label_path)

        pred_vol = _load_volume(pred_path)
        pred_idx = np.zeros_like(pred_vol, dtype=np.int32)
        for i, v in enumerate(LABEL_VALUES):
            pred_idx[pred_vol == v] = i

        for z in range(t1_vol.shape[2]):
            t1_slice = t1_vol[:, :, z].flatten()
            t2_slice = t2_vol[:, :, z].flatten()
            features = np.stack([t1_slice, t2_slice], axis=1)
            label_slice = label_vol[:, :, z].flatten()
            label_indices = np.array([VALUE_TO_INDEX.get(v, 0) for v in label_slice])
            pred_slice = pred_idx[:, :, z].flatten()

            df_slice = pd.DataFrame(features, columns=["T1", "T2"])
            df_slice["label"] = label_indices
            df_slice["prediction"] = pred_slice

            seg_features = compute_segmentation_features(pred_slice)
            for k, v in seg_features.items():
                df_slice[k] = v

            slices_list.append(df_slice)

    df_cur = pd.concat(slices_list, ignore_index=True)
    if len(df_cur) > MAX_ROWS:
        df_cur = df_cur.sample(MAX_ROWS, random_state=42)
    print(f"Current dataset shape: {df_cur.shape}")
    return df_cur

def main():
    print("Starting Evidently monitoring with new API...")

    df_ref = load_reference_dataset()
    df_cur = load_current_dataset()

    definition = DataDefinition(
        classification=[
            MulticlassClassification(
                target="label",
                prediction_labels="prediction"
            )
        ]
    )

    ref_dataset = Dataset.from_pandas(df_ref, data_definition=definition)
    cur_dataset = Dataset.from_pandas(df_cur, data_definition=definition)

    report = Report([DataDriftPreset(), ClassificationPreset()])
    eval_result = report.run(reference_data=ref_dataset, current_data=cur_dataset)

    html_path = os.path.join(MONITORING_OUTPUT, "evidently_report.html")
    json_path = os.path.join(MONITORING_OUTPUT, "evidently_report.json")
    eval_result.save_html(html_path)
    eval_result.save_json(json_path)
    print(f"Saved Evidently report HTML to: {html_path}")
    print(f"Saved Evidently report JSON to: {json_path}")

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("iSeg-4Class-Segmentation")
    with mlflow.start_run(run_name="monitoring"):
        mlflow.log_artifact(html_path)
        mlflow.log_artifact(json_path)

    try:
        result_dict = eval_result.dict()
    except AttributeError:
        result_dict = eval_result.as_dict()

    metrics_list = result_dict.get("metrics", [])
    data_drift_detected = any(
        m.get("dataset_drift", {}).get("data", {}).get("share_of_drifted_columns", 0) > 0.3
        for m in metrics_list
        if "dataset_drift" in m
    )
    perf_metrics = [
        m for m in metrics_list
        if "classification_performance" in m
    ]
    model_degradation_detected = False
    if perf_metrics:
        current_f1 = perf_metrics[0]["classification_performance"].get("current", {}).get("f1_score", 1.0)
        reference_f1 = perf_metrics[0]["classification_performance"].get("reference", {}).get("f1_score", 1.0)
        model_degradation_detected = current_f1 < reference_f1 * 0.9

    status = {
        "data_drift_detected": data_drift_detected,
        "model_degradation_detected": model_degradation_detected,
        "message": (
            "Warning: DATA DRIFT DETECTED" if data_drift_detected else "No data drift detected"
        ) + "; " + (
            "Warning: MODEL DEGRADATION DETECTED" if model_degradation_detected else "No model degradation detected"
        )
    }
    status_file = os.path.join(MONITORING_OUTPUT, "monitoring_status.json")
    with open(status_file, "w") as f:
        json.dump(status, f, indent=2)
    print(f"Monitoring status saved to: {status_file}")

if __name__ == "__main__":
    main()
