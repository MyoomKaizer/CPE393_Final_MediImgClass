import os
import glob
import re
import numpy as np
import nibabel as nib
from typing import List, Tuple, Dict
from tensorflow.keras.models import load_model

from .config import MODELS_DIR, DATA_DIR, TRAIN_SUBJECTS, VAL_SUBJECTS, TEST_SUBJECTS

LABEL_BG = 0
LABEL_CSF = 10
LABEL_GM = 150
LABEL_WM = 250


def _sorted_subject_paths() -> List[Dict[str, str]]:
    """Return list of dicts: {"t1": path, "t2": path, "label": path} sorted by subject id."""
    t1_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*T1*.hdr")))
    t2_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*T2*.hdr")))
    gt_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*label*.hdr")))

    def extract_id(p):
        m = re.search(r"subject-(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else -1

    t1_dict = {extract_id(p): p for p in t1_paths}
    t2_dict = {extract_id(p): p for p in t2_paths}
    gt_dict = {extract_id(p): p for p in gt_paths}

    subject_ids = sorted(set(t1_dict.keys()) & set(t2_dict.keys()) & set(gt_dict.keys()))
    subjects = []
    for sid in subject_ids:
        subjects.append({
            "id": sid,
            "t1": t1_dict[sid],
            "t2": t2_dict[sid],
            "label": gt_dict[sid],
        })
    return subjects


def load_subjects():
    """Load all subjects as (id, t1, t2, gt) numpy arrays."""
    subjects_paths = _sorted_subject_paths()
    subjects = []
    for info in subjects_paths:
        t1 = nib.load(info["t1"]).get_fdata()
        t2 = nib.load(info["t2"]).get_fdata()
        gt = nib.load(info["label"]).get_fdata()
        subjects.append((info["id"], t1, t2, gt))
    return subjects


def normalize_volume(vol: np.ndarray) -> np.ndarray:
    """Normalize volume over non-zero voxels. Gaussian normalization."""
    mask = vol > 0
    if not np.any(mask):
        return vol
    data = vol[mask]
    mean = data.mean()
    std = data.std() if data.std() > 0 else 1.0
    norm = np.zeros_like(vol, dtype=np.float32)
    norm[mask] = (vol[mask] - mean) / (5.0 * std)
    return norm


def normalize_subjects(subjects):
    norm_subjects = []
    for sid, t1, t2, gt in subjects:
        t1_n = normalize_volume(t1)
        t2_n = normalize_volume(t2)
        norm_subjects.append((sid, t1_n, t2_n, gt))
    return norm_subjects


def split_subjects(norm_subjects):
    """Return dicts train/val/test mapping id -> (t1,t2,gt)."""
    by_id = {sid: (t1, t2, gt) for sid, t1, t2, gt in norm_subjects}
    train = {sid: by_id[sid] for sid in TRAIN_SUBJECTS}
    val = {sid: by_id[sid] for sid in VAL_SUBJECTS}
    test = {sid: by_id[sid] for sid in TEST_SUBJECTS}
    return train, val, test


def _build_slices_for_stage_1(
    subjects: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    min_foreground: int = 100,
):
    """Create X, Y for a given stage from a dict {sid: (t1,t2,gt)}.

    Stage 1: X=(T1,T2), Y=foreground(gt>0)
    Stage 2: X=(T1,T2, gt>=10), Y=(gt>=150)
    Stage 3: X=(T1,T2, gt>=10, gt>=150), Y=(gt>=250)
    """
    X_list = []
    Y_list = []

    for sid, (t1, t2, gt) in subjects.items():
        sx, sy, sz = gt.shape

        for z in range(sz):
            gt_slice = gt[:, :, z]
            if np.sum(gt_slice > 0) < min_foreground:
                continue

            t1_s = t1[:, :, z]
            t2_s = t2[:, :, z]

            x = np.stack([t1_s, t2_s], axis=-1)
            y = (gt_slice > 0).astype(np.float32)[..., None]
            # elif stage == 2:
            #     fg = (gt_slice >= LABEL_CSF).astype(np.float32)
            #     x = np.stack([t1_s, t2_s, fg], axis=-1)
            #     y = (gt_slice >= LABEL_GM).astype(np.float32)[..., None]
            # else:  # stage 3
            #     fg = (gt_slice >= LABEL_CSF).astype(np.float32)
            #     gmwm = (gt_slice >= LABEL_GM).astype(np.float32)
            #     x = np.stack([t1_s, t2_s, fg, gmwm], axis=-1)
            #     y = (gt_slice >= LABEL_WM).astype(np.float32)[..., None]

            X_list.append(x)
            Y_list.append(y)

    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)
    return X, Y

def _build_slices_for_stage_2(
    subjects: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    min_foreground: int = 100,
):
    m1 = load_model(os.path.join(MODELS_DIR, "stage1.h5"), compile=False)
    X_list = []
    Y_list = []

    for sid, (t1, t2, gt) in subjects.items():
        sx, sy, sz = gt.shape

        for z in range(sz):
            gt_slice = gt[:, :, z]
            if np.sum(gt_slice > 0) < min_foreground:
                continue

            t1_s = t1[:, :, z]
            t2_s = t2[:, :, z]

            x1 = np.stack([t1_s, t2_s], axis=-1)[None, ...]
            pred_fg = m1.predict(x1)[..., 0] # shape (height, width)
            fg = (pred_fg > 0.5).astype(np.float32)

            x2 = np.stack([t1_s, t2_s, fg], axis=-1)
            y = (gt_slice >= LABEL_GM).astype(np.float32)[..., None]
            # else:  # stage 3
            #     fg = (gt_slice >= LABEL_CSF).astype(np.float32)
            #     gmwm = (gt_slice >= LABEL_GM).astype(np.float32)
            #     x = np.stack([t1_s, t2_s, fg, gmwm], axis=-1)
            #     y = (gt_slice >= LABEL_WM).astype(np.float32)[..., None]

            X_list.append(x2)
            Y_list.append(y)

    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)
    return X, Y

def _build_slices_for_stage_3(
    subjects: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    min_foreground: int = 100,
):
    m1 = load_model(os.path.join(MODELS_DIR, "stage1.h5"), compile=False)
    m2 = load_model(os.path.join(MODELS_DIR, "stage2.h5"), compile=False)

    X_list = []
    Y_list = []

    for sid, (t1, t2, gt) in subjects.items():
        sx, sy, sz = gt.shape

        for z in range(sz):
            gt_slice = gt[:, :, z]
            if np.sum(gt_slice > 0) < min_foreground:
                continue

            t1_s = t1[:, :, z]
            t2_s = t2[:, :, z]

            x1 = np.stack([t1_s, t2_s], axis=-1)[None, ...]
            pred_fg = m1.predict(x1)[..., 0] # shape (height, width)
            fg = (pred_fg > 0.5).astype(np.float32)

            x2 = np.stack([t1_s, t2_s, fg], axis=-1)[None, ...]
            pred_gmwm = m2.predict(x2)[..., 0]
            gmwm = (pred_gmwm > 0.5).astype(np.float32)

            x3 = np.stack([t1_s, t2_s, fg, gmwm], axis=-1)
            y = (gt_slice >= LABEL_WM).astype(np.float32)[..., None]

            X_list.append(x3)
            Y_list.append(y)

    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)
    return X, Y

def build_train_val_data(stage: int):
    subjects = load_subjects()
    norm_subjects = normalize_subjects(subjects)
    train_dict, val_dict, _ = split_subjects(norm_subjects)
    if stage == 1:
        X_train, Y_train = _build_slices_for_stage_1(train_dict)
        X_val, Y_val = _build_slices_for_stage_1(val_dict)
    elif stage == 2:
        X_train, Y_train = _build_slices_for_stage_2(train_dict)
        X_val, Y_val = _build_slices_for_stage_2(val_dict)
    else:
        X_train, Y_train = _build_slices_for_stage_3(train_dict)
        X_val, Y_val = _build_slices_for_stage_3(val_dict)
    return X_train, Y_train, X_val, Y_val


def build_test_volumes():
    """Return test dict {sid: (t1, t2, gt)} normalized, for volume-wise eval."""
    subjects = load_subjects()
    norm_subjects = normalize_subjects(subjects)
    _, _, test_dict = split_subjects(norm_subjects)
    return test_dict
