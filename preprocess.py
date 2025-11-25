"""
preprocess.py

Data loading and preprocessing utilities for iSeg 2017.

- Loads T1/T2/label volumes from iSeg-2017-Training
- Normalizes each volume
- Converts 3D volumes into 2D slices for training the U-Net (Stage 1).

This version supports 4-class segmentation with labels:
    0 (background), 10 (CSF), 150 (gray matter), 250 (white matter).
"""

import os
from glob import glob
from typing import List, Tuple

import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split

# Original iSeg label values (intensity values in label volumes)
LABEL_VALUES = [0, 10, 150, 250]
VALUE_TO_INDEX = {v: i for i, v in enumerate(LABEL_VALUES)}


def _load_volume(path: str) -> np.ndarray:
    """Load a NIfTI / Analyze volume and return a 3D numpy array.

    Handles both .hdr/.img and .nii/.nii.gz. Squeezes a trailing size-1
    dimension if present, so shapes become (H, W, Z).
    """
    img = nib.load(path)
    vol = img.get_fdata()

    # Some iSeg images are stored as (H, W, Z, 1)
    if vol.ndim == 4 and vol.shape[-1] == 1:
        vol = vol[..., 0]

    if vol.ndim != 3:
        raise ValueError(f"Unsupported volume shape {vol.shape} for file {path}")

    return vol


def normalize_volume(vol: np.ndarray) -> np.ndarray:
    """Normalize a volume to zero mean, unit variance (on non-zero voxels)."""
    vol = vol.astype(np.float32)
    mask = vol != 0
    if not np.any(mask):
        return vol

    mean = vol[mask].mean()
    std = vol[mask].std()
    if std < 1e-6:
        std = 1.0
    vol[mask] = (vol[mask] - mean) / std
    return vol


def load_subjects(data_dir: str):
    """
    Load subjects from iSeg-2017-Training folder.
    Correctly supports the .hdr/.img pairs:
        subject-1-T1.hdr
        subject-1-T2.hdr
        subject-1-label.hdr
    """

    subjects = []

    # Find all T1 files
    t1_list = sorted(glob(os.path.join(data_dir, "subject-*-T1.hdr")))

    if not t1_list:
        raise SystemExit("No T1 volumes found in dataset folder.")

    for t1_path in t1_list:
        # Extract subject number
        # e.g. subject-1-T1.hdr → subject-1
        base = t1_path.replace("-T1.hdr", "")

        t2_path = base + "-T2.hdr"
        lbl_path = base + "-label.hdr"

        if not os.path.isfile(t2_path):
            print(f"[WARN] Missing T2 for {t1_path}, skipping.")
            continue

        if not os.path.isfile(lbl_path):
            print(f"[WARN] Missing label for {t1_path}, skipping.")
            continue

        t1_vol = _load_volume(t1_path)
        t2_vol = _load_volume(t2_path)
        label_vol = _load_volume(lbl_path)

        subjects.append((t1_vol, t2_vol, label_vol))

    if not subjects:
        raise SystemExit("No valid subjects found (T1+T2+label).")

    print(f"Loaded {len(subjects)} subjects.")
    return subjects


def create_slice_dataset(
    subjects: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    test_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create 2D slice dataset from 3D subjects for 4-class segmentation.

    For each subject and axial slice index z:
        - X slice: stack of normalized T1 and T2 → shape (H, W, 2)
        - Y slice: class index (0..3) for each pixel → shape (H, W, 1)

    The label mapping is defined by LABEL_VALUES.
    """
    X_slices = []
    Y_slices = []

    for idx, (t1_vol, t2_vol, lbl_vol) in enumerate(subjects):
        print(f"Processing subject {idx+1}/{len(subjects)} with shape {t1_vol.shape}")

        t1_norm = normalize_volume(t1_vol)
        t2_norm = normalize_volume(t2_vol)

        if t1_norm.shape != t2_norm.shape or t1_norm.shape != lbl_vol.shape:
            raise ValueError(
                f"Shape mismatch in subject {idx}: "
                f"T1 {t1_norm.shape}, T2 {t2_norm.shape}, label {lbl_vol.shape}"
            )

        sx, sy, sz = t1_norm.shape

        for z in range(sz):
            t1_slice = t1_norm[:, :, z]
            t2_slice = t2_norm[:, :, z]
            lbl_slice = lbl_vol[:, :, z]

            # Map raw label values (0,10,150,250) to class indices 0..3
            class_idx = np.zeros_like(lbl_slice, dtype=np.int32)
            for value, class_id in VALUE_TO_INDEX.items():
                class_idx[lbl_slice == value] = class_id

            x = np.stack([t1_slice, t2_slice], axis=-1)  # (H, W, 2)
            y = class_idx[..., np.newaxis]  # (H, W, 1)

            X_slices.append(x)
            Y_slices.append(y)

    X = np.asarray(X_slices, dtype=np.float32)
    Y = np.asarray(Y_slices, dtype=np.int32)

    print("Total slices:", X.shape[0])
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=test_size, random_state=random_state, shuffle=True
    )
    print("Train slices:", X_train.shape[0])
    print("Val slices:", X_val.shape[0])

    return X_train, X_val, Y_train, Y_val


if __name__ == "__main__":
    # Quick test (you must have iSeg-2017-Training extracted)
    data_dir = "./iSeg-2017-Training"
    if not os.path.isdir(data_dir):
        raise SystemExit(f"Directory not found: {data_dir}")

    subjects = load_subjects(data_dir)
    X_train, X_val, Y_train, Y_val = create_slice_dataset(subjects)
    print("X_train:", X_train.shape)
    print("Y_train:", Y_train.shape)
    print("X_val:", X_val.shape)
    print("Y_val:", Y_val.shape)

    
