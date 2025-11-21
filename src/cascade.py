import os
import argparse
import numpy as np
import nibabel as nib
from tensorflow.keras.models import load_model

from .config import MODELS_DIR
from .data import build_test_volumes, LABEL_CSF, LABEL_GM, LABEL_WM


def run_cascade(out_path):
    subjects = build_test_volumes()

    m1 = load_model(os.path.join(MODELS_DIR, "stage1.h5"), compile=False)
    m2 = load_model(os.path.join(MODELS_DIR, "stage2.h5"), compile=False)
    m3 = load_model(os.path.join(MODELS_DIR, "stage3.h5"), compile=False)

    for sid, (t1, t2, gt) in subjects.items():
        sx, sy, sz = gt.shape
        prediction_final = np.zeros((sx, sy, sz), dtype=np.uint8)
        for z in range(sz):
            gt_slice = gt[:, :, z]
            if np.sum(gt_slice > 0) < 100:
                continue

            t1_s = t1[:, :, z]
            t2_s = t2[:, :, z]

            x1 = np.stack([t1_s, t2_s], axis=-1)[None, ...]
            pred_fg = m1.predict(x1)[..., 0] # shape (height, width)
            fg = (pred_fg > 0.5).astype(np.float32)

            x2 = np.stack([t1_s, t2_s, fg], axis=-1)[None, ...]
            pred_gmwm = m2.predict(x2)[..., 0]
            gmwm = (pred_gmwm > 0.5).astype(np.float32)

            x3 = np.stack([t1_s, t2_s, fg, gmwm], axis=-1)[None, ...]
            pred_gm = m3.predict(x3)[..., 0]
            gm = (pred_gm > 0.5).astype(np.float32)
                
            prediction_final[:, :, z] = 10*fg + 140*gmwm + 100*gm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run_cascade(args.t1, args.t2, args.out)


if __name__ == "__main__":
    main()
