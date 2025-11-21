import os
import argparse
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from .config import MODELS_DIR, STAGE1_CFG, STAGE2_CFG, STAGE3_CFG
from .data import build_train_val_data, normalize_volume, LABEL_CSF, LABEL_GM, LABEL_WM
from .models import unet


def train_unet_stage(stage: int):
    if stage == 1:
        cfg = STAGE1_CFG
    elif stage == 2:
        cfg = STAGE2_CFG
    else:
        cfg = STAGE3_CFG

    print(f"Training stage {stage} with cfg {cfg}")
    X_train, Y_train, X_val, Y_val = build_train_val_data(stage)

    input_shape = X_train.shape[1:]
    model = unet(input_shape, lr=cfg["lr"])

    save_path = os.path.join(MODELS_DIR, f"stage{stage}.h5")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(save_path, monitor="val_loss", save_best_only=True),
    ]

    model.fit(
        X_train,
        Y_train,
        validation_data=(X_val, Y_val),
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        callbacks=callbacks,
    )

    print(f"Saved stage {stage} model to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], required=True)
    args = parser.parse_args()
    train_unet_stage(args.stage)


if __name__ == "__main__":
    main()
