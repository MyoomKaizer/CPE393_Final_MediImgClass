import os

# Root directories
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "train")
MODELS_DIR = os.path.join(PROJECT_ROOT, "saved_models")

os.makedirs(MODELS_DIR, exist_ok=True)

# Subject indices (match your notebook)
TRAIN_SUBJECTS = list(range(0, 6))   # 0–5
VAL_SUBJECTS = [6, 7]                # 6–7
TEST_SUBJECTS = [8, 9]               # 8–9

# Training hyperparameters
STAGE1_CFG = {
    "epochs": 20,
    "batch_size": 1,
    "lr": 1e-3,
}

STAGE2_CFG = {
    "epochs": 20,
    "batch_size": 1,
    "lr": 1e-3,
}

STAGE3_CFG = {
    "epochs": 30,
    "batch_size": 1,
    "lr": 1e-3,
}
