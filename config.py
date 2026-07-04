"""
config.py
---------
Central configuration: all paths, constants, and feature lists.
Auto-detects whether running on Kaggle or local laptop.
Import this at the top of every other script.
"""

import os
from pathlib import Path
import torch

# ── Root directory ────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent

# ── Auto-detect environment ───────────────────────────────────────────────────
ON_KAGGLE = os.path.exists("/kaggle/working")

if ON_KAGGLE:
    # Kaggle paths
    RAW_DIR       = Path("/kaggle/input/kkbox-churn-prediction-challenge")
    PROCESSED_DIR = Path("/kaggle/working/processed")
    MODELS_DIR    = Path("/kaggle/working/models")
    RESULTS_DIR   = Path("/kaggle/working/results")
else:
    # Local laptop paths
    RAW_DIR       = ROOT / "data" / "raw"
    PROCESSED_DIR = ROOT / "data" / "processed"
    MODELS_DIR    = ROOT / "models"
    RESULTS_DIR   = ROOT / "results"

# Create output dirs (RAW_DIR already exists on Kaggle, don't touch it)
for _d in [PROCESSED_DIR, MODELS_DIR, RESULTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Temporal split constants ──────────────────────────────────────────────────
# All input features may only see data ON OR BEFORE this date
FEATURE_CUTOFF_INT   = 20161231
LTV_TARGET_START     = 20170101   # LTV target window starts here (POST cutoff)
LTV_TARGET_END       = 20170228   # LTV target window ends here
ENGAGEMENT_WIN_START = 20161202   # 30-day engagement window (pre-cutoff)
ENGAGEMENT_WIN_END   = 20161231

# ── Feature column groups ─────────────────────────────────────────────────────
CATEGORICAL_COLS = ["city", "gender", "registered_via", "payment_method_id"]

EMBEDDING_DIMS = {
    "city":              5,
    "gender":            2,
    "registered_via":    3,
    "payment_method_id": 8,
}

# z-score scaled (unbounded range)
SCALE_COLS = [
    "bd_clean",
    "registration_tenure_days",
    "avg_payment_plan_days",
    "avg_actual_amount_paid",
    "num_transactions",
    "total_secs_log",
    "daily_active_days",
]

# Already in [0,1] — left unscaled
UNSCALED_NUM_COLS = ["is_auto_renew_rate", "avg_song_completion"]

# Final column names fed into the model
NUM_COLS = [f"{c}_scaled" for c in SCALE_COLS] + UNSCALED_NUM_COLS
CAT_COLS = [f"{c}_enc"    for c in CATEGORICAL_COLS]

# ── Training hyperparameters ──────────────────────────────────────────────────
BATCH_SIZE   = 2048
EPOCHS       = 50
PATIENCE     = 10     # early stopping patience
LR_PATIENCE  = 5      # ReduceLROnPlateau patience
LR           = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 1.0
RANDOM_SEED  = 42
FM_K         = 8
BACKBONE_DIMS = (256, 128, 64)
DROPOUT_RATES = (0.3, 0.3, 0.2)

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Checkpoint paths ──────────────────────────────────────────────────────────
CKPT = {
    "exp1": str(MODELS_DIR / "exp1_churn_only.pt"),
    "exp2": str(MODELS_DIR / "exp2_ltv_only.pt"),
    "exp3": str(MODELS_DIR / "exp3_fixed_5050.pt"),
    "exp4": str(MODELS_DIR / "exp4_churn_dominant.pt"),
    "exp5": str(MODELS_DIR / "exp5_ltv_dominant.pt"),
    "exp6": str(MODELS_DIR / "exp6_uncertainty.pt"),
    "exp7": str(MODELS_DIR / "exp7_pcgrad.pt"),
}

# ── Helper ────────────────────────────────────────────────────────────────────
def parquet(name: str) -> str:
    return str(PROCESSED_DIR / f"{name}.parquet")


if __name__ == "__main__":
    print(f"Environment : {'Kaggle' if ON_KAGGLE else 'Local'}")
    print(f"Device      : {DEVICE}")
    print(f"RAW_DIR     : {RAW_DIR}")
    print(f"PROCESSED   : {PROCESSED_DIR}")
    print(f"MODELS      : {MODELS_DIR}")
    print(f"RESULTS     : {RESULTS_DIR}")
