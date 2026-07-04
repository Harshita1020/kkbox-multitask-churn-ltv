"""
utils.py
--------
Shared training and evaluation utilities used by all training scripts.

Functions:
  load_manifest()        — load feature_manifest.json
  load_data()            — load parquet splits + build DataLoaders
  get_pos_weight()       — compute BCEWithLogitsLoss pos_weight
  run_epoch()            — one forward pass (train or eval mode)
  train_model()          — full training loop with early stopping
  evaluate_full()        — all metrics on a DataLoader
  get_raw_predictions()  — raw numpy arrays (logits, labels, ltv_pred, ltv_true)
  compute_ece()          — Expected Calibration Error
  plot_training_curves() — save loss/AUC curve plots
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score, r2_score,
    roc_auc_score, roc_curve,
)
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE, DEVICE, EPOCHS, GRAD_CLIP, LR, LR_PATIENCE,
    PATIENCE, PROCESSED_DIR, RESULTS_DIR, WEIGHT_DECAY,
)
from models import KKBoxDataset


# ── Data loading ──────────────────────────────────────────────────────────────
def load_manifest() -> dict:
    with open(os.path.join(PROCESSED_DIR, "feature_manifest.json")) as f:
        return json.load(f)


def load_data(manifest: dict) -> tuple:
    """
    Returns:
      train_df, val_df, test_df
      train_loader, val_loader, test_loader
      cat_cols, num_cols, cardinalities, embed_dims
    """
    cat_cols = [v["column"] for v in manifest["categorical"].values()]
    num_cols = manifest["numerical_scaled"] + manifest["numerical_unscaled"]
    cardinalities = {v["column"]: v["cardinality"]
                     for v in manifest["categorical"].values()}
    embed_dims    = {v["column"]: v["embedding_dim"]
                     for v in manifest["categorical"].values()}

    train_df = pd.read_parquet(os.path.join(PROCESSED_DIR, "model_dataset_train.parquet"))
    val_df   = pd.read_parquet(os.path.join(PROCESSED_DIR, "model_dataset_val.parquet"))
    test_df  = pd.read_parquet(os.path.join(PROCESSED_DIR, "model_dataset_test.parquet"))

    def make_loader(df, shuffle):
        return DataLoader(KKBoxDataset(df, cat_cols, num_cols),
                          batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)

    return (train_df, val_df, test_df,
            make_loader(train_df, True),
            make_loader(val_df,   False),
            make_loader(test_df,  False),
            cat_cols, num_cols, cardinalities, embed_dims)


def get_pos_weight(train_df: pd.DataFrame) -> torch.Tensor:
    """
    pos_weight = N_negative / N_positive.
    Passed to BCEWithLogitsLoss to upweight the minority (churn) class.
    Improves ranking (AUC) but inflates predicted probabilities
    -> requires post-hoc calibration (see 06_calibration_business.py).
    """
    n_neg = (train_df["is_churn"] == 0).sum()
    n_pos = (train_df["is_churn"] == 1).sum()
    pw = torch.tensor(n_neg / n_pos, dtype=torch.float32)
    print(f"  pos_weight = {pw.item():.2f}  (N_neg={n_neg:,}  N_pos={n_pos:,})")
    return pw


# ── Training loop ─────────────────────────────────────────────────────────────
def run_epoch(model, loader, bce_fn, mse_fn,
              lambda_churn, lambda_ltv, optimizer=None):
    """
    One forward pass through all batches.
    optimizer=None  ->  eval mode (no gradients).
    optimizer given ->  train mode (backward + step).
    Returns: (avg_loss, churn_auc, ltv_rmse_log)
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total, n = 0.0, 0
    logits_l, churn_l, ltv_p_l, ltv_t_l = [], [], [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for x_num, x_cat, y_churn, y_ltv in loader:
            x_num, x_cat = x_num.to(DEVICE), x_cat.to(DEVICE)
            y_churn, y_ltv = y_churn.to(DEVICE), y_ltv.to(DEVICE)
            if is_train:
                optimizer.zero_grad()
            logit, ltv_pred = model(x_num, x_cat)
            loss = (lambda_churn * bce_fn(logit, y_churn)
                    + lambda_ltv * mse_fn(ltv_pred, y_ltv))
            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
            total += loss.item() * len(y_churn)
            n     += len(y_churn)
            logits_l.append(logit.detach().cpu())
            churn_l.append(y_churn.cpu())
            ltv_p_l.append(ltv_pred.detach().cpu())
            ltv_t_l.append(y_ltv.cpu())

    logits_np = torch.cat(logits_l).numpy()
    churn_np  = torch.cat(churn_l).numpy()
    ltv_p_np  = torch.cat(ltv_p_l).numpy()
    ltv_t_np  = torch.cat(ltv_t_l).numpy()
    probs = torch.sigmoid(torch.from_numpy(logits_np)).numpy()
    auc   = roc_auc_score(churn_np, probs)
    rmse  = float(np.sqrt(np.mean((ltv_p_np - ltv_t_np)**2)))
    return total / n, auc, rmse


def train_model(model, train_loader, val_loader,
                lambda_churn, lambda_ltv, pos_weight,
                checkpoint_path: str, exp_name: str = "") -> pd.DataFrame:
    """
    Full training loop with Adam + ReduceLROnPlateau + early stopping.
    Saves the best checkpoint and restores it before returning.
    Returns per-epoch history DataFrame.
    """
    model = model.to(DEVICE)
    bce_fn    = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    mse_fn    = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=LR_PATIENCE, factor=0.5, verbose=False
    )

    best_val, no_imp, history = float("inf"), 0, []

    for epoch in range(EPOCHS):
        tr_l, tr_auc, tr_r = run_epoch(
            model, train_loader, bce_fn, mse_fn,
            lambda_churn, lambda_ltv, optimizer)
        vl_l, vl_auc, vl_r = run_epoch(
            model, val_loader, bce_fn, mse_fn,
            lambda_churn, lambda_ltv)
        scheduler.step(vl_l)

        history.append({"epoch": epoch, "train_loss": tr_l, "val_loss": vl_l,
                         "val_auc": vl_auc, "val_rmse_log": vl_r})
        print(f"  [{exp_name}] ep {epoch:2d}  "
              f"tr={tr_l:.4f}  vl={vl_l:.4f}  "
              f"auc={vl_auc:.4f}  rmse={vl_r:.4f}")

        if vl_l < best_val - 1e-5:
            best_val, no_imp = vl_l, 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                print(f"  [{exp_name}] Early stop ep {epoch}  best_val={best_val:.4f}")
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE,
                                     weights_only=True))
    return pd.DataFrame(history)


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_full(model, loader) -> dict:
    """Compute all churn + LTV metrics on a DataLoader."""
    logits, churn_true, ltv_p_log, ltv_t_log = get_raw_predictions(model, loader)
    probs       = 1 / (1 + np.exp(-logits))
    ltv_p_raw   = np.expm1(ltv_p_log)
    ltv_t_raw   = np.expm1(ltv_t_log)
    return {
        "churn_auc_roc":    float(roc_auc_score(churn_true, probs)),
        "churn_auc_pr":     float(average_precision_score(churn_true, probs)),
        "ltv_rmse_log":     float(np.sqrt(np.mean((ltv_p_log - ltv_t_log)**2))),
        "ltv_rmse_raw_twd": float(np.sqrt(np.mean((ltv_p_raw - ltv_t_raw)**2))),
        "ltv_mae_raw_twd":  float(np.mean(np.abs(ltv_p_raw - ltv_t_raw))),
        "ltv_r2_raw":       float(r2_score(ltv_t_raw, ltv_p_raw)),
    }


def get_raw_predictions(model, loader) -> tuple:
    """Returns (logits, churn_true, ltv_pred_log, ltv_true_log) as numpy arrays."""
    model = model.to(DEVICE)
    model.eval()
    logits_l, churn_l, ltv_p_l, ltv_t_l = [], [], [], []
    with torch.no_grad():
        for x_num, x_cat, y_churn, y_ltv in loader:
            logit, ltv_pred = model(x_num.to(DEVICE), x_cat.to(DEVICE))
            logits_l.append(logit.cpu())
            churn_l.append(y_churn)
            ltv_p_l.append(ltv_pred.cpu())
            ltv_t_l.append(y_ltv)
    return (torch.cat(logits_l).numpy(),
            torch.cat(churn_l).numpy(),
            torch.cat(ltv_p_l).numpy(),
            torch.cat(ltv_t_l).numpy())


# ── Calibration ───────────────────────────────────────────────────────────────
def compute_ece(probs: np.ndarray, labels: np.ndarray,
                n_bins: int = 10) -> tuple:
    """
    Expected Calibration Error.
    ECE = sum_b (|B_b|/N) * |avg_confidence(b) - avg_accuracy(b)|
    Returns (ece: float, bin_stats: pd.DataFrame)
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece, n, rows = 0.0, len(probs), []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (probs >= lo) & (probs < hi if i < n_bins-1 else probs <= hi)
        if mask.sum() == 0:
            rows.append((lo, hi, 0, float("nan"), float("nan")))
            continue
        conf = probs[mask].mean()
        acc  = labels[mask].mean()
        ece += (mask.sum() / n) * abs(conf - acc)
        rows.append((lo, hi, int(mask.sum()), float(conf), float(acc)))
    return ece, pd.DataFrame(rows, columns=["bin_lo","bin_hi","n","avg_conf","avg_acc"])


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_training_curves(histories: dict, save_name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, hist in histories.items():
        axes[0].plot(hist["epoch"], hist["val_auc"],      label=name)
        axes[1].plot(hist["epoch"], hist["val_rmse_log"], label=name)
    axes[0].set_title("Val Churn AUC-ROC by epoch")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("AUC"); axes[0].legend(fontsize=7)
    axes[1].set_title("Val LTV RMSE (log) by epoch")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("RMSE"); axes[1].legend(fontsize=7)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, save_name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {path}")
