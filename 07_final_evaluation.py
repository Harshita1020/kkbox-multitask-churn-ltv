"""
07_final_evaluation.py
----------------------
Stage 7: Consolidated final evaluation of the selected model.

Selected model: Exp-6 (uncertainty weighting) + isotonic calibration.
Reason: only experiment that simultaneously matches or exceeds BOTH
        single-task ceilings (Exp-1 churn AUC and Exp-2 LTV RMSE).

HOW TO RUN (after Stage 6):
  python 07_final_evaluation.py

OUTPUTS:
  results/final_model_metrics.json
  results/final_roc_pr_curves.png
  results/final_ltv_scatter.png
  results/final_summary.txt
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score, precision_recall_curve,
    r2_score, roc_auc_score, roc_curve,
)

from config import CKPT, DEVICE, PROCESSED_DIR, RANDOM_SEED, RESULTS_DIR
from models import MultiTaskFMNet
from utils import get_raw_predictions, load_data, load_manifest


def main():
    print("=== Stage 7: Final Evaluation Summary ===")
    print("  Model: Exp-6 (uncertainty weighting) + isotonic calibration\n")

    manifest = load_manifest()
    (train_df, val_df, test_df,
     train_loader, val_loader, test_loader,
     cat_cols, num_cols, cardinalities, embed_dims) = load_data(manifest)

    # ── Load Exp-6 ────────────────────────────────────────────────────────────
    model = MultiTaskFMNet(cat_cols=cat_cols, cardinalities=cardinalities,
                           embed_dims=embed_dims, num_numerical=len(num_cols))
    model.load_state_dict(torch.load(CKPT["exp6"], map_location=DEVICE, weights_only=True))
    model = model.to(DEVICE).eval()

    # ── Predictions ───────────────────────────────────────────────────────────
    val_logits,  val_churn_true,  val_ltv_pred,  _  = get_raw_predictions(model, val_loader)
    test_logits, test_churn_true, test_ltv_pred, _  = get_raw_predictions(model, test_loader)

    # ── Isotonic calibration ──────────────────────────────────────────────────
    val_probs_raw  = 1 / (1 + np.exp(-val_logits))
    test_probs_raw = 1 / (1 + np.exp(-test_logits))
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val_probs_raw, val_churn_true)
    test_probs_cal = iso.predict(test_probs_raw)

    # ── Final metrics ─────────────────────────────────────────────────────────
    ltv_pred_raw = np.expm1(test_ltv_pred)
    ltv_true_raw = np.expm1(test_df["ltv"].values)

    metrics = {
        "model":            "exp6_uncertainty_isotonic_calibrated",
        "churn_auc_roc":    float(roc_auc_score(test_churn_true, test_probs_cal)),
        "churn_auc_pr":     float(average_precision_score(test_churn_true, test_probs_cal)),
        "churn_base_rate":  float(test_churn_true.mean()),
        "ltv_rmse_raw_twd": float(np.sqrt(np.mean((ltv_pred_raw - ltv_true_raw)**2))),
        "ltv_mae_raw_twd":  float(np.mean(np.abs(ltv_pred_raw - ltv_true_raw))),
        "ltv_r2_raw":       float(r2_score(ltv_true_raw, ltv_pred_raw)),
    }

    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │            FINAL MODEL METRICS (Test Set)           │")
    print("  ├─────────────────────────────────────────────────────┤")
    for k, v in metrics.items():
        if k != "model":
            print(f"  │  {k:35s}: {v:.4f}  │")
    print("  └─────────────────────────────────────────────────────┘")

    with open(os.path.join(RESULTS_DIR, "final_model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Compare with baselines ────────────────────────────────────────────────
    bpath = os.path.join(RESULTS_DIR, "baseline_results.json")
    if os.path.exists(bpath):
        with open(bpath) as f:
            bl = json.load(f)
        exp1_auc  = bl["exp1_churn_only"]["test"]["churn_auc_roc"]
        exp2_rmse = bl["exp2_ltv_only"]["test"]["ltv_rmse_raw_twd"]
        print(f"\n  vs single-task ceilings:")
        print(f"    AUC  : {metrics['churn_auc_roc']:.4f} vs {exp1_auc:.4f} "
              f"({'above' if metrics['churn_auc_roc']>=exp1_auc else 'below'} ceiling)")
        print(f"    RMSE : {metrics['ltv_rmse_raw_twd']:.2f} vs {exp2_rmse:.2f} TWD "
              f"({'better' if metrics['ltv_rmse_raw_twd']<=exp2_rmse else 'worse'})")

    # ── Plot 1: ROC + PR curves ───────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(test_churn_true, test_probs_cal)
    prec, rec, _ = precision_recall_curve(test_churn_true, test_probs_cal)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(fpr, tpr, color="#1565C0", lw=2,
                 label=f"AUC-ROC = {metrics['churn_auc_roc']:.4f}")
    axes[0].plot([0,1],[0,1], "k--", alpha=0.4, lw=1)
    axes[0].fill_between(fpr, tpr, alpha=0.08, color="#1565C0")
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve — Final Model (Test Set)"); axes[0].legend()

    axes[1].plot(rec, prec, color="#2E7D32", lw=2,
                 label=f"AUC-PR = {metrics['churn_auc_pr']:.4f}")
    axes[1].axhline(y=metrics["churn_base_rate"], color="gray", ls="--", alpha=0.6,
                    label=f"Base rate = {metrics['churn_base_rate']:.4f}")
    axes[1].fill_between(rec, prec, alpha=0.08, color="#2E7D32")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve — Final Model (Test Set)")
    axes[1].legend()

    fig.suptitle("Exp-6 (Uncertainty Weighting) + Isotonic Calibration",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "final_roc_pr_curves.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\n  ROC + PR curves saved.")

    # ── Plot 2: LTV scatter ───────────────────────────────────────────────────
    rng = np.random.default_rng(RANDOM_SEED)
    s   = rng.choice(len(ltv_pred_raw), size=min(20_000, len(ltv_pred_raw)), replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    lim = max(ltv_true_raw.max(), ltv_pred_raw.max())

    axes[0].scatter(ltv_true_raw[s], ltv_pred_raw[s], s=2, alpha=0.15, color="#1B5E20")
    axes[0].plot([0,lim],[0,lim], "r--", alpha=0.5, lw=1.5)
    axes[0].set_xlabel("Actual LTV (TWD)"); axes[0].set_ylabel("Predicted LTV (TWD)")
    axes[0].set_title(f"Full range | R²={metrics['ltv_r2_raw']:.4f}  "
                      f"RMSE={metrics['ltv_rmse_raw_twd']:.1f} TWD")

    mask = (ltv_true_raw[s] <= 500) & (ltv_pred_raw[s] <= 500)
    axes[1].scatter(ltv_true_raw[s][mask], ltv_pred_raw[s][mask],
                    s=3, alpha=0.2, color="#1B5E20")
    axes[1].plot([0,500],[0,500], "r--", alpha=0.5, lw=1.5)
    axes[1].set_xlabel("Actual LTV (TWD)"); axes[1].set_ylabel("Predicted LTV (TWD)")
    axes[1].set_title("Zoomed 0-500 TWD\n"
                      "(spike at 0 = users with no Jan-Feb 2017 transactions)")

    fig.suptitle("LTV Prediction Scatter — Final Model (Test Set)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "final_ltv_scatter.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  LTV scatter saved.")

    # ── Summary text ──────────────────────────────────────────────────────────
    pr_lift = metrics["churn_auc_pr"] / metrics["churn_base_rate"]
    summary = f"""
====================================================
  KKBOX MULTI-TASK MLP — FINAL RESULTS
====================================================
Model  : Exp-6 Uncertainty Weighting + Isotonic Calibration
Dataset: WSDM 2017 KKBox  |  992,931 users  |  70/15/15 split

CHURN (test):
  AUC-ROC  : {metrics['churn_auc_roc']:.4f}
  AUC-PR   : {metrics['churn_auc_pr']:.4f}  ({pr_lift:.1f}x vs base rate {metrics['churn_base_rate']:.4f})

LTV REGRESSION (test, raw TWD):
  RMSE     : {metrics['ltv_rmse_raw_twd']:.2f} TWD
  MAE      : {metrics['ltv_mae_raw_twd']:.2f} TWD
  R-squared: {metrics['ltv_r2_raw']:.4f}

BUSINESS LAYER:
  Retention Priority Score = P(churn_cal) x E[LTV]
  +750% to +1,300% revenue saved vs random (50K TWD budget)

KEY LEARNINGS:
  1. LTV leakage: always use temporal split for regression targets
  2. pos_weight improves AUC but inflates probs -> needs calibration
  3. Uncertainty weighting (Exp-6) auto-adapts task balance
  4. Isotonic regression fixes bias miscalibration; temp scaling does not
  5. DuckDB essential for 392M-row queries without full RAM load
====================================================
"""
    print(summary)
    with open(os.path.join(RESULTS_DIR, "final_summary.txt"), "w") as f:
        f.write(summary)

    print(f"=== Stage 7 complete. All outputs in {RESULTS_DIR} ===")


if __name__ == "__main__":
    main()
