"""
04_training_baselines.py
------------------------
Stage 4: Single-task baselines — establishes performance ceilings.

  Exp-1  lambda_churn=1.0  lambda_ltv=0.0  ->  best possible churn AUC
  Exp-2  lambda_churn=0.0  lambda_ltv=1.0  ->  best possible LTV RMSE

These ceilings tell us whether multi-task learning (Stage 5) hurts,
matches, or beats training each task in isolation.

HOW TO RUN (after Stage 2):
  python 04_training_baselines.py

OUTPUTS:
  models/exp1_churn_only.pt
  models/exp2_ltv_only.pt
  results/exp1_history.csv
  results/exp2_history.csv
  results/baseline_results.json
  results/training_curves_baselines.png
  results/roc_exp1.png
  results/ltv_scatter_exp2.png
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve

from config import CKPT, DEVICE, RANDOM_SEED, RESULTS_DIR
from models import MultiTaskFMNet
from utils import (
    evaluate_full, get_pos_weight, get_raw_predictions,
    load_data, load_manifest, plot_training_curves, train_model,
)


def main():
    print("=== Stage 4: Training Baselines (Exp-1 & Exp-2) ===")
    print(f"  Device: {DEVICE}")

    manifest = load_manifest()
    (train_df, val_df, test_df,
     train_loader, val_loader, test_loader,
     cat_cols, num_cols, cardinalities, embed_dims) = load_data(manifest)

    pos_weight = get_pos_weight(train_df)

    def new_model():
        m = MultiTaskFMNet(
            cat_cols=cat_cols, cardinalities=cardinalities,
            embed_dims=embed_dims, num_numerical=len(num_cols),
        )
        print(f"  Model parameters: {m.count_parameters():,}")
        return m

    # ── Exp-1: Churn-only ────────────────────────────────────────────────────
    print("\n--- Exp-1: Churn-Only (lambda_churn=1.0, lambda_ltv=0.0) ---")
    torch.manual_seed(RANDOM_SEED)
    model_exp1 = new_model()
    hist1 = train_model(
        model_exp1, train_loader, val_loader,
        lambda_churn=1.0, lambda_ltv=0.0,
        pos_weight=pos_weight,
        checkpoint_path=CKPT["exp1"],
        exp_name="Exp-1",
    )
    hist1.to_csv(os.path.join(RESULTS_DIR, "exp1_history.csv"), index=False)

    # ── Exp-2: LTV-only ──────────────────────────────────────────────────────
    print("\n--- Exp-2: LTV-Only (lambda_churn=0.0, lambda_ltv=1.0) ---")
    torch.manual_seed(RANDOM_SEED)
    model_exp2 = new_model()
    hist2 = train_model(
        model_exp2, train_loader, val_loader,
        lambda_churn=0.0, lambda_ltv=1.0,
        pos_weight=pos_weight,
        checkpoint_path=CKPT["exp2"],
        exp_name="Exp-2",
    )
    hist2.to_csv(os.path.join(RESULTS_DIR, "exp2_history.csv"), index=False)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    print("\n--- Evaluation on val + test ---")
    results = {
        "exp1_churn_only": {
            "val":  evaluate_full(model_exp1, val_loader),
            "test": evaluate_full(model_exp1, test_loader),
        },
        "exp2_ltv_only": {
            "val":  evaluate_full(model_exp2, val_loader),
            "test": evaluate_full(model_exp2, test_loader),
        },
    }
    with open(os.path.join(RESULTS_DIR, "baseline_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    for exp, res in results.items():
        print(f"\n  {exp}  (test):")
        for k, v in res["test"].items():
            if v is not None:
                print(f"    {k}: {v:.4f}")
    print("\n  Note: Exp-1 LTV metrics and Exp-2 churn AUC=0.5 are meaningless"
          " — those heads never received gradient.")

    # ── Training curves ───────────────────────────────────────────────────────
    plot_training_curves(
        {"Exp-1 (churn-only)": hist1, "Exp-2 (LTV-only)": hist2},
        "training_curves_baselines.png",
    )

    # ── ROC curve (Exp-1) ─────────────────────────────────────────────────────
    logits, churn_true, _, _ = get_raw_predictions(model_exp1, test_loader)
    probs = 1 / (1 + np.exp(-logits))
    fpr, tpr, _ = roc_curve(churn_true, probs)
    auc1 = results["exp1_churn_only"]["test"]["churn_auc_roc"]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="#1565C0", lw=2, label=f"AUC = {auc1:.4f}")
    ax.plot([0,1],[0,1], "k--", alpha=0.4, lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Exp-1 ROC Curve — Churn-Only Baseline (Test Set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "roc_exp1.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── LTV scatter (Exp-2) ───────────────────────────────────────────────────
    _, _, ltv_p_log, ltv_t_log = get_raw_predictions(model_exp2, test_loader)
    ltv_p = np.expm1(ltv_p_log)
    ltv_t = np.expm1(ltv_t_log)
    rng   = np.random.default_rng(RANDOM_SEED)
    s     = rng.choice(len(ltv_p), size=min(20_000, len(ltv_p)), replace=False)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(ltv_t[s], ltv_p[s], s=2, alpha=0.15, color="#2E7D32")
    lim = max(ltv_t.max(), ltv_p.max())
    ax.plot([0,lim],[0,lim], "r--", alpha=0.5, lw=1.5)
    ax.set_xlabel("Actual LTV (TWD)"); ax.set_ylabel("Predicted LTV (TWD)")
    r2 = results["exp2_ltv_only"]["test"]["ltv_r2_raw"]
    ax.set_title(f"Exp-2 Predicted vs Actual LTV (Test)\nR² = {r2:.4f}")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "ltv_scatter_exp2.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n=== Stage 4 complete. Results -> {RESULTS_DIR} ===")


if __name__ == "__main__":
    main()
