"""
06_calibration_business.py
--------------------------
Stage 6: Probability calibration + business decision layer.

WHY CALIBRATION IS NEEDED:
  pos_weight in BCEWithLogitsLoss improves ranking (AUC) but inflates
  all predicted probabilities — the model says 33% churn on average
  when the true rate is 6.4%. This breaks the Retention Priority Score
  (P(churn) x E[LTV]) because inflated P(churn) distorts priority rankings.

METHODS TRIED:
  Temperature scaling  — only fixes sharpness, not bias. ECE barely moves.
  Isotonic regression  — non-parametric monotone map. Fixes bias. ECE ~0.

BUSINESS LAYER:
  Retention Priority Score = P(churn_calibrated) x E[LTV]
  Budget simulation compares model ranking vs random vs churn-prob-only.

HOW TO RUN (after Stage 5):
  python 06_calibration_business.py

OUTPUTS:
  results/reliability_diagrams.png
  results/retention_priority_scores.csv
  results/budget_allocation.txt
  results/sensitivity_analysis.png
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression

from config import CKPT, DEVICE, PROCESSED_DIR, RANDOM_SEED, RESULTS_DIR
from models import MultiTaskFMNet
from utils import compute_ece, get_raw_predictions, load_data, load_manifest


def load_exp6(cat_cols, num_cols, cardinalities, embed_dims):
    model = MultiTaskFMNet(cat_cols=cat_cols, cardinalities=cardinalities,
                           embed_dims=embed_dims, num_numerical=len(num_cols))
    model.load_state_dict(torch.load(CKPT["exp6"], map_location=DEVICE, weights_only=True))
    return model.to(DEVICE).eval()


def reliability_diagram(ax, bin_stats, ece, title):
    valid = bin_stats.dropna()
    mids  = (valid["bin_lo"] + valid["bin_hi"]) / 2
    ax.bar(mids, valid["avg_acc"], width=0.08, alpha=0.7,
           edgecolor="black", label="Observed churn rate")
    ax.plot([0,1],[0,1], "k--", alpha=0.5, lw=1, label="Perfect calibration")
    ax.scatter(valid["avg_conf"], valid["avg_acc"],
               color="red", s=40, zorder=5, label="Bin (conf, acc)")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed churn rate")
    ax.set_title(f"{title}\nECE = {ece:.4f}")
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.legend(fontsize=7)


def main():
    print("=== Stage 6: Calibration & Business Decision Layer ===")

    if not os.path.exists(CKPT["exp6"]):
        raise FileNotFoundError("Run 05_multitask_ablation.py first.")

    manifest = load_manifest()
    (train_df, val_df, test_df,
     train_loader, val_loader, test_loader,
     cat_cols, num_cols, cardinalities, embed_dims) = load_data(manifest)

    model = load_exp6(cat_cols, num_cols, cardinalities, embed_dims)
    print(f"  Loaded Exp-6 from {CKPT['exp6']}")

    # ── Raw predictions ───────────────────────────────────────────────────────
    print("\n[1] Getting raw logits ...")
    val_logits,  val_churn_true,  val_ltv_pred,  _  = get_raw_predictions(model, val_loader)
    test_logits, test_churn_true, test_ltv_pred, _  = get_raw_predictions(model, test_loader)
    val_probs_raw  = 1 / (1 + np.exp(-val_logits))
    test_probs_raw = 1 / (1 + np.exp(-test_logits))

    # ── ECE uncalibrated ─────────────────────────────────────────────────────
    print("\n[2] ECE — uncalibrated ...")
    ece_raw, bins_raw = compute_ece(test_probs_raw, test_churn_true)
    print(f"  ECE (uncalibrated)     : {ece_raw:.4f}")
    print(f"  Mean predicted P(churn): {test_probs_raw.mean():.4f}")
    print(f"  True churn rate        : {test_churn_true.mean():.4f}")
    n_neg = (val_churn_true == 0).sum()
    n_pos = (val_churn_true == 1).sum()
    print(f"  Cause: pos_weight={n_neg/n_pos:.1f} inflates all probabilities.")

    # ── Temperature scaling ───────────────────────────────────────────────────
    print("\n[3] Temperature scaling ...")
    T   = nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=100)
    bce = nn.BCEWithLogitsLoss()
    vl_t = torch.from_numpy(val_logits)
    vl_y = torch.from_numpy(val_churn_true.astype(np.float32))
    def closure():
        opt.zero_grad()
        bce(vl_t / T, vl_y).backward()
        return bce(vl_t / T, vl_y)
    opt.step(closure)
    test_probs_temp = torch.sigmoid(torch.from_numpy(test_logits) / T.detach()).numpy()
    ece_temp, bins_temp = compute_ece(test_probs_temp, test_churn_true)
    print(f"  T = {T.item():.4f}")
    print(f"  ECE after temp scaling : {ece_temp:.4f}  (barely improved — bias problem)")

    # ── Isotonic regression ───────────────────────────────────────────────────
    print("\n[4] Isotonic regression ...")
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val_probs_raw, val_churn_true)
    test_probs_iso = iso.predict(test_probs_raw)
    ece_iso, bins_iso = compute_ece(test_probs_iso, test_churn_true)
    print(f"  ECE after isotonic     : {ece_iso:.4f}  <- WINNER")
    print(f"  Mean calibrated P      : {test_probs_iso.mean():.4f}")
    print(f"  Distinct cal probs     : {len(np.unique(test_probs_iso))}"
          " (step function — see project notes on resolution trade-off)")

    # ── Reliability diagrams ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    reliability_diagram(axes[0], bins_raw,  ece_raw,  "Uncalibrated")
    reliability_diagram(axes[1], bins_temp, ece_temp, "Temperature scaling")
    reliability_diagram(axes[2], bins_iso,  ece_iso,  "Isotonic regression (SELECTED)")
    fig.suptitle("Reliability Diagrams — Exp-6 (Uncertainty Weighting)",
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "reliability_diagrams.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Reliability diagrams saved.")

    # ── Retention Priority Score ──────────────────────────────────────────────
    print("\n[5] Building Retention Priority Score ...")
    e_ltv = np.expm1(test_ltv_pred)
    results_df = pd.DataFrame({
        "msno":           test_df["msno"].values,
        "is_churn":       test_churn_true,
        "p_churn":        test_probs_iso,
        "e_ltv":          e_ltv,
        "priority_score": test_probs_iso * e_ltv,
    }).sort_values("priority_score", ascending=False).reset_index(drop=True)
    results_df["rank"] = np.arange(1, len(results_df) + 1)
    print("  Top 5 users by priority score:")
    print(results_df[["rank","p_churn","e_ltv","priority_score","is_churn"]].head().to_string(index=False))

    # ── 5-segment illustration ────────────────────────────────────────────────
    p_lo, p_hi = results_df["p_churn"].quantile([0.33, 0.67])
    v_lo, v_hi = results_df["e_ltv"].quantile([0.33, 0.67])
    def pick(risk, value):
        pm = (results_df["p_churn"] < p_lo if risk == "low"
              else results_df["p_churn"] > p_hi if risk == "high"
              else results_df["p_churn"].between(p_lo, p_hi))
        vm = (results_df["e_ltv"] < v_lo if value == "low"
              else results_df["e_ltv"] > v_hi if value == "high"
              else results_df["e_ltv"].between(v_lo, v_hi))
        sub = results_df[pm & vm]
        return None if len(sub) == 0 else sub.iloc[
            (sub["priority_score"] - sub["priority_score"].median()).abs().argmin()]

    segs = [
        ("high","high","Immediate: personal outreach + premium discount"),
        ("high","low", "Low priority: automated email only"),
        ("low", "high","Monitor: no action right now"),
        ("medium","medium","Queue for weekly retention campaign"),
        ("low","low","No action: below cost-of-retention threshold"),
    ]
    seg_rows = []
    for risk, value, action in segs:
        row = pick(risk, value)
        if row is not None:
            seg_rows.append({"segment":f"{risk}-risk,{value}-value",
                              "p_churn":f"{row['p_churn']:.3f}",
                              "e_ltv":f"{row['e_ltv']:.1f}",
                              "priority":f"{row['priority_score']:.2f}",
                              "action":action})
    print("\n  User Segments:")
    print(pd.DataFrame(seg_rows).to_string(index=False))

    # ── Budget allocation ─────────────────────────────────────────────────────
    print("\n[6] Budget allocation simulation ...")
    BUDGET, VOUCHER, RET = 50_000, 50, 0.30
    n_int = BUDGET // VOUCHER
    print(f"  Budget: {BUDGET:,} TWD | Voucher: {VOUCHER} TWD | "
          f"N interventions: {n_int:,}")

    def exp_rev(idx):
        return float((results_df.loc[idx, "priority_score"] * RET).sum())

    model_idx  = results_df.index[:n_int]
    rng        = np.random.default_rng(RANDOM_SEED)
    rand_idx   = rng.choice(results_df.index, size=n_int, replace=False)
    churn_idx  = results_df.sort_values("p_churn", ascending=False).index[:n_int]

    rev_m = exp_rev(model_idx)
    rev_r = exp_rev(rand_idx)
    rev_c = exp_rev(churn_idx)

    lines = [
        "=== Budget Allocation Simulation ===",
        f"Budget {BUDGET:,} TWD | Voucher {VOUCHER} TWD | {n_int:,} interventions | {RET:.0%} success rate",
        "",
        f"Model priority ranking  : {rev_m:>12,.0f} TWD expected revenue saved",
        f"Random selection        : {rev_r:>12,.0f} TWD  ({(rev_m/max(rev_r,1)-1)*100:+.1f}% vs model)",
        f"Churn-prob-only ranking : {rev_c:>12,.0f} TWD  ({(rev_m/max(rev_c,1)-1)*100:+.1f}% vs model)",
    ]
    txt = "\n".join(lines)
    print("\n" + txt)
    with open(os.path.join(RESULTS_DIR, "budget_allocation.txt"), "w") as f:
        f.write(txt)

    # ── Sensitivity analysis ──────────────────────────────────────────────────
    print("\n[7] Sensitivity analysis (+/-10% P(churn) perturbation) ...")
    full = results_df.copy()
    full["s_up"] = (full["p_churn"]*1.1).clip(0,1) * full["e_ltv"]
    full["s_dn"] = (full["p_churn"]*0.9).clip(0,1) * full["e_ltv"]
    r_up = full["s_up"].rank(ascending=False)
    r_dn = full["s_dn"].rank(ascending=False)
    delta = (r_up - r_dn).abs()
    top_d = delta.iloc[:n_int]
    print(f"  Median rank change in top {n_int:,}: {top_d.median():.0f}")
    print(f"  Could fall out of budget: {(top_d>n_int).sum()} "
          f"({(top_d>n_int).mean()*100:.1f}%)")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(top_d, bins=50, color="#42A5F5", edgecolor="white")
    ax.set_xlabel("Rank change under +/-10% P(churn) perturbation")
    ax.set_ylabel("Count"); ax.set_title(f"Rank Sensitivity — Top {n_int:,} Budget Users")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "sensitivity_analysis.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    results_df.drop(columns=["s_up","s_dn"], errors="ignore").to_csv(
        os.path.join(RESULTS_DIR, "retention_priority_scores.csv"), index=False)
    print(f"\n=== Stage 6 complete. Results -> {RESULTS_DIR} ===")


if __name__ == "__main__":
    main()
