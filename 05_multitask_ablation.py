"""
05_multitask_ablation.py
------------------------
Stage 5: Five multi-task experiments on top of single-task ceilings.

  Exp-3  Fixed equal weight      lambda_churn=0.5  lambda_ltv=0.5
  Exp-4  Churn-dominant          lambda_churn=0.7  lambda_ltv=0.3
  Exp-5  LTV-dominant            lambda_churn=0.3  lambda_ltv=0.7
  Exp-6  Uncertainty weighting   Kendall, Gal & Cipolla (NeurIPS 2017)
  Exp-7  PCGrad                  Yu et al. (NeurIPS 2020)

HOW TO RUN (after Stage 4):
  python 05_multitask_ablation.py

OUTPUTS:
  models/exp3..7.pt
  results/exp3..7_history.csv
  results/ablation_results_table.csv
  results/pareto_frontier.png
  results/training_curves_multitask.png
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from config import (
    BATCH_SIZE, CKPT, DEVICE, EPOCHS, GRAD_CLIP,
    LR, LR_PATIENCE, PATIENCE, RANDOM_SEED, RESULTS_DIR, WEIGHT_DECAY,
)
from models import MultiTaskFMNet, PCGrad
from utils import (
    get_pos_weight, load_data, load_manifest, plot_training_curves,
)


# ── Quick eval helper ─────────────────────────────────────────────────────────
def eval_metrics(model, loader):
    model.to(DEVICE).eval()
    logits_l, ltv_p_l, churn_l, ltv_t_l = [], [], [], []
    with torch.no_grad():
        for x_num, x_cat, y_churn, y_ltv in loader:
            logit, ltv_pred = model(x_num.to(DEVICE), x_cat.to(DEVICE))
            logits_l.append(logit.cpu())
            ltv_p_l.append(ltv_pred.cpu())
            churn_l.append(y_churn)
            ltv_t_l.append(y_ltv)
    logits_np = torch.cat(logits_l).numpy()
    churn_np  = torch.cat(churn_l).numpy()
    ltv_p_np  = torch.cat(ltv_p_l).numpy()
    ltv_t_np  = torch.cat(ltv_t_l).numpy()
    probs = 1 / (1 + np.exp(-logits_np))
    auc   = roc_auc_score(churn_np, probs)
    rmse  = float(np.sqrt(np.mean((ltv_p_np - ltv_t_np)**2)))
    return auc, rmse


# ── Fixed weight training (Exp 3,4,5) ────────────────────────────────────────
def train_fixed(model, lc, ll, pos_weight, ckpt, name, train_loader, val_loader):
    model.to(DEVICE)
    bce_fn    = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    mse_fn    = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=LR_PATIENCE, factor=0.5, verbose=False)
    best, no_imp, history = float("inf"), 0, []

    for epoch in range(EPOCHS):
        model.train()
        for x_num, x_cat, y_churn, y_ltv in train_loader:
            optimizer.zero_grad()
            logit, pred = model(x_num.to(DEVICE), x_cat.to(DEVICE))
            loss = (lc * bce_fn(logit, y_churn.to(DEVICE))
                    + ll * mse_fn(pred, y_ltv.to(DEVICE)))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        model.eval()
        vb, vm, n = 0.0, 0.0, 0
        with torch.no_grad():
            for x_num, x_cat, y_churn, y_ltv in val_loader:
                logit, pred = model(x_num.to(DEVICE), x_cat.to(DEVICE))
                vb += bce_fn(logit, y_churn.to(DEVICE)).item() * len(y_churn)
                vm += mse_fn(pred,  y_ltv.to(DEVICE)).item()  * len(y_ltv)
                n  += len(y_churn)
        vb /= n; vm /= n
        vl = lc*vb + ll*vm
        auc, rmse = eval_metrics(model, val_loader)
        scheduler.step(vl)
        history.append({"epoch":epoch,"val_loss":vl,"val_auc":auc,"val_rmse_log":rmse})
        print(f"  [{name}] ep {epoch:2d}  vl={vl:.4f}  auc={auc:.4f}  rmse={rmse:.4f}")
        if vl < best - 1e-5:
            best, no_imp = vl, 0
            torch.save(model.state_dict(), ckpt)
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                print(f"  [{name}] Early stop ep {epoch}")
                break

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    return model, pd.DataFrame(history)


# ── Uncertainty weighting (Exp-6) ────────────────────────────────────────────
def train_uncertainty(model, pos_weight, ckpt, name, train_loader, val_loader):
    """
    Learns per-task log-variance s = log(sigma^2).
    Loss = 0.5*exp(-s)*L_task + 0.5*s
    Both log-variances are optimised jointly with model weights.
    """
    model.to(DEVICE)
    log_var_c = nn.Parameter(torch.zeros(1, device=DEVICE))
    log_var_l = nn.Parameter(torch.zeros(1, device=DEVICE))
    bce_fn    = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    mse_fn    = nn.MSELoss()
    optimizer = torch.optim.Adam(
        list(model.parameters()) + [log_var_c, log_var_l],
        lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=LR_PATIENCE, factor=0.5, verbose=False)
    best, no_imp, history = float("inf"), 0, []

    for epoch in range(EPOCHS):
        model.train()
        for x_num, x_cat, y_churn, y_ltv in train_loader:
            optimizer.zero_grad()
            logit, pred = model(x_num.to(DEVICE), x_cat.to(DEVICE))
            loss = (0.5*torch.exp(-log_var_c)*bce_fn(logit, y_churn.to(DEVICE)) + 0.5*log_var_c
                   +0.5*torch.exp(-log_var_l)*mse_fn(pred,  y_ltv.to(DEVICE))  + 0.5*log_var_l)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        model.eval()
        vb, vm, n = 0.0, 0.0, 0
        with torch.no_grad():
            for x_num, x_cat, y_churn, y_ltv in val_loader:
                logit, pred = model(x_num.to(DEVICE), x_cat.to(DEVICE))
                vb += bce_fn(logit, y_churn.to(DEVICE)).item() * len(y_churn)
                vm += mse_fn(pred,  y_ltv.to(DEVICE)).item()  * len(y_ltv)
                n  += len(y_churn)
        vb /= n; vm /= n
        with torch.no_grad():
            vl = float(0.5*torch.exp(-log_var_c)*vb + 0.5*log_var_c
                      +0.5*torch.exp(-log_var_l)*vm  + 0.5*log_var_l)
        auc, rmse = eval_metrics(model, val_loader)
        scheduler.step(vl)
        history.append({"epoch":epoch,"val_loss":vl,"val_auc":auc,"val_rmse_log":rmse,
                         "s_churn":float(log_var_c),"s_ltv":float(log_var_l)})
        print(f"  [{name}] ep {epoch:2d}  vl={vl:.4f}  auc={auc:.4f}  rmse={rmse:.4f}"
              f"  sc={float(log_var_c):.3f}  sl={float(log_var_l):.3f}")
        if vl < best - 1e-5:
            best, no_imp = vl, 0
            torch.save(model.state_dict(), ckpt)
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                print(f"  [{name}] Early stop ep {epoch}")
                break

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    return model, pd.DataFrame(history)


# ── PCGrad (Exp-7) ────────────────────────────────────────────────────────────
def train_pcgrad(model, pos_weight, ckpt, name, train_loader, val_loader):
    model.to(DEVICE)
    bce_fn    = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    mse_fn    = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=LR_PATIENCE, factor=0.5, verbose=False)
    pcgrad    = PCGrad(optimizer, model.parameters())
    best, no_imp, history = float("inf"), 0, []

    for epoch in range(EPOCHS):
        model.train()
        for x_num, x_cat, y_churn, y_ltv in train_loader:
            logit, pred  = model(x_num.to(DEVICE), x_cat.to(DEVICE))
            loss_c = bce_fn(logit, y_churn.to(DEVICE))
            loss_l = mse_fn(pred,  y_ltv.to(DEVICE))
            pcgrad.pc_backward([loss_c, loss_l])
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            pcgrad.step()

        model.eval()
        vb, vm, n = 0.0, 0.0, 0
        with torch.no_grad():
            for x_num, x_cat, y_churn, y_ltv in val_loader:
                logit, pred = model(x_num.to(DEVICE), x_cat.to(DEVICE))
                vb += bce_fn(logit, y_churn.to(DEVICE)).item() * len(y_churn)
                vm += mse_fn(pred,  y_ltv.to(DEVICE)).item()  * len(y_ltv)
                n  += len(y_churn)
        vb /= n; vm /= n
        vl = vb + vm
        auc, rmse = eval_metrics(model, val_loader)
        scheduler.step(vl)
        history.append({"epoch":epoch,"val_loss":vl,"val_auc":auc,"val_rmse_log":rmse})
        print(f"  [{name}] ep {epoch:2d}  vl={vl:.4f}  auc={auc:.4f}  rmse={rmse:.4f}")
        if vl < best - 1e-5:
            best, no_imp = vl, 0
            torch.save(model.state_dict(), ckpt)
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                print(f"  [{name}] Early stop ep {epoch}")
                break

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    return model, pd.DataFrame(history)


# ── Pareto frontier plot ──────────────────────────────────────────────────────
def plot_pareto(all_results, baseline_results):
    exp1_auc  = baseline_results["exp1_churn_only"]["test"]["churn_auc_roc"]
    exp2_rmse = baseline_results["exp2_ltv_only"]["test"]["ltv_rmse_log"]

    fig, ax = plt.subplots(figsize=(8, 6))
    stars = {"exp6_uncertainty", "exp7_pcgrad"}
    for name, res in all_results.items():
        auc, rmse = res.get("test_auc"), res.get("test_rmse_log")
        if auc is None or rmse is None:
            continue
        marker = "*" if name in stars else "o"
        sz     = 280 if name in stars else 100
        ax.scatter(auc, rmse, s=sz, marker=marker, zorder=5, label=name.replace("_"," "))
        ax.annotate(name.split("_")[0].upper(), (auc, rmse),
                    textcoords="offset points", xytext=(6,4), fontsize=8)
    ax.scatter(exp1_auc, exp2_rmse, s=200, marker="D", color="black", zorder=6,
               label="Single-task ceilings")
    ax.set_xlabel("Test Churn AUC-ROC (higher is better)")
    ax.set_ylabel("Test LTV RMSE log (lower is better)")
    ax.set_title("Pareto Frontier: Churn AUC vs LTV RMSE\n"
                 "Top-right = better on both tasks")
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "pareto_frontier.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Pareto plot saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=== Stage 5: Multi-Task Ablation (Exp-3 to Exp-7) ===")
    print(f"  Device: {DEVICE}")

    if not os.path.exists(os.path.join(RESULTS_DIR, "baseline_results.json")):
        raise FileNotFoundError("Run 04_training_baselines.py first.")

    with open(os.path.join(RESULTS_DIR, "baseline_results.json")) as f:
        baseline = json.load(f)

    manifest = load_manifest()
    (train_df, val_df, test_df,
     train_loader, val_loader, test_loader,
     cat_cols, num_cols, cardinalities, embed_dims) = load_data(manifest)
    pos_weight = get_pos_weight(train_df)

    def new_model():
        return MultiTaskFMNet(cat_cols=cat_cols, cardinalities=cardinalities,
                              embed_dims=embed_dims, num_numerical=len(num_cols))

    all_results, all_histories = {}, {}

    # ── Exp 3, 4, 5 ──────────────────────────────────────────────────────────
    fixed = {
        "exp3": ("exp3_fixed_5050",    0.5, 0.5),
        "exp4": ("exp4_churn_dominant",0.7, 0.3),
        "exp5": ("exp5_ltv_dominant",  0.3, 0.7),
    }
    for key, (fname, lc, ll) in fixed.items():
        print(f"\n--- {fname} (lc={lc}, ll={ll}) ---")
        torch.manual_seed(RANDOM_SEED)
        model, hist = train_fixed(new_model(), lc, ll, pos_weight,
                                  CKPT[key], fname, train_loader, val_loader)
        hist.to_csv(os.path.join(RESULTS_DIR, f"{fname}_history.csv"), index=False)
        auc, rmse = eval_metrics(model, test_loader)
        all_results[fname]   = {"test_auc":auc,"test_rmse_log":rmse,"lambda_churn":lc,"lambda_ltv":ll}
        all_histories[fname] = hist
        print(f"  TEST: AUC={auc:.4f}  RMSE_log={rmse:.4f}")

    # ── Exp-6 ────────────────────────────────────────────────────────────────
    print("\n--- Exp-6: Uncertainty Weighting (Kendall 2018) ---")
    torch.manual_seed(RANDOM_SEED)
    model6, hist6 = train_uncertainty(new_model(), pos_weight, CKPT["exp6"],
                                      "Exp-6", train_loader, val_loader)
    hist6.to_csv(os.path.join(RESULTS_DIR, "exp6_history.csv"), index=False)
    auc6, rmse6 = eval_metrics(model6, test_loader)
    all_results["exp6_uncertainty"]   = {"test_auc":auc6,"test_rmse_log":rmse6}
    all_histories["exp6_uncertainty"] = hist6
    print(f"  TEST: AUC={auc6:.4f}  RMSE_log={rmse6:.4f}")

    # ── Exp-7 ────────────────────────────────────────────────────────────────
    print("\n--- Exp-7: PCGrad (Yu et al. 2020) ---")
    torch.manual_seed(RANDOM_SEED)
    model7, hist7 = train_pcgrad(new_model(), pos_weight, CKPT["exp7"],
                                 "Exp-7", train_loader, val_loader)
    hist7.to_csv(os.path.join(RESULTS_DIR, "exp7_history.csv"), index=False)
    auc7, rmse7 = eval_metrics(model7, test_loader)
    all_results["exp7_pcgrad"]   = {"test_auc":auc7,"test_rmse_log":rmse7}
    all_histories["exp7_pcgrad"] = hist7
    print(f"  TEST: AUC={auc7:.4f}  RMSE_log={rmse7:.4f}")

    # ── Ablation table ────────────────────────────────────────────────────────
    rows = [
        {"experiment":"exp1_churn_only","lambda_churn":1.0,"lambda_ltv":0.0,
         "test_auc":baseline["exp1_churn_only"]["test"]["churn_auc_roc"],"test_rmse_log":None},
        {"experiment":"exp2_ltv_only","lambda_churn":0.0,"lambda_ltv":1.0,
         "test_auc":None,"test_rmse_log":baseline["exp2_ltv_only"]["test"]["ltv_rmse_log"]},
    ]
    for name, res in all_results.items():
        rows.append({"experiment": name, **res})
    df_abl = pd.DataFrame(rows)
    df_abl.to_csv(os.path.join(RESULTS_DIR, "ablation_results_table.csv"), index=False)

    print("\n--- Full Ablation Table ---")
    print(df_abl.to_string(index=False))

    # ── Winner ────────────────────────────────────────────────────────────────
    exp1_auc  = baseline["exp1_churn_only"]["test"]["churn_auc_roc"]
    exp2_rmse = baseline["exp2_ltv_only"]["test"]["ltv_rmse_log"]
    print(f"\n  Single-task ceilings: AUC={exp1_auc:.4f}  RMSE_log={exp2_rmse:.4f}")
    print(f"  Exp-6 (SELECTED):     AUC={auc6:.4f}  RMSE_log={rmse6:.4f}")
    if auc6 >= exp1_auc and rmse6 <= exp2_rmse:
        print("  -> Exp-6 BEATS BOTH single-task ceilings simultaneously!")
    print(f"  Exp-7 (PCGrad):       AUC={auc7:.4f}  RMSE_log={rmse7:.4f}")

    plot_pareto(all_results, baseline)
    plot_training_curves(all_histories, "training_curves_multitask.png")
    print(f"\n=== Stage 5 complete. Results -> {RESULTS_DIR} ===")


if __name__ == "__main__":
    main()
