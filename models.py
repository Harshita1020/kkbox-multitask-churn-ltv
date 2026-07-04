"""
models.py
---------
All PyTorch model classes — imported by every training script.

Classes:
  KKBoxDataset       — PyTorch Dataset wrapping the parquet splits
  FMInteractionLayer — Factorization Machine O(k*d) pairwise interaction
  MultiTaskFMNet     — Embeddings + FM + shared MLP backbone + dual heads
  PCGrad             — Projecting Conflicting Gradients optimizer wrapper
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────────────────────
class KKBoxDataset(Dataset):
    """
    Wraps one of the model_dataset_{train/val/test}.parquet DataFrames
    into a PyTorch-compatible Dataset.

    Returns per sample:
      x_num   (float32) : scaled + unscaled numerical features
      x_cat   (int64)   : label-encoded categorical indices
      y_churn (float32) : binary label 0/1
      y_ltv   (float32) : log1p(forward LTV in TWD)
    """
    def __init__(self, df, cat_cols: list, num_cols: list):
        self.x_cat   = torch.tensor(df[cat_cols].values, dtype=torch.long)
        self.x_num   = torch.tensor(df[num_cols].values, dtype=torch.float32)
        self.y_churn = torch.tensor(df["is_churn"].values,  dtype=torch.float32)
        self.y_ltv   = torch.tensor(df["log1p_ltv"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.y_churn)

    def __getitem__(self, idx):
        return self.x_num[idx], self.x_cat[idx], self.y_churn[idx], self.y_ltv[idx]


# ─────────────────────────────────────────────────────────────────────────────
class FMInteractionLayer(nn.Module):
    """
    Factorization Machine interaction layer.
    Computes all pairwise feature interactions in O(k*d) via:

      FM(x) = 0.5 * [ (sum_i V_i * x_i)^2  -  sum_i (V_i * x_i)^2 ]

    This is the 'sum-of-squares minus square-of-sums' identity (Rendle 2010).
    Output shape: (batch, k)  where k is the number of latent factors.
    """
    def __init__(self, input_dim: int, k: int = 8):
        super().__init__()
        self.V = nn.Parameter(torch.randn(input_dim, k) * 0.01)

    def forward(self, x):
        xV      = x.unsqueeze(2) * self.V.unsqueeze(0)   # (batch, d, k)
        sum_sq  = xV.sum(dim=1).pow(2)                    # (batch, k)
        sq_sum  = xV.pow(2).sum(dim=1)                    # (batch, k)
        return 0.5 * (sum_sq - sq_sum)                    # (batch, k)


# ─────────────────────────────────────────────────────────────────────────────
class MultiTaskFMNet(nn.Module):
    """
    Full multi-task model architecture:

    Input
      -> Embedding lookup for each categorical column
      -> Concatenate embeddings + numerical features  (combined_dim = 27)
      -> FM interaction layer                         (fm_k = 8 dims)
      -> Concatenate [combined, fm_out]               (35 dims)
      -> Shared MLP: 256->128->64 (BN + ReLU + Dropout each layer)
      -> Churn head: 64->32->1   (raw logit; sigmoid at inference)
      -> LTV head:   64->32->1   (predicts log1p(LTV))
    """
    def __init__(
        self,
        cat_cols:      list,
        cardinalities: dict,
        embed_dims:    dict,
        num_numerical: int,
        fm_k:          int   = 8,
        backbone_dims: tuple = (256, 128, 64),
        dropout_rates: tuple = (0.3, 0.3, 0.2),
    ):
        super().__init__()
        self.cat_cols = cat_cols

        # Embedding tables
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(cardinalities[col], embed_dims[col])
            for col in cat_cols
        })

        embed_total  = sum(embed_dims[c] for c in cat_cols)
        combined_dim = embed_total + num_numerical   # 18 + 9 = 27

        # FM layer
        self.fm = FMInteractionLayer(combined_dim, k=fm_k)

        # Shared backbone
        backbone_in = combined_dim + fm_k            # 27 + 8 = 35
        layers, prev = [], backbone_in
        for dim, p in zip(backbone_dims, dropout_rates):
            layers += [
                nn.Linear(prev, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(p),
            ]
            prev = dim
        self.backbone = nn.Sequential(*layers)

        # Task heads
        self.churn_head = nn.Sequential(
            nn.Linear(prev, 32), nn.ReLU(), nn.Linear(32, 1)
        )
        self.ltv_head = nn.Sequential(
            nn.Linear(prev, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x_num, x_cat):
        embeds  = [self.embeddings[col](x_cat[:, i])
                   for i, col in enumerate(self.cat_cols)]
        x       = torch.cat(embeds + [x_num], dim=1)
        fm_out  = self.fm(x)
        h       = torch.cat([x, fm_out], dim=1)
        shared  = self.backbone(h)
        return (self.churn_head(shared).squeeze(-1),
                self.ltv_head(shared).squeeze(-1))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
class PCGrad:
    """
    Projecting Conflicting Gradients — Yu et al., NeurIPS 2020.

    When task i's gradient conflicts with task j's gradient for a shared
    parameter (negative cosine similarity), PCGrad removes the conflicting
    component from task i's gradient before summing.

    Usage:
        pcgrad = PCGrad(optimizer, model.parameters())
        pcgrad.pc_backward([loss_churn, loss_ltv])
        pcgrad.step()

    Bug fix vs original paper pseudocode:
        The pseudocode filters None gradients with a list comprehension,
        de-syncing indices for task-specific parameters.
        This implementation keeps fixed-length lists with explicit Nones
        and only skips projection where either gradient is None.
    """
    def __init__(self, optimizer, params):
        self.optimizer = optimizer
        self.params    = list(params)

    def pc_backward(self, losses: list) -> None:
        # Collect per-task gradients
        grads_per_task = []
        for i, loss in enumerate(losses):
            self.optimizer.zero_grad()
            retain = (i < len(losses) - 1)
            loss.backward(retain_graph=retain)
            grads_per_task.append([
                p.grad.clone() if p.grad is not None else None
                for p in self.params
            ])

        # Project conflicting gradients
        projected = [list(g) for g in grads_per_task]
        for i in range(len(losses)):
            for j in range(len(losses)):
                if i == j:
                    continue
                for k in range(len(self.params)):
                    g_i = projected[i][k]
                    g_j = grads_per_task[j][k]
                    if g_i is None or g_j is None:
                        continue
                    gi_f = g_i.flatten()
                    gj_f = g_j.flatten()
                    dot  = torch.dot(gi_f, gj_f)
                    if dot < 0:
                        gi_f = gi_f - (dot / (gj_f.dot(gj_f) + 1e-12)) * gj_f
                        projected[i][k] = gi_f.view_as(g_i)

        # Sum projected gradients and assign
        self.optimizer.zero_grad()
        for k, p in enumerate(self.params):
            total = None
            for i in range(len(losses)):
                g = projected[i][k]
                if g is None:
                    continue
                total = g if total is None else total + g
            p.grad = total

    def step(self):
        self.optimizer.step()
