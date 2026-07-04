"""
02_feature_engineering.py
--------------------------
Stage 2: Build the 13-feature model-ready table.

KEY RULES:
  1. Hard temporal cutoff: features see data <= 2016-12-31 only.
     LTV target = forward revenue Jan-Feb 2017 (post-cutoff).
     This prevents the model from algebraically reconstructing the LTV target
     from its own input features (leakage that caused R2=0.999 in early runs).

  2. Split BEFORE fitting encoders / scalers to prevent leakage from
     val/test statistics into the training process.

HOW TO RUN:
  python 02_feature_engineering.py

OUTPUTS (data/processed/ or /kaggle/working/processed/):
  model_dataset_train.parquet
  model_dataset_val.parquet
  model_dataset_test.parquet
  categorical_encoders.joblib
  numerical_scaler.joblib
  feature_manifest.json
"""

import json
import os
import warnings
warnings.filterwarnings("ignore")

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import (
    PROCESSED_DIR, RANDOM_SEED, parquet,
    FEATURE_CUTOFF_INT, LTV_TARGET_START, LTV_TARGET_END,
    ENGAGEMENT_WIN_START, ENGAGEMENT_WIN_END,
    CATEGORICAL_COLS, EMBEDDING_DIMS, SCALE_COLS, UNSCALED_NUM_COLS,
)

FEATURE_CUTOFF_TS = pd.Timestamp("2016-12-31")
con = duckdb.connect()


# ── Step 1: Merge ─────────────────────────────────────────────────────────────
def build_raw_table() -> pd.DataFrame:
    print("[1/6] Merging tables via DuckDB ...")
    log_tbl = "user_logs" if os.path.exists(parquet("user_logs")) else "user_logs_v2"
    print(f"      Using logs table: {log_tbl}")

    query = f"""
        WITH txn_agg AS (
            SELECT msno,
                   COUNT(*)                   AS num_transactions,
                   AVG(payment_plan_days)      AS avg_payment_plan_days,
                   AVG(actual_amount_paid)     AS avg_actual_amount_paid,
                   AVG(is_auto_renew)          AS is_auto_renew_rate
            FROM '{parquet("transactions")}'
            WHERE transaction_date <= {FEATURE_CUTOFF_INT}
            GROUP BY msno
        ),
        latest_txn AS (
            SELECT msno, payment_method_id
            FROM (
                SELECT msno, payment_method_id,
                       ROW_NUMBER() OVER (
                           PARTITION BY msno ORDER BY transaction_date DESC
                       ) rn
                FROM '{parquet("transactions")}'
                WHERE transaction_date <= {FEATURE_CUTOFF_INT}
            ) WHERE rn = 1
        ),
        ltv_target AS (
            -- FORWARD-LOOKING: strictly after feature cutoff
            SELECT msno, SUM(actual_amount_paid) AS ltv
            FROM '{parquet("transactions")}'
            WHERE transaction_date BETWEEN {LTV_TARGET_START} AND {LTV_TARGET_END}
            GROUP BY msno
        ),
        logs_agg AS (
            -- total_secs clipped to [0,86400] to remove corrupted extremes
            SELECT msno,
                   COUNT(DISTINCT date)                        AS daily_active_days,
                   SUM(GREATEST(LEAST(total_secs,86400),0))   AS total_secs_sum,
                   SUM(num_25)  AS sum25, SUM(num_50)  AS sum50,
                   SUM(num_75)  AS sum75, SUM(num_985) AS sum985,
                   SUM(num_100) AS sum100
            FROM '{parquet(log_tbl)}'
            WHERE date BETWEEN {ENGAGEMENT_WIN_START} AND {ENGAGEMENT_WIN_END}
            GROUP BY msno
        )
        SELECT t.msno, t.is_churn,
               COALESCE(lv.ltv, 0)                  AS ltv,
               m.city, m.bd, m.gender, m.registered_via,
               m.registration_init_time,
               COALESCE(tx.num_transactions, 0)      AS num_transactions,
               tx.avg_payment_plan_days,
               tx.avg_actual_amount_paid,
               COALESCE(tx.is_auto_renew_rate, 0)    AS is_auto_renew_rate,
               lt.payment_method_id,
               COALESCE(la.daily_active_days, 0)     AS daily_active_days,
               COALESCE(la.total_secs_sum, 0)        AS total_secs_sum,
               COALESCE(la.sum25,  0) AS sum25,
               COALESCE(la.sum50,  0) AS sum50,
               COALESCE(la.sum75,  0) AS sum75,
               COALESCE(la.sum985, 0) AS sum985,
               COALESCE(la.sum100, 0) AS sum100
        FROM '{parquet("train")}' t
        LEFT JOIN '{parquet("members")}' m    USING (msno)
        LEFT JOIN txn_agg tx                 USING (msno)
        LEFT JOIN latest_txn lt              USING (msno)
        LEFT JOIN ltv_target lv              USING (msno)
        LEFT JOIN logs_agg la               USING (msno)
    """
    df = con.execute(query).df()
    print(f"      Merged: {len(df):,} rows x {df.shape[1]} cols")
    return df


# ── Step 2: Engineer features ─────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("[2/6] Engineering derived features ...")

    # Days from registration to feature cutoff
    reg_date = pd.to_datetime(
        df["registration_init_time"].astype("Int64").astype(str),
        format="%Y%m%d", errors="coerce"
    )
    df["registration_tenure_days"] = (FEATURE_CUTOFF_TS - reg_date).dt.days

    # Weighted song completion rate (0=never finishes, 1=always finishes)
    song_totals = df[["sum25","sum50","sum75","sum985","sum100"]].sum(axis=1)
    df["avg_song_completion"] = (
        0.25*df["sum25"] + 0.50*df["sum50"] + 0.75*df["sum75"]
        + 0.985*df["sum985"] + 1.0*df["sum100"]
    ) / (song_totals + 1)

    # Log-transform listening time (right-skewed -> more normal)
    df["total_secs_log"] = np.log1p(df["total_secs_sum"])

    # Log-transform LTV (same reason)
    df["log1p_ltv"] = np.log1p(df["ltv"])

    # Clean age: invalid values -> NaN (imputed after split)
    df["bd_clean"] = df["bd"].where(df["bd"].between(1, 100))

    # Fill missing gender
    df["gender"] = df["gender"].fillna("unknown")

    return df


# ── Step 3: Split ─────────────────────────────────────────────────────────────
def split_data(df: pd.DataFrame) -> dict:
    print("[3/6] Train/Val/Test split 70/15/15 (stratified) ...")
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["is_churn"], random_state=RANDOM_SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["is_churn"], random_state=RANDOM_SEED
    )
    splits = {"train": train_df, "val": val_df, "test": test_df}
    for name, split in splits.items():
        print(f"      {name:6s}  n={len(split):>7,}  churn={split['is_churn'].mean():.4f}")
    return splits


# ── Step 4: Imputation ────────────────────────────────────────────────────────
def impute(splits: dict) -> dict:
    print("[4/6] Median imputation (fit on train only) ...")
    train = splits["train"]
    meds = {
        "bd_clean":                 train["bd_clean"].median(),
        "registration_tenure_days": train["registration_tenure_days"].median(),
        "avg_payment_plan_days":    train["avg_payment_plan_days"].median(),
        "avg_actual_amount_paid":   train["avg_actual_amount_paid"].median(),
    }
    for split in splits.values():
        for col, med in meds.items():
            split[col] = split[col].fillna(med)
    print(f"      Medians: {meds}")
    return splits


# ── Step 5: Encode + Scale ────────────────────────────────────────────────────
def encode_and_scale(splits: dict) -> tuple:
    print("[5/6] Label-encoding + StandardScaling (fit on train only) ...")

    def fit_encoder(col):
        cats = sorted(col.dropna().unique().tolist())
        m = {cat: i for i, cat in enumerate(cats)}
        m["__unknown__"] = len(cats)
        return m

    def apply_encoder(col, mapping):
        return col.map(mapping).fillna(mapping["__unknown__"]).astype(int)

    encoders = {}
    train = splits["train"]
    for col in CATEGORICAL_COLS:
        enc = fit_encoder(train[col])
        encoders[col] = enc
        for split in splits.values():
            split[f"{col}_enc"] = apply_encoder(split[col], enc)
        print(f"      {col:25s}  cardinality: {len(enc)}")

    scaler = StandardScaler()
    scaler.fit(train[SCALE_COLS])
    for split in splits.values():
        scaled = scaler.transform(split[SCALE_COLS])
        for i, col in enumerate(SCALE_COLS):
            split[f"{col}_scaled"] = scaled[:, i]

    joblib.dump(scaler,   str(PROCESSED_DIR / "numerical_scaler.joblib"))
    joblib.dump(encoders, str(PROCESSED_DIR / "categorical_encoders.joblib"))
    return splits, encoders


# ── Step 6: Save ──────────────────────────────────────────────────────────────
def save_outputs(splits: dict, encoders: dict) -> None:
    print("[6/6] Saving splits + manifest ...")

    cat_enc  = [f"{c}_enc"    for c in CATEGORICAL_COLS]
    num_sc   = [f"{c}_scaled" for c in SCALE_COLS]
    out_cols = (["msno","is_churn","log1p_ltv","ltv"]
                + cat_enc + num_sc + UNSCALED_NUM_COLS)

    for name, split in splits.items():
        path = str(PROCESSED_DIR / f"model_dataset_{name}.parquet")
        split[out_cols].to_parquet(path, engine="pyarrow",
                                   compression="zstd", index=False)
        print(f"      {name:6s}: {len(split):,} rows -> {path}")

    manifest = {
        "feature_cutoff":    str(FEATURE_CUTOFF_TS.date()),
        "ltv_target_window": [LTV_TARGET_START, LTV_TARGET_END],
        "engagement_window": [ENGAGEMENT_WIN_START, ENGAGEMENT_WIN_END],
        "targets": {"churn":"is_churn","ltv_log":"log1p_ltv","ltv_raw":"ltv"},
        "categorical": {
            col: {
                "column":        f"{col}_enc",
                "cardinality":   len(encoders[col]),
                "embedding_dim": EMBEDDING_DIMS[col],
                "unknown_index": encoders[col]["__unknown__"],
            }
            for col in CATEGORICAL_COLS
        },
        "numerical_scaled":   num_sc,
        "numerical_unscaled": UNSCALED_NUM_COLS,
    }
    mpath = str(PROCESSED_DIR / "feature_manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"      Manifest -> {mpath}")

    n_features = len(cat_enc) + len(num_sc) + len(UNSCALED_NUM_COLS)
    print(f"      Total feature columns: {n_features}")


# ── Verify ────────────────────────────────────────────────────────────────────
def verify():
    print("\n--- Verification ---")
    for name in ["train", "val", "test"]:
        df = pd.read_parquet(str(PROCESSED_DIR / f"model_dataset_{name}.parquet"))
        nulls = df.drop(columns=["msno"]).isna().sum().sum()
        print(f"  {name:6s}  shape={df.shape}  "
              f"churn={df['is_churn'].mean():.4f}  "
              f"ltv_med={df['ltv'].median():.0f}  "
              f"nulls={nulls}")
    if nulls == 0:
        print("  OK: zero null cells.")


def main():
    print("=== Stage 2: Feature Engineering & Preprocessing ===")
    df = build_raw_table()
    df = engineer_features(df)
    splits = split_data(df)
    splits = impute(splits)
    splits, encoders = encode_and_scale(splits)
    save_outputs(splits, encoders)
    verify()
    print("\n=== Stage 2 complete ===")


if __name__ == "__main__":
    main()
