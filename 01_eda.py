"""
01_eda.py
---------
Stage 1: Exploratory Data Analysis using DuckDB.
Queries Parquet files directly on disk — no full RAM load needed even
for the 392M-row user_logs file.

HOW TO RUN:
  python 01_eda.py

OUTPUTS (saved to results/):
  eda_churn_balance.png
  eda_member_demographics.png
  eda_churn_by_attributes.png
  eda_transaction_signals.png
  eda_engagement_vs_churn.png
"""

import os
import warnings
warnings.filterwarnings("ignore")

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import RESULTS_DIR, parquet

sns.set_theme(style="whitegrid", palette="deep")
con = duckdb.connect()


def save(fig, name: str) -> None:
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def section(title: str) -> None:
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


# ── 1. Row counts ─────────────────────────────────────────────────────────────
def eda_row_counts():
    section("1. Dataset Row Counts")
    tables = ["members", "train", "train_v2",
              "transactions", "transactions_v2", "user_logs_v2"]
    if os.path.exists(parquet("user_logs")):
        tables.append("user_logs")
    for name in tables:
        n = con.execute(f"SELECT COUNT(*) FROM '{parquet(name)}'").fetchone()[0]
        print(f"  {name:25s}  {n:>15,} rows")


# ── 2. Churn label balance ────────────────────────────────────────────────────
def eda_churn_balance():
    section("2. Churn Label Balance (train_v2)")
    df = con.execute(
        f"SELECT is_churn, COUNT(*) n FROM '{parquet('train_v2')}' GROUP BY 1 ORDER BY 1"
    ).df()
    df["pct"] = df["n"] / df["n"].sum() * 100
    print(df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.barplot(data=df, x="is_churn", y="n", ax=ax,
                palette=["#42A5F5", "#EF5350"])
    for bar, pct in zip(ax.patches, df["pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5000,
                f"{pct:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_title("Churn Label Balance (~9% positive class)", fontweight="bold")
    ax.set_xlabel("is_churn  (0=retained, 1=churned)")
    ax.set_ylabel("Count")
    save(fig, "eda_churn_balance.png")


# ── 3. Member demographics ────────────────────────────────────────────────────
def eda_member_demographics():
    section("3. Member Demographics")
    members = con.execute(
        f"SELECT city, bd, gender, registered_via, registration_init_time "
        f"FROM '{parquet('members')}'"
    ).df()
    valid_bd = members[(members["bd"] > 0) & (members["bd"] <= 100)]
    pct_valid   = len(valid_bd) / len(members) * 100
    pct_unknown = members["gender"].isna().sum() / len(members) * 100
    print(f"  Valid age (1-100) : {pct_valid:.1f}%")
    print(f"  Gender unknown    : {pct_unknown:.1f}%")

    members["reg_year"] = (members["registration_init_time"]
                           .astype(str).str[:4].astype("Int64"))

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    axes[0].hist(valid_bd["bd"], bins=50, color="#42A5F5", edgecolor="white")
    axes[0].set_title("Age (valid 1-100 only)")

    gc = members["gender"].fillna("unknown").value_counts()
    axes[1].bar(gc.index, gc.values, color=["#42A5F5", "#EF5350", "#BDBDBD"])
    axes[1].set_title(f"Gender ({pct_unknown:.0f}% unknown)")

    yr = (members[members["reg_year"].between(2004, 2017)]["reg_year"]
          .value_counts().sort_index())
    axes[2].bar(yr.index.astype(str), yr.values, color="#66BB6A")
    axes[2].set_title("Registrations by year")
    axes[2].tick_params(axis="x", rotation=45)

    fig.tight_layout()
    save(fig, "eda_member_demographics.png")


# ── 4. Churn rate by attributes ───────────────────────────────────────────────
def eda_churn_by_attributes():
    section("4. Churn Rate by Member Attributes")

    by_gender = con.execute(f"""
        SELECT COALESCE(m.gender,'unknown') gender,
               COUNT(*) n, AVG(t.is_churn)*100 churn_pct
        FROM '{parquet('train_v2')}' t
        LEFT JOIN '{parquet('members')}' m USING (msno)
        GROUP BY 1
    """).df()

    by_channel = con.execute(f"""
        SELECT m.registered_via, COUNT(*) n, AVG(t.is_churn)*100 churn_pct
        FROM '{parquet('train_v2')}' t
        LEFT JOIN '{parquet('members')}' m USING (msno)
        GROUP BY 1 ORDER BY n DESC LIMIT 8
    """).df()

    by_age = con.execute(f"""
        SELECT CASE WHEN m.bd BETWEEN 1  AND 17  THEN '1-17'
                    WHEN m.bd BETWEEN 18 AND 24  THEN '18-24'
                    WHEN m.bd BETWEEN 25 AND 34  THEN '25-34'
                    WHEN m.bd BETWEEN 35 AND 44  THEN '35-44'
                    WHEN m.bd BETWEEN 45 AND 100 THEN '45-100'
                    ELSE 'unknown' END age_bucket,
               COUNT(*) n, AVG(t.is_churn)*100 churn_pct
        FROM '{parquet('train_v2')}' t
        LEFT JOIN '{parquet('members')}' m USING (msno)
        GROUP BY 1 ORDER BY n DESC
    """).df()

    print("  Churn by gender:\n", by_gender.to_string(index=False))
    print("\n  Churn by channel:\n", by_channel.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    sns.barplot(data=by_gender,  x="gender",         y="churn_pct", ax=axes[0])
    sns.barplot(data=by_channel, x="registered_via", y="churn_pct", ax=axes[1])
    sns.barplot(data=by_age,     x="age_bucket",     y="churn_pct", ax=axes[2])
    axes[2].tick_params(axis="x", rotation=30)
    for ax, ttl in zip(axes, ["by gender", "by channel", "by age"]):
        ax.set_title(f"Churn rate {ttl}"); ax.set_ylabel("churn rate (%)")
    fig.tight_layout()
    save(fig, "eda_churn_by_attributes.png")


# ── 5. Transaction signals ────────────────────────────────────────────────────
def eda_transaction_signals():
    section("5. Subscription Behaviour vs Churn (strongest signals)")
    txn = con.execute(f"""
        WITH latest AS (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY msno ORDER BY transaction_date DESC) rn
            FROM '{parquet('transactions_v2')}'
        )
        SELECT l.is_auto_renew, l.is_cancel,
               COUNT(*) n, ROUND(AVG(t.is_churn)*100,1) churn_pct
        FROM latest l
        JOIN '{parquet('train_v2')}' t USING (msno)
        WHERE l.rn = 1
        GROUP BY 1, 2 ORDER BY 1, 2
    """).df()
    print(txn.to_string(index=False))

    txn["segment"] = txn.apply(
        lambda r: f"renew={int(r.is_auto_renew)}\ncancel={int(r.is_cancel)}", axis=1)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.bar(txn["segment"], txn["churn_pct"],
                  color=["#66BB6A", "#42A5F5", "#FFA726", "#EF5350"])
    for bar, pct in zip(bars, txn["churn_pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5, f"{pct:.1f}%",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_title("Churn rate by auto-renew / cancel status")
    ax.set_ylabel("Churn rate (%)")
    fig.tight_layout()
    save(fig, "eda_transaction_signals.png")


# ── 6. Engagement vs churn ────────────────────────────────────────────────────
def eda_engagement():
    section("6. Listening Engagement vs Churn")
    log_tbl = "user_logs" if os.path.exists(parquet("user_logs")) else "user_logs_v2"
    print(f"  Using: {log_tbl}")

    eng = con.execute(f"""
        WITH agg AS (
            SELECT msno,
                   COUNT(DISTINCT date) active_days,
                   SUM(total_secs)/3600.0 total_hours
            FROM '{parquet(log_tbl)}' GROUP BY msno
        )
        SELECT t.is_churn, COUNT(*) n,
               ROUND(AVG(a.active_days),1)  avg_active_days,
               ROUND(AVG(a.total_hours),1)  avg_total_hours
        FROM '{parquet('train_v2')}' t
        LEFT JOIN agg a USING (msno)
        GROUP BY 1 ORDER BY 1
    """).df()
    print(eng.to_string(index=False))

    pct_none = con.execute(f"""
        SELECT ROUND(AVG(CASE WHEN a.msno IS NULL THEN 1 ELSE 0 END)*100,1)
        FROM '{parquet('train_v2')}' t
        LEFT JOIN (SELECT DISTINCT msno FROM '{parquet(log_tbl)}') a USING (msno)
    """).fetchone()[0]
    print(f"  Users with zero activity rows: {pct_none}%")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    labels = ["Retained", "Churned"]
    axes[0].bar(labels, eng["avg_active_days"],  color=["#42A5F5", "#EF5350"])
    axes[0].set_title("Avg active days"); axes[0].set_ylabel("Days")
    axes[1].bar(labels, eng["avg_total_hours"],  color=["#42A5F5", "#EF5350"])
    axes[1].set_title("Avg total listening hours"); axes[1].set_ylabel("Hours")
    fig.suptitle("Listening engagement: retained vs churned users", fontweight="bold")
    fig.tight_layout()
    save(fig, "eda_engagement_vs_churn.png")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=== Stage 1: Exploratory Data Analysis ===")
    eda_row_counts()
    eda_churn_balance()
    eda_member_demographics()
    eda_churn_by_attributes()
    eda_transaction_signals()
    eda_engagement()
    print(f"\n=== EDA complete. Plots saved to {RESULTS_DIR} ===")


if __name__ == "__main__":
    main()
