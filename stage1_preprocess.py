"""
stage1_preprocess.py
────────────────────
Stage 1: Data loading, EDA, preprocessing pipeline, and temporal split.

Run:
    python stage1_preprocess.py --data data/creditcard.csv

Outputs (all in outputs/):
    outputs/artifacts/preprocessor.joblib   ← fitted pipeline, used by every stage
    outputs/artifacts/X_train.npy           ← preprocessed feature arrays
    outputs/artifacts/X_val.npy
    outputs/artifacts/X_test.npy
    outputs/artifacts/y_train.npy           ← label arrays
    outputs/artifacts/y_val.npy
    outputs/artifacts/y_test.npy
    outputs/plots/eda_*.png                 ← EDA figures (also logged to MLflow)

MLflow:
    Experiment : fraud_detection
    Run name   : stage1_preprocessing
    Logs       : dataset stats, split sizes, class ratios, all EDA plots
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import mlflow

# ── project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.data.preprocessing import (
    temporal_split,
    fit_and_apply,
    save_preprocessor,
    get_feature_names,
)
from src.utils.mlflow_utils import setup_mlflow, log_figure, end_run_if_active

warnings.filterwarnings("ignore")

PLOTS_DIR     = Path("outputs/plots")
ARTIFACTS_DIR = Path("outputs/artifacts")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# ─── EDA functions ────────────────────────────────────────────────────────────

def eda_class_distribution(df: pd.DataFrame) -> plt.Figure:
    """Bar chart of class balance — shows the extreme imbalance clearly."""
    counts = df["Class"].value_counts().sort_index()
    labels = ["Legitimate (0)", "Fraud (1)"]
    colors = ["#4A90D9", "#E05C5C"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Raw counts
    axes[0].bar(labels, counts.values, color=colors, width=0.5, edgecolor="white")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 500, f"{v:,}", ha="center", fontsize=10)
    axes[0].set_title("Transaction counts by class", fontsize=12)
    axes[0].set_ylabel("Count")
    axes[0].grid(axis="y", alpha=0.3)

    # Percentages
    pct = counts / counts.sum() * 100
    axes[1].bar(labels, pct.values, color=colors, width=0.5, edgecolor="white")
    for i, v in enumerate(pct.values):
        axes[1].text(i, v + 0.05, f"{v:.3f}%", ha="center", fontsize=10)
    axes[1].set_title("Class distribution (%)", fontsize=12)
    axes[1].set_ylabel("Percentage")
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Class imbalance  —  {counts[1]} fraud in {len(df):,} transactions "
        f"({counts[1]/len(df)*100:.3f}%)",
        fontsize=13, y=1.01
    )
    fig.tight_layout()
    return fig


def eda_amount_by_class(df: pd.DataFrame) -> plt.Figure:
    """Distribution of transaction amounts split by fraud / legitimate."""
    fraud  = df[df["Class"] == 1]["Amount"]
    legit  = df[df["Class"] == 0]["Amount"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Log-scale histogram overlay
    axes[0].hist(legit,  bins=80, alpha=0.6, color="#4A90D9", label="Legitimate", density=True, log=True)
    axes[0].hist(fraud,  bins=80, alpha=0.6, color="#E05C5C", label="Fraud",      density=True, log=True)
    axes[0].set_xlabel("Amount ($)")
    axes[0].set_ylabel("Density (log scale)")
    axes[0].set_title("Amount distribution (log y)")
    axes[0].legend()

    # Box plots side by side
    axes[1].boxplot(
        [np.log1p(legit), np.log1p(fraud)],
        labels=["Legitimate", "Fraud"],
        patch_artist=True,
        boxprops=dict(facecolor="none"),
        medianprops=dict(color="black", linewidth=1.5),
    )
    axes[1].set_ylabel("log(1 + Amount)")
    axes[1].set_title("log(Amount) box plot")

    # Summary statistics table
    stats = pd.DataFrame({
        "Legitimate": legit.describe().round(2),
        "Fraud":      fraud.describe().round(2),
    })
    axes[2].axis("off")
    tbl = axes[2].table(
        cellText=stats.values,
        rowLabels=stats.index,
        colLabels=stats.columns,
        cellLoc="center", loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.1, 1.4)
    axes[2].set_title("Amount statistics", fontsize=11)

    fig.suptitle("Transaction amount analysis by class", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


def eda_time_patterns(df: pd.DataFrame) -> plt.Figure:
    """
    Transaction volume and fraud rate over time.
    Time is in seconds — we convert to hours for readability.
    """
    df_copy = df.copy()
    df_copy["hour"] = (df_copy["Time"] / 3600).astype(int)

    hourly = df_copy.groupby("hour").agg(
        total=("Class", "count"),
        fraud=("Class", "sum"),
    ).reset_index()
    hourly["fraud_rate"] = hourly["fraud"] / hourly["total"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].bar(hourly["hour"], hourly["total"], color="#4A90D9", alpha=0.7, width=0.8)
    axes[0].set_ylabel("Number of transactions")
    axes[0].set_title("Transaction volume over time")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].plot(hourly["hour"], hourly["fraud_rate"] * 100, color="#E05C5C",
                 linewidth=1.5, marker="o", markersize=3)
    axes[1].set_xlabel("Hours elapsed (from dataset start)")
    axes[1].set_ylabel("Fraud rate (%)")
    axes[1].set_title("Fraud rate over time (per hour bucket)")
    axes[1].grid(alpha=0.3)

    # Mark the temporal split points
    n = len(df)
    train_end_row = int(n * 0.70)
    val_end_row   = int(n * 0.85)
    train_end_hr  = df_copy.iloc[train_end_row]["hour"]
    val_end_hr    = df_copy.iloc[val_end_row]["hour"]

    for ax in axes:
        ax.axvline(train_end_hr, color="green",  linestyle="--", linewidth=1.0,
                   label=f"Train end (row {train_end_row:,})")
        ax.axvline(val_end_hr,   color="orange", linestyle="--", linewidth=1.0,
                   label=f"Val end (row {val_end_row:,})")

    axes[0].legend(fontsize=8)
    fig.suptitle("Temporal patterns — the dashed lines show the train/val/test boundaries",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    return fig


def eda_feature_importance_by_fraud(df: pd.DataFrame) -> plt.Figure:
    """
    For each V feature, shows the mean absolute difference between
    fraud and legitimate distributions. This reveals which V features
    carry the most discriminative signal.
    """
    v_cols = [f"V{i}" for i in range(1, 29)]
    fraud_means = df[df["Class"] == 1][v_cols].mean()
    legit_means = df[df["Class"] == 0][v_cols].mean()
    diff = (fraud_means - legit_means).abs().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#E05C5C" if d > diff.median() else "#4A90D9" for d in diff.values]
    ax.bar(diff.index, diff.values, color=colors, edgecolor="white", width=0.7)
    ax.set_xlabel("Feature")
    ax.set_ylabel("|mean(fraud) − mean(legit)|")
    ax.set_title("Feature discriminability — absolute mean difference between classes")
    ax.axhline(diff.median(), linestyle="--", color="gray", linewidth=0.8,
               label=f"Median ({diff.median():.2f})")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def eda_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Correlation of V features with the Class label."""
    v_cols = [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    corr = df[v_cols].corr()["Class"].drop("Class").sort_values()

    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ["#E05C5C" if v < 0 else "#4A90D9" for v in corr.values]
    ax.barh(corr.index, corr.values, color=colors, edgecolor="white", height=0.7)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson correlation with Class label")
    ax.set_title("Feature–Class correlations")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def eda_split_summary(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> plt.Figure:
    """Table summarising the three splits: sizes and fraud rates."""
    rows = []
    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        n_total = len(split)
        n_fraud = split["Class"].sum()
        rows.append([
            name,
            f"{n_total:,}",
            f"{n_fraud:,}",
            f"{n_fraud / n_total * 100:.3f}%",
            f"{split['Time'].min():.0f}s",
            f"{split['Time'].max():.0f}s",
        ])

    fig, ax = plt.subplots(figsize=(9, 2))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows,
        colLabels=["Split", "Transactions", "Fraud cases", "Fraud rate",
                   "Time start", "Time end"],
        cellLoc="center", loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.8)
    # Header row styling
    for j in range(6):
        tbl[(0, j)].set_facecolor("#2C2C2A")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")
    ax.set_title("Temporal split summary", fontsize=12, pad=20)
    fig.tight_layout()
    return fig


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(data_path: str, val_frac: float = 0.15, test_frac: float = 0.15) -> None:
    print("\n" + "═" * 60)
    print("  STAGE 1 — DATA PREPROCESSING & EDA")
    print("═" * 60)

    # ── 1. Load data ──────────────────────────────────────────────
    print(f"\n[1/6] Loading data from {data_path}")
    df = pd.read_csv(data_path)
    print(f"      Shape : {df.shape}")
    print(f"      Fraud : {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")
    print(f"      Nulls : {df.isnull().sum().sum()}")

    assert df.isnull().sum().sum() == 0, "Dataset has null values — check your CSV."
    assert "Class" in df.columns, "Expected a 'Class' column in the dataset."
    assert "Time" in df.columns and "Amount" in df.columns

    # ── 2. Temporal split ─────────────────────────────────────────
    print(f"\n[2/6] Temporal split  (val={val_frac:.0%}, test={test_frac:.0%})")
    train_df, val_df, test_df = temporal_split(df, val_frac=val_frac, test_frac=test_frac)
    print(f"      Train : {len(train_df):>7,}  ({train_df['Class'].sum()} fraud, "
          f"{train_df['Class'].mean()*100:.3f}%)")
    print(f"      Val   : {len(val_df):>7,}  ({val_df['Class'].sum()} fraud, "
          f"{val_df['Class'].mean()*100:.3f}%)")
    print(f"      Test  : {len(test_df):>7,}  ({test_df['Class'].sum()} fraud, "
          f"{test_df['Class'].mean()*100:.3f}%)")

    # ── 3. EDA plots ──────────────────────────────────────────────
    print("\n[3/6] Generating EDA plots...")
    eda_figures = {
        "eda_class_distribution":      eda_class_distribution(df),
        "eda_amount_by_class":          eda_amount_by_class(df),
        "eda_time_patterns":            eda_time_patterns(df),
        "eda_feature_discriminability": eda_feature_importance_by_fraud(df),
        "eda_feature_class_corr":       eda_correlation_heatmap(df),
        "eda_split_summary":            eda_split_summary(train_df, val_df, test_df),
    }
    for name, fig in eda_figures.items():
        path = PLOTS_DIR / f"{name}.png"
        fig.savefig(path, bbox_inches="tight", dpi=130)
        plt.close(fig)
        print(f"      Saved → {path}")

    # ── 4. Preprocessing pipeline ─────────────────────────────────
    print("\n[4/6] Fitting preprocessing pipeline...")
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = fit_and_apply(
        train_df, val_df, test_df
    )
    print(f"      Feature names : {get_feature_names()}")
    print(f"      X_train shape : {X_train.shape}  dtype={X_train.dtype}")
    print(f"      X_val shape   : {X_val.shape}")
    print(f"      X_test shape  : {X_test.shape}")

    # ── 5. Save artifacts ─────────────────────────────────────────
    print("\n[5/6] Saving artifacts...")
    save_preprocessor(preprocessor, str(ARTIFACTS_DIR / "preprocessor.joblib"))
    for name, arr in [
        ("X_train", X_train), ("X_val", X_val), ("X_test", X_test),
        ("y_train", y_train), ("y_val", y_val), ("y_test", y_test),
    ]:
        np.save(str(ARTIFACTS_DIR / f"{name}.npy"), arr)
        print(f"      Saved → outputs/artifacts/{name}.npy")

    # ── 6. MLflow logging ─────────────────────────────────────────
    print("\n[6/6] Logging to MLflow...")
    setup_mlflow()
    end_run_if_active()

    with mlflow.start_run(run_name="stage1_preprocessing"):
        # Dataset stats
        mlflow.log_params({
            "data_path":     data_path,
            "n_total":       len(df),
            "n_fraud":       int(df["Class"].sum()),
            "fraud_rate":    round(float(df["Class"].mean()), 6),
            "val_frac":      val_frac,
            "test_frac":     test_frac,
            "n_features":    X_train.shape[1],
        })
        mlflow.log_metrics({
            "train_size":       len(train_df),
            "val_size":         len(val_df),
            "test_size":        len(test_df),
            "train_fraud_rate": round(float(train_df["Class"].mean()), 6),
            "val_fraud_rate":   round(float(val_df["Class"].mean()), 6),
            "test_fraud_rate":  round(float(test_df["Class"].mean()), 6),
        })
        # EDA plots
        for name in eda_figures:
            mlflow.log_artifact(str(PLOTS_DIR / f"{name}.png"), artifact_path="eda_plots")
        # Preprocessor
        mlflow.log_artifact(str(ARTIFACTS_DIR / "preprocessor.joblib"), artifact_path="pipeline")

    print("\n" + "─" * 60)
    print("  Stage 1 complete.")
    print("  Next step: python stage2_train_models.py")
    print("─" * 60 + "\n")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1 — preprocessing and EDA")
    parser.add_argument(
        "--data",
        type=str,
        default="data/creditcard.csv",
        help="Path to creditcard.csv",
    )
    parser.add_argument("--val-frac",  type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    args = parser.parse_args()

    main(
        data_path=args.data,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
    )