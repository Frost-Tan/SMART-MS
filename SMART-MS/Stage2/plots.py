"""Confusion matrix, metric bar chart, and cascaded feature importance."""
from __future__ import annotations

import os
from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_results(
    y_true_type: list,
    y_pred_type: list,
    le: Any,
    mean_stats: pd.Series,
    output_dir: str,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true_type, y_pred_type)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
    )
    plt.title("Confusion Matrix of Saponin Types")
    plt.ylabel("True Skeleton")
    plt.xlabel("Predicted Skeleton")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_confusion_matrix_type.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(12, 6))
    known_site_accs = [
        mean_stats.get(f"{s}_Site_Acc", 0)
        for s in ["C3", "C6", "C20", "C26", "C28", "Other_sugar"]
        if pd.notna(mean_stats.get(f"{s}_Site_Acc", 0))
    ]
    avg_known_site_acc = float(np.mean(known_site_accs)) if known_site_accs else 0.0
    metrics_to_plot = {
        "Full Match Acc": mean_stats.get("FullMatch_Acc", 0),
        "Type Acc": mean_stats.get("Type_Acc", 0),
        "Total Sugar Acc": mean_stats.get("Sugar_Total_Acc", 0),
        "Other F1": mean_stats.get("Other_F1", 0),
        "Known Site Acc (Avg)": avg_known_site_acc,
    }
    sns.barplot(x=list(metrics_to_plot.keys()), y=list(metrics_to_plot.values()), palette="viridis")
    plt.title("Overall Model Performance Metrics")
    plt.ylim(0, 1.05)
    for i, v in enumerate(metrics_to_plot.values()):
        plt.text(i, v + 0.02, f"{v:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_overall_metrics.png"), dpi=300)
    plt.close()


def plot_and_export_importances(
    clf_type: Any,
    clf_sugar: Any,
    clf_site: Any,
    feature_names: list[str],
    work_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    imp_type = clf_type.feature_importances_
    imp_sugar = clf_sugar.feature_importances_
    imp_site = clf_site.feature_importances_

    max_len = len(imp_site)
    imp_type_pad = np.pad(imp_type, (0, max_len - len(imp_type)))
    imp_sugar_pad = np.pad(imp_sugar, (0, max_len - len(imp_sugar)))

    df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Stage_1": imp_type_pad / (imp_type_pad.sum() + 1e-6),
            "Stage_2": imp_sugar_pad / (imp_sugar_pad.sum() + 1e-6),
            "Stage_3": imp_site / (imp_site.sum() + 1e-6),
        }
    )

    df_st1_top = df[["Feature", "Stage_1"]].sort_values("Stage_1", ascending=False).head(10)
    df_st2_top = df[["Feature", "Stage_2"]].sort_values("Stage_2", ascending=False).head(10)
    df_st3_top = df[["Feature", "Stage_3"]].sort_values("Stage_3", ascending=False).head(10)

    def _plot_single_stage(data: pd.DataFrame, val_col: str, title: str, filename: str, color: str) -> None:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=data, y="Feature", x=val_col, color=color)
        plt.title(title)
        plt.xlabel("Normalized Importance")
        plt.grid(axis="x", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(work_dir, filename), dpi=300)
        plt.close()

    _plot_single_stage(
        df_st1_top,
        "Stage_1",
        "Stage 1 (Skeleton Type) - Top 10 Features",
        "feature_importance_stage1_type.png",
        "#4C72B0",
    )
    _plot_single_stage(
        df_st2_top,
        "Stage_2",
        "Stage 2 (Total Sugar) - Top 10 Features",
        "feature_importance_stage2_sugar.png",
        "#DD8452",
    )
    _plot_single_stage(
        df_st3_top,
        "Stage_3",
        "Stage 3 (Specific Sites) - Top 10 Features",
        "feature_importance_stage3_site.png",
        "#55A868",
    )

    top_n_per_stage = 5
    combined_features = list(
        set(
            df_st1_top.head(top_n_per_stage)["Feature"].tolist()
            + df_st2_top.head(top_n_per_stage)["Feature"].tolist()
            + df_st3_top.head(top_n_per_stage)["Feature"].tolist()
        )
    )

    df_combined = df[df["Feature"].isin(combined_features)].copy()
    df_combined.columns = ["Feature", "Stage 1: Skeleton", "Stage 2: Total Sugar", "Stage 3: Sites"]

    df_plot = df_combined.melt(id_vars="Feature", var_name="Model Stage", value_name="Importance Score")
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df_plot, y="Feature", x="Importance Score", hue="Model Stage", palette="muted")
    plt.title("Cascaded Feature Evolution (Union of Stage-Specific Top Features)")
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(work_dir, "cascaded_importance_evolution.png"), dpi=300)
    plt.close()

    return df_st1_top, df_st2_top, df_st3_top, df_combined
