"""ROC and feature importance plotting after training."""
from __future__ import annotations

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, roc_curve


def plot_metrics(
    cfg: Dict,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    feature_importances: np.ndarray,
    feature_names: List[str],
) -> None:
    _ = y_pred
    plt.style.use("seaborn-v0_8-whitegrid")
    out = cfg["OUTPUT_DIR"]

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(out, "ROC_Curve_MH_Baseline.png"), dpi=300, bbox_inches="tight")
    plt.close()

    top_n = 20
    idx = np.argsort(feature_importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in idx]
    top_scores = feature_importances[idx]

    plt.figure(figsize=(10, 8))
    sns.barplot(x=top_scores, y=top_features, palette="mako")
    plt.title(f"Top {top_n} Features (Based on [M-H]- Mass Baseline)")
    plt.xlabel("LightGBM Feature Importance (Gain)")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "Feature_Importance_MH_Baseline.png"), dpi=300)
    plt.close()
