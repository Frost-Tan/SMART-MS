"""LightGBM cross-validation training, model saving, and report generation."""
from __future__ import annotations

import os
import warnings
from typing import Any, Dict

warnings.filterwarnings("ignore")

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from features import (
    extract_features,
    generate_feature_names,
    get_mh_baseline_mass,
    parse_spectrum_and_filter,
)
from plots import plot_metrics


def run_training(cfg: Dict[str, Any]) -> None:
    for d in [cfg["OUTPUT_DIR"], cfg["MODEL_DIR"]]:
        os.makedirs(d, exist_ok=True)

    print(f">>> Loading dataset: {cfg['DATA_PATH']}")
    df = pd.read_csv(cfg["DATA_PATH"])

    x_list, y_list = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Features"):
        mz_list, int_list = parse_spectrum_and_filter(row["spectrum"], cfg)
        if not mz_list:
            continue
        mh_baseline = get_mh_baseline_mass(row["precursor_mz"], row.get("adduct", ""), cfg)
        feat_vec = extract_features(mz_list, int_list, row["precursor_mz"], mh_baseline, cfg)
        x_list.append(feat_vec)
        y_list.append(int(row["is_saponin"]))

    x_arr = np.array(x_list)
    y_arr = np.array(y_list)
    feature_names = generate_feature_names(cfg)

    print(f"\n>>> Feature construction complete | Samples: {x_arr.shape[0]} | Dimensions: {x_arr.shape[1]}")

    skf = StratifiedKFold(n_splits=cfg["CV_FOLDS"], shuffle=True, random_state=cfg["RANDOM_SEED"])

    oof_preds = np.zeros(len(x_arr))
    oof_probs = np.zeros(len(x_arr))
    fold_metrics = []
    feature_importances = np.zeros(x_arr.shape[1])

    clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        importance_type="gain",
        random_state=cfg["RANDOM_SEED"],
        n_jobs=-1,
        verbose=-1,
    )

    print(f"\n>>> Starting {cfg['CV_FOLDS']}-Fold cross-validation training and saving models...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(x_arr, y_arr)):
        clf.fit(x_arr[train_idx], y_arr[train_idx])

        probs = clf.predict_proba(x_arr[val_idx])[:, 1]
        preds = clf.predict(x_arr[val_idx])
        oof_probs[val_idx], oof_preds[val_idx] = probs, preds
        feature_importances += clf.feature_importances_ / cfg["CV_FOLDS"]

        model_save_path = os.path.join(cfg["MODEL_DIR"], f"lgbm_mh_baseline_fold_{fold + 1}.joblib")
        joblib.dump(clf, model_save_path)

        fold_auc = roc_auc_score(y_arr[val_idx], probs)
        fold_metrics.append(
            [
                fold + 1,
                fold_auc,
                precision_score(y_arr[val_idx], preds),
                recall_score(y_arr[val_idx], preds),
                f1_score(y_arr[val_idx], preds),
            ]
        )
        print(f"  Fold {fold + 1}: AUC={fold_auc:.4f} (saved)")

    overall_auc = roc_auc_score(y_arr, oof_probs)
    print("\n>>> Generating visualization images and reports...")
    plot_metrics(cfg, y_arr, oof_probs, oof_preds, feature_importances, feature_names)

    report_path = os.path.join(cfg["OUTPUT_DIR"], "Saponin_MH_Baseline_Report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 50 + "\n  Saponin Model ([M-H]- Baseline) - LGBM Report\n" + "=" * 50 + "\n\n")
        f.write("[Model Configuration]\nData Path     : " + cfg["DATA_PATH"] + "\n")
        f.write(f"Spec Bin Size : {cfg['BIN_SIZE_SPEC']} Da\nLoss Bin Size : {cfg['BIN_SIZE_LOSS']} Da\n\n")
        f.write("[Per-Fold Performance]\n")
        f.write(f"{'Fold':<6} {'AUC':<10} {'Precision':<12} {'Recall':<10} {'F1-Score'}\n")
        for m in fold_metrics:
            f.write(f"{m[0]:<6} {m[1]:<10.4f} {m[2]:<12.4f} {m[3]:<10.4f} {m[4]:.4f}\n")
        f.write("\n" + "-" * 50 + "\n[Overall Out-of-Fold (OOF) Metrics]\n")
        f.write(f"ROC-AUC   : {overall_auc:.4f}\nPrecision : {precision_score(y_arr, oof_preds):.4f}\n")
        f.write(f"Recall    : {recall_score(y_arr, oof_preds):.4f}\nF1-Score  : {f1_score(y_arr, oof_preds):.4f}\n")
        f.write("\n" + "=" * 50 + "\n[Top 20 Most Important Features (Gain)]\n" + "=" * 50 + "\n")
        idx = np.argsort(feature_importances)[::-1][:20]
        for i, idx_val in enumerate(idx):
            f.write(
                f"Top {i + 1:02d}: {feature_names[idx_val]:<20} | Importance: {feature_importances[idx_val]:.4f}\n"
            )

    print(f"\n✅ Execution complete! Models saved to {cfg['MODEL_DIR']}")
