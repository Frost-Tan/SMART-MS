"""Stage2 Cascaded LightGBM Cross-Validation and Three Ablation Main Loop."""
from __future__ import annotations

import os
import warnings
from typing import Any, Dict, List

warnings.filterwarnings("ignore")

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from config import STRATEGIES
from features import (
    build_feature_names,
    extract_advanced_features,
    parse_spectrum,
    precursor_mass_for_features,
)
from plots import plot_and_export_importances, plot_results
from solver import solve_optimal_sugar_distribution


def _lgbm_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "n_estimators": int(cfg["LGBM_N_ESTIMATORS"]),
        "learning_rate": float(cfg["LGBM_LEARNING_RATE"]),
        "num_leaves": int(cfg["LGBM_NUM_LEAVES"]),
        "feature_fraction": float(cfg["LGBM_FEATURE_FRACTION"]),
        "bagging_fraction": float(cfg["LGBM_BAGGING_FRACTION"]),
        "bagging_freq": int(cfg["LGBM_BAGGING_FREQ"]),
        "min_child_samples": int(cfg["LGBM_MIN_CHILD_SAMPLES"]),
        "random_state": int(cfg["RANDOM_SEED"]),
        "class_weight": "balanced",
        "verbose": -1,
    }
    if str(cfg.get("LGBM_DEVICE", "cpu")).lower() == "gpu":
        base.update(
            {
                "device": "gpu",
                "max_bin": 63,
                "gpu_platform_id": int(cfg.get("LGBM_GPU_PLATFORM_ID", 0)),
                "gpu_device_id": int(cfg.get("LGBM_GPU_DEVICE_ID", 0)),
                "n_jobs": 1,
            }
        )
    else:
        base["n_jobs"] = -1
    return base


def run_ablation(cfg: Dict[str, Any]) -> None:
    out_root = cfg["OUTPUT_DIR"]
    os.makedirs(out_root, exist_ok=True)

    strategy_ids: List[int] = [1, 2, 3] if cfg.get("RUN_ALL_ABLATIONS") else [3]

    print("==================================================")
    print("  Saponin Cascaded Model Stage2 (Ablation Optional)")
    print("==================================================\n")
    print(f"Data: {cfg['DATA_PATH']}")
    print(f"Output Root: {out_root}")
    print(f"Running Strategies: {', '.join(STRATEGIES[sid] for sid in strategy_ids)}\n")

    try:
        full_data = pd.read_csv(cfg["DATA_PATH"])
    except Exception as e:
        print(f"Error: Data loading failed: {e}")
        return

    target_sugar = "total_suger_count"
    site_cols_map = {
        "C3": "C3_sugar",
        "C6": "C6_sugar",
        "C20": "C20_sugar",
        "C26": "C26_sugar",
        "C28": "C28_sugar",
        "Other_sugar": "Other_sugar",
    }

    for col in [target_sugar] + list(site_cols_map.values()):
        full_data[col] = pd.to_numeric(full_data[col], errors="coerce").fillna(0).astype(int)

    min_peaks = int(cfg["MIN_PEAKS"])
    x_list: List[np.ndarray] = []
    valid_indices: List[Any] = []
    for idx, row in tqdm(full_data.iterrows(), total=len(full_data), desc="[1/4] Extracting Features"):
        mz, inten = parse_spectrum(row.get("spectrum", ""))
        if len(mz) <= min_peaks:
            continue
        pmass = precursor_mass_for_features(row)
        x_list.append(extract_advanced_features(mz, inten, pmass, cfg))
        valid_indices.append(idx)

    x_arr = np.array(x_list)
    y_data = full_data.loc[valid_indices].copy().reset_index(drop=True)

    feature_names = build_feature_names(cfg)
    full_feature_names = feature_names + ["OOF_Type_Result", "OOF_Sugar_Result"]

    print(f"\n[2/4] Total valid samples: {len(x_arr)}. Starting training...")

    lgbm_params = _lgbm_params(cfg)
    le_type = LabelEncoder()
    y_data["type_encoded"] = le_type.fit_transform(y_data["type"].astype(str))
    idx_other = list(le_type.classes_).index("Other") if "Other" in le_type.classes_ else -1
    known_skeletons = [cls for cls in le_type.classes_ if cls != "Other"]

    global_comparison_results: List[Dict[str, Any]] = []
    outer_folds = int(cfg["OUTER_FOLDS"])
    inner_folds = int(cfg["INNER_FOLDS"])
    seed = int(cfg["RANDOM_SEED"])

    for strategy_id in strategy_ids:
        strategy_name = STRATEGIES[strategy_id]
        print(f"\n>>> Starting Strategy {strategy_id}: {strategy_name}")
        work_dir = os.path.join(out_root, strategy_name)
        os.makedirs(work_dir, exist_ok=True)
        joblib.dump(le_type, os.path.join(work_dir, "le_type.joblib"))

        fold_stats: List[Dict[str, Any]] = []
        all_fold_predictions: List[pd.DataFrame] = []
        y_true_type_all: List[Any] = []
        y_pred_type_all: List[Any] = []
        last_clf_type = last_clf_sugar = last_clf_site = None

        skf = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
        y_type_all = y_data["type_encoded"].values
        y_sugar_all = y_data[target_sugar].values
        y_sites_all = {site: y_data[col].values for site, col in site_cols_map.items()}

        for fold, (train_idx, val_idx) in enumerate(skf.split(x_arr, y_data["type"].astype(str))):
            print(f"    -> Running Fold {fold + 1}/{outer_folds}...")
            x_train, x_val = x_arr[train_idx], x_arr[val_idx]
            y_type_train, y_type_val = y_type_all[train_idx], y_type_all[val_idx]
            y_sugar_train, y_sugar_val = y_sugar_all[train_idx], y_sugar_all[val_idx]

            clf_type = lgb.LGBMClassifier(**lgbm_params)
            clf_type.fit(x_train, y_type_train)
            joblib.dump(clf_type, os.path.join(work_dir, f"model_type_fold{fold}.joblib"))
            pred_type_val_idx = np.argmax(clf_type.predict_proba(x_val), axis=1)
            pred_type_names = le_type.inverse_transform(pred_type_val_idx)

            y_true_type_all.extend(y_type_val)
            y_pred_type_all.extend(pred_type_val_idx)

            if strategy_id in (2, 3):
                clf_type_oof = lgb.LGBMClassifier(**lgbm_params)
                pred_type_train_oof = cross_val_predict(
                    clf_type_oof, x_train, y_type_train, cv=inner_folds, n_jobs=1
                )
                x_train_sugar = np.hstack([x_train, pred_type_train_oof.reshape(-1, 1)])
                x_val_sugar = np.hstack([x_val, pred_type_val_idx.reshape(-1, 1)])
            else:
                x_train_sugar, x_val_sugar = x_train, x_val

            clf_sugar = lgb.LGBMClassifier(**lgbm_params)
            clf_sugar.fit(x_train_sugar, y_sugar_train)
            joblib.dump(clf_sugar, os.path.join(work_dir, f"model_total_sugar_fold{fold}.joblib"))
            pred_sugar_val_val = clf_sugar.classes_[np.argmax(clf_sugar.predict_proba(x_val_sugar), axis=1)]

            if strategy_id in (2, 3):
                clf_sugar_oof = lgb.LGBMClassifier(**lgbm_params)
                pred_sugar_train_oof = cross_val_predict(
                    clf_sugar_oof, x_train_sugar, y_sugar_train, cv=inner_folds, n_jobs=1
                )
                x_train_sites = np.hstack([x_train_sugar, pred_sugar_train_oof.reshape(-1, 1)])
                x_val_sites = np.hstack([x_val_sugar, pred_sugar_val_val.reshape(-1, 1)])
            else:
                x_train_sites, x_val_sites = x_train, x_val

            site_probs: Dict[str, np.ndarray] = {}
            site_classes: Dict[str, np.ndarray] = {}
            y_sites_val: Dict[str, np.ndarray] = {}
            train_mask_known = y_type_train != idx_other
            x_train_sites_known = x_train_sites[train_mask_known]

            for site, y_all in y_sites_all.items():
                y_site_train_known = y_all[train_idx][train_mask_known]
                y_sites_val[site] = y_all[val_idx]
                clf_site = lgb.LGBMClassifier(**lgbm_params)

                if len(y_site_train_known) > 0 and len(np.unique(y_site_train_known)) > 1:
                    clf_site.fit(x_train_sites_known, y_site_train_known)
                    probs = clf_site.predict_proba(x_val_sites)
                    classes = clf_site.classes_
                    joblib.dump(clf_site, os.path.join(work_dir, f"model_site_{site}_fold{fold}.joblib"))
                    if site == "C3":
                        last_clf_site = clf_site
                else:
                    probs = np.zeros((len(x_val_sites), 1))
                    probs[:, 0] = 1.0
                    classes = np.array([0])
                    joblib.dump(None, os.path.join(work_dir, f"model_site_{site}_fold{fold}.joblib"))

                site_probs[site] = probs
                site_classes[site] = classes

            last_clf_type, last_clf_sugar = clf_type, clf_sugar

            final_preds = {s: [] for s in site_cols_map.keys()}

            if strategy_id in (1, 2):
                for i in range(len(x_val)):
                    is_other = str(pred_type_names[i]).strip().upper() == "OTHER"
                    for s in site_cols_map.keys():
                        if is_other:
                            final_preds[s].append(0)
                        else:
                            best_class_idx = int(np.argmax(site_probs[s][i]))
                            final_preds[s].append(int(site_classes[s][best_class_idx]))
            else:
                for i in range(len(x_val)):
                    best_combo = solve_optimal_sugar_distribution(
                        pred_sugar_val_val[i],
                        {s: site_probs[s][i] for s in site_cols_map.keys()},
                        site_classes,
                        pred_type_names[i],
                    )
                    for s_idx, s in enumerate(site_cols_map.keys()):
                        final_preds[s].append(best_combo[s_idx])

            is_other_true = y_type_val == idx_other
            is_known_true = ~is_other_true

            match_known = (y_type_val == pred_type_val_idx) & (y_sugar_val == pred_sugar_val_val)
            for s in site_cols_map.keys():
                match_known &= y_sites_val[s] == np.array(final_preds[s])
            match_other = (y_type_val == pred_type_val_idx) & (y_sugar_val == pred_sugar_val_val)
            matches = np.where(is_other_true, match_other, match_known)

            fold_res: Dict[str, Any] = {
                "Fold": fold + 1,
                "FullMatch_Acc": float(np.mean(matches)),
                "Type_Acc": float(accuracy_score(y_type_val, pred_type_val_idx)),
                "Sugar_Total_Acc": float(accuracy_score(y_sugar_val, pred_sugar_val_val)),
                "Type_Macro_F1": float(
                    f1_score(y_type_val, pred_type_val_idx, average="macro", zero_division=0)
                ),
                "Sugar_Total_Macro_F1": float(
                    f1_score(y_sugar_val, pred_sugar_val_val, average="macro", zero_division=0)
                ),
            }

            for cls in known_skeletons:
                c_idx = list(le_type.classes_).index(cls)
                fold_res[f"{cls}_Recall"] = float(
                    recall_score(
                        y_type_val,
                        pred_type_val_idx,
                        labels=[c_idx],
                        average=None,
                        zero_division=0,
                    )[0]
                )

            for s in site_cols_map.keys():
                if np.sum(is_known_true) > 0:
                    fold_res[f"{s}_Site_Acc"] = float(
                        accuracy_score(
                            y_sites_val[s][is_known_true],
                            np.array(final_preds[s])[is_known_true],
                        )
                    )
                    fold_res[f"{s}_Macro_F1"] = float(
                        f1_score(
                            y_sites_val[s][is_known_true],
                            np.array(final_preds[s])[is_known_true],
                            average="macro",
                            zero_division=0,
                        )
                    )
                else:
                    fold_res[f"{s}_Site_Acc"], fold_res[f"{s}_Macro_F1"] = np.nan, np.nan

            if idx_other != -1:
                fold_res["Other_Precision"] = float(
                    precision_score(
                        y_type_val,
                        pred_type_val_idx,
                        labels=[idx_other],
                        average=None,
                        zero_division=0,
                    )[0]
                )
                fold_res["Other_Recall"] = float(
                    recall_score(
                        y_type_val,
                        pred_type_val_idx,
                        labels=[idx_other],
                        average=None,
                        zero_division=0,
                    )[0]
                )
                fold_res["Other_F1"] = float(
                    f1_score(
                        y_type_val,
                        pred_type_val_idx,
                        labels=[idx_other],
                        average=None,
                        zero_division=0,
                    )[0]
                )
                if np.sum(is_other_true) > 0:
                    fold_res["Other_Sugar_Total_Acc"] = float(
                        accuracy_score(
                            y_sugar_val[is_other_true],
                            pred_sugar_val_val[is_other_true],
                        )
                    )
                    fold_res["Other_Sugar_Total_MAE"] = float(
                        mean_absolute_error(
                            y_sugar_val[is_other_true],
                            pred_sugar_val_val[is_other_true],
                        )
                    )
                else:
                    fold_res["Other_Sugar_Total_Acc"], fold_res["Other_Sugar_Total_MAE"] = np.nan, np.nan

            fold_stats.append(fold_res)

            df_fold_pred = y_data.iloc[val_idx].copy()
            df_fold_pred["Pred_Type"] = pred_type_names
            df_fold_pred["Pred_Total_Sugar"] = pred_sugar_val_val
            for s in site_cols_map.keys():
                df_fold_pred[f"Pred_{s}"] = final_preds[s]
            df_fold_pred["Is_Perfect_Match"] = matches
            all_fold_predictions.append(df_fold_pred)

        df_stats = pd.DataFrame(fold_stats)
        mean_stats = df_stats.mean(numeric_only=True)
        std_stats = df_stats.std(numeric_only=True)

        plot_results(y_true_type_all, y_pred_type_all, le_type, mean_stats, work_dir)

        st1_imp = st2_imp = st3_imp = combined_imp = None
        if strategy_id == 3 and last_clf_site is not None:
            st1_imp, st2_imp, st3_imp, combined_imp = plot_and_export_importances(
                last_clf_type, last_clf_sugar, last_clf_site, full_feature_names, work_dir
            )

        report_lines: List[str] = []
        report_lines.append("==================================================")
        report_lines.append("   ADVANCED SAPONIN IDENTIFICATION MODEL REPORT   ")
        report_lines.append(f"          (STRATEGY: {strategy_name})             ")
        report_lines.append("==================================================\n")

        for res in fold_stats:
            report_lines.append(f"--- FOLD {res['Fold']} ---")
            report_lines.append(f"FullMatch Acc           : {res['FullMatch_Acc']:.4f}")
            report_lines.append(f"Type Acc                : {res['Type_Acc']:.4f}")
            report_lines.append(f"Total Sugar Acc         : {res['Sugar_Total_Acc']:.4f}")
            for s in site_cols_map.keys():
                val = res.get(f"{s}_Site_Acc", np.nan)
                if pd.notna(val):
                    report_lines.append(f"{s} Site Acc             : {val:.4f}")
            for cls in known_skeletons:
                val = res.get(f"{cls}_Recall", np.nan)
                if pd.notna(val):
                    report_lines.append(f"{cls} Recall              : {val:.4f}")
            report_lines.append(f"Type Macro-F1           : {res['Type_Macro_F1']:.4f}")
            report_lines.append(f"Sugar Total Macro-F1    : {res['Sugar_Total_Macro_F1']:.4f}")
            for s in site_cols_map.keys():
                val = res.get(f"{s}_Macro_F1", np.nan)
                if pd.notna(val):
                    report_lines.append(f"{s} Macro-F1             : {val:.4f}")
            report_lines.append("")

        report_lines.append("==================================================")
        report_lines.append("           OVERALL PERFORMANCE (MEAN ± STD)       ")
        report_lines.append("==================================================\n")

        report_lines.append("--- GLOBAL PERFORMANCE ---")
        keys_global = [
            "FullMatch_Acc",
            "Type_Acc",
            "Sugar_Total_Acc",
            "Type_Macro_F1",
            "Sugar_Total_Macro_F1",
        ]
        for k in keys_global:
            if k in mean_stats and pd.notna(mean_stats[k]):
                report_lines.append(f"{k:<23} : {mean_stats[k]:.4f} ± {std_stats.get(k, 0):.4f}")

        report_lines.append("\n--- SKELETON RECALL ---")
        for cls in known_skeletons:
            k = f"{cls}_Recall"
            if k in mean_stats and pd.notna(mean_stats[k]):
                report_lines.append(f"{k:<23} : {mean_stats[k]:.4f} ± {std_stats.get(k, 0):.4f}")

        report_lines.append("\n--- KNOWN SKELETON SITE PERFORMANCE (MASKED) ---")
        for s in site_cols_map.keys():
            k_acc = f"{s}_Site_Acc"
            k_f1 = f"{s}_Macro_F1"
            if k_acc in mean_stats and pd.notna(mean_stats[k_acc]):
                report_lines.append(f"{k_acc:<23} : {mean_stats[k_acc]:.4f} ± {std_stats.get(k_acc, 0):.4f}")
            if k_f1 in mean_stats and pd.notna(mean_stats[k_f1]):
                report_lines.append(f"{k_f1:<23} : {mean_stats[k_f1]:.4f} ± {std_stats.get(k_f1, 0):.4f}")

        if idx_other != -1:
            report_lines.append("\n--- OTHER CLASS SPECIFIC PERFORMANCE ---")
            keys_other = [
                "Other_Precision",
                "Other_Recall",
                "Other_F1",
                "Other_Sugar_Total_Acc",
                "Other_Sugar_Total_MAE",
            ]
            for k in keys_other:
                if k in mean_stats and pd.notna(mean_stats[k]):
                    report_lines.append(f"{k:<23} : {mean_stats[k]:.4f} ± {std_stats.get(k, 0):.4f}")

        if strategy_id == 3 and st1_imp is not None:
            report_lines.append("\n==================================================")
            report_lines.append("     INDIVIDUAL STAGE FEATURE IMPORTANCE (TOP 10) ")
            report_lines.append("==================================================")

            report_lines.append("\n[STAGE 1: SKELETON TYPE]")
            report_lines.append(f"{'Feature':<25} | {'Importance':<10}")
            report_lines.append("-" * 40)
            for _, row in st1_imp.iterrows():
                report_lines.append(f"{row['Feature']:<25} | {row['Stage_1']:.6f}")

            report_lines.append("\n[STAGE 2: TOTAL SUGAR]")
            report_lines.append(f"{'Feature':<25} | {'Importance':<10}")
            report_lines.append("-" * 40)
            for _, row in st2_imp.iterrows():
                report_lines.append(f"{row['Feature']:<25} | {row['Stage_2']:.6f}")

            report_lines.append("\n[STAGE 3: SPECIFIC SITES]")
            report_lines.append(f"{'Feature':<25} | {'Importance':<10}")
            report_lines.append("-" * 40)
            for _, row in st3_imp.iterrows():
                report_lines.append(f"{row['Feature']:<25} | {row['Stage_3']:.6f}")

            report_lines.append("\n==================================================")
            report_lines.append("       COMBINED EVOLUTION FEATURE IMPORTANCE      ")
            report_lines.append("==================================================")
            report_lines.append(
                f"{'Feature':<25} | {'Stage 1 (Type)':<15} | {'Stage 2 (Sugar)':<15} | {'Stage 3 (Site)':<15}"
            )
            report_lines.append("-" * 79)
            for _, row in combined_imp.iterrows():
                report_lines.append(
                    f"{row['Feature']:<25} | {row['Stage 1: Skeleton']:.6f}        | "
                    f"{row['Stage 2: Total Sugar']:.6f}         | {row['Stage 3: Sites']:.6f}"
                )
            report_lines.append("==================================================\n")

        report_content = "\n".join(report_lines)
        print(f"\n{report_content}\n")

        with open(os.path.join(work_dir, "model_performance_report.txt"), "w", encoding="utf-8") as f:
            f.write(report_content)
        pd.concat(all_fold_predictions, ignore_index=True).to_csv(
            os.path.join(work_dir, "cv_prediction_detailed.csv"), index=False
        )
        df_stats.to_csv(os.path.join(work_dir, "cv_metrics_summary.csv"), index=False)

        global_comparison_results.append(
            {
                "Strategy": strategy_name,
                "FullMatch_Acc": mean_stats.get("FullMatch_Acc", 0),
                "Type_Acc": mean_stats.get("Type_Acc", 0),
                "Sugar_Total_Acc": mean_stats.get("Sugar_Total_Acc", 0),
            }
        )

    if cfg.get("RUN_ALL_ABLATIONS") and len(global_comparison_results) > 1:
        print("\n[3/4] Generating cross-strategy performance comparison files in output directory...")
        df_compare = pd.DataFrame(global_comparison_results)
        df_compare.to_csv(os.path.join(out_root, "ablation_comparison_metrics.csv"), index=False)

        plt.figure(figsize=(10, 6))
        df_melt = df_compare.melt(id_vars="Strategy", var_name="Metric", value_name="Score")
        sns.barplot(data=df_melt, x="Strategy", y="Score", hue="Metric", palette="Set2")
        plt.title("Ablation Study: Metrics Comparison Across Strategies")
        plt.ylim(0, 1.05)
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(os.path.join(out_root, "ablation_performance_bar.png"), dpi=300)
        plt.close()

        with open(os.path.join(out_root, "ablation_overall_report.txt"), "w", encoding="utf-8") as f:
            f.write("=========================================\n")
            f.write("       ABLATION STUDY OVERALL REPORT     \n")
            f.write("=========================================\n\n")
            f.write(df_compare.to_string(index=False))

    print(f"\n[4/4] Execution complete! Results directory: {out_root}")
