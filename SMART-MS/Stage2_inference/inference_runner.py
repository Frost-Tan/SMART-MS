"""Stage2 cascaded inference main workflow."""
from __future__ import annotations

import os
import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from distribution_solver import solve_optimal_distribution
from feature_extract import extract_advanced_features_from_spec
from loaders import load_database, load_models
from mgf_parser import parse_mgf

warnings.filterwarnings("ignore")


def run_inference(cfg: Dict[str, Any]) -> None:
    os.makedirs(cfg["OUTPUT_DIR"], exist_ok=True)

    site_cols: List[str] = list(cfg["SITE_COLS"])
    adduct_modes: List[str] = list(cfg["ADDUCT_MODES"])

    le_type, type_models, sugar_models, site_models = load_models(cfg["MODEL_DIR"], cfg)
    db_list = load_database(cfg["DB_FILE"])
    spectra = parse_mgf(cfg["MGF_FILE"], cfg)
    print(f"Found {len(spectra)} valid spectra.")
    results_list: List[Dict[str, Any]] = []

    h_plus = float(cfg["H_PLUS_MASS"])
    fa_ion = float(cfg["FA_ION_MASS"])
    ppm_tol = float(cfg["MASS_TOLERANCE_PPM"])

    for adduct_mode in adduct_modes:
        print(f"\n🚀 Running Adduct: {adduct_mode}")
        valid_specs: List[Dict[str, Any]] = []
        X_list: List[np.ndarray] = []
        for s in spectra:
            feat = extract_advanced_features_from_spec(s, cfg, adduct_mode)
            if feat is not None:
                valid_specs.append(s)
                X_list.append(feat)
        if not X_list:
            continue

        X_base = np.array(X_list)
        n = X_base.shape[0]

        avg_prob_type = np.mean([clf.predict_proba(X_base) for clf in type_models], axis=0)
        pred_type_idx = np.argmax(avg_prob_type, axis=1)
        pred_type_name = le_type.inverse_transform(pred_type_idx)
        prob_t = np.max(avg_prob_type, axis=1)

        X_sugar = np.hstack([X_base, pred_type_idx.reshape(-1, 1)])
        sugar_union = np.unique(np.concatenate([clf.classes_ for clf in sugar_models]))
        probs_sugar_folds = []
        for clf in sugar_models:
            p = clf.predict_proba(X_sugar)
            if p.shape[1] < len(sugar_union):
                pad_p = np.zeros((n, len(sugar_union)))
                for i, c in enumerate(clf.classes_):
                    pad_p[:, np.where(sugar_union == c)[0][0]] = p[:, i]
                p = pad_p
            probs_sugar_folds.append(p)
        avg_prob_sugar = np.mean(probs_sugar_folds, axis=0)
        pred_sugar_val = np.array([sugar_union[idx] for idx in np.argmax(avg_prob_sugar, axis=1)])
        prob_s = np.max(avg_prob_sugar, axis=1)

        X_sites = np.hstack([X_sugar, pred_sugar_val.reshape(-1, 1)])
        site_probs_avg: Dict[str, np.ndarray] = {}
        site_classes_dict: Dict[str, np.ndarray] = {}
        for s in site_cols:
            v_clfs = [c for c in site_models[s] if c is not None]
            union_labels = np.unique(np.concatenate([clf.classes_ for clf in v_clfs]))
            ps_folds = []
            for clf in v_clfs:
                p = clf.predict_proba(X_sites)
                if p.shape[1] < len(union_labels):
                    pad_p = np.zeros((n, len(union_labels)))
                    for i, c in enumerate(clf.classes_):
                        pad_p[:, np.where(union_labels == c)[0][0]] = p[:, i]
                    p = pad_p
                ps_folds.append(p)
            site_probs_avg[s], site_classes_dict[s] = np.mean(ps_folds, axis=0), union_labels

        for i in tqdm(range(n), desc="Generating Detailed CSV"):
            raw_p = valid_specs[i]["PEPMASS"]

            db_hits = []
            for c in db_list:
                if adduct_mode == "[M-H]-":
                    target_mz = c["mass"] - h_plus
                elif adduct_mode in ["[M+HCOO]-", "[M+FA-H]-"]:
                    target_mz = c["mass"] + fa_ion
                else:
                    continue

                ppm_err = abs(target_mz - raw_p) / target_mz * 1e6
                if ppm_err <= ppm_tol:
                    db_hits.append(c)

            structs = solve_optimal_distribution(
                pred_sugar_val[i],
                {s: site_probs_avg[s][i] for s in site_cols},
                site_classes_dict,
                pred_type_name[i],
                cfg,
            )

            det_hits = []
            for st in structs:
                st_str = " ".join([f"{s}:{st[s]}" for s in site_cols])
                cands = [
                    c["name"]
                    for c in db_hits
                    if c["type"] == pred_type_name[i].upper()
                    and c["total_sugar"] == pred_sugar_val[i]
                    and all(c[s] == st[s] for s in site_cols)
                ]
                det_hits.append(
                    f"[{st_str} (p={st['prob']:.4f})] => " + (" ; ".join(cands) if cands else "No matching DB saponin")
                )

            results_list.append(
                {
                    "ID": valid_specs[i]["ID"],
                    "PEPMASS": raw_p,
                    "RT": valid_specs[i].get("RT", ""),
                    "Adduct": adduct_mode,
                    "Pred_Type": f"{pred_type_name[i]} (p={prob_t[i]:.3f})",
                    "Pred_Total_Sugar": f"{pred_sugar_val[i]} (p={prob_s[i]:.3f})",
                    "Pred_Sites_Combinations": " | ".join(
                        [
                            f"{' '.join([f'{s}:{st[s]}' for s in site_cols])} (p={st['prob']:.4f})"
                            for st in structs
                        ]
                    ),
                    "Possible_DB_Ginsenosides (Mass Hit Only)": " ; ".join([c["name"] for c in db_hits])
                    if db_hits
                    else "No Mass Match",
                    "Final_DB_Ginsenosides (Detailed Hit)": "  |  ".join(det_hits),
                }
            )

    path_all = os.path.join(cfg["OUTPUT_DIR"], cfg["OUTPUT_CSV_ALL"])
    path_matched = os.path.join(cfg["OUTPUT_DIR"], cfg["OUTPUT_CSV_MATCHED"])

    pd.DataFrame(results_list).to_csv(path_all, index=False, encoding="utf-8-sig")
    df_m = pd.DataFrame(results_list)
    df_m = df_m[~df_m["Final_DB_Ginsenosides (Detailed Hit)"].str.contains("No matching DB saponin", na=False)]
    df_m.to_csv(path_matched, index=False, encoding="utf-8-sig")
    print(f"\n✅ Inference complete! High-precision proton constants and ion-level matching logic applied.")
    print(f"Detailed results: {path_all}\nMatched subset: {path_matched}")
