"""Load models, extract features, and write three inference result CSVs."""
from __future__ import annotations

import os
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from feature_extract import extract_features_vectorized, get_mh_baseline_mass_hypothesis
from mgf_parser import parse_mgf, parse_mgf_title


def run_inference(cfg: Dict[str, Any]) -> None:
    os.makedirs(cfg["OUTPUT_DIR"], exist_ok=True)

    model_files = [f for f in os.listdir(cfg["MODELS_DIR"]) if f.endswith(".joblib")]
    models = [joblib.load(os.path.join(cfg["MODELS_DIR"], mf)) for mf in model_files]
    print(f">>> Loaded {len(models)} ensemble models.")

    raw_spectra = parse_mgf(cfg["INPUT_MGF"])
    X_list_H, X_list_FA, meta_list = [], [], []

    print(f">>> Extracting dual-pathway features and parsing MGF info (total {len(raw_spectra)} spectra) ...")
    for idx, spec in tqdm(enumerate(raw_spectra), total=len(raw_spectra)):
        title_info = parse_mgf_title(spec.get("TITLE", ""))

        pep_str = spec.get("PEPMASS", title_info["TITLE_MZ"])
        try:
            pepmass = float(pep_str.split()[0]) if pep_str != "N/A" else 0.0
        except (ValueError, TypeError, AttributeError):
            pepmass = 0.0

        rt_raw = spec.get("RTINMINUTES", spec.get("RT", title_info["TITLE_RT"]))
        if rt_raw == "N/A":
            sec_rt = spec.get("RTINSECONDS", "N/A")
            if sec_rt != "N/A":
                try:
                    rt_raw = str(float(sec_rt) / 60.0)
                except (ValueError, TypeError):
                    pass

        try:
            rt_formatted = f"{float(rt_raw):.2f}"
        except (ValueError, TypeError):
            rt_formatted = "N/A"

        baseline_H = get_mh_baseline_mass_hypothesis(pepmass, "H")
        X_list_H.append(
            extract_features_vectorized(spec["mz_list"], spec["int_list"], pepmass, baseline_H, cfg)
        )

        baseline_FA = get_mh_baseline_mass_hypothesis(pepmass, "FA")
        X_list_FA.append(
            extract_features_vectorized(spec["mz_list"], spec["int_list"], pepmass, baseline_FA, cfg)
        )

        meta_list.append(
            {
                "ID": title_info["ID"] if title_info["ID"] != "N/A" else f"S_{idx+1}",
                "PEPMASS_MZ": f"{pepmass:.4f}",
                "RT_Min": rt_formatted,
                "Isotope": title_info["ISOTOPE"],
                "Peak_Count": len(spec["mz_list"]),
            }
        )

    X_batch_H = np.vstack(X_list_H)
    X_batch_FA = np.vstack(X_list_FA)

    print(">>> Running batch matrix inference ...")
    probs_H = np.mean([m.predict_proba(X_batch_H)[:, 1] for m in models], axis=0)
    probs_FA = np.mean([m.predict_proba(X_batch_FA)[:, 1] for m in models], axis=0)

    rows_H: List[Dict[str, Any]] = []
    rows_FA: List[Dict[str, Any]] = []
    rows_Best: List[Dict[str, Any]] = []

    for i, meta in enumerate(meta_list):
        pH, pFA = probs_H[i], probs_FA[i]

        row_h = meta.copy()
        row_h.update(
            {"Prediction_H": "Saponin (+)" if pH >= 0.5 else "Non-Saponin (-)", "Conf_H": f"{pH:.2%}"}
        )
        rows_H.append(row_h)

        row_fa = meta.copy()
        row_fa.update(
            {
                "Prediction_FA": "Saponin (+)" if pFA >= 0.5 else "Non-Saponin (-)",
                "Conf_FA": f"{pFA:.2%}",
            }
        )
        rows_FA.append(row_fa)

        row_best = meta.copy()
        best_p = max(pH, pFA)
        row_best.update(
            {
                "Best_Adduct": "[M-H]-" if pH >= pFA else "[M+HCOO]-",
                "Final_Prediction": "Saponin (+)" if best_p >= 0.5 else "Non-Saponin (-)",
                "Final_Conf": f"{best_p:.2%}",
                "Status": "High Conf" if (best_p > 0.9 or best_p < 0.1) else "Review Needed",
                "Score_H": f"{pH:.4f}",
                "Score_FA": f"{pFA:.4f}",
            }
        )
        row_best = {
            k: row_best[k]
            for k in [
                "ID",
                "Final_Prediction",
                "Final_Conf",
                "Status",
                "Best_Adduct",
                "PEPMASS_MZ",
                "RT_Min",
                "Score_H",
                "Score_FA",
                "Isotope",
                "Peak_Count",
            ]
        }
        rows_Best.append(row_best)

    path_h = os.path.join(cfg["OUTPUT_DIR"], cfg["OUT_H_ONLY"])
    path_fa = os.path.join(cfg["OUTPUT_DIR"], cfg["OUT_FA_ONLY"])
    path_best = os.path.join(cfg["OUTPUT_DIR"], cfg["OUT_BEST_MATCH"])

    pd.DataFrame(rows_H).to_csv(path_h, index=False)
    pd.DataFrame(rows_FA).to_csv(path_fa, index=False)
    pd.DataFrame(rows_Best).to_csv(path_best, index=False)

    print(
        f"\n✅ Inference complete! Three parallel result files generated:\n1. {path_h}\n2. {path_fa}\n3. {path_best}"
    )
