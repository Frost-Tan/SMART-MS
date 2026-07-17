"""Model and saponin database loading."""
from __future__ import annotations

import os
from typing import Any, Dict, List

import joblib
import pandas as pd


def load_models(model_dir: str, cfg: Dict[str, Any]):
    site_cols: List[str] = list(cfg["SITE_COLS"])
    outer_folds = int(cfg["OUTER_FOLDS"])
    print(f"Loading LightGBM models from {model_dir} ...")
    le_type = joblib.load(os.path.join(model_dir, "le_type.joblib"))
    type_models = []
    sugar_models = []
    site_models: Dict[str, list] = {s: [] for s in site_cols}
    for fold in range(outer_folds):
        type_models.append(joblib.load(os.path.join(model_dir, f"model_type_fold{fold}.joblib")))
        sugar_models.append(
            joblib.load(os.path.join(model_dir, f"model_total_sugar_fold{fold}.joblib"))
        )
        for s in site_cols:
            mp = os.path.join(model_dir, f"model_site_{s}_fold{fold}.joblib")
            site_models[s].append(joblib.load(mp) if os.path.exists(mp) else None)
    return le_type, type_models, sugar_models, site_models


def load_database(db_file: str) -> List[Dict[str, Any]]:
    print(f"Loading database from {db_file} ...")
    try:
        df = pd.read_csv(db_file, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(db_file, encoding="gb18030")

    db_list: List[Dict[str, Any]] = []
    mapping = {"C3": "C3", "C6": "C6", "C20": "C20", "C26": "C26", "C28": "C28", "Other": "Other_sugar"}
    for _, row in df.iterrows():
        cand: Dict[str, Any] = {
            "name": row.get("id", "Unknown"),
            "mass": 0.0,
            "type": str(row["type"]).strip().upper() if pd.notna(row.get("type")) else "UNKNOWN",
        }
        try:
            cand["mass"] = float(row["exact_mass"])
        except (ValueError, TypeError, KeyError):
            cand["mass"] = 0.0

        total_sugar = 0
        for db_col, code_col in mapping.items():
            val, parsed_val = row.get(db_col, 0), 0
            if pd.notna(val) and str(val).strip() != "":
                val_str = str(val).strip()
                if db_col == "Other":
                    try:
                        parsed_val = int(float(val_str))
                    except (ValueError, TypeError):
                        for p in val_str.replace(";", ",").split(","):
                            if ":" in p:
                                try:
                                    parsed_val += int(float(p.split(":")[1]))
                                except (ValueError, TypeError, IndexError):
                                    pass
                else:
                    try:
                        parsed_val = int(float(val_str))
                    except (ValueError, TypeError):
                        parsed_val = 0
            cand[code_col] = parsed_val
            total_sugar += parsed_val
        cand["total_sugar"] = total_sugar
        db_list.append(cand)
    return db_list
