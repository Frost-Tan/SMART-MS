"""Site-specific sugar distribution combinatorial optimization (consistent with the original Stage2_inference script)."""
from __future__ import annotations

import itertools
from typing import Any, Dict, List

import numpy as np


def solve_optimal_distribution(
    total_sugar: Any,
    site_probs_dict: Dict[str, np.ndarray],
    site_classes_dict: Dict[str, np.ndarray],
    pred_type: Any,
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    site_cols: List[str] = list(cfg["SITE_COLS"])
    top_k = int(cfg["TOP_K_RESULTS"])
    prob_ratio = float(cfg["PROB_RATIO_THRESHOLD"])

    if str(pred_type).strip().upper() == "OTHER":
        return [{"prob": 1.0, **{s: 0 for s in site_cols}}]

    val_prob_lists = [
        [(v, p) for v, p in zip(site_classes_dict[s], site_probs_dict[s])] for s in site_cols
    ]
    valid_combos: List[Dict[str, Any]] = []
    for combo in itertools.product(*val_prob_lists):
        vals = [item[0] for item in combo]
        if sum(vals) == total_sugar:
            jp = np.prod([item[1] for item in combo])
            if pred_type == "PPD" and vals[1] > 0:
                jp *= 0.0001
            valid_combos.append({"prob": float(jp), **{site_cols[i]: vals[i] for i in range(len(site_cols))}})

    valid_combos.sort(key=lambda x: x["prob"], reverse=True)
    if not valid_combos:
        return []
    return [valid_combos[0]] + [
        c
        for c in valid_combos[1:top_k]
        if c["prob"] >= valid_combos[0]["prob"] * prob_ratio
    ]
