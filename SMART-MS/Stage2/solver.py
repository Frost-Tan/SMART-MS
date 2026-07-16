"""Strategy 3: Joint decoding of site-specific sugar distribution."""
from __future__ import annotations

import itertools
from typing import Any, Dict, Tuple

import numpy as np


def solve_optimal_sugar_distribution(
    total_sugar: Any,
    site_probs_dict: Dict[str, np.ndarray],
    site_classes_dict: Dict[str, np.ndarray],
    pred_type: Any,
) -> Tuple[int, ...]:
    sites = ["C3", "C6", "C20", "C26", "C28", "Other_sugar"]
    if str(pred_type).strip().upper() == "OTHER":
        return tuple([0] * len(sites))
    val_prob_lists = [
        [(val, prob) for val, prob in zip(site_classes_dict[s], site_probs_dict[s])]
        for s in sites
    ]

    best_combo = tuple([0] * len(sites))
    max_prob = -1.0
    for combo_items in itertools.product(*val_prob_lists):
        vals = [item[0] for item in combo_items]
        if sum(vals) == total_sugar:
            joint_prob = float(np.prod([item[1] for item in combo_items]))
            if pred_type == "PPD" and vals[1] > 0:
                joint_prob *= 0.0001
            if joint_prob > max_prob:
                max_prob = joint_prob
                best_combo = tuple(vals)
    return best_combo
