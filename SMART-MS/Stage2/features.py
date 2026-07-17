"""Mass spectrum parsing and feature vector construction."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def parse_spectrum(spectrum_str: Any) -> Tuple[List[float], List[float]]:
    mz_list: List[float] = []
    int_list: List[float] = []
    if pd.isna(spectrum_str) or spectrum_str == "":
        return mz_list, int_list
    pairs = str(spectrum_str).strip().split()
    for pair in pairs:
        if ":" in pair:
            try:
                mz, intensity = pair.split(":")
                mz_list.append(float(mz))
                int_list.append(float(intensity))
            except ValueError:
                pass
    return mz_list, int_list


def extract_advanced_features(
    mz_list: List[float],
    int_list: List[float],
    pmass: float,
    cfg: Dict[str, Any],
) -> np.ndarray:
    max_mz = float(cfg["MAX_MZ"])
    bin_size = float(cfg["BIN_SIZE"])
    loss_bin_size = float(cfg["LOSS_BIN_SIZE"])
    max_loss = float(cfg["MAX_LOSS"])

    n_bins_spec = int(max_mz / bin_size) + 1
    n_bins_loss = int(max_loss / loss_bin_size) + 1
    vec_spec = np.zeros(n_bins_spec, dtype=np.float32)
    vec_loss = np.zeros(n_bins_loss, dtype=np.float32)
    max_int_mz, weighted_mean_mz, n_peaks = 0.0, 0.0, 0
    max_int_aglycone, max_int_sugar = 0.0, 0.0

    if mz_list and int_list:
        int_array = np.array(int_list)
        mz_array = np.array(mz_list)
        if np.max(int_array) > 0:
            int_array = int_array / np.max(int_array)
        n_peaks = len(mz_array)
        max_int_idx = int(np.argmax(int_array))
        max_int_mz = float(mz_array[max_int_idx])
        weighted_mean_mz = float(np.sum(mz_array * int_array) / np.sum(int_array))

        for mz, intensity in zip(mz_array, int_array):
            if mz < max_mz and mz >= 0:
                bi = int(mz / bin_size)
                vec_spec[bi] = max(vec_spec[bi], float(intensity))
            if pmass > mz:
                loss = pmass - mz
                if loss < max_loss and loss >= 0:
                    li = int(loss / loss_bin_size)
                    vec_loss[li] = max(vec_loss[li], float(intensity))
            if 400 <= mz <= 500:
                max_int_aglycone = max(max_int_aglycone, float(intensity))
            if 130 <= mz <= 180:
                max_int_sugar = max(max_int_sugar, float(intensity))

    scalars = [
        float(pmass),
        float(max_int_mz),
        float(weighted_mean_mz),
        float(n_peaks),
        float(max_int_aglycone),
        float(max_int_sugar),
    ]
    return np.concatenate([vec_spec, vec_loss, np.array(scalars, dtype=np.float32)])


def precursor_mass_for_features(row: pd.Series) -> float:
    pmass = float(row.get("precursor_mz", 0.0))
    adduct_type = str(row.get("adduct", "")).strip().upper()
    if "[M+HCOO]-" in adduct_type or "[M+FA-H]-" in adduct_type:
        pmass -= 46.005
    return pmass


def build_feature_names(cfg: Dict[str, Any]) -> list[str]:
    max_mz = float(cfg["MAX_MZ"])
    bin_size = float(cfg["BIN_SIZE"])
    max_loss = float(cfg["MAX_LOSS"])
    loss_bin_size = float(cfg["LOSS_BIN_SIZE"])
    n_spec = int(max_mz / bin_size) + 1
    n_loss = int(max_loss / loss_bin_size) + 1
    names = [f"Spec_Bin_{i}" for i in range(n_spec)]
    names += [f"Loss_Bin_{i}" for i in range(n_loss)]
    names += [
        "Precursor_MZ",
        "Max_Int_MZ",
        "Weighted_MZ",
        "Peak_Count",
        "Aglycone_Region",
        "Sugar_Region",
    ]
    return names
