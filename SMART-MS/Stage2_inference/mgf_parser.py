"""MGF parsing (consistent with the original Stage2_inference script logic)."""
from __future__ import annotations

import re
from typing import Any, Dict, List

import numpy as np


def parse_mgf(file_path: str, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    min_pmass = float(cfg["MIN_PMASS"])
    min_peaks = int(cfg["MIN_PEAKS"])
    spectra: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        spec: Dict[str, Any]
        mz_list: List[float]
        int_list: List[float]
        spec, mz_list, int_list = {}, [], []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == "BEGIN IONS":
                spec, mz_list, int_list = {}, [], []
            elif line.startswith("TITLE="):
                t_content = line.split("=", 1)[1]
                id_m = re.search(r"ID=([\w.-]+)", t_content)
                if id_m:
                    spec["ID"] = id_m.group(1)
                rt_m = re.search(r"RT=([\d.]+)", t_content)
                if rt_m:
                    spec["RT"] = rt_m.group(1)
            elif line.startswith("PEPMASS="):
                spec["PEPMASS"] = float(line.split("=", 1)[1].split()[0])
            elif line.startswith("RTINSECONDS="):
                spec["RT"] = float(line.split("=", 1)[1])
            elif line.startswith("RTINMINUTES="):
                spec["RT"] = float(line.split("=", 1)[1])
            elif line.startswith("SCANS="):
                spec["ID"] = line.split("=", 1)[1]
            elif line == "END IONS":
                spec["mz"] = np.array(mz_list)
                spec["intensity"] = np.array(int_list)
                if spec.get("PEPMASS", 0) > min_pmass and len(mz_list) >= min_peaks:
                    if "ID" not in spec:
                        spec["ID"] = f"RT_{spec.get('RT', 'unknown')}_{spec['PEPMASS']:.4f}"
                    spectra.append(spec)
            elif "=" not in line:
                try:
                    parts = line.split()
                    mz_list.append(float(parts[0]))
                    int_list.append(float(parts[1]))
                except (ValueError, IndexError):
                    pass
    return spectra
