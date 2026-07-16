"""MGF parsing: TITLE field and spectrum block reading."""


def parse_mgf_title(title_str):
    """Precisely parse the MGF TITLE field."""
    parsed = {"ID": "N/A", "TITLE_MZ": "N/A", "TITLE_RT": "N/A", "PEAKID": "N/A", "ISOTOPE": "N/A"}
    if not isinstance(title_str, str) or not title_str:
        return parsed

    parts = title_str.split("|")
    for part in parts:
        if "=" in part:
            k, v = part.split("=", 1)
            k = k.strip().upper()
            v = v.strip()
            if k == "ID":
                parsed["ID"] = v
            elif k == "MZ":
                parsed["TITLE_MZ"] = v
            elif k == "RT":
                parsed["TITLE_RT"] = v
            elif k == "PEAKID":
                parsed["PEAKID"] = v
            elif k == "ISOTOPE":
                parsed["ISOTOPE"] = v
    return parsed


def parse_mgf(mgf_path):
    spectra = []
    current_spectrum, peaks_mz, peaks_int = {}, [], []
    with open(mgf_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == "BEGIN IONS":
                current_spectrum, peaks_mz, peaks_int = {}, [], []
            elif line == "END IONS":
                current_spectrum["mz_list"] = peaks_mz
                current_spectrum["int_list"] = peaks_int
                spectra.append(current_spectrum)
            elif "=" in line:
                key, val = line.split("=", 1)
                current_spectrum[key.upper()] = val
            elif line[0].isdigit():
                parts = line.split()
                if len(parts) >= 2:
                    peaks_mz.append(float(parts[0]))
                    peaks_int.append(float(parts[1]))
    return spectra
