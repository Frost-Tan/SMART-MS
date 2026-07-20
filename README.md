# SMART-MS — Saponin Mass Spectrometry Analysis Toolkit

SMART-MS is a LightGBM-based saponin identification and structural elucidation tool that employs a two-stage cascaded strategy for end-to-end analysis from raw mass spectrometry data to saponin structural annotation.

---

## Project Overview

This project targets saponin mass spectrometry data in negative ion mode and implements a complete identification workflow:

| Stage | Module | Function |
|------|------|------|
| **Stage1** | `Stage1/` | Binary classification training — determine whether a compound is a saponin |
| **Stage1 Inference** | `Stage1_inference/` | Screen MGF spectra for saponin candidates using Stage1 model |
| **Stage2** | `Stage2/` | Cascaded model training — type classification → total sugar prediction → site-specific sugar prediction |
| **Stage2 Inference** | `Stage2_inference/` | Complete structural elucidation and database matching using Stage2 model |

### Stage1 — Binary Screening
- Input: MGF format mass spectrometry data
- Model: LightGBM binary classifier (5-fold cross-validation ensemble)
- Output: Saponin probabilities for both [M-H]⁻ and [M+HCOO]⁻ adduct modes

### Stage2 — Cascaded Structural Elucidation (Three Ablation Strategies)
- **Strategy1_Independent**: Independent prediction of type, total sugar, and site sugars
- **Strategy2_Cascaded_NoConstraint**: Type → total sugar → site cascaded prediction, no constrained decoding
- **Strategy3_Full_Pipeline** (default): Type → total sugar → site cascaded prediction + total sugar constrained decoding (recommended)
- Cascade structure: Previous stage predictions serve as features for the next stage
- Inference combines with a custom saponin database for matching and scoring

---

## Project Structure

```
<repo_root>/
├── dataset/                      # Input data directory
│   ├── FINAL_task1_neg.csv       # Stage1 training data
│   ├── FINAL_task2_neg.csv       # Stage2 training data
│   ├── merged_saponin_named.csv  # Saponin structure database (for Stage2 inference)
│   └── 46.mgf                    # Example input spectrum file (replace with your own MGF)
├── model/                        # Pretrained models (download from GitHub Release, see "Model Download" below)
│   ├── Stage1_saved_models/      # Stage1 model files
│   └── Stage2_saved_models/      # Stage2 model files
├── SMART-MS/                     # Source code directory
│   ├── Stage1/                   # Stage1 Training Module
│   │   ├── main.py               # Training entry point
│   │   ├── config.py             # Configuration (paths, hyperparameters)
│   │   ├── features.py           # Feature engineering
│   │   ├── training.py           # Training logic
│   │   ├── plots.py              # Visualization
│   ├── Stage1_inference/         # Stage1 Inference Module
│   │   ├── main.py               # Inference entry point
│   │   ├── config.py             # Configuration
│   │   ├── feature_extract.py    # Feature extraction
│   │   ├── inference_runner.py   # Inference runner
│   │   ├── mgf_parser.py         # MGF file parser
│   ├── Stage2/                   # Stage2 Training Module
│   │   ├── main.py               # Training entry point
│   │   ├── config.py             # Configuration (with three ablation strategies)
│   │   ├── features.py           # Feature engineering
│   │   ├── solver.py             # Decoder / solver
│   │   ├── training.py           # Training logic
│   │   ├── plots.py              # Visualization
│   └── Stage2_inference/         # Stage2 Inference Module
│       ├── main.py               # Inference entry point
│       ├── config.py             # Configuration
│       ├── distribution_solver.py # Sugar distribution solver
│       ├── feature_extract.py    # Feature extraction
│       ├── inference_runner.py   # Inference runner
│       ├── loaders.py            # Model loader
│       ├── mgf_parser.py         # MGF file parser
├── README.md                     # Project README (this file)
├── environment.yml               # General environment config (recommended)
├── environment_original.yml      # Original experiment environment config
└── Command_Reference.txt         # Complete command line reference
```

---

## Environment Setup

This project provides two conda environment configuration files for different use cases:

| Configuration File | Description | Use Case |
|--------------------|-------------|----------|
| `environment.yml` | **General config (recommended)** | Suitable for most computers, uses relaxed version constraints for better compatibility |
| `environment_original.yml` | **Original experiment config** | Precisely reproduce the Python environment and dependency versions from the original experiment (requires Windows + Python 3.10) |

### Method 1: Use general config (recommended)

```bash
# Create conda environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate smartms
```

### Method 2: Use original experiment config

```bash
# Create conda environment from environment_original.yml (exact reproduction of original experiment)
conda env create -f environment_original.yml

# Activate the environment
conda activate smartms
```

### Method 3: Manual installation

```bash
conda create -n smartms python=3.10
conda activate smartms
pip install pandas numpy lightgbm scikit-learn matplotlib seaborn joblib tqdm
```

### Dependency Overview

| Package | Original Version | Purpose |
|---------|-----------------|---------|
| `pandas` | 2.3.2 | Data processing and CSV I/O |
| `numpy` | 2.2.6 | Numerical computing |
| `lightgbm` | 4.6.0 | Model training and inference engine |
| `scikit-learn` | 1.7.2 | Cross-validation, evaluation metrics |
| `matplotlib` | 3.10.6 | Data visualization (with `seaborn` 0.13.2) |
| `joblib` | 1.5.2 | Model persistence |
| `tqdm` | 4.67.1 | Progress bar display |

---

## Model Download

Pretrained model files are published separately on GitHub Release due to their large size.

**Download URL**: https://github.com/Frost-Tan/SMART-MS/releases/tag/v1_model

Download `model.zip` and extract it to the repository root directory, ensuring the `model/` folder is located at `<repo_root>/model/` with the following structure:

```
<repo_root>/
└── model/
    ├── Stage1_saved_models/
    └── Stage2_saved_models/
```

> **Note**: If you choose to train models yourself instead of using the pretrained ones, the training scripts will automatically generate model files under `SMART-MS/Stage1/output/` and `SMART-MS/Stage2/output/`, so manual download is not required.

---

## Quick Start

### 1. Environment Preparation

```bash
conda activate smartms
cd /d <repo_root>
```

### 2. Stage1 Training (Binary Classification Model)

```bash
python SMART-MS\Stage1\main.py
```

Reads `dataset\FINAL_task1_neg.csv` by default; outputs to `SMART-MS\Stage1\output\`.

### 3. Stage2 Training (Cascaded Model)

```bash
# Default: run Strategy 3 only (recommended)
python SMART-MS\Stage2\main.py

# Run all three ablation strategies
python SMART-MS\Stage2\main.py --all-ablations
```

### 4. Stage1 Inference (Saponin Screening)

```bash
python SMART-MS\Stage1_inference\main.py --mgf dataset\your_file.mgf
```

### 5. Stage2 Inference (Structural Elucidation)

```bash
python SMART-MS\Stage2_inference\main.py --mgf dataset\your_file.mgf --db dataset\your_db.csv
```

---

## Detailed Command Reference

Please refer to: [Command_Reference.txt](Command_Reference.txt)

The document consolidates complete command line instructions for all four modules, including:
- All optional parameters and their default values
- CMD / PowerShell usage examples
- Input and output file descriptions
- Typical workflow

---

## Output File Description

### Stage1 Training Output (`Stage1/output/`)

| File | Description |
|------|-------------|
| `ROC_Curve_MH_Baseline.png` | ROC curve plot |
| `Feature_Importance_MH_Baseline.png` | Feature importance plot |
| `Saponin_MH_Baseline_Report.txt` | Training evaluation report |
| `lgbm_mh_baseline_fold_*.joblib` | 5-fold cross-validation model files |

### Stage1 Inference Output (`Stage1_inference/output/`)

| File | Description |
|------|-------------|
| `Task1_inference_H_hypothesis_only.csv` | [M-H]⁻ adduct inference results |
| `Task1_inference_FA_hypothesis_only.csv` | [M+HCOO]⁻ adduct inference results |
| `Task1_inference_Best_Match_Final.csv` | Best match consolidated results |

### Stage2 Training Output (`Stage2/output/Strategy3_Full_Pipeline/`)

| File | Description |
|------|-------------|
| `plot_confusion_matrix_type.png` | Type classification confusion matrix |
| `plot_overall_metrics.png` | Overall evaluation metrics plot |
| `model_performance_report.txt` | Model performance report |
| `cv_prediction_detailed.csv` | Cross-validation detailed predictions |
| `cv_metrics_summary.csv` | Metrics summary |
| `le_type.joblib` | Type label encoder |
| `model_type_fold*.joblib` | Type classification models (per fold) |
| `model_total_sugar_fold*.joblib` | Total sugar regression models (per fold) |
| `model_site_*_fold*.joblib` | Site-specific sugar regression models (per fold) |
| `feature_importance_stage*.png` | Feature importance per cascade stage |
| `cascaded_importance_evolution.png` | Cascaded feature importance evolution |

### Stage2 Inference Output (`Stage2_inference/output/`)

| File | Description |
|------|-------------|
| `Task2_inference_results_detailed_final.csv` | Complete structural elucidation results |
| `Task2_inference_results_matched_only.csv` | Database-matched subset |

---

## Model Architecture

### Stage1 Model
- **Algorithm**: LightGBM binary classification
- **Features**: Mass spectrum peak binning features + neutral loss features
- **Ensemble**: 5-fold cross-validation model averaging
- **Adduct modes**: [M-H]⁻, [M+HCOO]⁻ (auto-detection)

### Stage2 Model (Cascaded Architecture)
- **Stage I**: Saponin type multi-class classification (LightGBM)
- **Stage II**: Total sugar count regression (LightGBM, input includes Stage I prediction probabilities)
- **Stage III**: Site-specific (C3/C6/C20/C26/C28/Other) sugar count regression (independent LightGBM per site, input includes Stage I + Stage II predictions)
- **Decoding**: Strategy3 uses constrained optimization to match sugar distributions in the database
- **Ensemble**: 10-fold outer CV × 10-fold inner OOF

---

## Parameter Tuning

| Parameter | Stage1 | Stage2 | Description |
|-----------|--------|--------|-------------|
| `--cv-folds` / `--outer-folds` | 5 | 10 | Cross-validation folds |
| `--inner-folds` | - | 10 | Inner OOF folds |
| `--seed` | 42 | 42 | Random seed |
| `--device` | - | cpu/gpu | Compute device (Stage2 supports GPU) |

---

## Technical Requirements

- **Python**: 3.10 (original environment) / 3.9~3.11 (general environment)
- **Operating System**: Windows / Linux / macOS
- **Memory**: ≥ 16 GB recommended
- **GPU (optional)**: Stage2 training supports LightGBM GPU acceleration

---

## Citation

For citation, please contact the authors.

---

## License

This project is for research use only.