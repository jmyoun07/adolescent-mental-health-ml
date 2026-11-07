# Adolescent Mental Health Risk Prediction

This repository contains reproducible analysis scripts for developing and evaluating machine-learning classifiers for adolescent depressed mood and suicidality using national school-based survey data.

## Study Scope

The scripts support two analysis tracks:

- Domestic model development and temporal validation using Korea Youth Risk Behavior Survey (KYRBS) data.
- Shared-feature external validation using the US Youth Risk Behavior Surveillance System (YRBSS) data.

The target outcomes are:

- Depressed mood
- Suicidal ideation
- Suicide plan
- Suicide attempt

## Repository Contents

- `analysis/`: Data preparation, model training, external validation, bootstrap uncertainty estimation, calibration, and sensitivity-analysis scripts.
- `metadata/`: Non-identifiable variable-selection tables, YRBSS column metadata, and harmonized feature documentation used by the analysis scripts.
- `data/`: Placeholder directory for local data files. Raw and processed survey data are not included.
- `outputs/`: Placeholder directory for locally generated model outputs. Results are not included.
- `requirements.txt`: Python package versions used for the reproducible analysis environment.

## Data Availability

Raw KYRBS and YRBSS data are not distributed in this repository. Users should obtain the datasets from the official data providers and place them under the expected local paths before running the scripts.

Expected local data layout:

```text
data/
  raw_sav/
    kyrbs2020.sav
    kyrbs2021.sav
    kyrbs2022.sav
    kyrbs2023.sav
    kyrbs2024.sav
  external_yrbss/
    2023_national/
      yrbs2023_national_raw.csv
      yrbs2023_national_columns.csv
  processed/
```

## Environment

Create a Python environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Main Workflow

Run scripts from the repository root.

```bash
python analysis/make_reduced_40_dataset.py
python analysis/run_lit_reduced_revised_ml.py
python analysis/run_yrbss_revised_external_validation.py
python analysis/bootstrap_ci_recalibration.py
python analysis/compute_tree_shap_contributions.py
```

Additional sensitivity analyses:

```bash
python analysis/run_kyrbs_no_depressed_sensitivity.py
python analysis/run_kyrbs_behavior_only_sensitivity.py
python analysis/run_yrbss_external_sensitivity.py
python analysis/run_lit_reduced_revised_sensitivity.py
```

## Notes

- Generated files should be written to `outputs/`.
- Raw data, processed analytic datasets, trained model objects, and large output files are intentionally excluded from version control.
- File paths in the scripts are relative to the repository root.
