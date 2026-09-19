# Flood_Project — Pipeline Scripts (split from the original notebook)

This folder splits the original Colab notebook into 7 standalone scripts
plus one shared helper module, so they can be read and run directly from
a GitHub repository without relying on Google Drive.

## File structure

| File | Content | Required raw input | Output |
|---|---|---|---|
| `utils_github_loader.py` | Helpers for downloading raw data, models, and the scaler from GitHub, plus a journal-style plotting helper | — | — |
| `01_data_preparation.py` | Load + clean the representative dataset (STEPs 1-3), feature engineering, flood labeling (Q95), train/valid/test split (STEPs 4-7) | `RepData.zip` | `Results/ML_Training_Input/...` |
| `02_model_training.py` | REFERENCE ONLY — must not be re-run. Baseline training + Optuna tuning + saving the final models | Output of script 01 | Trained `.pkl` models |
| `03_model_evaluation.py` | Evaluation of every model on the test set, feature importance, calibration, threshold optimization | Output of script 01 + models from GitHub | Metrics and plots |
| `04_spatial_transfer_data_preparation.py` | Supplementary extreme-event analysis + cleaning/feature engineering of the transfer data | `spatial_transfer_all.zip` + scaler/models from GitHub | `Results/Spatial_Transfer/Input/...` |
| `05_spatial_transfer_evaluation.py` | Prediction on the spatial-transfer catchments + catchment environmental distance analysis | Output of script 04 + models from GitHub | `Results/Spatial_Transfer/Predictions/...`, `Results/Spatial_Transfer/Environmental_Distance/Gauge_Attribute_Zscore_Heatmap.png` |
| `06_shap_interpretability.py` | SHAP analysis (global + gauge-wise + success vs failure) | Output of script 04 + models from GitHub | Plots, feature rankings + `SHAP_Importance_Difference.png` |
| `07_probability_failure_analysis.py` | Predicted-probability distribution + gauge-wise FP/FN failure analysis | Output of scripts 04/05 + models from GitHub | Tables, plots + `Gauge_Probability_Separation.png` |

## Run order

```
01 -> 02 (optional / reference only) -> 03 -> 04 -> 05 -> 06 -> 07
```

`01_data_preparation.py` is meant to be run in Google Colab and is
internally organized as STEP 1 through STEP 7 (loading and cleaning
the raw representative dataset, then feature engineering, labeling,
and splitting). No manual upload or Google Drive mount is needed -
the raw dataset is downloaded automatically at the start of the
script.

Scripts 03 through 07 download the final trained models directly from
GitHub instead of re-running training (`ensure_models()` /
`ensure_all_trained_models()` in `utils_github_loader.py`). Script 02 is
kept only for reviewing the training procedure and is never called from
the other scripts.

Every script stores data and models under the local folder
`./flood_project_workspace/` (the equivalent of `/content/` in the
original Colab notebook).

## Output folder structure

Every script writes under a single root folder, `./flood_project_workspace/`
(the equivalent of `/content/flood_project_workspace/` when run in Colab).
All outputs are consistently nested under `Results/`:

```
flood_project_workspace/
├── RepData.zip                          (downloaded raw data)
├── spatial_transfer_all.zip             (downloaded raw transfer data)
└── Results/
    ├── Data_Quality/                    (01 - STEP 1)
    ├── Data_Cleaning/                   (01 - STEP 2)
    ├── ML_Preparation/                  (01 - STEP 3)
    ├── ML_Features/                     (01 - STEP 4)
    ├── Flood_Threshold_Analysis/        (01 - STEP 5/6)
    ├── ML_Training_Input/               (01 - STEP 7)  <- read by 03, 04, 05
    ├── ML_Models/
    │   ├── Baseline/                    (02)
    │   ├── Advanced/                    (02)
    │   └── Optuna_Optimization/         (02)  <- downloaded from GitHub by 03-07
    ├── Model_Evaluation/                (03)
    ├── Threshold_Correction/            (03)  <- read by 04, 05, 07
    └── Spatial_Transfer/
        ├── Data_Quality/                (04)
        ├── Data_Cleaning/               (04)
        ├── Preparation/                 (04)
        ├── ML_Features/                 (04)
        ├── Input/                       (04)  <- read by 05, 06, 07
        │   └── Gauge_Wise_Evaluation/   (04)
        ├── Environmental_Distance/      (05)
        ├── PCA_Visualization/           (05)
        ├── Predictions/                 (05)  <- read by 06
        │   └── SHAP_Analysis/           (06)
        ├── Probability_Distribution/    (07)
        └── Failure_Analysis_By_Gauge/   (07)
```



```bash
pip install -r requirements.txt
```

## Configuration (`utils_github_loader.py`)

If the exact file names in your GitHub repository differ from the
defaults below, just edit the `CONFIG` section at the top of
`utils_github_loader.py`:

- `RAW_MAIN_DATA_ZIP = "RepData.zip"`
- `RAW_TRANSFER_DATA_ZIP = "spatial_transfer_all.zip"`
- `MODEL_XGBOOST_FILE = "Best_XGBoost.pkl"`
- `MODEL_CATBOOST_FILE = "Best_CatBoost.pkl"`
- `SCALER_FILE = "StandardScaler.pkl"`
- `ALL_TRAINED_MODELS` — mapping of baseline/advanced model file names to their subfolder (used by script 03)

**Important**: the names `Logistic_Regression.pkl`, `Random_Forest.pkl`,
`Gradient_Boosting.pkl`, `XGBoost_Model.pkl`, `LightGBM_Model.pkl`, and
`CatBoost_Model.pkl` were inferred from the naming used inside the
original training code (script 02). Please make sure these files were
uploaded under the same names at the root of your GitHub repository
before running script 03; otherwise update `ALL_TRAINED_MODELS` in
`utils_github_loader.py` to match the real file names.

## New journal-style figures

Three figures were added to the transfer / failure-analysis scripts,
matching the reference plots supplied for the project, and using
`apply_journal_style()` (larger fonts, bold titles, 300 DPI) for a
publication-ready look:

- `Gauge_Attribute_Zscore_Heatmap.png` (script 05) — absolute Z-score of
  each catchment attribute per gauge, log color scale.
- `SHAP_Importance_Difference.png` (script 06) — SHAP importance
  difference between limited-detection and successful gauges.
- `Gauge_Probability_Separation.png` (script 07) — mean predicted
  probability for flood vs. non-flood samples, per gauge, for the
  optimized XGBoost and CatBoost models.

## Other notes

- Colab-specific parts (`drive.mount`, the `!zip` / `!unzip` / `!cp`
  commands at the end of the original notebook) were removed, since
  they were only used for backing up results to Google Drive and are
  not required to run this pipeline.
- `!pip install ...` magic commands were replaced with the safe
  `pip_install(...)` helper, which works both as a plain script and
  inside Colab/Jupyter.
- `display(...)` falls back to `print(...)` when IPython is not
  available, so the scripts run both in Colab/Jupyter and as plain
  Python scripts.
