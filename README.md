# Flood Prediction and Spatial Transferability — Code

This repository contains the code for a study on regional flood-event prediction using gradient-boosting models (XGBoost, CatBoost) trained on CAMELS-US catchments, with a particular focus on how well those models generalize to catchments outside the training set. The paper is currently unpublished; this README covers only how to run the code.

## Repository layout

```
Flood_Project/
├── catchment.py                          run once, manually, in Colab
├── utils_github_loader.py                shared download helpers — every script below imports this
├── 01_data_preparation.py
├── 02_model_training.py                  reference only, see note below
├── 03_model_evaluation.py
├── 04_spatial_transfer_data_preparation.py
├── 05_spatial_transfer_evaluation.py
├── 06_shap_interpretability.py
├── 07_probability_failure_analysis.py
├── requirements.txt
│
├── RepData.zip                           daily records for the 40 representative catchments
├── spatial_transfer_all.zip              daily records for the 7 transfer catchments
├── Best_XGBoost.pkl, Best_CatBoost.pkl    optimized models (output of script 02)
├── Logistic_Regression.pkl, Random_Forest.pkl (or .zip, see below), Gradient_Boosting.pkl,
│   XGBoost_Model.pkl, LightGBM_Model.pkl, CatBoost_Model.pkl   baseline/advanced models
└── StandardScaler.pkl                    fitted scaler (output of script 01)
```

`catchment.py` is a separate, one-time step: it takes the raw CAMELS-US attribute tables and decides which gauge IDs belong to the 40-catchment representative set and which belong to the 7-catchment spatial-transfer set. It doesn't touch the daily forcing/streamflow records themselves and isn't part of the pipeline below — `RepData.zip` and `spatial_transfer_all.zip` (the actual daily records for those selected catchments) are prepared separately and are what scripts `01` and `04` consume.

Everything else — `01` through `07` — is a self-contained pipeline. Each script downloads whatever raw data, scaler, or trained model it needs directly from this repository the first time it runs, so nothing has to be fetched or placed by hand beyond the scripts themselves.

## Running everything in Colab

**Step 0 — catchment selection (only if you're redoing it).** Open a Colab notebook, paste the full contents of `catchment.py` into a cell, and run it. If you're just re-running the modeling pipeline against the existing 40+7 catchments, skip this — `RepData.zip` and `spatial_transfer_all.zip` are already in the repository.

**Step 1 — get the pipeline scripts into Colab.** Either clone the whole repository:

```python
!git clone https://github.com/saeedbalahang-collab/Flood_Project.git
%cd Flood_Project
```

or, if you'd rather not pull down the large data/model files at clone time, just upload the eight `.py` files (the seven numbered scripts plus `utils_github_loader.py`) through Colab's file panel. Either way, data and models are downloaded on demand by the scripts themselves.

**Step 2 — install what Colab doesn't already have:**

```python
!pip install -q -r requirements.txt
```

**Step 3 — run the scripts in order, with `%run`, not `!python`:**

```python
%run 01_data_preparation.py
%run 03_model_evaluation.py
%run 04_spatial_transfer_data_preparation.py
%run 05_spatial_transfer_evaluation.py
%run 06_shap_interpretability.py
%run 07_probability_failure_analysis.py
```

`%run` executes inside the notebook's own kernel, so `display()` calls render properly as tables; `!python file.py` spawns a separate process and you'd only see the `print()` output.

Script `02` is intentionally left out of that sequence. It retrains all five models from scratch and re-runs the Optuna hyperparameter search — none of which is needed to reproduce the results, since the trained models it would produce are already in the repository and get downloaded automatically by the other scripts. Run it only if you want to inspect or re-verify the training procedure itself.

Once everything has run, results live under `./flood_project_workspace/Results/` inside the Colab VM. To pull them back down as a single archive:

```python
!zip -r Results.zip flood_project_workspace/Results
from google.colab import files
files.download("Results.zip")
```

Scripts `01`, `03`, `04`, and `05` typically finish in a couple of minutes each; `06` and parts of `07` take longer because SHAP is computed sample-by-sample over the full evaluation set.

## What ends up where

Every script writes into the same local root, `./flood_project_workspace/` (i.e. `/content/flood_project_workspace/` inside Colab), and everything is nested under `Results/`:

```
flood_project_workspace/Results/
├── Data_Quality/, Data_Cleaning/, ML_Preparation/, ML_Features/,
│   Flood_Threshold_Analysis/, ML_Training_Input/        (01)
├── ML_Models/Baseline/ · Advanced/ · Optuna_Optimization/  (02)
├── Model_Evaluation/, Threshold_Correction/,
│   Extreme_Event_Analysis/                                (03)
└── Spatial_Transfer/
    ├── Data_Quality/, Data_Cleaning/, Preparation/, ML_Features/,
    │   Input/ (+ Gauge_Wise_Evaluation/)                   (04)
    ├── Environmental_Distance/, PCA_Visualization/,
    │   Predictions/ (+ SHAP_Analysis/)                     (05, 06)
    └── Probability_Distribution/, Failure_Analysis_By_Gauge/ (07)
```


