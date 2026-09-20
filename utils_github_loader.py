# -*- coding: utf-8 -*-
"""
utils_github_loader.py
------------------------------------------------------------
Shared helper module for downloading raw data, trained models,
and the fitted scaler from the project's GitHub repository.

All pipeline scripts (01 through 07) import this module so that
paths that used to point to "/content/..." in the original Colab
notebook now resolve automatically inside a local BASE_DIR, and
so that raw data / models / scaler are fetched from GitHub instead
of requiring a manual upload or a Google Drive mount.

If the exact file names in your repository differ from the
defaults below, just edit the CONFIG section; none of the other
scripts need to change.
------------------------------------------------------------
"""

import os
import shutil
import zipfile
import urllib.request

# Safe fallback for display(), which only exists inside Jupyter/Colab
try:
    from IPython.display import display  # noqa: F401
except ImportError:
    def display(x):
        print(x)


# ============================================================
# CONFIG - edit this section to match the actual file names in your repository
# ============================================================

GITHUB_USER = "saeedbalahang-collab"
GITHUB_REPO = "Flood_Project"
GITHUB_BRANCH = "main"

# File names expected at the root of the GitHub repository
RAW_MAIN_DATA_ZIP = "RepData.zip"                   # raw input data for the feature-engineering stage
RAW_TRANSFER_DATA_ZIP = "spatial_transfer_all.zip"  # raw data for the spatial-transfer catchments
MODEL_XGBOOST_FILE = "Best_XGBoost.pkl"
MODEL_CATBOOST_FILE = "Best_CatBoost.pkl"
SCALER_FILE = "StandardScaler.pkl"                  # edit if the scaler was uploaded under a different name

# Mapping of every trained model (produced by script 02) to its local
# subfolder. Edit this if a file name or subfolder on GitHub differs.
# key = file name on GitHub (assumed to be at the repo root)
# value = local subfolder under Results/ML_Models/
ALL_TRAINED_MODELS = {
    "Logistic_Regression.pkl": "Baseline",
    "Random_Forest.pkl": "Baseline",
    "Gradient_Boosting.pkl": "Baseline",
    "XGBoost_Model.pkl": "Advanced",
    "LightGBM_Model.pkl": "Advanced",
    "CatBoost_Model.pkl": "Advanced",
    MODEL_XGBOOST_FILE: "Optuna_Optimization",
    MODEL_CATBOOST_FILE: "Optuna_Optimization",
}

# Some models are too large to upload as a raw .pkl to GitHub and are
# instead uploaded as a .zip (containing that single .pkl file).
# key   = the .pkl file name as it appears in ALL_TRAINED_MODELS above
# value = the .zip file name uploaded at the root of the repository
# Add an entry here for every model you had to zip before uploading.
ZIPPED_MODELS = {
    "Random_Forest.pkl": "Random_Forest.zip",
}

# GitHub's web upload limits a plain repo file to ~25-100 MB. If even
# the zipped model is still too large for that, upload it as a GitHub
# "Release" asset instead (Releases allow files up to 2 GB, no Git LFS
# needed) and put its direct download URL here. This takes priority
# over github_raw_url(...) for that exact file name - everything else
# about ensure_all_trained_models() (including ZIPPED_MODELS above)
# still applies on top of it.
# key   = the file name as used in ZIPPED_MODELS / ALL_TRAINED_MODELS
# value = the full direct-download URL
# Example, after creating a Release with tag "v1-models" and attaching
# Random_Forest.zip to it:
#   MODEL_URL_OVERRIDES = {
#       "Random_Forest.zip": "https://github.com/saeedbalahang-collab/"
#                             "Flood_Project/releases/download/"
#                             "v1-models/Random_Forest.zip",
#   }
MODEL_URL_OVERRIDES = {
}

# All local paths (equivalent to "/content" in the original Colab code)
# are created inside this folder.
BASE_DIR = os.path.abspath("./flood_project_workspace")


# ============================================================
# Base helpers
# ============================================================

def github_raw_url(filename_in_repo_root: str) -> str:
    """
    Convert a file name located at the repository root into a
    downloadable "raw" GitHub URL.
    Example:
        blob: https://github.com/USER/REPO/blob/main/Best_CatBoost.pkl
        raw : https://raw.githubusercontent.com/USER/REPO/main/Best_CatBoost.pkl
    """
    return (
        f"https://raw.githubusercontent.com/"
        f"{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/{filename_in_repo_root}"
    )


def resolve_url(filename: str) -> str:
    """
    Return the URL to download `filename` from: MODEL_URL_OVERRIDES[filename]
    if that file was uploaded somewhere other than the repo root (e.g. a
    GitHub Release asset), otherwise the usual github_raw_url(filename).
    """
    if filename in MODEL_URL_OVERRIDES:
        return MODEL_URL_OVERRIDES[filename]
    return github_raw_url(filename)


def blob_url_to_raw(blob_url: str) -> str:
    """Convert a full GitHub "blob" link into its "raw" equivalent."""
    return blob_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")


def download_file(url: str, dest_path: str, force: bool = False) -> str:
    """Download a single file, skipping it if it was already downloaded."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    if os.path.exists(dest_path) and not force:
        print(f"[OK] Already downloaded: {dest_path}")
        return dest_path

    print(f"[DOWNLOAD] {url}\n        -> {dest_path}")
    urllib.request.urlretrieve(url, dest_path)
    print(f"[DONE] {dest_path}  ({os.path.getsize(dest_path)/1024:.1f} KB)")
    return dest_path


def download_and_extract_zip(url: str, extract_to: str, force: bool = False) -> str:
    """Download a zip file and extract it. Returns the extraction folder path."""
    os.makedirs(extract_to, exist_ok=True)
    zip_path = os.path.join(extract_to, os.path.basename(url))

    if not (os.path.exists(zip_path) and not force):
        download_file(url, zip_path, force=force)

    marker = os.path.join(extract_to, ".extracted")
    if not os.path.exists(marker) or force:
        print(f"[EXTRACT] {zip_path} -> {extract_to}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_to)
        with open(marker, "w") as f:
            f.write("done")
    else:
        print(f"[OK] Already extracted: {extract_to}")

    return extract_to


def find_first_csv(folder: str) -> str:
    """Recursively find the first .csv file inside a folder."""
    for root, _dirs, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".csv"):
                return os.path.join(root, f)
    raise FileNotFoundError(f"No .csv file found under {folder}.")


# ============================================================
# High-level helpers - each script only needs to call these
# ============================================================

def ensure_main_raw_data_zip() -> str:
    """
    Download the main raw representative dataset (RepData.zip) from
    GitHub and place it at BASE_DIR/RepData.zip - i.e. exactly where
    the original notebook's own file-discovery logic (STEP 1) expects
    to find it locally (the equivalent of "/content/RepData.zip").
    The discovery/extraction logic in STEP 1 of script 01 takes it
    from there (pandas can also read a .zip containing a single CSV
    directly, which STEP 3 relies on).
    """
    dest_zip = os.path.join(BASE_DIR, "RepData.zip")
    return download_file(resolve_url(RAW_MAIN_DATA_ZIP), dest_zip)


def ensure_transfer_raw_dataset() -> str:
    """
    Download the raw spatial-transfer dataset (spatial_transfer_all.zip)
    from GitHub and place it exactly where the original code expects it:
    BASE_DIR/spatial_transfer_all.zip
    (the original code extracts this zip itself, so only the download
    is needed here).
    """
    dest_zip = os.path.join(BASE_DIR, RAW_TRANSFER_DATA_ZIP)
    return download_file(resolve_url(RAW_TRANSFER_DATA_ZIP), dest_zip)


def ensure_scaler(target_path: str = None) -> str:
    if target_path is None:
        target_path = os.path.join(BASE_DIR, "Results", "ML_Training_Input", "StandardScaler.pkl")
    return download_file(resolve_url(SCALER_FILE), target_path)


def ensure_models(target_dir: str = None):
    """
    Download Best_XGBoost.pkl and Best_CatBoost.pkl from GitHub.
    The default target directory is exactly where the original code
    calls joblib.load(...) from.
    Returns: (xgboost_path, catboost_path)
    """
    if target_dir is None:
        target_dir = os.path.join(BASE_DIR, "Results", "ML_Models", "Optuna_Optimization")

    xgb_path = os.path.join(target_dir, MODEL_XGBOOST_FILE)
    cat_path = os.path.join(target_dir, MODEL_CATBOOST_FILE)

    download_file(resolve_url(MODEL_XGBOOST_FILE), xgb_path)
    download_file(resolve_url(MODEL_CATBOOST_FILE), cat_path)

    return xgb_path, cat_path


def ensure_all_trained_models(base_dir: str = None):
    """
    Download every trained model (baseline + advanced + optimized) from
    GitHub and place each one exactly where the evaluation code (script
    03) expects it:
    BASE_DIR/Results/ML_Models/<Baseline|Advanced|Optuna_Optimization>/<file>.pkl

    Use this whenever you want to evaluate all models (not just the two
    final ones) without re-running training.

    Models listed in ZIPPED_MODELS are downloaded as a .zip and
    extracted, since a raw .pkl was too large to upload to GitHub
    directly.
    """
    if base_dir is None:
        base_dir = BASE_DIR

    model_root = os.path.join(base_dir, "Results", "ML_Models")

    for filename, subfolder in ALL_TRAINED_MODELS.items():
        dest = os.path.join(model_root, subfolder, filename)

        if os.path.exists(dest):
            print(f"[OK] Already downloaded: {dest}")
            continue

        try:
            if filename in ZIPPED_MODELS:
                zip_name = ZIPPED_MODELS[filename]
                extract_dir = os.path.join(base_dir, "_downloads", filename.replace(".pkl", ""))
                download_and_extract_zip(resolve_url(zip_name), extract_dir)

                pkl_path = None
                for root, _dirs, files in os.walk(extract_dir):
                    for f in files:
                        if f.lower().endswith(".pkl"):
                            pkl_path = os.path.join(root, f)
                            break
                    if pkl_path:
                        break

                if pkl_path is None:
                    raise FileNotFoundError(f"No .pkl file found inside {zip_name}")

                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy(pkl_path, dest)
                print(f"[OK] Extracted {pkl_path} -> {dest}")
            else:
                download_file(resolve_url(filename), dest)
        except Exception as e:
            print(f"[WARN] Could not download {filename}: {e}. "
                  f"If this file was uploaded under a different name, "
                  f"update ALL_TRAINED_MODELS (and ZIPPED_MODELS, if "
                  f"applicable) at the top of this file.")


def pip_install(*packages):
    """Safe replacement for the "!pip install ..." Colab magic command."""
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *packages], check=False)


# ============================================================
# Journal-style plotting helper
# ============================================================

def apply_journal_style():
    """
    Apply larger, publication-style matplotlib defaults (bigger fonts,
    bold titles, high DPI) so that every figure produced by the pipeline
    matches typical journal formatting requirements. Call this once at
    the top of any script that creates plots, before the first
    plt.figure() call.
    """
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 17,
        "axes.titleweight": "bold",
        "axes.labelsize": 15,
        "axes.labelweight": "bold",
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "legend.title_fontsize": 14,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "serif",
    })
