# ============================================================
# Script 04 of 07 - Full preparation of the raw spatial-transfer data
# (cleaning + feature engineering + scaling)
#
# Requires: the raw spatial_transfer_all.zip data, downloaded
# automatically from GitHub, plus the fitted scaler and the two
# optimized models (only loaded - training is never repeated).
# ============================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_github_loader import (  # noqa: F401
    BASE_DIR, ensure_models, ensure_scaler,
    ensure_transfer_raw_dataset, display,
)

ensure_models()
ensure_scaler()
ensure_transfer_raw_dataset()

from utils_github_loader import apply_journal_style
apply_journal_style()  # publication-style fonts for every figure in this script

"""CELL 31
# Load spatial transfer Dataset
# Check Missing Streamflow Records
"""

# ============================================================
# CELL 31
# Load spatial transfer Dataset
# Check Missing Streamflow Records
# ============================================================

import os
import zipfile
import numpy as np
import pandas as pd

print("-"*50)
print("LOADING spatial transfer DATASET")
print("-"*50)

# ============================================================
# Search for uploaded file
# ============================================================

search_paths = [
    "./flood_project_workspace/spatial_transfer_all.csv",
    "./flood_project_workspace/spatial_transfer_all.zip",
]

existing_file = None

for f in search_paths:
    if os.path.exists(f):
        existing_file = f
        break

if existing_file is None:

    files = os.listdir("./flood_project_workspace")

    csvs = [f for f in files if f.lower().endswith(".csv")]
    zips = [f for f in files if f.lower().endswith(".zip")]

    if len(csvs) > 0:
        existing_file = "./flood_project_workspace/" + csvs[0]

    elif len(zips) > 0:
        existing_file = "./flood_project_workspace/" + zips[0]

    else:
        raise FileNotFoundError("No CSV or ZIP file found in ./flood_project_workspace")

print("Detected file:")
print(existing_file)

# ============================================================
# Read CSV directly
# ============================================================

if existing_file.lower().endswith(".csv"):

    RepData = pd.read_csv(existing_file)

# ============================================================
# Read ZIP
# ============================================================

elif existing_file.lower().endswith(".zip"):

    try:

        extract_folder="./flood_project_workspace/spatial_transfer"

        os.makedirs(extract_folder,exist_ok=True)

        with zipfile.ZipFile(existing_file,"r") as zip_ref:
            zip_ref.extractall(extract_folder)

        csv_files=[]

        for root,dirs,files in os.walk(extract_folder):
            for f in files:
                if f.lower().endswith(".csv"):
                    csv_files.append(os.path.join(root,f))

        if len(csv_files)==0:
            raise Exception("No CSV found inside ZIP.")

        RepData=pd.read_csv(csv_files[0])

    except zipfile.BadZipFile:

        print()
        print("WARNING:")
        print("Uploaded file is not a valid ZIP.")
        print("Trying to read it as CSV...")

        RepData=pd.read_csv(existing_file)

else:

    raise Exception("Unsupported file format.")

print("-"*50)
print("Dataset Loaded Successfully")
print("-"*50)

print("Shape:",RepData.shape)

display(RepData.head())

# ============================================================
# CHECK STREAMFLOW DATA QUALITY
# ============================================================

print()
print("-"*50)
print("CHECKING STREAMFLOW DATA QUALITY")
print("-"*50)

if "Q" not in RepData.columns:
    raise Exception("Column 'Q' not found.")

if "gauge_id" not in RepData.columns:
    raise Exception("Column 'gauge_id' not found.")

summary=[]

for gid,group in RepData.groupby("gauge_id"):

    total=len(group)

    missing=group["Q"].isna().sum()

    minus999=(group["Q"]==-999).sum()

    valid=total-missing-minus999

    summary.append([gid,total,valid,missing,minus999])

quality_df=pd.DataFrame(
    summary,
    columns=[
        "gauge_id",
        "Total_Records",
        "Valid_Q",
        "Missing_Q",
        "Minus999_Q"
    ]
)

quality_df["Missing_%"]=100*quality_df["Missing_Q"]/quality_df["Total_Records"]

quality_df["Minus999_%"]=100*quality_df["Minus999_Q"]/quality_df["Total_Records"]

quality_df=quality_df.sort_values(
    ["Missing_Q","Minus999_Q"],
    ascending=False
)

display(quality_df)

print()

print("Total gauges :",quality_df.shape[0])

print("Total records:",len(RepData))

print("Missing Q :",RepData["Q"].isna().sum())

print("Minus999 Q:",(RepData["Q"]==-999).sum())

# ============================================================
# Save Results
# ============================================================

results_dir="./flood_project_workspace/Results/Spatial_Transfer/Data_Quality"

os.makedirs(results_dir,exist_ok=True)

quality_df.to_csv(
    os.path.join(results_dir,
    "Q_Data_Quality_By_Gauge_spatial_transfer_data.csv"),
    index=False
)

print()

print("Quality report saved to:")

print(results_dir)

print("-"*50)
print("CELL COMPLETED SUCCESSFULLY")
print("-"*50)

# ============================================================
# CELL 32
# Streamflow Cleaning and Dataset Preparation
# ============================================================

import os
import numpy as np
import pandas as pd


print("-"*60)
print("STREAMFLOW CLEANING STARTED")
print("-"*60)


# ------------------------------------------------------------
# Check dataframe
# ------------------------------------------------------------

if "RepData" in globals():
    df = RepData.copy()
    print("Using dataframe: RepData")

elif "rep_df" in globals():
    df = rep_df.copy()
    print("Using dataframe: rep_df")

else:
    raise NameError(
        "Representative dataframe not found. Load RepData.csv first."
    )


print("Original shape:", df.shape)


# ------------------------------------------------------------
# Check Q column
# ------------------------------------------------------------

if "Q" not in df.columns:
    raise KeyError(
        "Column 'Q' not found. Please check streamflow column name."
    )


# ------------------------------------------------------------
# Convert invalid values
# ------------------------------------------------------------

print("-"*60)
print("Cleaning Q values")
print("-"*60)


# Count before cleaning

missing_before = df["Q"].isna().sum()

minus999_before = (df["Q"] == -999).sum()


print("Missing Q before:", missing_before)
print("Q=-999 before:", minus999_before)


# Convert -999 to NaN

df["Q"] = df["Q"].replace(-999, np.nan)


# ------------------------------------------------------------
# Quality summary after conversion
# ------------------------------------------------------------

quality_after = (
    df.groupby("gauge_id")
    .agg(
        Total_records=("Q","count"),
        Missing_Q=("Q", lambda x: x.isna().sum()),
        Mean_Q=("Q","mean"),
        Min_Q=("Q","min"),
        Max_Q=("Q","max")
    )
    .reset_index()
)


quality_after["Missing_percent"] = (
    quality_after["Missing_Q"] /
    df.groupby("gauge_id").size().values
)*100


print("-"*60)
print("Quality after cleaning")
print("-"*60)


display(quality_after)


# ------------------------------------------------------------
# Remove invalid records only
# ------------------------------------------------------------

df_clean = df.dropna(subset=["Q"]).copy()


print("-"*60)
print("Cleaning summary")
print("-"*60)


print("Original records :", len(df))
print("Removed records  :", len(df)-len(df_clean))
print("Final records    :", len(df_clean))
print("Remaining gauges :", df_clean["gauge_id"].nunique())


# ------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------

output_folder = "./flood_project_workspace/Results/Spatial_Transfer/Data_Cleaning"

os.makedirs(
    output_folder,
    exist_ok=True
)


# save cleaned dataset

clean_path = os.path.join(
    output_folder,
    "spatial_transfer_Data_Cleaned.csv"
)


df_clean.to_csv(
    clean_path,
    index=False
)


# save quality report

quality_path = os.path.join(
    output_folder,
    "Streamflow_Quality_After_Cleaning_spatial_transfer.csv"
)


quality_after.to_csv(
    quality_path,
    index=False
)


print("-"*60)
print("FILES SAVED")
print("-"*60)

print(clean_path)
print(quality_path)


# ------------------------------------------------------------
# Store final dataframe
# ------------------------------------------------------------

RepData_clean = df_clean.copy()


print("-"*60)
print("CELL 15C COMPLETED SUCCESSFULLY")
print("-"*60)

"""C33"""

# ============================================================
# CELL 33
# Final Dataset Preparation Before Machine Learning
# ============================================================

import pandas as pd
import numpy as np
import os


print("-"*60)
print("FINAL ML DATASET PREPARATION STARTED")
print("-"*60)


# ------------------------------------------------------------
# Load representative dataset
# ------------------------------------------------------------

input_file = "./flood_project_workspace/spatial_transfer_all.csv"

if not os.path.exists(input_file):
    input_file = "./flood_project_workspace/spatial_transfer_all.zip"


df = pd.read_csv(input_file)

print("Original dataset size:")
print(df.shape)



# ------------------------------------------------------------
# Check streamflow column
# ------------------------------------------------------------

if "Q" not in df.columns:
    raise ValueError(
        "Streamflow column Q not found. Please check column name."
    )


# ------------------------------------------------------------
# Streamflow quality filtering
# ------------------------------------------------------------

initial_records = len(df)


missing_q = df["Q"].isna().sum()

minus999_q = (df["Q"] == -999).sum()


print("--------------------------------")
print("Streamflow filtering")
print("--------------------------------")

print("Missing Q:", missing_q)
print("Q = -999:", minus999_q)



# save removed records report

removed_report = pd.DataFrame({
    "Category":[
        "Missing Q",
        "Q equals -999"
    ],
    "Number_of_records":[
        missing_q,
        minus999_q
    ]
})


# ------------------------------------------------------------
# Remove invalid streamflow
# ------------------------------------------------------------

df_clean = df[
    (df["Q"].notna()) &
    (df["Q"] != -999)
].copy()



print("--------------------------------")
print("After filtering")
print("--------------------------------")

print("Remaining records:", len(df_clean))

print(
    "Removed records:",
    initial_records-len(df_clean)
)



# ------------------------------------------------------------
# Create datetime variables
# ------------------------------------------------------------

df_clean["Date"] = pd.to_datetime(
    dict(
        year=df_clean["Year"],
        month=df_clean["Mnth"],
        day=df_clean["Day"]
    )
)


df_clean["Day_of_Year"] = (
    df_clean["Date"].dt.dayofyear
)


df_clean["Season"] = (
    (df_clean["Mnth"]%12 + 3)//3
)



# ------------------------------------------------------------
# Sort chronologically
# ------------------------------------------------------------

df_clean = df_clean.sort_values(
    [
        "gauge_id",
        "Date"
    ]
).reset_index(drop=True)



# ------------------------------------------------------------
# Record availability per basin
# ------------------------------------------------------------

gauge_summary = (
    df_clean
    .groupby("gauge_id")
    .agg(
        Valid_Records=("Q","count"),
        Start_Date=("Date","min"),
        End_Date=("Date","max")
    )
    .reset_index()
)


print("--------------------------------")
print("Gauge availability")
print("--------------------------------")

display(gauge_summary)



# ------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------

output_folder = (
    "./flood_project_workspace/Results/Spatial_Transfer/Preparation"
)

os.makedirs(
    output_folder,
    exist_ok=True
)


df_clean.to_csv(
    output_folder+"/Final_spatial_transfer_for_ML.csv",
    index=False
)


gauge_summary.to_csv(
    output_folder+"/Gauge_Record_Count_After_Filtering_spatial_transfer_for_ML.csv",
    index=False
)


removed_report.to_csv(
    output_folder+"/Removed_Streamflow_Report_spatial_transfer.csv",
    index=False
)



print("--------------------------------")
print("FILES SAVED")
print("--------------------------------")

print(
    output_folder
)


print("""
1. Final_spatial_transfer_for_ML.csv
2. Gauge_Record_Count_After_Filtering_spatial_transfer_for_ML.csv
3. Removed_Streamflow_Report_spatial_transfer.csv
""")


print("--------------------------------")
print("CELL 16 COMPLETED SUCCESSFULLY")
print("--------------------------------")

"""C34"""

# ============================================================
# CELL 34
# Hydrological Feature Engineering
# Flood-oriented ML Dataset Construction
# ============================================================

import pandas as pd
import numpy as np
import os

print("------------------------------------------------------------")
print("HYDROLOGICAL FEATURE ENGINEERING STARTED")
print("------------------------------------------------------------")


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

input_file = "./flood_project_workspace/Results/Spatial_Transfer/Preparation/Final_spatial_transfer_for_ML.csv"

output_folder = "./flood_project_workspace/Results/Spatial_Transfer/ML_Features"

os.makedirs(output_folder, exist_ok=True)


# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------

df = pd.read_csv(input_file)

print("Original dataset:")
print(df.shape)


# ------------------------------------------------------------
# Create datetime index
# ------------------------------------------------------------

df["Date"] = pd.to_datetime(
    dict(
        year=df["Year"],
        month=df["Mnth"],
        day=df["Day"]
    )
)

df = df.sort_values(
    ["gauge_id", "Date"]
).reset_index(drop=True)


# ------------------------------------------------------------
# Antecedent precipitation features
# ------------------------------------------------------------

print("--------------------------------")
print("Creating precipitation memory features")
print("--------------------------------")


for lag in [1,3,7,14,30]:

    df[f"prcp_lag_{lag}"] = (
        df.groupby("gauge_id")["prcp(mm/day)"]
        .shift(lag)
    )


for window in [3,7,14,30]:

    df[f"prcp_acc_{window}"] = (
        df.groupby("gauge_id")["prcp(mm/day)"]
        .rolling(window)
        .sum()
        .reset_index(level=0, drop=True)
    )


# ------------------------------------------------------------
# Rainfall intensity indicators
# ------------------------------------------------------------

df["prcp_max_7day"] = (
    df.groupby("gauge_id")["prcp(mm/day)"]
    .rolling(7)
    .max()
    .reset_index(level=0, drop=True)
)


df["prcp_max_30day"] = (
    df.groupby("gauge_id")["prcp(mm/day)"]
    .rolling(30)
    .max()
    .reset_index(level=0, drop=True)
)


# ------------------------------------------------------------
# Temperature indices
# ------------------------------------------------------------

df["tmean(C)"] = (
    df["tmax(C)"] +
    df["tmin(C)"]
) / 2


df["positive_temperature"] = np.maximum(
    df["tmean(C)"],
    0
)


# ------------------------------------------------------------
# Seasonal encoding
# ------------------------------------------------------------

df["month_sin"] = np.sin(
    2*np.pi*df["Mnth"]/12
)

df["month_cos"] = np.cos(
    2*np.pi*df["Mnth"]/12
)

# ------------------------------------------------------------
# Flood threshold calculation
# Basin-specific Q95
# ------------------------------------------------------------

print("--------------------------------")
print("Creating flood labels")
print("--------------------------------")


q95 = (
    df.groupby("gauge_id")["Q"]
    .quantile(0.95)
    .rename("Q95")
)


df = df.merge(
    q95,
    on="gauge_id",
    how="left"
)


df["Flood_Event"] = (
    df["Q"] >= df["Q95"]
).astype(int)


# ------------------------------------------------------------
# Remove incomplete rows
# ------------------------------------------------------------

before = len(df)

df = df.dropna()

after = len(df)


print("--------------------------------")
print("Missing feature removal")
print("--------------------------------")

print("Before:", before)
print("After :", after)
print("Removed:", before-after)


# ------------------------------------------------------------
# Define feature groups
# ------------------------------------------------------------

dynamic_features = [

    "dayl(s)",
    "prcp(mm/day)",
    "srad(W/m2)",
    "swe(mm)",
    "tmax(C)",
    "tmin(C)",
    "vp(Pa)",

    "prcp_lag_1",
    "prcp_lag_3",
    "prcp_lag_7",
    "prcp_lag_14",
    "prcp_lag_30",

    "prcp_acc_3",
    "prcp_acc_7",
    "prcp_acc_14",
    "prcp_acc_30",

    "prcp_max_7day",
    "prcp_max_30day",

    "tmean(C)",
    "positive_temperature",

    "month_sin",
    "month_cos"

]


static_features = [

    "p_mean",
    "pet_mean",

    "frac_snow",
    "aridity",

    "high_prec_freq",
    "high_prec_dur",

    "elev_mean",
    "slope_mean",

    "area_gages2",

    "soil_depth_pelletier",
    "soil_porosity",
    "soil_conductivity",

    "sand_frac",
    "clay_frac",

    "frac_forest",

    "lai_max",
    "lai_diff",

    "gvf_max",
    "gvf_diff",

    "carbonate_rocks_frac",
    "geol_permeability"

]


target_variables = [


    "Flood_Event"

]


final_columns = (
    dynamic_features +
    static_features +
    target_variables +
    ["gauge_id","Date","Q"]
)


ml_dataset = df[final_columns].copy()


# ------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------

ml_dataset.to_csv(
    f"{output_folder}/Flood_ML_Dataset_spatial_transfer.csv",
    index=False
)


pd.DataFrame(
    {
        "Dynamic_Features": pd.Series(dynamic_features),
        "Static_Features": pd.Series(static_features)
    }
).to_csv(
    f"{output_folder}/Feature_Definition_spatial_transfer.csv",
    index=False
)


pd.DataFrame(
    {
        "Target":
        [

            "Flood_Event"
        ],
        "Description":
        [

            "Flood occurrence based on basin-specific Q95 threshold"
        ]
    }
).to_csv(
    f"{output_folder}/Target_Definition_spatial_transfer.csv",
    index=False
)


print("------------------------------------------------------------")
print("FINAL ML DATASET CREATED")
print("------------------------------------------------------------")

print("Rows:", len(ml_dataset))
print("Columns:", len(ml_dataset.columns))

print("--------------------------------")
print("Files saved:")
print("--------------------------------")

print("Flood_ML_Dataset_spatial_transfer.csv")
print("Feature_Definition_spatial_transfer.csv")
print("Target_Definition_spatial_transfer.csv")

print("------------------------------------------------------------")
print("CELL 17 COMPLETED SUCCESSFULLY")
print("------------------------------------------------------------")








# ============================================================
# CELL: SPATIAL TRANSFER DATASET PREPARATION
# Leakage-free Q95 thresholding using 1980-2008 only
# Apply threshold on 2012-2014 transfer period
# ============================================================


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib


print("="*80)
print("SPATIAL TRANSFER DATASET PREPARATION STARTED")
print("="*80)


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

input_file = "./flood_project_workspace/Results/Spatial_Transfer/ML_Features/Flood_ML_Dataset_spatial_transfer.csv"

output_dir = "./flood_project_workspace/Results/Spatial_Transfer/Input"

os.makedirs(output_dir, exist_ok=True)


# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------

df = pd.read_csv(input_file)


print("Dataset shape:")
print(df.shape)


# ------------------------------------------------------------
# Convert Date
# ------------------------------------------------------------

df["Date"] = pd.to_datetime(df["Date"])


# ------------------------------------------------------------
# Define periods
# ------------------------------------------------------------

train_start = "1980-01-01"
train_end   = "2008-12-31"

transfer_start = "2012-01-01"
transfer_end   = "2014-12-31"



# ------------------------------------------------------------
# Load scaler fitted on representative training catchments
# ------------------------------------------------------------

scaler_path = "./flood_project_workspace/Results/ML_Training_Input/StandardScaler.pkl"

scaler = joblib.load(
    scaler_path
)

print("Scaler loaded successfully")



# ------------------------------------------------------------
# Identify transfer catchments
#
# Modify here if specific gauge list exists
# ------------------------------------------------------------

all_gauges = df["gauge_id"].unique()

print("Number of gauges:")
print(len(all_gauges))



# ------------------------------------------------------------
# Calculate Q95 using ONLY 1980-2008
# ------------------------------------------------------------


threshold_records = []

transfer_records = []


for gauge in all_gauges:


    gauge_df = df[
        df["gauge_id"] == gauge
    ].copy()


    train_df = gauge_df[
        (gauge_df["Date"] >= train_start)
        &
        (gauge_df["Date"] <= train_end)
    ]


    transfer_df = gauge_df[
        (gauge_df["Date"] >= transfer_start)
        &
        (gauge_df["Date"] <= transfer_end)
    ]



    # Skip if no data

    if len(train_df)==0 or len(transfer_df)==0:
        continue



    # -----------------------------
    # Q95 from training period only
    # -----------------------------

    q95_train = np.percentile(
        train_df["Q"].dropna(),
        95
    )



    # -----------------------------
    # Apply Q95 to transfer period
    # -----------------------------

    transfer_df["Flood_Event"] = (
        transfer_df["Q"] >= q95_train
    ).astype(int)



    threshold_records.append({

        "gauge_id": gauge,

        "Q95_train": q95_train,

        "Training_samples": len(train_df),

        "Transfer_samples": len(transfer_df),

        "Transfer_Flood_Events":
            transfer_df["Flood_Event"].sum()

    })


    transfer_records.append(
        transfer_df
    )



# ------------------------------------------------------------
# Combine transfer dataset
# ------------------------------------------------------------

transfer_final = pd.concat(
    transfer_records,
    ignore_index=True
)



threshold_df = pd.DataFrame(
    threshold_records
)



# ------------------------------------------------------------
# Save thresholds
# ------------------------------------------------------------

threshold_df.to_csv(
    f"{output_dir}/Transfer_Thresholds_By_Gauge.csv",
    index=False
)



# ------------------------------------------------------------
# Define feature columns
# Remove target and metadata
# ------------------------------------------------------------


remove_cols = [

    "Date",
    "Q",
    "gauge_id",
    "Flood_Event"

]


feature_columns = [
    c for c in transfer_final.columns
    if c not in remove_cols
]


print("Number of features:")
print(len(feature_columns))



# ------------------------------------------------------------
# Create X and y
# ------------------------------------------------------------


X_transfer = transfer_final[
    feature_columns
]


y_transfer = transfer_final[
    "Flood_Event"
]



# ------------------------------------------------------------
# Apply existing scaler
# NO FIT !!!
# ------------------------------------------------------------
import joblib
scaler = joblib.load('/content/Flood_Project/StandardScaler.pkl')

X_transfer_scaled = scaler.transform(
    X_transfer
)


X_transfer_scaled = pd.DataFrame(
    X_transfer_scaled,
    columns=feature_columns
)



# ------------------------------------------------------------
# Save datasets
# ------------------------------------------------------------


X_transfer.to_csv(
    f"{output_dir}/X_transfer.csv",
    index=False
)


y_transfer.to_csv(
    f"{output_dir}/y_transfer.csv",
    index=False
)


X_transfer_scaled.to_csv(
    f"{output_dir}/X_transfer_scaled.csv",
    index=False
)



transfer_final.to_csv(
    f"{output_dir}/Spatial_Transfer_Final_Dataset.csv",
    index=False
)



# ------------------------------------------------------------
# Statistics
# ------------------------------------------------------------


stats = pd.DataFrame({

    "Statistic":[

        "Total transfer samples",
        "Total flood events",
        "Flood percentage",
        "Number of gauges"

    ],

    "Value":[

        len(transfer_final),

        transfer_final["Flood_Event"].sum(),

        transfer_final["Flood_Event"].mean()*100,

        transfer_final["gauge_id"].nunique()

    ]

})


stats.to_csv(
    f"{output_dir}/Transfer_Streamflow_Statistics.csv",
    index=False
)



# ------------------------------------------------------------
# Plot 1
# Flood event distribution
# ------------------------------------------------------------


plt.figure(figsize=(6,4))


counts = transfer_final["Flood_Event"].value_counts()


plt.bar(
    counts.index.astype(str),
    counts.values
)


plt.xlabel("Flood Event")
plt.ylabel("Number of samples")
plt.title(
    "Flood Event Distribution in Transfer Dataset"
)


plt.tight_layout()


plt.savefig(
    f"{output_dir}/Transfer_Flood_Event_Distribution.png",
    dpi=300
)


plt.close()



# ------------------------------------------------------------
# Plot 2
# Q95 thresholds
# ------------------------------------------------------------


plt.figure(figsize=(10,5))


plt.bar(
    threshold_df["gauge_id"].astype(str),
    threshold_df["Q95_train"]
)


plt.xlabel("Gauge ID")
plt.ylabel("Q95 from 1980-2008")
plt.title(
    "Gauge-specific Q95 Thresholds for Spatial Transfer"
)


plt.xticks(rotation=45)


plt.tight_layout()


plt.savefig(
    f"{output_dir}/Transfer_Q95_Thresholds.png",
    dpi=300
)


plt.close()



# ------------------------------------------------------------
# Print summary
# ------------------------------------------------------------


print("="*80)
print("TRANSFER DATASET CREATED")
print("="*80)


print("\nThreshold summary:")
display(threshold_df)


print("\nFinal transfer dataset:")
print(
    transfer_final.shape
)


print("\nFiles saved:")

for f in os.listdir(output_dir):
    print(f)


print("="*80)
print("CELL COMPLETED SUCCESSFULLY")
print("="*80)

# =============================================================================
# GAUGE-WISE SPATIAL TRANSFERABILITY EVALUATION (UPDATED)
# =============================================================================

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef
)


print("="*70)
print("GAUGE-WISE SPATIAL TRANSFERABILITY EVALUATION")
print("="*70)


# =============================================================================
# Paths
# =============================================================================

BASE="./flood_project_workspace/Results"

TRANSFER_DIR=os.path.join(
    BASE,
    "Spatial_Transfer/Input"
)


OUTPUT_DIR=os.path.join(
    TRANSFER_DIR,
    "Gauge_Wise_Evaluation"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# =============================================================================
# Load thresholds
# =============================================================================

threshold_df = pd.read_csv(
    "./flood_project_workspace/Results/Threshold_Correction/Corrected_Threshold_Results.csv"
)


threshold_dict = dict(
    zip(
        threshold_df["Model"],
        threshold_df["Validation_Optimal_Threshold"]
    )
)



# =============================================================================
# Load original dataset (metadata)
# =============================================================================

data=pd.read_csv(
    os.path.join(
        TRANSFER_DIR,
        "Spatial_Transfer_Final_Dataset.csv"
    )
)


print("\nOriginal Dataset:")
print(data.shape)



# =============================================================================
# Load scaled transfer data
# =============================================================================


X_transfer = pd.read_csv(
    os.path.join(
        TRANSFER_DIR,
        "X_transfer_scaled.csv"
    )
)


y_transfer = pd.read_csv(
    os.path.join(
        TRANSFER_DIR,
        "y_transfer.csv"
    )
).values.ravel()



print("\nScaled X:")
print(X_transfer.shape)

print("Target:")
print(y_transfer.shape)



# =============================================================================
# Check alignment
# =============================================================================

assert len(data)==len(X_transfer)==len(y_transfer), \
"Dataset length mismatch"



# =============================================================================
# Add index mapping
# =============================================================================


data = data.reset_index(drop=True)

X_transfer.index = data.index



# =============================================================================
# Load models
# =============================================================================


print("\nLoading models...")


xgb_model=joblib.load(
    "./flood_project_workspace/Results/ML_Models/Optuna_Optimization/Best_XGBoost.pkl"
)


cat_model=joblib.load(
    "./flood_project_workspace/Results/ML_Models/Optuna_Optimization/Best_CatBoost.pkl"
)


print("Models loaded")



# =============================================================================
# Evaluation Function
# =============================================================================


def evaluate_model(
        model,
        model_name,
        X_test,
        y_test):


    probability = model.predict_proba(
        X_test
    )[:,1]


    threshold = threshold_dict[model_name]


    prediction = (
        probability >= threshold
    ).astype(int)



    cm = confusion_matrix(
        y_test,
        prediction,
        labels=[0,1]
    )


    TN, FP, FN, TP = cm.ravel()



    metrics={


        "Model":model_name,


        "Samples":len(y_test),


        "Flood_Events":int(y_test.sum()),


        "Accuracy":
        accuracy_score(
            y_test,
            prediction
        ),


        "Precision":
        precision_score(
            y_test,
            prediction,
            zero_division=0
        ),


        "Recall":
        recall_score(
            y_test,
            prediction,
            zero_division=0
        ),


        "F1":
        f1_score(
            y_test,
            prediction,
            zero_division=0
        ),


        "ROC_AUC":
        roc_auc_score(
            y_test,
            probability
        )
        if len(np.unique(y_test))>1
        else np.nan,


        "PR_AUC":
        average_precision_score(
            y_test,
            probability
        ),


        "MCC":
        matthews_corrcoef(
            y_test,
            prediction
        ),


        "TP":TP,
        "TN":TN,
        "FP":FP,
        "FN":FN

    }


    return metrics, prediction, probability, cm




# =============================================================================
# Gauge-wise evaluation
# =============================================================================


all_results=[]


gauges=data["gauge_id"].unique()


print("\nNumber of gauges:",len(gauges))



models=[

    (
        xgb_model,
        "Optimized_XGBoost"
    ),


    (
        cat_model,
        "Optimized_CatBoost"
    )

]



for gauge in gauges:


    print("\n--------------------------------")
    print("Processing Gauge:",gauge)
    print("--------------------------------")


    idx = (
        data["gauge_id"] == gauge
    )


    # Scaled input
    X_gauge = X_transfer.loc[idx]


    # Target
    y_gauge = y_transfer[idx.values]



    gauge_data = data.loc[idx].copy()



    for model,model_name in models:



        metrics,pred,prob,cm = evaluate_model(
            model,
            model_name,
            X_gauge,
            y_gauge
        )



        metrics["Gauge_ID"]=gauge


        all_results.append(
            metrics
        )



        # =================================================
        # Save predictions
        # =================================================


        prediction_df=gauge_data[
            [
                "gauge_id",
                "Date",
                "Q",
                "Flood_Event"
            ]
        ].copy()



        prediction_df["Predicted_Event"]=pred

        prediction_df["Flood_Probability"]=prob

        prediction_df["Model"]=model_name



        prediction_df.to_csv(

            os.path.join(
                OUTPUT_DIR,
                f"{model_name}_Gauge_{gauge}_Prediction.csv"
            ),

            index=False
        )



        # =================================================
        # Save confusion matrix
        # =================================================


        pd.DataFrame(
            cm,
            index=[
                "Observed_0",
                "Observed_1"
            ],
            columns=[
                "Predicted_0",
                "Predicted_1"
            ]

        ).to_csv(

            os.path.join(
                OUTPUT_DIR,
                f"{model_name}_Gauge_{gauge}_Confusion_Matrix.csv"
            )

        )



# =============================================================================
# Save final results
# =============================================================================


results_df=pd.DataFrame(
    all_results
)


results_df=results_df.sort_values(
    [
        "Gauge_ID",
        "Model"
    ]
)



results_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "Gauge_Wise_Model_Performance.csv"
    ),

    index=False
)



print("\nEvaluation completed")

print(results_df)

