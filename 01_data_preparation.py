# -*- coding: utf-8 -*-
# ============================================================
# Script 01 of 07 - Main dataset preparation
# (load + clean the representative dataset, hydrological feature
#  engineering, flood labeling, train/valid/test split)
#
# This script must always be run first. Its outputs are stored in the
# local folder ./flood_project_workspace/Results/... and are required
# by every later script (02 through 07).
#
# This script is meant to be run in Google Colab. The main raw
# representative dataset (RepData.zip) is downloaded automatically
# from GitHub into /content, so no manual upload or Google Drive
# mount is required.
# ============================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_github_loader import BASE_DIR, ensure_main_raw_data_zip, display  # noqa: F401

ensure_main_raw_data_zip()

# ============================================================
# STEP 1
# Load Representative Dataset
# Check Missing Streamflow Records
# ============================================================

import os
import zipfile
import numpy as np
import pandas as pd

print("-"*50)
print("LOADING REPRESENTATIVE DATASET")
print("-"*50)

# ============================================================
# Search for uploaded file
# ============================================================

search_paths = [
    "./flood_project_workspace/drive/MyDrive/Flood_Project_Backups/Flood_Project_updated_main/Representative_Data/RepData.csv",
    "./flood_project_workspace/RepData.zip",
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

        extract_folder="./flood_project_workspace/Representative_Data"

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

results_dir="./flood_project_workspace/Results/Data_Quality"

os.makedirs(results_dir,exist_ok=True)

quality_df.to_csv(
    os.path.join(results_dir,
    "Q_Data_Quality_By_Gauge.csv"),
    index=False
)

print()

print("Quality report saved to:")

print(results_dir)

print("-"*50)
print("STEP 1 COMPLETED SUCCESSFULLY")
print("-"*50)


# ============================================================
# STEP 2
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

output_folder = "./flood_project_workspace/Results/Data_Cleaning"

os.makedirs(
    output_folder,
    exist_ok=True
)


# save cleaned dataset

clean_path = os.path.join(
    output_folder,
    "RepData_Cleaned.csv"
)


df_clean.to_csv(
    clean_path,
    index=False
)


# save quality report

quality_path = os.path.join(
    output_folder,
    "Streamflow_Quality_After_Cleaning.csv"
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
print("STEP 2 COMPLETED SUCCESSFULLY")
print("-"*60)


# ============================================================
# STEP 3
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

input_file = "./flood_project_workspace/drive/MyDrive/Flood_Project_Backups/Flood_Project_updated_main/Representative_Data/RepData.csv"

if not os.path.exists(input_file):
    input_file = "./flood_project_workspace/RepData.zip"


df = pd.read_csv(
    input_file,
    low_memory=False
)

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
# ------------------------------------------------------------
# Convert date components to numeric
# ------------------------------------------------------------

for col in ["Year", "Mnth", "Day"]:
    df_clean[col] = pd.to_numeric(
        df_clean[col],
        errors="coerce"
    )



# Remove records with invalid date components

df_clean = df_clean.dropna(
    subset=["Year", "Mnth", "Day"]
).copy()


df_clean["Year"] = df_clean["Year"].astype(int)
df_clean["Mnth"] = df_clean["Mnth"].astype(int)
df_clean["Day"] = df_clean["Day"].astype(int)
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
    "./flood_project_workspace/Results/ML_Preparation"
)

os.makedirs(
    output_folder,
    exist_ok=True
)


df_clean.to_csv(
    output_folder+"/Final_ML_Dataset.csv",
    index=False
)


gauge_summary.to_csv(
    output_folder+"/Gauge_Record_Count_After_Filtering.csv",
    index=False
)


removed_report.to_csv(
    output_folder+"/Removed_Streamflow_Report.csv",
    index=False
)



print("--------------------------------")
print("FILES SAVED")
print("--------------------------------")

print(
    output_folder
)


print("""
1. Final_ML_Dataset.csv
2. Gauge_Record_Count_After_Filtering.csv
3. Removed_Streamflow_Report.csv
""")


print("--------------------------------")
print("STEP 3 COMPLETED SUCCESSFULLY")
print("--------------------------------")


"""Copy of Untitled14.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1W57bDM0UIQMm0Uhx60o4spv2BPIjChK_

# **C16**
"""

# ============================================================
# STEP 4
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

input_file = "./flood_project_workspace/Results/ML_Preparation/Final_ML_Dataset.csv"

output_folder = "./flood_project_workspace/Results/ML_Features"

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

    "Q",
    "Flood_Event"

]


final_columns = (
    dynamic_features +
    static_features +
    target_variables +
    ["gauge_id","Date"]
)


ml_dataset = df[final_columns].copy()


# ------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------

ml_dataset.to_csv(
    f"{output_folder}/Flood_ML_Dataset.csv",
    index=False
)


pd.DataFrame(
    {
        "Dynamic_Features": pd.Series(dynamic_features),
        "Static_Features": pd.Series(static_features)
    }
).to_csv(
    f"{output_folder}/Feature_Definition.csv",
    index=False
)


pd.DataFrame(
    {
        "Target":
        [
            "Q",
            "Flood_Event"
        ],
        "Description":
        [
            "Continuous streamflow prediction",
            "Flood occurrence based on basin-specific Q95 threshold"
        ]
    }
).to_csv(
    f"{output_folder}/Target_Definition.csv",
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

print("Flood_ML_Dataset.csv")
print("Feature_Definition.csv")
print("Target_Definition.csv")

print("------------------------------------------------------------")
print("STEP 4 COMPLETED SUCCESSFULLY")
print("------------------------------------------------------------")

# ============================================================
# STEP 5
# FLOOD THRESHOLD ANALYSIS (LEAKAGE-FREE)
# Thresholds are computed ONLY from TRAIN data
# ============================================================

import os
import numpy as np
import pandas as pd

print("="*70)
print("FLOOD THRESHOLD ANALYSIS (TRAIN-ONLY)")
print("="*70)

# ------------------------------------------------------------
# Locate dataset
# ------------------------------------------------------------

candidate_files = [

    "./flood_project_workspace/Results/ML_Features/Flood_ML_Dataset.csv"
]

input_file = None

for f in candidate_files:
    if os.path.exists(f):
        input_file = f
        break

if input_file is None:
    raise FileNotFoundError("Flood dataset not found.")

df = pd.read_csv(input_file)

print(f"Dataset loaded: {df.shape}")

# ------------------------------------------------------------
# Date processing
# ------------------------------------------------------------

df["Date"] = pd.to_datetime(df["Date"])

train_df = df[
    (df["Date"].dt.year >= 1980) &
    (df["Date"].dt.year <= 2008)
].copy()

val_df = df[
    (df["Date"].dt.year >= 2009) &
    (df["Date"].dt.year <= 2011)
].copy()

test_df = df[
    (df["Date"].dt.year >= 2012) &
    (df["Date"].dt.year <= 2014)
].copy()

print("-"*70)
print("Dataset Split")
print("-"*70)

print(f"Train samples      : {len(train_df):,}")
print(f"Validation samples : {len(val_df):,}")
print(f"Test samples       : {len(test_df):,}")

# ------------------------------------------------------------
# Output directory
# ------------------------------------------------------------

output_folder = "./flood_project_workspace/Results/Flood_Threshold_Analysis"

os.makedirs(
    output_folder,
    exist_ok=True
)

# ------------------------------------------------------------
# Compute thresholds ONLY from TRAIN
# ------------------------------------------------------------

print("-"*70)
print("Computing thresholds from TRAIN only")
print("-"*70)

thresholds = []

for gauge, group in train_df.groupby("gauge_id"):

    q = group["Q"].dropna()

    if len(q) == 0:
        continue

    q90 = np.percentile(q,90)
    q95 = np.percentile(q,95)
    q98 = np.percentile(q,98)
    q99 = np.percentile(q,99)

    thresholds.append({

        "gauge_id":gauge,

        "Train_Records":len(q),

        "Q90":q90,
        "Q95":q95,
        "Q98":q98,
        "Q99":q99,

        "Flood90_Train":(q>=q90).sum(),
        "Flood95_Train":(q>=q95).sum(),
        "Flood98_Train":(q>=q98).sum(),
        "Flood99_Train":(q>=q99).sum()

    })

thresholds = pd.DataFrame(thresholds)

display(thresholds.head())

threshold_file = os.path.join(

    output_folder,
    "Flood_Thresholds_TrainOnly.csv"

)

thresholds.to_csv(

    threshold_file,
    index=False

)

print()
print(f"Threshold file saved:")
print(threshold_file)

# ------------------------------------------------------------
# Overall statistics
# ------------------------------------------------------------

overall = pd.DataFrame({

"Statistic":[

"Total Samples",

"Training Samples",

"Validation Samples",

"Testing Samples",

"Number of Basins",

"Mean Q",

"Median Q",

"Maximum Q",

"Minimum Q"

],

"Value":[

len(df),

len(train_df),

len(val_df),

len(test_df),

df["gauge_id"].nunique(),

df["Q"].mean(),

df["Q"].median(),

df["Q"].max(),

df["Q"].min()

]

})

display(overall)

overall.to_csv(

os.path.join(

output_folder,
"Overall_Streamflow_Statistics.csv"

),

index=False

)

# ------------------------------------------------------------
# Flood counts using TRAIN thresholds
# (Train / Validation / Test)
# ------------------------------------------------------------

threshold_dict = thresholds.set_index("gauge_id")

records = []

for dataset_name, dataset in zip(

    ["Train","Validation","Test"],

    [train_df,val_df,test_df]

):

    merged = dataset.merge(

        threshold_dict[
            ["Q90","Q95","Q98","Q99"]
        ],

        left_on="gauge_id",

        right_index=True,

        how="left"

    )

    records.append({

        "Dataset":dataset_name,

        "Threshold":"Q90",

        "Flood_Events":(merged["Q"]>=merged["Q90"]).sum()

    })

    records.append({

        "Dataset":dataset_name,

        "Threshold":"Q95",

        "Flood_Events":(merged["Q"]>=merged["Q95"]).sum()

    })

    records.append({

        "Dataset":dataset_name,

        "Threshold":"Q98",

        "Flood_Events":(merged["Q"]>=merged["Q98"]).sum()

    })

    records.append({

        "Dataset":dataset_name,

        "Threshold":"Q99",

        "Flood_Events":(merged["Q"]>=merged["Q99"]).sum()

    })

event_counts = pd.DataFrame(records)

display(event_counts)

event_counts.to_csv(

os.path.join(

output_folder,
"Flood_Event_Counts_By_Split.csv"

),

index=False

)

print("="*70)
print("PART 1 COMPLETED")
print("="*70)

print("Files created:")
print("1. Flood_Thresholds_TrainOnly.csv")
print("2. Overall_Streamflow_Statistics.csv")
print("3. Flood_Event_Counts_By_Split.csv")

# ============================================================
# STEP 6
# FIGURES AND VISUALIZATION
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("="*70)
print("GENERATING FIGURES")
print("="*70)

# ------------------------------------------------------------
# Histogram (All data)
# ------------------------------------------------------------

plt.figure(figsize=(8,5))

plt.hist(
    df["Q"],
    bins=100
)

plt.xlabel("Daily Streamflow")

plt.ylabel("Frequency")

plt.title("Distribution of Daily Streamflow (All Samples)")

plt.tight_layout()

plt.savefig(

os.path.join(
output_folder,
"Histogram_Q_All.png"
),

dpi=300

)

plt.close()

# ------------------------------------------------------------
# Histogram (Train only)
# ------------------------------------------------------------

plt.figure(figsize=(8,5))

plt.hist(

train_df["Q"],
bins=100

)

plt.xlabel("Daily Streamflow")

plt.ylabel("Frequency")

plt.title("Distribution of Daily Streamflow (Training Dataset)")

plt.tight_layout()

plt.savefig(

os.path.join(
output_folder,
"Histogram_Q_Train.png"
),

dpi=300

)

plt.close()

# ------------------------------------------------------------
# Empirical CDF
# ------------------------------------------------------------

q=np.sort(df["Q"].values)

cdf=np.arange(1,len(q)+1)/len(q)

plt.figure(figsize=(8,5))

plt.plot(q,cdf)

plt.xlabel("Daily Streamflow")

plt.ylabel("Empirical CDF")

plt.title("Empirical CDF of Streamflow")

plt.tight_layout()

plt.savefig(

os.path.join(
output_folder,
"CDF_Q.png"
),

dpi=300

)

plt.close()

# ------------------------------------------------------------
# Boxplot
# ------------------------------------------------------------

plt.figure(figsize=(5,7))

plt.boxplot(

df["Q"],
vert=True

)

plt.ylabel("Daily Streamflow")

plt.tight_layout()

plt.savefig(

os.path.join(
output_folder,
"Boxplot_Q.png"
),

dpi=300

)

plt.close()

# ------------------------------------------------------------
# Threshold distributions
# ------------------------------------------------------------

plt.figure(figsize=(9,6))

plt.hist(
thresholds["Q90"],
bins=20,
alpha=0.6,
label="Q90"
)

plt.hist(
thresholds["Q95"],
bins=20,
alpha=0.6,
label="Q95"
)

plt.hist(
thresholds["Q98"],
bins=20,
alpha=0.6,
label="Q98"
)

plt.hist(
thresholds["Q99"],
bins=20,
alpha=0.6,
label="Q99"
)

plt.xlabel("Threshold Discharge")

plt.ylabel("Number of Catchments")

plt.title("Distribution of Basin-Specific Flood Thresholds")

plt.legend()

plt.tight_layout()

plt.savefig(

os.path.join(
output_folder,
"Threshold_Distributions.png"
),

dpi=300

)

plt.close()

# ------------------------------------------------------------
# Flood event counts by split
# ------------------------------------------------------------

pivot = event_counts.pivot(

index="Threshold",

columns="Dataset",

values="Flood_Events"

)

pivot.to_csv(

os.path.join(
output_folder,
"Flood_Event_Counts_Pivot.csv"
)

)

pivot.plot(

kind="bar",
figsize=(7,5)

)

plt.ylabel("Number of Flood Events")

plt.title("Flood Events Using Train-Derived Thresholds")

plt.tight_layout()

plt.savefig(

os.path.join(
output_folder,
"Flood_Event_Counts.png"
),

dpi=300

)

plt.close()

# ------------------------------------------------------------
# Total flood events
# ------------------------------------------------------------

total_counts = (

event_counts
.groupby("Threshold")["Flood_Events"]
.sum()
.reset_index()

)

total_counts.to_csv(

os.path.join(

output_folder,

"Flood_Event_Counts_Total.csv"

),

index=False

)

# ------------------------------------------------------------
# Threshold summary statistics
# ------------------------------------------------------------

summary_stats = pd.DataFrame({

"Statistic":[

"Mean Q90",
"Mean Q95",
"Mean Q98",
"Mean Q99",

"Median Q90",
"Median Q95",
"Median Q98",
"Median Q99"

],

"Value":[

thresholds["Q90"].mean(),
thresholds["Q95"].mean(),
thresholds["Q98"].mean(),
thresholds["Q99"].mean(),

thresholds["Q90"].median(),
thresholds["Q95"].median(),
thresholds["Q98"].median(),
thresholds["Q99"].median()

]

})

summary_stats.to_csv(

os.path.join(

output_folder,

"Threshold_Summary_Statistics.csv"

),

index=False

)

print("="*70)
print("FILES SAVED")
print("="*70)

saved_files=[

"Flood_Thresholds_TrainOnly.csv",
"Overall_Streamflow_Statistics.csv",
"Flood_Event_Counts_By_Split.csv",
"Flood_Event_Counts_Total.csv",
"Flood_Event_Counts_Pivot.csv",
"Threshold_Summary_Statistics.csv",
"Histogram_Q_All.png",
"Histogram_Q_Train.png",
"CDF_Q.png",
"Boxplot_Q.png",
"Threshold_Distributions.png",
"Flood_Event_Counts.png"

]

for i,f in enumerate(saved_files,1):
    print(f"{i}. {f}")

print("="*70)
print("STEP 5/6 COMPLETED SUCCESSFULLY")
print("="*70)

# ============================================================
# STEP 7
# CREATE FLOOD_EVENT USING TRAIN-DERIVED THRESHOLDS
# (LEAKAGE-FREE)
# ============================================================

import os
import numpy as np
import pandas as pd

print("="*70)
print("CREATING FLOOD_EVENT (TRAIN-DERIVED Q95)")
print("="*70)

# ------------------------------------------------------------
# Input files
# ------------------------------------------------------------

dataset_file = "./flood_project_workspace/Results/ML_Features/Flood_ML_Dataset.csv"

threshold_file = "./flood_project_workspace/Results/Flood_Threshold_Analysis/Flood_Thresholds_TrainOnly.csv"

output_folder = "./flood_project_workspace/Results/ML_Features"

os.makedirs(output_folder, exist_ok=True)

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

df = pd.read_csv(dataset_file)

thresholds = pd.read_csv(threshold_file)

print(f"Dataset shape      : {df.shape}")
print(f"Threshold basins   : {len(thresholds)}")

# ------------------------------------------------------------
# Merge thresholds
# ------------------------------------------------------------

df = df.merge(

    thresholds[
        [
            "gauge_id",
            "Q95"
        ]
    ],

    on="gauge_id",

    how="left"

)

# ------------------------------------------------------------
# Check missing thresholds
# ------------------------------------------------------------

missing = df["Q95"].isna().sum()

if missing > 0:

    raise ValueError(
        f"{missing} rows have no corresponding TRAIN threshold."
    )

# ------------------------------------------------------------
# Create Flood_Event
# ------------------------------------------------------------

df["Flood_Event"] = (

    df["Q"] >= df["Q95"]

).astype(int)

# ------------------------------------------------------------
# Statistics
# ------------------------------------------------------------

overall = pd.DataFrame({

"Statistic":[

"Total Samples",

"Flood Events",

"Non-Flood Events",

"Flood Percentage (%)"

],

"Value":[

len(df),

int(df["Flood_Event"].sum()),

int((df["Flood_Event"]==0).sum()),

100*df["Flood_Event"].mean()

]

})

print()
print(overall)

# ------------------------------------------------------------
# Statistics by split
# ------------------------------------------------------------

df["Date"] = pd.to_datetime(df["Date"])

split_stats=[]

for name,start,end in [

("Train",1980,2008),

("Validation",2009,2011),

("Test",2012,2014)

]:

    d=df[

        (df["Date"].dt.year>=start) &
        (df["Date"].dt.year<=end)

    ]

    split_stats.append({

        "Dataset":name,

        "Samples":len(d),

        "Flood_Events":int(d["Flood_Event"].sum()),

        "Flood_Percentage(%)":

        round(100*d["Flood_Event"].mean(),3)

    })

split_stats=pd.DataFrame(split_stats)

print()
print(split_stats)

# ------------------------------------------------------------
# Basin statistics
# ------------------------------------------------------------

basin_stats=(

df.groupby("gauge_id")

.agg(

Samples=("Flood_Event","size"),

Flood_Events=("Flood_Event","sum")

)

.reset_index()

)

basin_stats["Flood_Percentage(%)"] = (
    100 * basin_stats["Flood_Events"] / basin_stats["Samples"]
).round(3)
# ------------------------------------------------------------
# Remove temporary threshold column
# ------------------------------------------------------------

df.drop(

columns=["Q95"],

inplace=True

)

# ------------------------------------------------------------
# Save dataset
# ------------------------------------------------------------

output_dataset=os.path.join(

output_folder,

"Flood_ML_Dataset.csv"

)

df.to_csv(

output_dataset,

index=False

)

overall.to_csv(

os.path.join(

output_folder,

"Flood_Event_Overall_Statistics.csv"

),

index=False

)

split_stats.to_csv(

os.path.join(

output_folder,

"Flood_Event_Split_Statistics.csv"

),

index=False

)

basin_stats.to_csv(

os.path.join(

output_folder,

"Flood_Event_Basin_Statistics.csv"

),

index=False

)

print("="*70)
print("FILES SAVED")
print("="*70)

print("1. Flood_ML_Dataset.csv")
print("2. Flood_Event_Overall_Statistics.csv")
print("3. Flood_Event_Split_Statistics.csv")
print("4. Flood_Event_Basin_Statistics.csv")

print("="*70)
print("STEP 7 COMPLETED SUCCESSFULLY")
print("="*70)

import os
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],

    # Font sizes
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,

    # Font weights
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",

    # Line widths
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,

    # Figure quality
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05
})

pivot = event_counts.pivot(

index="Threshold",

columns="Dataset",

values="Flood_Events"

)

pivot.to_csv(

os.path.join(
output_folder,
"Flood_Event_Counts_Pivot.csv"
)

)

pivot.plot(

kind="bar",
figsize=(7,5)

)

plt.ylabel("Number of Flood Events")

plt.title("Flood Events Using Train-Derived Thresholds")

plt.tight_layout()

plt.savefig(

os.path.join(
output_folder,
"Flood_Event_Counts.png"
),

dpi=300

)

plt.close()

import os
import matplotlib.pyplot as plt

pivot = event_counts.pivot(
    index="Threshold",
    columns="Dataset",
    values="Flood_Events"
)

pivot.to_csv(
    os.path.join(
        output_folder,
        "Flood_Event_Counts_Pivot.csv"
    )
)

# Set overall font sizes
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15
})

ax = pivot.plot(
    kind="bar",
    figsize=(9, 6),
    width=0.8
)

ax.set_ylabel("Number of Flood Events", fontsize=15)
ax.set_xlabel("Threshold", fontsize=15)

ax.set_title(
    "Flood Events Using Train-Derived Thresholds",
    fontsize=18,
    pad=15
)

# Set font size for the x and y axis tick labels
ax.tick_params(
    axis="both",
    labelsize=14
)

# Set legend font size
ax.legend(
    fontsize=13,
    title_fontsize=14
)

plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "Flood_Event_Counts.png"
    ),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(7.2, 5.0))

ax.hist(
    thresholds["Q90"],
    bins=20,
    alpha=0.55,
    edgecolor="black",
    linewidth=0.5,
    label="Q90"
)

ax.hist(
    thresholds["Q95"],
    bins=20,
    alpha=0.55,
    edgecolor="black",
    linewidth=0.5,
    label="Q95"
)

ax.hist(
    thresholds["Q98"],
    bins=20,
    alpha=0.55,
    edgecolor="black",
    linewidth=0.5,
    label="Q98"
)

ax.hist(
    thresholds["Q99"],
    bins=20,
    alpha=0.55,
    edgecolor="black",
    linewidth=0.5,
    label="Q99"
)

ax.set_xlabel(
    "Threshold Discharge",
    fontweight="bold"
)

ax.set_ylabel(
    "Number of Catchments",
    fontweight="bold"
)

ax.set_title(
    "Distribution of Basin-Specific Flood Thresholds",
    fontweight="bold"
)

legend = ax.legend(frameon=False)

for txt in legend.get_texts():
    txt.set_fontweight("bold")

for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight("bold")

ax.tick_params(
    axis="both",
    which="major",
    width=1.2,
    length=5
)

fig.tight_layout()

fig.savefig(
    os.path.join(
        output_folder,
        "Threshold_Edited_Distributions.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)



from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib



output_folder = "./flood_project_workspace/Results/ML_Training_Input"

os.makedirs(
    output_folder,
    exist_ok=True
)
# ------------------------------------------------------------
# Load target labels from updated Flood_ML_Dataset.csv
# ------------------------------------------------------------

dataset_path = "./flood_project_workspace/Results/ML_Features/Flood_ML_Dataset.csv"

df = pd.read_csv(dataset_path)

df["Date"] = pd.to_datetime(df["Date"])

# Chronological split
train_df = df[
    (df["Date"] >= "1980-01-01") &
    (df["Date"] <= "2008-12-31")
].copy()

valid_df = df[
    (df["Date"] >= "2009-01-01") &
    (df["Date"] <= "2011-12-31")
].copy()

test_df = df[
    (df["Date"] >= "2012-01-01") &
    (df["Date"] <= "2014-12-31")
].copy()

# Target labels
y_train = train_df["Flood_Event"].to_numpy()

y_valid = valid_df["Flood_Event"].to_numpy()

y_test = test_df["Flood_Event"].to_numpy()

y_train = pd.DataFrame(y_train, columns=["Flood_Event"])

y_valid = pd.DataFrame(y_valid, columns=["Flood_Event"])

y_test = pd.DataFrame(y_test, columns=["Flood_Event"])

print("Target sizes")
print("Train:", len(y_train))
print("Validation:", len(y_valid))
print("Test:", len(y_test))

drop_columns = [

    "Flood_Event",
    "Q",
    "gauge_id",
    "Date"

]


feature_columns = [

    c for c in train_df.columns

    if c not in drop_columns

]


print("-"*70)
print("Feature definition")
print("-"*70)

print("Number of features:",
      len(feature_columns))


# ------------------------------------------------------------
# Create X and y
# ------------------------------------------------------------

X_train = train_df[feature_columns]


X_valid = valid_df[feature_columns]


X_test = test_df[feature_columns]



# ------------------------------------------------------------
# Feature Scaling
# Only TRAIN is used for fitting
# ------------------------------------------------------------

print("-"*70)
print("Feature scaling")
print("-"*70)


scaler = StandardScaler()
joblib.dump(
    scaler,
    "./flood_project_workspace/Results/ML_Training_Input/StandardScaler.pkl"
)

X_train_scaled = scaler.fit_transform(
    X_train
)


X_valid_scaled = scaler.transform(
    X_valid
)


X_test_scaled = scaler.transform(
    X_test
)



# ------------------------------------------------------------
# Convert back to dataframe
# ------------------------------------------------------------

X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=feature_columns
)


X_valid_scaled = pd.DataFrame(
    X_valid_scaled,
    columns=feature_columns
)


X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=feature_columns
)



# ------------------------------------------------------------
# Save datasets
# ------------------------------------------------------------

X_train_scaled.to_csv(
    output_folder+"/X_train_scaled.csv",
    index=False
)


X_valid_scaled.to_csv(
    output_folder+"/X_validation_scaled.csv",
    index=False
)


X_test_scaled.to_csv(
    output_folder+"/X_test_scaled.csv",
    index=False
)


y_train.to_csv(
    output_folder+"/y_train.csv",
    index=False
)


y_valid.to_csv(
    output_folder+"/y_validation.csv",
    index=False
)


y_test.to_csv(
    output_folder+"/y_test.csv",
    index=False)


train_df.to_csv(
    output_folder+"/Temporal_Train.csv",
    index=False)
test_df.to_csv(
    output_folder+"/Temporal_Test.csv",
    index=False)
valid_df.to_csv(
    output_folder+"/Temporal_Validation.csv",
    index=False
)

train_df_scaled = pd.concat([X_train_scaled,train_df[drop_columns],
y_train])

test_df_scaled = pd.concat([X_test_scaled,test_df[drop_columns],
y_test])

train_df_scaled.to_csv(
    output_folder+"/Temporal_Train_scaled.csv",
    index=False
)

test_df_scaled.to_csv(
    output_folder+"/Temporal_Test_scaled.csv",
    index=False
)

