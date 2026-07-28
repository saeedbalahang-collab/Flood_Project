import os
import zipfile
import pandas as pd
import numpy as np
print("-"*70)
print("CHECKING STREAMFLOW DATA QUALITY FOR ALL CAMELS BASINS")
print("-"*70)
# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
import requests
url = "https://github.com/saeedbalahang-collab/Flood_Project/releases/download/v1.0.0/CAMELS_Streamflow.zip"
response = requests.get(url)
with open("CAMELS_Streamflow.zip", "wb") as f:
    f.write(response.content)
print("Downloaded!")
import zipfile
with zipfile.ZipFile("CAMELS_Streamflow.zip", 'r') as zip_ref:
    zip_ref.extractall("CAMELS_Streamflow")

print("ZIP extracted successfully")

EXTRACT_DIR = "/content/CAMELS_Streamflow"
# ------------------------------------------------------------
# Find txt files
# ------------------------------------------------------------

txt_files = []

for root, dirs, files in os.walk(EXTRACT_DIR):

    for f in files:

        if f.lower().endswith(".txt"):

            txt_files.append(
                os.path.join(root, f)
            )


print(f"Total txt files found: {len(txt_files)}")


# ------------------------------------------------------------
# Complete CAMELS-US daily period
# ------------------------------------------------------------

FULL_DATES = pd.date_range(
    start="1980-10-01",
    end="2014-09-30",
    freq="D"
)


EXPECTED_RECORDS = len(FULL_DATES)

print(
    f"Expected records per basin: {EXPECTED_RECORDS}"
)


# ------------------------------------------------------------
# Quality assessment
# ------------------------------------------------------------

summary = []


for file in txt_files:

    try:

        gauge_id = os.path.basename(file).split(".")[0]


        # -----------------------------
        # Read streamflow file
        # -----------------------------

        df = pd.read_csv(
            file,
            sep=r"\s+",
            header=None,
            names=[
                "gauge_id",
                "year",
                "month",
                "day",
                "Q",
                "Flag"
            ]
        )


        # -----------------------------
        # Create date column
        # -----------------------------

        df["Date"] = pd.to_datetime(
            df[["year","month","day"]]
        )


        # Keep only required columns

        df = df[
            [
                "Date",
                "Q"
            ]
        ]


        # -----------------------------
        # Create complete time series
        # -----------------------------

        complete_df = pd.DataFrame(
            {
                "Date": FULL_DATES
            }
        )


        complete_df = complete_df.merge(
            df,
            on="Date",
            how="left"
        )


        # -----------------------------
        # Identify missing values
        # -----------------------------

        missing_mask = (
            complete_df["Q"].isna()
            |
            (complete_df["Q"] == -999)
        )


        missing_records = (
            missing_mask.sum()
        )


        valid_records = (
            EXPECTED_RECORDS -
            missing_records
        )


        missing_percent = (
            100 *
            missing_records /
            EXPECTED_RECORDS
        )



        ################
        # ------------------------------------------------------------
        # 2012-2014 Quality Assessment
        # ------------------------------------------------------------

        MODEL_DATES = pd.date_range(
            start="2012-01-01",
            end="2014-12-31",
            freq="D"
        )

        MODEL_EXPECTED_RECORDS = len(MODEL_DATES)


        model_df = pd.DataFrame(
            {
                "Date": MODEL_DATES
            }
        )


        model_df = model_df.merge(
            df,
            on="Date",
            how="left"
        )


        model_missing_mask = (
            model_df["Q"].isna()
            |
            (model_df["Q"] == -999)
        )


        model_missing_records = (
            model_missing_mask.sum()
        )


        model_valid_records = (
            MODEL_EXPECTED_RECORDS -
            model_missing_records
        )


        model_missing_percent = (
            100 *
            model_missing_records /
            MODEL_EXPECTED_RECORDS
        )



        # -----------------------------
        # Save results
        # -----------------------------
        summary.append({

          "gauge_id": gauge_id,

          "Expected_Records":
              EXPECTED_RECORDS,

          "Available_File_Records":
              len(df),

          "Valid_Records":
              valid_records,

          "Missing_Records":
              missing_records,

          "Missing_Percent":
              round(
                  missing_percent,
                  3
              ),

          "2012_2014_Expected_Records":
              MODEL_EXPECTED_RECORDS,

          "2012_2014_Valid_Records":
              model_valid_records,

          "2012_2014_Missing_Records":
              model_missing_records,

          "2012_2014_Missing_Percent":
              round(
                  model_missing_percent,
                  3
              )

      })



    except Exception as e:

        print(
            f"Error processing {file}: {e}"
        )



# ------------------------------------------------------------
# Create dataframe
# ------------------------------------------------------------

quality_df = pd.DataFrame(summary)


quality_df = quality_df.sort_values(
    "Missing_Percent",
    ascending=False
)


# ------------------------------------------------------------
# Threshold summaries
# ------------------------------------------------------------

under10 = quality_df[
    quality_df["Missing_Percent"] < 10
]


under20 = quality_df[
    quality_df["Missing_Percent"] < 20
]


print("-"*70)
print("SUMMARY")
print("-"*70)


print(
    f"Total gauges: {len(quality_df)}"
)


print(
    f"<10% missing: {len(under10)}"
)


print(
    f"<20% missing: {len(under20)}"
)


print(
    f">=20% missing: "
    f"{(quality_df['Missing_Percent']>=20).sum()}"
)

display(
    quality_df.head(20)
)



# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

OUTPUT_DIR = (
    "/content/Results/Streamflow_Quality"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


quality_df.to_csv(
    f"{OUTPUT_DIR}/All_Gauge_Streamflow_Quality.csv",
    index=False
)


under10.to_csv(
    f"{OUTPUT_DIR}/Gauge_LessThan10PercentMissing.csv",
    index=False
)


under20.to_csv(
    f"{OUTPUT_DIR}/Gauge_LessThan20PercentMissing.csv",
    index=False
)
print("-"*70)
print("FILES SAVED")
print("-"*70)

print(OUTPUT_DIR)


# ============================================================
# CELL: Match Quality File with CAMELS Attributes
#
# Quality criteria:
# 1) Full CAMELS period (1980-2014): Missing < 20%
# 2) Test period (2012-2014): Missing < 10%
#
# Remove gauges not available in CAMELS attributes
#
# ============================================================

import pandas as pd
import os


print("-"*70)
print("FILTERING CAMELS BASINS BASED ON STREAMFLOW QUALITY")
print("-"*70)


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

QUALITY_FILE = (
    "/content/"
    "All_Gauge_Streamflow_Quality (1).csv"
)

ATTRIBUTE_FILE = (
    "/content/Data.csv"
)


OUTPUT_DIR = (
    "/content/Results/CAMELS_Filtered"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)



# ------------------------------------------------------------
# Read files
# ------------------------------------------------------------

quality_df = pd.read_csv(
    QUALITY_FILE,
    dtype={
        "gauge_id": str
    }
)


data_df = pd.read_csv(
    ATTRIBUTE_FILE,
    dtype={
        "gauge_id": str
    }
)



print(
    f"Quality file basins: {len(quality_df)}"
)

print(
    f"Attribute file basins: {len(data_df)}"
)



# ------------------------------------------------------------
# Ensure gauge_id format
# ------------------------------------------------------------

quality_df["gauge_id"] = (
    quality_df["gauge_id"]
    .astype(str)
    .str.zfill(7)
)


data_df["gauge_id"] = (
    data_df["gauge_id"]
    .astype(str)
    .str.zfill(7)
)



# ------------------------------------------------------------
# Match quality file with CAMELS attributes
# Remove 3 gauges without attributes
# ------------------------------------------------------------

quality_matched = quality_df[
    quality_df["gauge_id"].isin(
        data_df["gauge_id"]
    )
].copy()



print("-"*70)

print(
    f"Quality gauges after CAMELS matching: "
    f"{len(quality_matched)}"
)



# ------------------------------------------------------------
# Quality filtering
#
# Full period:
# Missing_Percent < 20%
#
# Test period:
# 2012-2014 Missing_Percent < 10%
# ------------------------------------------------------------

MAX_FULL_PERIOD_MISSING = 20

MAX_TEST_PERIOD_MISSING = 10



quality_filtered = quality_matched[

    (quality_matched["Missing_Percent"]
     < MAX_FULL_PERIOD_MISSING)

    &

    (quality_matched["2012_2014_Missing_Percent"]
     < MAX_TEST_PERIOD_MISSING)

].copy()



print(
    f"Basins after quality filtering: "
    f"{len(quality_filtered)}"
)



# ------------------------------------------------------------
# Extract final attribute basins
# ------------------------------------------------------------

final_basins = data_df[
    data_df["gauge_id"].isin(
        quality_filtered["gauge_id"]
    )
].copy()



print("-"*70)

print(
    f"Final basins for clustering: "
    f"{len(final_basins)}"
)



# ------------------------------------------------------------
# Consistency check
# ------------------------------------------------------------

check = (
    set(final_basins["gauge_id"])
    ==
    set(quality_filtered["gauge_id"])
)


print(
    f"Gauge consistency check: {check}"
)



# ------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------

quality_filtered.to_csv(
    f"{OUTPUT_DIR}/Quality_Filtered_Gauges.csv",
    index=False
)


final_basins.to_csv(
    f"{OUTPUT_DIR}/Final_CAMELS_Basins_For_Clustering.csv",
    index=False
)



print("-"*70)

print("FILES SAVED")

print("-"*70)

print(OUTPUT_DIR)

# ============================================================
# Cell 0: Project Configuration
#
# Purpose:
#   Define all project parameters and create output folders
#
# This cell controls:
#   - Data path
#   - Output structure
#   - Statistical thresholds
#   - Clustering settings
#   - Reproducibility
#
# ============================================================


import os
import random
import numpy as np


# -----------------------------
# Random Seed (Reproducibility)
# -----------------------------

RANDOM_STATE = 42

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)



# -----------------------------
# File Paths
# -----------------------------

# Upload Data.csv to Colab first
DATA_PATH = "/content/Results/CAMELS_Filtered/Final_CAMELS_Basins_For_Clustering.csv"


# Main output folder
OUTPUT_DIR = "/content/Results"





# -----------------------------
# Create Output Structure
# -----------------------------

folders = [

    OUTPUT_DIR,

    f"{OUTPUT_DIR}/01_Data_Quality",
    f"{OUTPUT_DIR}/02_Missing_Values",
    f"{OUTPUT_DIR}/03_Outliers",
    f"{OUTPUT_DIR}/04_Correlation",
    f"{OUTPUT_DIR}/05_VIF",
    f"{OUTPUT_DIR}/06_Scaling",
    f"{OUTPUT_DIR}/07_PCA",
    f"{OUTPUT_DIR}/08_Cluster_Selection",
    f"{OUTPUT_DIR}/09_Final_Clustering",
    f"{OUTPUT_DIR}/10_Validation",
    f"{OUTPUT_DIR}/11_Final_Selected_Basins",
    f"{OUTPUT_DIR}/12_Final_Validation",
    f"{OUTPUT_DIR}/13_Transfer_Basins",

]

for folder in folders:
    os.makedirs(folder, exist_ok=True)



# -----------------------------
# Column Definitions
# -----------------------------


# Identification columns
ID_COLUMNS = [
    "gauge_id",
    "gauge_lat",
    "gauge_lon"
]


# Features used for clustering
FEATURE_COLUMNS = [

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

    "runoff_ratio",
    "baseflow_index",
    "stream_elas",

    "carbonate_rocks_frac",
    "geol_permeability"

]



# -----------------------------
# Statistical Parameters
# -----------------------------


# Correlation threshold
CORRELATION_THRESHOLD = 0.90


# VIF threshold
VIF_THRESHOLD = 10



# Missing value thresholds

MISSING_LOW = 0.05
MISSING_HIGH = 0.20



# -----------------------------
# PCA Settings
# -----------------------------

PCA_VARIANCE_THRESHOLD = 0.90



# -----------------------------
# Clustering Settings
# -----------------------------


# Search range for optimal clusters

MIN_CLUSTERS = 2
MAX_CLUSTERS = 20



# KMeans optimization parameters

KMEANS_INIT_OPTIONS = [
    "k-means++",
    "random"
]


KMEANS_N_INIT_OPTIONS = [
    10,
    50,
    100
]


KMEANS_MAX_ITER_OPTIONS = [
    300,
    500,
    1000
]



# -----------------------------
# Basin Selection
# -----------------------------


NUMBER_OF_SELECTED_BASINS = 40
SELECTION_METHOD = "proportional"
ENABLE_RANDOM_BASELINE = True
RANDOM_SELECTION_REPEATS = 100



print("Configuration loaded successfully.")
print(f"Data path: {DATA_PATH}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Number of clustering features: {len(FEATURE_COLUMNS)}")

"""Cell 1 — Import Libraries"""

# ============================================================
# Cell 1: Import Required Libraries
#
# Purpose:
#   Import all scientific computing and visualization packages
#
# ============================================================


# -----------------------------
# Data manipulation
# -----------------------------

import pandas as pd
import numpy as np



# -----------------------------
# Visualization
# -----------------------------

import matplotlib.pyplot as plt
import seaborn as sns



# -----------------------------
# Statistics
# -----------------------------

from scipy import stats



# -----------------------------
# Machine Learning
# -----------------------------

from sklearn.preprocessing import StandardScaler

from sklearn.impute import (
    SimpleImputer,
    KNNImputer
)

from sklearn.decomposition import PCA


from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering
)


from sklearn.metrics import (

    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score

)



from sklearn.metrics.pairwise import (

    cosine_similarity

)



# -----------------------------
# VIF Calculation
# -----------------------------

from statsmodels.stats.outliers_influence import variance_inflation_factor



# -----------------------------
# Optimization
# -----------------------------

from itertools import product



# -----------------------------
# File Handling
# -----------------------------

import warnings

warnings.filterwarnings("ignore")



print("All libraries imported successfully.")

"""Cell 2 — Load Data + Initial Validation"""

# ============================================================
# Cell 2: Load Dataset and Initial Validation
#
# Purpose:
#   - Load Data.csv
#   - Check dimensions
#   - Verify required columns
#   - Check data types
#   - Detect duplicate basin IDs
#
# Output:
#   Results/01_Data_Quality/
#
# ============================================================


# -----------------------------
# Load dataset
# -----------------------------

df = pd.read_csv(DATA_PATH)



# -----------------------------
# Basic information
# -----------------------------

print("Dataset loaded successfully.")
print("--------------------------------")

print(f"Number of basins : {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")



# -----------------------------
# Display first rows
# -----------------------------

display(df.head())



# -----------------------------
# Check required columns
# -----------------------------


required_columns = ID_COLUMNS + FEATURE_COLUMNS


missing_columns = [

    col for col in required_columns
    if col not in df.columns

]


if len(missing_columns) == 0:

    print("✓ All required columns are available.")

else:

    print("Missing columns:")
    print(missing_columns)



# -----------------------------
# Data types
# -----------------------------

dtype_report = pd.DataFrame({

    "Column": df.columns,

    "Data_Type": df.dtypes.astype(str)

})


dtype_report.to_csv(

    f"{OUTPUT_DIR}/01_Data_Quality/Data_Types_Report.csv",

    index=False

)



display(dtype_report)



# -----------------------------
# Duplicate basin IDs
# -----------------------------


duplicate_ids = df[

    df["gauge_id"].duplicated(keep=False)

]


print("--------------------------------")

print(

    f"Duplicate gauge_id count: {duplicate_ids.shape[0]}"

)


duplicate_ids.to_csv(

    f"{OUTPUT_DIR}/01_Data_Quality/Duplicate_Gauge_IDs.csv",

    index=False

)



# -----------------------------
# Basic statistical summary
# -----------------------------


summary = df[FEATURE_COLUMNS].describe().T


summary.to_csv(

    f"{OUTPUT_DIR}/01_Data_Quality/Feature_Descriptive_Statistics.csv"

)


display(summary)



print("--------------------------------")
print("Cell 2 completed successfully.")

"""Cell 3 — Data Quality Assessment (Missing Values)"""

# ============================================================
# Cell 3: Missing Value Analysis
#
# Purpose:
#   Quantify missing values before preprocessing
#
# Output:
#   - Missing value report
#   - Missing value plot
#
# ============================================================



# -----------------------------
# Missing value calculation
# -----------------------------


missing_report = pd.DataFrame({

    "Missing_Count": df.isnull().sum(),

    "Missing_Percentage":
        (df.isnull().sum() / len(df)) * 100

})


missing_report = (

    missing_report

    .sort_values(

        by="Missing_Percentage",

        ascending=False

    )

)



# Save report

missing_report.to_csv(

    f"{OUTPUT_DIR}/02_Missing_Values/Missing_Value_Report.csv"

)



display(missing_report)



# -----------------------------
# Plot missing values
# -----------------------------


plt.figure(figsize=(12,6))


sns.barplot(

    x=missing_report.index,

    y=missing_report["Missing_Percentage"]

)


plt.xticks(

    rotation=90

)


plt.ylabel(

    "Missing Percentage (%)"

)


plt.xlabel(

    "Feature"

)


plt.title(

    "Percentage of Missing Values Across Features"

)


plt.tight_layout()



plt.savefig(

    f"{OUTPUT_DIR}/02_Missing_Values/Missing_Value_Barplot.png",

    dpi=300

)



plt.show()



# -----------------------------
# Missing values in features only
# -----------------------------


feature_missing = missing_report.loc[

    missing_report.index.isin(FEATURE_COLUMNS)

]


feature_missing.to_csv(

    f"{OUTPUT_DIR}/02_Missing_Values/Feature_Missing_Report.csv"

)



print("--------------------------------")
print("Cell 3 completed successfully.")

"""Cell 4 — Missing Value Investigation + Imputation"""

# ============================================================
# Cell 4: Missing Value Investigation and Imputation
#
# Purpose:
#   - Identify basins containing missing values
#   - Apply justified imputation
#   - Save cleaned dataset
#
# ============================================================


# -----------------------------
# Identify rows with missing values
# -----------------------------

missing_rows = df[

    df[FEATURE_COLUMNS]
    .isnull()
    .any(axis=1)

]


print("Basins containing missing values:")
display(missing_rows)



# Save missing basin report

missing_rows.to_csv(

    f"{OUTPUT_DIR}/02_Missing_Values/Basins_With_Missing_Values.csv",

    index=False

)



# -----------------------------
# Median Imputation
# -----------------------------


df_clean = df.copy()


for col in FEATURE_COLUMNS:

    if df_clean[col].isnull().sum() > 0:

        median_value = df_clean[col].median()

        df_clean[col] = df_clean[col].fillna(median_value)

        print(
            f"{col}: missing values replaced by median = {median_value}"
        )



# -----------------------------
# Verify no missing values remain
# -----------------------------


remaining_missing = (

    df_clean[FEATURE_COLUMNS]
    .isnull()
    .sum()

)


print("\nRemaining missing values:")
display(

    remaining_missing[
        remaining_missing > 0
    ]

)



# -----------------------------
# Save cleaned dataset
# -----------------------------


df_clean.to_csv(

    f"{OUTPUT_DIR}/02_Missing_Values/Data_Cleaned.csv",

    index=False

)



print("--------------------------------")
print("Missing value treatment completed.")

"""Cell 5 — Outlier Detection"""

# ============================================================
# Cell 5: Outlier Detection
#
# Purpose:
#   Detect unusual catchments without removing them
#
# Methods:
#   1) IQR method
#   2) Z-score method
#   3) Isolation Forest
#
# Output:
#   Results/03_Outliers/
#
# ============================================================


from sklearn.ensemble import IsolationForest



# ------------------------------------------------------------
# Prepare feature dataset
# ------------------------------------------------------------

X_features = df_clean[FEATURE_COLUMNS].copy()



# ------------------------------------------------------------
# Method 1: IQR Outlier Detection
# ------------------------------------------------------------


iqr_results = []


for feature in FEATURE_COLUMNS:

    Q1 = X_features[feature].quantile(0.25)

    Q3 = X_features[feature].quantile(0.75)

    IQR = Q3 - Q1


    lower_bound = Q1 - 1.5 * IQR

    upper_bound = Q3 + 1.5 * IQR


    outlier_indices = X_features[

        (X_features[feature] < lower_bound) |

        (X_features[feature] > upper_bound)

    ].index


    for idx in outlier_indices:

        iqr_results.append({

            "Index": idx,

            "Gauge_ID": df_clean.loc[idx,"gauge_id"],

            "Feature": feature,

            "Value": X_features.loc[idx,feature],

            "Lower_Bound": lower_bound,

            "Upper_Bound": upper_bound

        })



iqr_report = pd.DataFrame(iqr_results)



iqr_report.to_csv(

    f"{OUTPUT_DIR}/03_Outliers/IQR_Outlier_Report.csv",

    index=False

)



print(
    f"IQR detected {len(iqr_report)} unusual observations."
)



# ------------------------------------------------------------
# Method 2: Z-score Detection
# ------------------------------------------------------------


zscore_results = []


for feature in FEATURE_COLUMNS:


    z_scores = np.abs(

        stats.zscore(

            X_features[feature],

            nan_policy="omit"

        )

    )


    outlier_indices = np.where(

        z_scores > 3

    )[0]


    for idx in outlier_indices:


        zscore_results.append({

            "Index": idx,

            "Gauge_ID": df_clean.loc[idx,"gauge_id"],

            "Feature": feature,

            "Value": X_features.loc[idx,feature],

            "Z_score": z_scores[idx]

        })



zscore_report = pd.DataFrame(zscore_results)



zscore_report.to_csv(

    f"{OUTPUT_DIR}/03_Outliers/Zscore_Outlier_Report.csv",

    index=False

)



print(

    f"Z-score detected {len(zscore_report)} unusual observations."

)



# ------------------------------------------------------------
# Method 3: Isolation Forest
# ------------------------------------------------------------


# Standardize first

scaler_outlier = StandardScaler()


X_scaled_outlier = scaler_outlier.fit_transform(

    X_features

)



iso_forest = IsolationForest(

    contamination="auto",

    random_state=RANDOM_STATE

)



iso_labels = iso_forest.fit_predict(

    X_scaled_outlier

)



iso_scores = iso_forest.decision_function(

    X_scaled_outlier

)



isolation_report = pd.DataFrame({

    "Gauge_ID": df_clean["gauge_id"],

    "Isolation_Label": iso_labels,

    "Isolation_Score": iso_scores

})



# -1 means anomaly

isolation_report["Is_Outlier"] = (

    isolation_report["Isolation_Label"] == -1

)



isolation_report.to_csv(

    f"{OUTPUT_DIR}/03_Outliers/IsolationForest_Outliers.csv",

    index=False

)



print(

    f"Isolation Forest detected {isolation_report['Is_Outlier'].sum()} basins."

)



# ------------------------------------------------------------
# Summary Report
# ------------------------------------------------------------


outlier_summary = pd.DataFrame({

    "Method":[

        "IQR",

        "Z-score",

        "Isolation Forest"

    ],

    "Number_of_Detected_Outliers":[

        len(iqr_report),

        len(zscore_report),

        isolation_report["Is_Outlier"].sum()

    ]

})


outlier_summary.to_csv(

    f"{OUTPUT_DIR}/03_Outliers/Outlier_Summary.csv",

    index=False

)



display(outlier_summary)



print("--------------------------------")
print("Cell 5 completed successfully.")

# ============================================================
# Cell 6: Correlation Analysis
#
# Purpose:
#   Identify highly correlated features before clustering
#
# Method:
#   Pearson correlation coefficient
#
# Outputs:
#   - Correlation matrix
#   - Correlation heatmap
#   - Highly correlated feature pairs
#
# Note:
#   No feature is removed in this stage.
#
# ============================================================


# ------------------------------------------------------------
# Prepare feature matrix
# ------------------------------------------------------------

X_corr = df_clean[FEATURE_COLUMNS].copy()



# ------------------------------------------------------------
# Pearson correlation matrix
# ------------------------------------------------------------

corr_matrix = X_corr.corr(method="pearson")



# Save correlation matrix

corr_matrix.to_csv(

    f"{OUTPUT_DIR}/04_Correlation/Correlation_Matrix.csv"

)



# ------------------------------------------------------------
# Correlation Heatmap
# ------------------------------------------------------------


plt.figure(figsize=(16,12))


sns.heatmap(

    corr_matrix,

    cmap="coolwarm",

    center=0,

    linewidths=0.5,

    square=True

)


plt.title(

    "Pearson Correlation Matrix of Catchment Attributes",

    fontsize=14

)


plt.tight_layout()



plt.savefig(

    f"{OUTPUT_DIR}/04_Correlation/Correlation_Heatmap.png",

    dpi=300,

    bbox_inches="tight"

)


plt.show()



# ------------------------------------------------------------
# Extract highly correlated pairs
# ------------------------------------------------------------


correlation_pairs = []


for i in range(len(corr_matrix.columns)):

    for j in range(i+1, len(corr_matrix.columns)):


        feature_1 = corr_matrix.columns[i]

        feature_2 = corr_matrix.columns[j]


        correlation_value = corr_matrix.iloc[i,j]


        if abs(correlation_value) >= CORRELATION_THRESHOLD:


            correlation_pairs.append({

                "Feature_1": feature_1,

                "Feature_2": feature_2,

                "Correlation": correlation_value

            })



high_corr_df = pd.DataFrame(

    correlation_pairs

)



# Sort by absolute correlation

if len(high_corr_df) > 0:

    high_corr_df["Absolute_Correlation"] = (

        high_corr_df["Correlation"].abs()

    )


    high_corr_df = high_corr_df.sort_values(

        by="Absolute_Correlation",

        ascending=False

    )



# Save report

high_corr_df.to_csv(

    f"{OUTPUT_DIR}/04_Correlation/Highly_Correlated_Features.csv",

    index=False

)



# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------


print("--------------------------------")

print(

    f"Number of highly correlated feature pairs "
    f"(|r| >= {CORRELATION_THRESHOLD}): "
    f"{len(high_corr_df)}"

)


if len(high_corr_df) > 0:

    display(high_corr_df)

else:

    print("No highly correlated feature pairs detected.")



print("--------------------------------")
print("Cell 6 completed successfully.")

# ============================================================
# Cell 7: Variance Inflation Factor (VIF) Analysis
#
# Purpose:
#   Detect multicollinearity among features before clustering
#
# Method:
#   Iterative VIF elimination
#
# Rule:
#   Remove feature with highest VIF if VIF > threshold
#
# Output:
#   - Initial VIF
#   - Removed features log
#   - Final VIF
#   - Final feature list
#
# ============================================================


from statsmodels.stats.outliers_influence import variance_inflation_factor



# ------------------------------------------------------------
# Function to calculate VIF
# ------------------------------------------------------------


def calculate_vif(dataframe):

    vif_data = pd.DataFrame()

    vif_data["Feature"] = dataframe.columns


    vif_data["VIF"] = [

        variance_inflation_factor(

            dataframe.values,

            i

        )

        for i in range(dataframe.shape[1])

    ]


    vif_data = vif_data.sort_values(

        by="VIF",

        ascending=False

    )


    return vif_data




# ------------------------------------------------------------
# Prepare initial feature dataset
# ------------------------------------------------------------


X_vif = df_clean[FEATURE_COLUMNS].copy()



# ------------------------------------------------------------
# Standardize before VIF
# ------------------------------------------------------------

scaler_vif = StandardScaler()


X_vif_scaled = pd.DataFrame(

    scaler_vif.fit_transform(X_vif),

    columns=X_vif.columns

)



# ------------------------------------------------------------
# Initial VIF
# ------------------------------------------------------------


initial_vif = calculate_vif(

    X_vif_scaled

)


initial_vif.to_csv(

    f"{OUTPUT_DIR}/05_VIF/Initial_VIF.csv",

    index=False

)


print("Initial VIF:")
display(initial_vif)



# ------------------------------------------------------------
# Iterative VIF elimination
# ------------------------------------------------------------


features_remaining = list(

    X_vif_scaled.columns

)


removed_features = []


iteration = 1



while True:


    vif_current = calculate_vif(

        X_vif_scaled[features_remaining]

    )


    max_vif = vif_current.iloc[0]["VIF"]

    max_feature = vif_current.iloc[0]["Feature"]



    print(

        f"Iteration {iteration}: "

        f"Maximum VIF = {max_vif:.2f} "

        f"({max_feature})"

    )



    if max_vif <= VIF_THRESHOLD:

        break



    removed_features.append({

        "Iteration": iteration,

        "Removed_Feature": max_feature,

        "VIF_Value": max_vif

    })


    features_remaining.remove(

        max_feature

    )


    iteration += 1




# ------------------------------------------------------------
# Save removed features log
# ------------------------------------------------------------


removed_df = pd.DataFrame(

    removed_features

)


removed_df.to_csv(

    f"{OUTPUT_DIR}/05_VIF/Removed_Features.csv",

    index=False

)



# ------------------------------------------------------------
# Final VIF
# ------------------------------------------------------------


final_vif = calculate_vif(

    X_vif_scaled[features_remaining]

)



final_vif.to_csv(

    f"{OUTPUT_DIR}/05_VIF/Final_VIF.csv",

    index=False

)



# ------------------------------------------------------------
# Final Feature List
# ------------------------------------------------------------


final_feature_list = pd.DataFrame({

    "Selected_Features": features_remaining

})


final_feature_list.to_csv(

    f"{OUTPUT_DIR}/05_VIF/Final_Feature_List.csv",

    index=False

)



# ------------------------------------------------------------
# Decision log
# ------------------------------------------------------------


with open(

    f"{OUTPUT_DIR}/05_VIF/VIF_Decision_Log.txt",

    "w"

) as file:


    file.write(

        "VIF Feature Selection Log\n"

    )

    file.write(

        "=========================\n\n"

    )


    if len(removed_features)==0:


        file.write(

            "No features removed.\n"

        )


    else:


        for item in removed_features:


            file.write(

                f"Iteration {item['Iteration']}: "

                f"Removed {item['Removed_Feature']} "

                f"(VIF={item['VIF_Value']:.3f})\n"

            )



# ------------------------------------------------------------
# Final report
# ------------------------------------------------------------


print("--------------------------------")

print(

    f"Initial number of features: "

    f"{len(FEATURE_COLUMNS)}"

)


print(

    f"Final number of features: "

    f"{len(features_remaining)}"

)


print("--------------------------------")


print("Final VIF values:")

display(final_vif)



print("--------------------------------")
print("Cell 7 completed successfully.")

# ============================================================
# Cell 8: Feature Scaling
#
# Purpose:
#   Standardize final selected catchment attributes
#   before PCA and clustering
#
# Method:
#   StandardScaler (zero mean, unit variance)
#
# Input:
#   Final 21 features selected after VIF
#
# Output:
#   - Scaled dataset
#   - Scaling statistics
#
# ============================================================


# ------------------------------------------------------------
# Prepare final feature matrix
# ------------------------------------------------------------


X_final = df_clean[features_remaining].copy()



# Save original final feature dataset

X_final.to_csv(

    f"{OUTPUT_DIR}/06_Scaling/Final_Features_Before_Scaling.csv",

    index=False

)



# ------------------------------------------------------------
# Apply Standard Scaling
# ------------------------------------------------------------


scaler = StandardScaler()



X_scaled_array = scaler.fit_transform(

    X_final

)



X_scaled = pd.DataFrame(

    X_scaled_array,

    columns=features_remaining,

    index=df_clean.index

)



# ------------------------------------------------------------
# Save scaled dataset
# ------------------------------------------------------------


X_scaled.to_csv(

    f"{OUTPUT_DIR}/06_Scaling/Scaled_Features.csv",

    index=False

)



# ------------------------------------------------------------
# Scaling statistics
# ------------------------------------------------------------


scaling_statistics = pd.DataFrame({

    "Feature": features_remaining,

    "Mean_before_scaling": scaler.mean_,

    "Std_before_scaling": scaler.scale_

})



scaling_statistics.to_csv(

    f"{OUTPUT_DIR}/06_Scaling/Scaling_Statistics.csv",

    index=False

)



# ------------------------------------------------------------
# Quality check
# ------------------------------------------------------------


check_statistics = pd.DataFrame({

    "Mean_after_scaling":

        X_scaled.mean(),

    "Std_after_scaling":

        X_scaled.std()

})



print("--------------------------------")

print("Scaling quality check:")

display(check_statistics)



print("--------------------------------")


print(

    "Number of final features:",

    X_scaled.shape[1]

)


print(

    "Number of catchments:",

    X_scaled.shape[0]

)


print("--------------------------------")

print("Cell 8 completed successfully.")

# ============================================================
# Cell 9: Principal Component Analysis (PCA)
#
# Purpose:
#   Analyze effective dimensionality of catchment attributes
#
# Input:
#   Scaled 21 selected features
#
# Outputs:
#   - Explained variance
#   - PCA loadings
#   - PCA scores
#   - Scree plot
#
# ============================================================

from sklearn.decomposition import PCA







# ------------------------------------------------------------
# Apply PCA
# ------------------------------------------------------------


pca = PCA()



pca_scores = pca.fit_transform(

    X_scaled

)



# ------------------------------------------------------------
# Explained variance table
# ------------------------------------------------------------


explained_variance = pd.DataFrame({

    "Component":

        np.arange(1, len(pca.explained_variance_ratio_)+1),

    "Explained_Variance_Ratio":

        pca.explained_variance_ratio_,

    "Cumulative_Variance":

        np.cumsum(

            pca.explained_variance_ratio_

        )

})



explained_variance.to_csv(

    f"{OUTPUT_DIR}/07_PCA/PCA_Explained_Variance.csv",

    index=False

)



display(explained_variance.head(10))



# ------------------------------------------------------------
# Scree plot
# ------------------------------------------------------------


plt.figure(figsize=(8,5))


plt.plot(

    explained_variance["Component"],

    explained_variance["Explained_Variance_Ratio"],

    marker="o"

)


plt.xlabel(

    "Principal Component"

)


plt.ylabel(

    "Explained Variance Ratio"

)


plt.title(

    "PCA Scree Plot"

)


plt.grid(True)


plt.tight_layout()


plt.savefig(

    f"{OUTPUT_DIR}/07_PCA/Scree_Plot.png",

    dpi=300,

    bbox_inches="tight"

)


plt.show()



# ------------------------------------------------------------
# Cumulative variance plot
# ------------------------------------------------------------


plt.figure(figsize=(8,5))


plt.plot(

    explained_variance["Component"],

    explained_variance["Cumulative_Variance"],

    marker="o"

)


plt.axhline(

    y=0.90,

    linestyle="--",

    label="90% variance"

)


plt.axhline(

    y=0.95,

    linestyle="--",

    label="95% variance"

)



plt.xlabel(

    "Number of Principal Components"

)


plt.ylabel(

    "Cumulative Explained Variance"

)


plt.title(

    "Cumulative PCA Explained Variance"

)


plt.legend()


plt.grid(True)


plt.tight_layout()



plt.savefig(

    f"{OUTPUT_DIR}/07_PCA/Cumulative_Variance_Plot.png",

    dpi=300,

    bbox_inches="tight"

)


plt.show()



# ------------------------------------------------------------
# PCA Loadings
# ------------------------------------------------------------


pca_loadings = pd.DataFrame(

    pca.components_.T,

    columns=[

        f"PC{i}"

        for i in range(1, len(pca.components_)+1)

    ],

    index=features_remaining

)



pca_loadings.to_csv(

    f"{OUTPUT_DIR}/07_PCA/PCA_Loadings.csv"

)



# ------------------------------------------------------------
# PCA transformed dataset
# ------------------------------------------------------------


pca_scores_df = pd.DataFrame(

    pca_scores,

    columns=[

        f"PC{i}"

        for i in range(1, pca_scores.shape[1]+1)

    ]

)



# Add basin identifier

pca_scores_df.insert(

    0,

    "gauge_id",

    df_clean["gauge_id"].values

)



pca_scores_df.to_csv(

    f"{OUTPUT_DIR}/07_PCA/PCA_Transformed_Data.csv",

    index=False

)



# ------------------------------------------------------------
# Determine effective number of PCs
# ------------------------------------------------------------


pc90 = np.argmax(

    explained_variance["Cumulative_Variance"] >= 0.90

) + 1



pc95 = np.argmax(

    explained_variance["Cumulative_Variance"] >= 0.95

) + 1



print("--------------------------------")

print(

    f"Number of PCs explaining >=90% variance: {pc90}"

)


print(

    f"Number of PCs explaining >=95% variance: {pc95}"

)


print("--------------------------------")

print("Cell 9 completed successfully.")

# ============================================================
# Cell 10: Optimal Cluster Number Selection
#
# Purpose:
#   Determine optimal K for clustering
#
# Method:
#   KMeans + multiple clustering validation indices
#
# Input:
#   PCA transformed space
#
# Output:
#   Cluster evaluation metrics
#   Optimization plots
#
# ============================================================


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.cluster import KMeans

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)



# ------------------------------------------------------------
# Create output directory
# ------------------------------------------------------------






# ------------------------------------------------------------
# Select PCA dimensions explaining 90%
# ------------------------------------------------------------


n_pca_components = pc90


X_cluster = pca_scores[:, :n_pca_components]



print("--------------------------------")

print(

    "Number of PCA components used:",

    n_pca_components

)

print(

    "Clustering dataset shape:",

    X_cluster.shape

)

print("--------------------------------")



# ------------------------------------------------------------
# Range of K values
# ------------------------------------------------------------


k_range = range(2,61)



results = []



# ------------------------------------------------------------
# Evaluate different K values
# ------------------------------------------------------------


for k in k_range:


    print(f"Testing k={k}")


    kmeans = KMeans(

        n_clusters=k,

        random_state=42,

        n_init=50

    )


    labels = kmeans.fit_predict(

        X_cluster

    )
    from sklearn.metrics import adjusted_rand_score

    stability_scores = []

    for seed in [10,20,30,40,50]:

        km_temp = KMeans(
            n_clusters=k,
            random_state=seed,
            n_init=50
        )

        labels_temp = km_temp.fit_predict(X_cluster)

        stability_scores.append(
            adjusted_rand_score(labels, labels_temp)
        )

    cluster_stability = np.mean(stability_scores)



    inertia = kmeans.inertia_


    silhouette = silhouette_score(

        X_cluster,

        labels

    )


    db_index = davies_bouldin_score(

        X_cluster,

        labels

    )


    ch_index = calinski_harabasz_score(

        X_cluster,

        labels

    )



    results.append({

        "Number_of_Clusters": k,

        "Inertia": inertia,

        "Silhouette_Score": silhouette,

        "Davies_Bouldin_Index": db_index,

        "Calinski_Harabasz_Index": ch_index,
        "Cluster_Stability": cluster_stability

    })



# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------


cluster_results = pd.DataFrame(results)



cluster_results.to_csv(

    f"{OUTPUT_DIR}/08_Cluster_Selection/Cluster_Evaluation.csv",

    index=False

)



display(cluster_results.head())



# ------------------------------------------------------------
# Plot 1: Elbow Method
# ------------------------------------------------------------


plt.figure(figsize=(8,5))


plt.plot(

    cluster_results["Number_of_Clusters"],

    cluster_results["Inertia"],

    marker="o"

)


plt.xlabel("Number of Clusters")

plt.ylabel("Within Cluster Sum of Squares")


plt.title("KMeans Elbow Method")


plt.grid(True)


plt.tight_layout()


plt.savefig(

    f"{OUTPUT_DIR}/08_Cluster_Selection/Elbow_Plot.png",

    dpi=300,

    bbox_inches="tight"

)


plt.show()



# ------------------------------------------------------------
# Plot 2: Silhouette Score
# ------------------------------------------------------------


plt.figure(figsize=(8,5))


plt.plot(

    cluster_results["Number_of_Clusters"],

    cluster_results["Silhouette_Score"],

    marker="o"

)


plt.xlabel("Number of Clusters")

plt.ylabel("Silhouette Score")


plt.title("Silhouette Score vs Number of Clusters")


plt.grid(True)


plt.tight_layout()



plt.savefig(

    f"{OUTPUT_DIR}/08_Cluster_Selection/Silhouette_Plot.png",

    dpi=300,

    bbox_inches="tight"

)


plt.show()



# ------------------------------------------------------------
# Plot 3: Davies-Bouldin
# ------------------------------------------------------------


plt.figure(figsize=(8,5))


plt.plot(

    cluster_results["Number_of_Clusters"],

    cluster_results["Davies_Bouldin_Index"],

    marker="o"

)


plt.xlabel("Number of Clusters")

plt.ylabel("Davies-Bouldin Index")


plt.title("Davies-Bouldin Index vs Number of Clusters")


plt.grid(True)


plt.tight_layout()



plt.savefig(

    f"{OUTPUT_DIR}/08_Cluster_Selection/DB_Index_Plot.png",

    dpi=300,

    bbox_inches="tight"

)


plt.show()



# ------------------------------------------------------------
# Plot 4: Calinski-Harabasz
# ------------------------------------------------------------


plt.figure(figsize=(8,5))


plt.plot(

    cluster_results["Number_of_Clusters"],

    cluster_results["Calinski_Harabasz_Index"],

    marker="o"

)


plt.xlabel("Number of Clusters")

plt.ylabel("Calinski-Harabasz Index")


plt.title("Calinski-Harabasz Index vs Number of Clusters")


plt.grid(True)


plt.tight_layout()



plt.savefig(

    f"{OUTPUT_DIR}/08_Cluster_Selection/CH_Index_Plot.png",

    dpi=300,

    bbox_inches="tight"

)


plt.show()



# ------------------------------------------------------------
# Identify best candidates
# ------------------------------------------------------------


best_silhouette = cluster_results.loc[

    cluster_results["Silhouette_Score"].idxmax()

]


best_db = cluster_results.loc[

    cluster_results["Davies_Bouldin_Index"].idxmin()

]


best_ch = cluster_results.loc[

    cluster_results["Calinski_Harabasz_Index"].idxmax()

]

# ------------------------------------------------------------
# Final K selection
# ------------------------------------------------------------

candidate_clusters = cluster_results[

    (cluster_results["Silhouette_Score"] >=
     cluster_results["Silhouette_Score"].max()*0.95)

]


# ------------------------------------------------------------
# Final K selection based on highest Silhouette
# with stability consideration
# ------------------------------------------------------------

best_final_cluster = cluster_results.loc[
    cluster_results["Silhouette_Score"].idxmax()
]


print("Final selected K:")
display(best_final_cluster)
print("--------------------------------")

print("Best Silhouette K:")

display(best_silhouette)


print("Best Davies-Bouldin K:")

display(best_db)


print("Best Calinski-Harabasz K:")

display(best_ch)



print("--------------------------------")
print(
    "Selected number of clusters:",
    int(best_final_cluster["Number_of_Clusters"])
)
print("Cell 10 completed successfully.")

print(
    "Selected number of clusters:",
    int(best_final_cluster["Number_of_Clusters"])
)

SELECTED_K = int(best_final_cluster["Number_of_Clusters"])

print(
    f"Selected clustering level: K={SELECTED_K}"
)
best_final_cluster.to_csv(
    f"{OUTPUT_DIR}/08_Cluster_Selection/Selected_K_Justification.csv",
    index=False
)

from geopy.distance import geodesic

# ============================================================
# Cell 11: Final Clustering and Selection of 40 Representative Basins
#
# Purpose:
#   1. Apply optimized KMeans (K=14)
#   2. Assign basins to clusters
#   3. Select 40 representative basins
#      considering:
#        - Hydrological similarity
#        - Spatial diversity
#
# ============================================================


import os
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances



# ------------------------------------------------------------
# Output directory
# ------------------------------------------------------------


FINAL_DIR = f"{OUTPUT_DIR}/09_Final_Clustering"



# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------


OPTIMAL_K = 7

TARGET_BASINS = 40



# ------------------------------------------------------------
# Run final KMeans
# ------------------------------------------------------------


kmeans_final = KMeans(

    n_clusters=OPTIMAL_K,

    random_state=42,

    n_init=100

)


cluster_labels = kmeans_final.fit_predict(

    X_cluster

)



print("--------------------------------")

print("Final KMeans completed")

print("Number of clusters:", OPTIMAL_K)

print("--------------------------------")



# ------------------------------------------------------------
# Basin cluster assignment
# ------------------------------------------------------------


cluster_assignment = pd.DataFrame()


cluster_assignment["gauge_id"] = df_clean["gauge_id"].values


cluster_assignment["latitude"] = df_clean["gauge_lat"].values


cluster_assignment["longitude"] = df_clean["gauge_lon"].values


cluster_assignment["Cluster"] = cluster_labels



cluster_assignment.to_csv(

    f"{FINAL_DIR}/Basin_Cluster_Assignment.csv",

    index=False

)



display(cluster_assignment.head())



# ------------------------------------------------------------
# Cluster size summary
# ------------------------------------------------------------


cluster_summary = (

    cluster_assignment

    .groupby("Cluster")

    .size()

    .reset_index(name="Number_of_Basins")

)



cluster_summary["Selection_Target"] = (

    cluster_summary["Number_of_Basins"]

    /

    len(cluster_assignment)

    *

    TARGET_BASINS

)



cluster_summary.to_csv(

    f"{FINAL_DIR}/Cluster_Summary.csv",

    index=False

)



display(cluster_summary)



# ------------------------------------------------------------
# Calculate distances to cluster centroids
# ------------------------------------------------------------


distances = pairwise_distances(

    X_cluster,

    kmeans_final.cluster_centers_

)



min_distance = distances.min(axis=1)



basin_distance = pd.DataFrame({

    "gauge_id":

        df_clean["gauge_id"].values,


    "Cluster":

        cluster_labels,


    "Distance_to_Centroid":

        min_distance

})



basin_distance.to_csv(

    f"{FINAL_DIR}/Basin_to_Centroid_Distance.csv",

    index=False

)



# ------------------------------------------------------------
# Representative selection
# Spatial diversity constraint
# ------------------------------------------------------------


selected = []



# allocate number of basins per cluster

cluster_summary["Final_Number"] = np.maximum(
    1,
    np.round(cluster_summary["Selection_Target"])
).astype(int)



# correction to reach 40

difference = (
    TARGET_BASINS -
    cluster_summary["Final_Number"].sum()
)


if difference > 0:

    cluster_summary.loc[
        cluster_summary["Selection_Target"]
        .sort_values(ascending=False)
        .index[:difference],
        "Final_Number"
    ] += 1


elif difference < 0:

    removable = cluster_summary[
        cluster_summary["Final_Number"] > 1
    ].sort_values(
        "Selection_Target",
        ascending=True
    )

    cluster_summary.loc[
        removable.index[:abs(difference)],
        "Final_Number"
    ] -= 1






# ------------------------------------------------------------
# Select within each cluster
# ------------------------------------------------------------


for _, row in cluster_summary.iterrows():


    cluster_id = row["Cluster"]

    n_select = row["Final_Number"]



    candidates = basin_distance[

        basin_distance["Cluster"]

        == cluster_id

    ].copy()



    candidates = candidates.sort_values(

        "Distance_to_Centroid"

    )



    chosen = []



    for idx, candidate in candidates.iterrows():


        if len(chosen) >= n_select:

            break



        lat1 = cluster_assignment.loc[

            cluster_assignment.gauge_id

            == candidate.gauge_id,

            "latitude"

        ].values[0]


        lon1 = cluster_assignment.loc[

            cluster_assignment.gauge_id

            == candidate.gauge_id,

            "longitude"

        ].values[0]



        accept = True



        for c in chosen:


            lat2 = cluster_assignment.loc[

                cluster_assignment.gauge_id

                == c,

                "latitude"

            ].values[0]


            lon2 = cluster_assignment.loc[

                cluster_assignment.gauge_id

                == c,

                "longitude"

            ].values[0]


            distance_km = geodesic(
            (lat1, lon1),
            (lat2, lon2)
            ).km


            if distance_km < 100:

                accept=False



        if accept:

            chosen.append(candidate.gauge_id)



    selected.extend(chosen)



# ------------------------------------------------------------
# Final correction if less than 40
# ------------------------------------------------------------





# ------------------------------------------------------------
# Save final 40 basins
# ------------------------------------------------------------


representative_basins = cluster_assignment[

    cluster_assignment.gauge_id.isin(selected)

].copy()



representative_basins.to_csv(

    f"{FINAL_DIR}/Representative_40_Basins.csv",

    index=False

)



print("--------------------------------")

print(

    "Number of selected representative basins:",

    len(representative_basins)

)

print("--------------------------------")



display(representative_basins.head())



print("Cell 11 completed successfully.")

# ============================================================
# Cell 12: Basin Similarity Matrix and Spatial Validation
#
# Purpose:
#   1. Generate hydrological similarity matrix
#   2. Validate representative basin selection
#   3. Analyze spatial distribution
#
# ============================================================


from sklearn.metrics import pairwise_distances



# ------------------------------------------------------------
# Output folder
# ------------------------------------------------------------

SIM_DIR = f"{OUTPUT_DIR}/10_Validation"





# ------------------------------------------------------------
# 1. Calculate PCA distance matrix
# ------------------------------------------------------------


print("--------------------------------")
print("Calculating basin distance matrix...")
print("--------------------------------")


distance_matrix = pairwise_distances(

    X_cluster,

    metric="euclidean"

)



distance_df = pd.DataFrame(

    distance_matrix,

    index=df_clean["gauge_id"],

    columns=df_clean["gauge_id"]

)



distance_df.to_csv(

    f"{SIM_DIR}/Basin_Distance_Matrix.csv"

)



# ------------------------------------------------------------
# 2. Convert distance to similarity
#
# Similarity ranges between 0 and 1
# ------------------------------------------------------------


sigma = np.median(distance_matrix)

similarity_matrix = np.exp(
    -(distance_matrix**2) /
    (2 * sigma**2)
)



similarity_df = pd.DataFrame(

    similarity_matrix,

    index=df_clean["gauge_id"],

    columns=df_clean["gauge_id"]

)



similarity_df.to_csv(

    f"{SIM_DIR}/Basin_Similarity_Matrix.csv"

)



print("Similarity matrix generated")



# ------------------------------------------------------------
# 3. Extract representative basin similarity matrix
# ------------------------------------------------------------


representative_ids = representative_basins["gauge_id"].values



rep_similarity = similarity_df.loc[

    representative_ids,

    representative_ids

]



rep_similarity.to_csv(

    f"{SIM_DIR}/Representative_Basin_Similarity.csv"

)



# ------------------------------------------------------------
# 4. Cluster coverage check
# ------------------------------------------------------------


cluster_coverage = (

    representative_basins

    .groupby("Cluster")

    .size()

    .reset_index(name="Representative_Count")

)

cluster_coverage = cluster_summary[
    [
        "Cluster",
        "Number_of_Basins"
    ]
].merge(
    cluster_coverage,
    on="Cluster"
)

cluster_coverage.to_csv(

    f"{SIM_DIR}/Cluster_Coverage.csv",

    index=False

)

missing_clusters = set(
    cluster_assignment["Cluster"]
) - set(
    representative_basins["Cluster"]
)


print(
    "Clusters without representatives:",
    missing_clusters
)

print("--------------------------------")

print("Cluster coverage:")

display(cluster_coverage)



# ------------------------------------------------------------
# 5. Spatial distance analysis
# ------------------------------------------------------------


coords = representative_basins[

    [

        "latitude",

        "longitude"

    ]

].values



geo_distance = np.zeros(

    (len(coords), len(coords))

)


for i in range(len(coords)):

    for j in range(len(coords)):

        geo_distance[i,j] = geodesic(

            coords[i],

            coords[j]

        ).km



np.fill_diagonal(

    geo_distance,

    np.nan

)



nearest_neighbor = np.nanmin(

    geo_distance,

    axis=1

)



spatial_summary = pd.DataFrame({

    "gauge_id":

        representative_basins["gauge_id"],


    "Nearest_Neighbor_Distance_km":

        nearest_neighbor

})



spatial_summary.to_csv(

    f"{SIM_DIR}/Representative_Spatial_Distance.csv",

    index=False

)



print("--------------------------------")

print(

    "Mean nearest neighbor distance:",

    np.nanmean(nearest_neighbor)

)



# ------------------------------------------------------------
# 6. Spatial map of selected basins
# ------------------------------------------------------------


plt.figure(figsize=(9,6))


plt.scatter(

    df_clean["gauge_lon"],

    df_clean["gauge_lat"],

    s=8,

    alpha=0.25,

    label="All Basins"

)



plt.scatter(

    representative_basins["longitude"],

    representative_basins["latitude"],

    s=50,

    marker="*",


    label="Selected 40 Basins"

)



plt.xlabel("Longitude")

plt.ylabel("Latitude")


plt.title(

    "Spatial Distribution of Representative Basins"

)


plt.legend()


plt.grid(True)


plt.tight_layout()



plt.savefig(

    f"{SIM_DIR}/Representative_Basins_Map.png",

    dpi=300,

    bbox_inches="tight"

)


plt.show()



# ------------------------------------------------------------
# 7. Similarity statistics
# ------------------------------------------------------------


upper_triangle = rep_similarity.values[

    np.triu_indices(

        len(rep_similarity),

        k=1

    )

]



similarity_statistics = pd.DataFrame({

    "Mean_similarity":

        [np.mean(upper_triangle)],


    "Minimum_similarity":

        [np.min(upper_triangle)],


    "Maximum_similarity":

        [np.max(upper_triangle)],


    "Std_similarity":

        [np.std(upper_triangle)]

})



similarity_statistics.to_csv(

    f"{SIM_DIR}/Similarity_Statistics.csv",

    index=False

)



display(similarity_statistics)



print("--------------------------------")

print("Cell 12 completed successfully.")

# ============================================================
# Cell 13: Spatial Validation and Final Representative Export
# ============================================================

import os
import matplotlib.pyplot as plt
import pandas as pd


print("--------------------------------")
print("Spatial validation started")
print("--------------------------------")
FINAL_DIR = f"{OUTPUT_DIR}/11_Final_Selected_Basins"

# ------------------------------------------------------------
# Find final representative basin dataframe
# ------------------------------------------------------------

possible_names = [
    "representative_basins",
    "selected_representatives",
    "final_representative_basins",
    "selected_basins"
]

basin_df = representative_basins

print("Using dataframe: representative_basins")

# ------------------------------------------------------------
# Spatial summary
# ------------------------------------------------------------

spatial_summary = basin_df.groupby("Cluster").agg(
    Number_of_Basins=("gauge_id", "count"),
    Mean_Latitude=("latitude", "mean"),
    Mean_Longitude=("longitude", "mean"),
    Min_Latitude=("latitude", "min"),
    Max_Latitude=("latitude", "max"),
    Min_Longitude=("longitude", "min"),
    Max_Longitude=("longitude", "max")
).reset_index()

spatial_summary["Spatial_Range_Lat"] = (
    spatial_summary["Max_Latitude"]
    -
    spatial_summary["Min_Latitude"]
)

spatial_summary["Spatial_Range_Lon"] = (
    spatial_summary["Max_Longitude"]
    -
    spatial_summary["Min_Longitude"]
)
display(spatial_summary)



# ------------------------------------------------------------
# Spatial plot
# ------------------------------------------------------------

plt.figure(figsize=(10,7))

scatter = plt.scatter(
    basin_df["longitude"],
    basin_df["latitude"],
    c=basin_df["Cluster"],
    cmap="tab20",
    s=60,
    edgecolors="black"
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(
    "Spatial Distribution of Representative Basins"
)

plt.colorbar(
    scatter,
    label="Cluster"
)

plt.grid(True)
plt.savefig(
    f"{FINAL_DIR}/Representative_Basin_Spatial_Map.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------


basin_df.to_csv(
    f"{FINAL_DIR}/Final_Representative_Basins.csv",
    index=False
)


spatial_summary.to_csv(
    f"{FINAL_DIR}/Spatial_Validation_Summary.csv",
    index=False
)



print("--------------------------------")
print("Spatial validation completed")
print("--------------------------------")

print(
    f"Final representative basins: {len(basin_df)}"
)

print(
    f"Clusters represented: {basin_df['Cluster'].nunique()}"
)

print("--------------------------------")

# ============================================================
# Cell 14: Basin Characterization of Selected Basins
# Corrected Version
# ============================================================

import pandas as pd
import numpy as np
import os


print("--------------------------------")
print("Basin characterization started")
print("--------------------------------")


# ------------------------------------------------------------
# Identify feature dataset
# ------------------------------------------------------------

feature_df = X_final.copy()

selected_ids = representative_basins["gauge_id"]


print(
    f"Selected representative basins: {len(selected_ids)}"
)



# ------------------------------------------------------------
# Recover selected rows using original index
# ------------------------------------------------------------

selected_idx = df_clean.index[
    df_clean["gauge_id"].isin(selected_ids)
]


selected_features = feature_df.loc[
    selected_idx
].copy()

print(
    f"Feature dataset size: {len(feature_df)}"
)

print(
    f"Selected feature rows: {len(selected_features)}"
)



# ------------------------------------------------------------
# Numerical features
# ------------------------------------------------------------

numeric_features = feature_df.select_dtypes(
    include=np.number
).columns


comparison = pd.DataFrame(
    {
        "Feature": numeric_features,
        "All_Mean": [
            feature_df[c].mean()
            for c in numeric_features
        ],
        "Selected_Mean": [
            selected_features[c].mean()
            for c in numeric_features
        ],
        "All_STD": [
            feature_df[c].std()
            for c in numeric_features
        ],
        "Selected_STD": [
            selected_features[c].std()
            for c in numeric_features
        ]
    }
)


comparison["Mean_Difference_%"] = (
    abs(
        comparison["Selected_Mean"]
        -
        comparison["All_Mean"]
    )
    /
    (abs(comparison["All_Mean"]) + 1e-10)
    *
    100
)
comparison["STD_Difference_%"] = (
    abs(
        comparison["Selected_STD"]
        -
        comparison["All_STD"]
    )
    /
    (abs(comparison["All_STD"]) + 1e-10)
    *
    100
)

comparison = comparison.sort_values(
    "Mean_Difference_%",
    ascending=False
)


display(
    comparison
)



# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

CHAR_DIR = FINAL_DIR


comparison.to_csv(
    f"{CHAR_DIR}/Selected_vs_All_Basin_Statistics.csv",
    index=False
)


print("--------------------------------")
print("Characterization completed")
print("--------------------------------")

print(
    f"All basins: {len(feature_df)}"
)

print(
    f"Selected basins: {len(selected_features)}"
)
from scipy.stats import ks_2samp


ks_results = []

for c in numeric_features:

    stat, p_value = ks_2samp(
        feature_df[c],
        selected_features[c]
    )

    ks_results.append({

        "Feature": c,
        "KS_Statistic": stat,
        "p_value": p_value

    })


ks_results = pd.DataFrame(ks_results)


ks_results.to_csv(
    f"{CHAR_DIR}/KS_Test_Selected_vs_All.csv",
    index=False
)


display(ks_results)
print("--------------------------------")

# ============================================================
# Cell 15: Final Validation of Representative Catchments
#
# Purpose:
#   1. Evaluate feature-space coverage
#   2. Summarize representative statistics
#   3. Export validation results
#
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import pairwise_distances

print("--------------------------------")
print("FINAL VALIDATION STARTED")
print("--------------------------------")


# ============================================================
# Output folder
# ============================================================

VALID_DIR = f"{OUTPUT_DIR}/12_Final_Validation"



# ============================================================
# Prepare datasets
# ============================================================

feature_df = X_final.copy()

selected_ids = representative_basins["gauge_id"]

selected_idx = df_clean.index[
    df_clean["gauge_id"].isin(selected_ids)
]

selected_features = feature_df.loc[selected_idx].copy()

print(f"Total basins      : {len(feature_df)}")
print(f"Selected basins   : {len(selected_features)}")


# ============================================================
# Feature-space coverage
# ============================================================

distance_matrix = pairwise_distances(
    selected_features,
    feature_df,
    metric="euclidean"
)

nearest_distance = distance_matrix.min(axis=0)

coverage_summary = pd.DataFrame({

    "Metric":[
        "Mean_Nearest_Distance",
        "Median_Nearest_Distance",
        "Maximum_Nearest_Distance",
        "Minimum_Nearest_Distance",
        "STD_Nearest_Distance"
    ],

    "Value":[
        nearest_distance.mean(),
        np.median(nearest_distance),
        nearest_distance.max(),
        nearest_distance.min(),
        nearest_distance.std()
    ]

})

print("--------------------------------")
print("Feature-space coverage")
display(coverage_summary)


# ============================================================
# Coverage distribution
# ============================================================

plt.figure(figsize=(7,5))

plt.hist(
    nearest_distance,
    bins=25,
    edgecolor="black"
)

plt.xlabel("Nearest Representative Distance")
plt.ylabel("Number of Catchments")
plt.title("Coverage of CAMELS Catchments by Representative Basins")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    f"{VALID_DIR}/Nearest_Distance_Distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# Cluster representation summary
# ============================================================

cluster_validation = representative_basins.groupby(
    "Cluster"
).size().reset_index(
    name="Representative_Count"
)

cluster_validation = cluster_summary[
    [
        "Cluster",
        "Number_of_Basins",
        "Final_Number"
    ]
].merge(
    cluster_validation,
    on="Cluster"
)

display(cluster_validation)


# ============================================================
# Export
# ============================================================

coverage_summary.to_csv(
    f"{VALID_DIR}/Feature_Space_Coverage.csv",
    index=False
)

cluster_validation.to_csv(
    f"{VALID_DIR}/Cluster_Representation.csv",
    index=False
)


print("--------------------------------")
print("Validation files exported")
print("--------------------------------")

print("Files created:")

print("1. Feature_Space_Coverage.csv")
print("2. Cluster_Representation.csv")
print("3. Nearest_Distance_Distribution.png")

print("--------------------------------")
print("CELL COMPLETED SUCCESSFULLY")
print("--------------------------------")


# ============================================================
# Cell 16: Selection of Spatial Transfer Basins
#
# Purpose:
#   Select one independent transfer basin from each cluster
#   for spatial transferability evaluation.
#
# Method:
#   - Exclude representative (training) basins
#   - Rank remaining basins by distance to cluster centroid
#   - Select the farthest basin in each cluster
#
# Output:
#   Transfer_Basins.csv
#
# ============================================================

import os
import pandas as pd

# ------------------------------------------------------------
# Output directory
# ------------------------------------------------------------

TRANSFER_DIR = f"{OUTPUT_DIR}/13_Transfer_Basins"


# ------------------------------------------------------------
# Representative basin IDs
# ------------------------------------------------------------

representative_ids = set(
    representative_basins["gauge_id"]
)

transfer_list = []

# ------------------------------------------------------------
# Select one transfer basin from each cluster
# ------------------------------------------------------------

for cluster in sorted(cluster_summary["Cluster"].unique()):

    candidates = basin_distance[
        basin_distance["Cluster"] == cluster
    ].copy()

    # Remove representative basins
    candidates = candidates[
        ~candidates["gauge_id"].isin(representative_ids)
    ]

    # Rank by distance from centroid
    candidates = candidates.sort_values(
        "Distance_to_Centroid",
        ascending=False
    )

    selected = candidates.iloc[0]

    basin_info = cluster_assignment[
        cluster_assignment["gauge_id"] == selected["gauge_id"]
    ].iloc[0]

    transfer_list.append({

        "Cluster": cluster,

        "gauge_id": basin_info["gauge_id"],

        "latitude": basin_info["latitude"],

        "longitude": basin_info["longitude"],

        "Distance_to_Centroid":
            selected["Distance_to_Centroid"]

    })

# ------------------------------------------------------------
# Final dataframe
# ------------------------------------------------------------

transfer_basins = pd.DataFrame(
    transfer_list
)

transfer_basins.to_csv(
    f"{TRANSFER_DIR}/Transfer_Basins.csv",
    index=False
)

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print("--------------------------------")
print("Spatial transfer basin selection completed")
print("--------------------------------")
print("Number of transfer basins:",
      len(transfer_basins))
print("--------------------------------")

display(transfer_basins)

print("Cell 16 completed successfully.")


