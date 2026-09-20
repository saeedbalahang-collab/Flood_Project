# ============================================================
# Script 05 of 07 - Model evaluation on the spatial-transfer catchments
# (prediction on transfer catchments + catchment environmental
#  distance analysis)
#
# Requires: the output of script 04 (Spatial_Transfer_Final_Dataset.csv, ...)
# Models are only loaded from GitHub - training is never repeated.
# ============================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_github_loader import BASE_DIR, ensure_models, ensure_scaler, display  # noqa: F401

ensure_models()
ensure_scaler()

from utils_github_loader import apply_journal_style
apply_journal_style()  # publication-style fonts for every figure in this script

# =====================================================================
# CELL 1
# PREDICTION ON SPATIAL TRANSFER CATCHMENTS (UPDATED)
# =====================================================================

import os
import joblib
import numpy as np
import pandas as pd


print("="*70)
print("SPATIAL TRANSFER PREDICTION")
print("="*70)



# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

base = "./flood_project_workspace/Results"

data_path = (
    f"{base}/Spatial_Transfer/Input/"
    "Spatial_Transfer_Final_Dataset.csv"
)


model_path = (
    f"{base}/ML_Models/Optuna_Optimization/"
)


output_path = (
    f"{base}/Spatial_Transfer/Predictions"
)


os.makedirs(
    output_path,
    exist_ok=True
)



# ---------------------------------------------------------------------
# Load dataset metadata
# ---------------------------------------------------------------------

df = pd.read_csv(
    data_path
)


df = df.reset_index(drop=True)


print("Dataset:")
print(df.shape)



# ---------------------------------------------------------------------
# Load scaled transfer data
# ---------------------------------------------------------------------

X = pd.read_csv(
    "./flood_project_workspace/Results/Spatial_Transfer/Input/X_transfer_scaled.csv"
)


y = pd.read_csv(
    "./flood_project_workspace/Results/Spatial_Transfer/Input/y_transfer.csv"
).values.ravel()



print("\nX shape:", X.shape)
print("y shape:", y.shape)



# ---------------------------------------------------------------------
# Check alignment
# ---------------------------------------------------------------------

assert len(df)==len(X)==len(y), \
"Mismatch between dataset and scaled data"



# ---------------------------------------------------------------------
# Load thresholds
# ---------------------------------------------------------------------

threshold_df = pd.read_csv(
    "./flood_project_workspace/Results/Threshold_Correction/Corrected_Threshold_Results.csv"
)


threshold_dict = dict(
    zip(
        threshold_df["Model"],
        threshold_df["Validation_Optimal_Threshold"]
    )
)



# ---------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------

models = {


    "Optimized_XGBoost":
        joblib.load(
            f"{model_path}/Best_XGBoost.pkl"
        ),



    "Optimized_CatBoost":
        joblib.load(
            f"{model_path}/Best_CatBoost.pkl"
        )

}



print("\nModels loaded")



# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------

for model_name, model in models.items():


    print("-"*60)

    print(model_name)



    probability = model.predict_proba(
        X
    )[:,1]



    threshold = threshold_dict[model_name]



    prediction = (
        probability >= threshold
    ).astype(int)



    out = df[
        [
            "Date",
            "gauge_id",
            "Q",
            "Flood_Event"
        ]
    ].copy()



    out["Probability"] = probability

    out["Prediction"] = prediction



    out.to_csv(

        f"{output_path}/{model_name}_Predictions.csv",

        index=False

    )



    print(out.head())



print("="*70)

print("Prediction files saved.")

print("="*70)





# =====================================================================
# CELL 2
# PERFORMANCE OF EACH TRANSFER CATCHMENT (UPDATED)
# =====================================================================


import numpy as np
import pandas as pd


from sklearn.metrics import (

    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix

)



print("="*70)

print("PER-GAUGE PERFORMANCE")

print("="*70)



# ---------------------------------------------------------------------

base = "./flood_project_workspace/Results"

output_path = (
    f"{base}/Spatial_Transfer/Predictions"
)


models = [

    "Optimized_XGBoost",

    "Optimized_CatBoost"

]



results=[]



# ---------------------------------------------------------------------

for model_name in models:


    pred_file = pd.read_csv(

        f"{output_path}/{model_name}_Predictions.csv"

    )



    print("-"*60)

    print(model_name)



    for gauge in sorted(
        pred_file["gauge_id"].unique()
    ):



        basin = pred_file[
            pred_file["gauge_id"] == gauge
        ].copy()



        y_true = basin["Flood_Event"]

        y_pred = basin["Prediction"]

        y_prob = basin["Probability"]




        cm = confusion_matrix(

            y_true,

            y_pred,

            labels=[0,1]

        )


        TN,FP,FN,TP = cm.ravel()



        result = {


            "Gauge_ID":gauge,


            "Model":model_name,


            "Samples":len(basin),


            "Flood_Events":
            int(y_true.sum()),



            "Accuracy":
            accuracy_score(
                y_true,
                y_pred
            ),



            "Precision":
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            ),



            "Recall":
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            ),



            "F1":
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            ),



            "ROC_AUC":

            roc_auc_score(
                y_true,
                y_prob
            )
            if len(
                np.unique(y_true)
            ) > 1
            else np.nan,



            "PR_AUC":

            average_precision_score(
                y_true,
                y_prob
            ),



            "MCC":

            matthews_corrcoef(
                y_true,
                y_pred
            ),



            "TP":TP,

            "TN":TN,

            "FP":FP,

            "FN":FN

        }



        results.append(result)




# ---------------------------------------------------------------------

results = pd.DataFrame(results)



results = results.sort_values(

    [
        "Gauge_ID",
        "Model"
    ]

)



results.to_csv(

    f"{output_path}/Per_Gauge_Performance.csv",

    index=False

)



print(results)



print("="*70)

print("Results saved.")

print("="*70)

# ============================================================
# CELL: Catchment Environmental Distance Analysis
# Z-score Distance + Mahalanobis Distance
# for Spatial Transferability Assessment
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import mahalanobis


print("="*70)
print("CATCHMENT ENVIRONMENTAL DISTANCE ANALYSIS")
print("="*70)


# ============================================================
# Paths
# ============================================================

BASE = "./flood_project_workspace/Results"

train_file = (
    "./flood_project_workspace/Results/ML_Training_Input/Temporal_Train.csv"
)

transfer_file = (
    "./flood_project_workspace/Results/Spatial_Transfer/Input/"
    "Spatial_Transfer_Final_Dataset.csv"
)


output_dir = (
    "./flood_project_workspace/Results/Spatial_Transfer/"
    "Environmental_Distance"
)

os.makedirs(
    output_dir,
    exist_ok=True
)



# ============================================================
# Load datasets
# ============================================================

train = pd.read_csv(train_file)

transfer = pd.read_csv(transfer_file)


print("Training dataset:")
print(train.shape)

print("Transfer dataset:")
print(transfer.shape)



# ============================================================
# Define static catchment descriptors
# ============================================================

static_features = [

    "clay_frac",
    "area_gages2",
    "high_prec_freq",
    "frac_snow",
    "gvf_diff",
    "gvf_max",
    "elev_mean",
    "p_mean",
    "soil_porosity",
    "soil_depth_pelletier",
    "soil_conductivity",
    "geol_permeability",
    "aridity",
    "slope_mean",
    "frac_forest",
    "carbonate_rocks_frac",
    "sand_frac",
    "lai_max",
    "lai_diff"

]


static_features = [
    f for f in static_features
    if f in train.columns
]


print("\nStatic features used:")
print(static_features)



# ============================================================
# Extract one value per gauge
# ============================================================

# Training catchments
# each gauge has repeated daily values
# keep one representative value

train_gauge = (
    train
    .groupby("gauge_id")[static_features]
    .first()
    .reset_index()
)



# Transfer catchments

transfer_gauge = (
    transfer
    .groupby("gauge_id")[static_features]
    .first()
    .reset_index()
)



print("\nTraining gauges:",
      len(train_gauge))

print("Transfer gauges:",
      len(transfer_gauge))



# ============================================================
# Standardization based ONLY on training catchments
# ============================================================

scaler = StandardScaler()


X_train_static = scaler.fit_transform(
    train_gauge[static_features]
)


X_transfer_static = scaler.transform(
    transfer_gauge[static_features]
)



# ============================================================
# 1. Euclidean Z-score distance
# ============================================================

train_mean = np.mean(
    X_train_static,
    axis=0
)


zscore_distance=[]


for row in X_transfer_static:

    distance = np.sqrt(
        np.sum(
            (row-train_mean)**2
        )
    )

    zscore_distance.append(distance)



# ============================================================
# 2. Mahalanobis distance
# ============================================================

cov_matrix = np.cov(
    X_train_static,
    rowvar=False
)


# regularization to avoid singular matrix

cov_matrix += (
    np.eye(cov_matrix.shape[0])
    * 1e-6
)


cov_inv = np.linalg.inv(
    cov_matrix
)



mahal_distance=[]


for row in X_transfer_static:

    d = mahalanobis(
        row,
        train_mean,
        cov_inv
    )

    mahal_distance.append(d)



# ============================================================
# 3. Feature-wise absolute Z-score deviation
# ============================================================

feature_zscore = pd.DataFrame(
    np.abs(X_transfer_static),
    columns=[
        f+"_Abs_Zscore"
        for f in static_features
    ]
)


# ============================================================
# Final output
# ============================================================


results = transfer_gauge[
    ["gauge_id"]
].copy()


results["Zscore_Euclidean_Distance"] = (
    zscore_distance
)


results["Mahalanobis_Distance"] = (
    mahal_distance
)



results = pd.concat(
    [
        results.reset_index(drop=True),
        feature_zscore.reset_index(drop=True)
    ],
    axis=1
)



results = results.sort_values(
    "Mahalanobis_Distance",
    ascending=False
)



results.to_csv(
    os.path.join(
        output_dir,
        "Gauge_Environmental_Distance.csv"
    ),
    index=False
)


# ============================================================
# Figure: gauge-wise absolute Z-score heatmap (journal style)
# ============================================================

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from utils_github_loader import apply_journal_style

apply_journal_style()

zscore_cols = [f + "_Abs_Zscore" for f in static_features]

heatmap_matrix = results.set_index("gauge_id")[zscore_cols].copy()
heatmap_matrix.columns = [c.replace("_Abs_Zscore", "") for c in heatmap_matrix.columns]

# Avoid non-positive values on the log color scale
plot_values = heatmap_matrix.clip(lower=1e-3)

fig, ax = plt.subplots(
    figsize=(max(12, 0.55 * len(heatmap_matrix.columns)), max(6, 0.6 * len(heatmap_matrix)))
)

sns.heatmap(
    plot_values,
    cmap="viridis",
    norm=mcolors.LogNorm(vmin=plot_values.values.min(), vmax=plot_values.values.max()),
    linewidths=0.5,
    linecolor="white",
    cbar_kws={"label": "Absolute Z-score (log scale)"},
    ax=ax,
)

ax.set_title("Catchment Attribute Shift by Gauge (Absolute Z-score)")
ax.set_xlabel("Catchment Attributes")
ax.set_ylabel("Gauge ID")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "Gauge_Attribute_Zscore_Heatmap.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close()



# ============================================================
# Feature contribution summary
# ============================================================


feature_summary = pd.DataFrame({

    "Feature":
    static_features,


    "Mean_Absolute_Zscore":
    np.mean(
        np.abs(X_transfer_static),
        axis=0
    )

})


feature_summary = feature_summary.sort_values(
    "Mean_Absolute_Zscore",
    ascending=False
)


feature_summary.to_csv(
    os.path.join(
        output_dir,
        "Feature_Environmental_Shift.csv"
    ),
    index=False
)



print("="*70)
print("COMPLETED")
print("="*70)


print("\nGauge-level environmental distance:")
display(results)


print("\nMost shifted environmental variables:")
display(feature_summary)


print("\nFiles saved:")
for f in os.listdir(output_dir):
    print(f)



# ==========================================================
# CELL 5
# CATCHMENT-LEVEL PCA VISUALIZATION
# TRAIN vs TRANSFER ENVIRONMENTAL SPACE
# ==========================================================

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


print("="*70)
print("CELL 5 - CATCHMENT LEVEL PCA VISUALIZATION")
print("="*70)


output_dir="./flood_project_workspace/Results/Spatial_Transfer/PCA_Visualization"

os.makedirs(
    output_dir,
    exist_ok=True
)


# ----------------------------------------------------------
# Load datasets
# ----------------------------------------------------------

train=pd.read_csv(
"./flood_project_workspace/Results/ML_Training_Input/Temporal_Train.csv"
)


transfer=pd.read_csv(
"./flood_project_workspace/Results/Spatial_Transfer/Input/Spatial_Transfer_Final_Dataset.csv"
)



# ----------------------------------------------------------
# Static environmental features
# ----------------------------------------------------------

static_features = [

"clay_frac",
"area_gages2",
"high_prec_freq",
"frac_snow",
"gvf_diff",
"gvf_max",
"elev_mean",
"p_mean",
"soil_porosity",
"soil_depth_pelletier",
"soil_conductivity",
"geol_permeability",
"aridity",
"slope_mean",
"frac_forest",
"carbonate_rocks_frac",
"sand_frac",
"lai_max",
"lai_diff"

]


print("Static features:")
print(static_features)



# ----------------------------------------------------------
# Convert daily data to catchment level
# ----------------------------------------------------------

train_catchment = (
    train
    .groupby("gauge_id")[static_features]
    .mean()
    .reset_index()
)


transfer_catchment = (
    transfer
    .groupby("gauge_id")[static_features]
    .mean()
    .reset_index()
)



print("\nTraining catchments:",
      len(train_catchment))

print("Transfer catchments:",
      len(transfer_catchment))



# ----------------------------------------------------------
# Scaling
# ----------------------------------------------------------
import joblib
scaler = joblib.load('./flood_project_workspace/Results/ML_Training_Input/StandardScaler.pkl')


X_train = scaler.fit_transform(
    train_catchment[static_features]
)


X_transfer = scaler.transform(
    transfer_catchment[static_features]
)



# ----------------------------------------------------------
# PCA
# FIT ONLY ON TRAINING CATCHMENTS
# ----------------------------------------------------------

pca=PCA(
    n_components=2,
    random_state=42
)


train_pca = pca.fit_transform(
    X_train
)


transfer_pca = pca.transform(
    X_transfer
)



print("\nExplained variance ratio:")
print(pca.explained_variance_ratio_)

print(
"Total explained variance:",
pca.explained_variance_ratio_.sum()
)



# ----------------------------------------------------------
# Save coordinates
# ----------------------------------------------------------

pca_df=pd.DataFrame({

"gauge_id":
list(train_catchment["gauge_id"])
+
list(transfer_catchment["gauge_id"]),


"PC1":
list(train_pca[:,0])
+
list(transfer_pca[:,0]),


"PC2":
list(train_pca[:,1])
+
list(transfer_pca[:,1]),


"Dataset":
["Train"]*len(train_pca)
+
["Transfer"]*len(transfer_pca)

})


pca_df.to_csv(
f"{output_dir}/Catchment_Level_PCA_Coordinates.csv",
index=False
)



# ----------------------------------------------------------
# Plot
# ----------------------------------------------------------

plt.figure(figsize=(8,6))


plt.scatter(
train_pca[:,0],
train_pca[:,1],
s=80,
alpha=0.7,
label="Training catchments"
)



plt.scatter(
transfer_pca[:,0],
transfer_pca[:,1],
s=120,
alpha=0.9,
label="Transfer catchments"
)



# Add gauge labels for transfer basins

for i,gauge in enumerate(
    transfer_catchment["gauge_id"]
):

    plt.text(
        transfer_pca[i,0],
        transfer_pca[i,1],
        str(gauge),
        fontsize=11
    )



plt.xlabel(
f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)", fontsize = 14
)


plt.ylabel(
f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)", fontsize = 14
)



plt.title(
"Catchment Environmental Space: Training vs Transfer", fontsize = 16
)


plt.legend()


plt.tight_layout()


plt.savefig(
f"{output_dir}/Catchment_Level_PCA_Train_vs_Transfer.png",
dpi=300
)


plt.close()



print("="*70)
print("PCA visualization completed")
print("="*70)

print("Saved:")
print("Catchment_Level_PCA_Coordinates.csv")
print("Catchment_Level_PCA_Train_vs_Transfer.png")
