# ============================================================
# Script 07 of 07 - Prediction probability distribution and gauge-wise
# failure analysis (FP/FN)
#
# Requires: the output of scripts 04 and 05 (Spatial_Transfer/Input/...,
# Threshold_Correction/Corrected_Threshold_Results.csv)
# Models are only loaded from GitHub - training is never repeated.
# ============================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_github_loader import BASE_DIR, ensure_models, display  # noqa: F401

ensure_models()

from utils_github_loader import apply_journal_style
apply_journal_style()  # publication-style fonts for every figure in this script

# ==========================================================
# CELL 7
# PROBABILITY DISTRIBUTION ANALYSIS
# ==========================================================

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils_github_loader import apply_journal_style
apply_journal_style()

print("="*70)
print("CELL 7 - PROBABILITY DISTRIBUTION ANALYSIS")
print("="*70)

output_dir="./flood_project_workspace/Results/Spatial_Transfer/Probability_Distribution"
os.makedirs(output_dir,exist_ok=True)

# ----------------------------------------------------------
# Load scaled transfer dataset
# ----------------------------------------------------------

transfer=pd.read_csv(
"./flood_project_workspace/Results/Spatial_Transfer/Input/Spatial_Transfer_Final_Dataset.csv"
)

ignore_cols=[
"Flood_Event",
"Date",
"gauge_id",
"Q"
]

features=[
c for c in transfer.columns
if c not in ignore_cols
]

X=pd.read_csv(
"./flood_project_workspace/Results/Spatial_Transfer/Input/X_transfer_scaled.csv"
)

# ----------------------------------------------------------
# Load models
# ----------------------------------------------------------

models={

"Optimized_XGBoost":
joblib.load(
"./flood_project_workspace/Results/ML_Models/Optuna_Optimization/Best_XGBoost.pkl"
),

"Optimized_CatBoost":
joblib.load(
"./flood_project_workspace/Results/ML_Models/Optuna_Optimization/Best_CatBoost.pkl"
)

}

# ----------------------------------------------------------
# Probability distribution
# ----------------------------------------------------------

all_gauge_summaries = {}

for name,model in models.items():

    prob=model.predict_proba(X)[:,1]

    df=pd.DataFrame({

        "Gauge_ID":transfer["gauge_id"],
        "Probability":prob,
        "Flood_Event":transfer["Flood_Event"]

    })


    df.to_csv(
        f"{output_dir}/{name}_Probability_Distribution.csv",
        index=False
    )


    gauge_summary=[]


    for gid,g in df.groupby("Gauge_ID"):


        flood_prob=g.loc[
            g["Flood_Event"]==1,
            "Probability"
        ]

        nonflood_prob=g.loc[
            g["Flood_Event"]==0,
            "Probability"
        ]


        gauge_summary.append({

            "Gauge_ID":gid,

            "Flood_samples":
            len(flood_prob),

            "Mean_Flood_Probability":
            flood_prob.mean()
            if len(flood_prob)>0 else np.nan,


            "Mean_NonFlood_Probability":
            nonflood_prob.mean(),


            "Max_Probability":
            g["Probability"].max(),


            "Probability_Separation":
            flood_prob.mean()-nonflood_prob.mean()
            if len(flood_prob)>0 else np.nan

        })


    summary=pd.DataFrame(gauge_summary)


    summary.to_csv(
        f"{output_dir}/{name}_Gauge_Probability_Summary.csv",
        index=False
    )

    all_gauge_summaries[name] = summary


# ============================================================
# Figure: gauge-level probability separation (journal style)
# ============================================================

model_display_names = {
    "Optimized_XGBoost": "Optimized XGBoost",
    "Optimized_CatBoost": "Optimized CatBoost",
}

fig, axes = plt.subplots(
    1, len(all_gauge_summaries),
    figsize=(7 * len(all_gauge_summaries), 6),
    sharey=True,
)

if len(all_gauge_summaries) == 1:
    axes = [axes]

for ax, (name, summary) in zip(axes, all_gauge_summaries.items()):

    plot_df = summary.sort_values("Gauge_ID").copy()
    gauge_labels = plot_df["Gauge_ID"].astype(str)
    x = np.arange(len(gauge_labels))
    width = 0.38

    ax.bar(x - width / 2, plot_df["Mean_NonFlood_Probability"],
           width=width, label="Non-flood", color="#4C72B0")
    ax.bar(x + width / 2, plot_df["Mean_Flood_Probability"],
           width=width, label="Flood", color="#DD8452")

    ax.set_xticks(x)
    ax.set_xticklabels(gauge_labels, rotation=45, ha="right")
    ax.set_xlabel("Gauge ID")
    ax.set_title(model_display_names.get(name, name))
    ax.set_ylim(0, 1.0)
    ax.legend(title="Event")

axes[0].set_ylabel("Mean Predicted Probability")

fig.suptitle("Gauge-level Probability Separation Between Flood and Non-flood Samples")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(
    os.path.join(output_dir, "Gauge_Probability_Separation.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# ==========================================================
# CELL 8
# FAILURE ANALYSIS BY GAUGE
# ==========================================================

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

print("="*70)
print("CELL 8 - FAILURE ANALYSIS")
print("="*70)

output_dir="./flood_project_workspace/Results/Spatial_Transfer/Failure_Analysis_By_Gauge"
os.makedirs(output_dir,exist_ok=True)

# ----------------------------------------------------------
# Load scaled transfer dataset
# ----------------------------------------------------------

transfer=pd.read_csv(
"./flood_project_workspace/Results/Spatial_Transfer/Input/Spatial_Transfer_Final_Dataset.csv"
)

ignore_cols=[
"Flood_Event",
"Date",
"gauge_id",
"Q"
]

features=[
c for c in transfer.columns
if c not in ignore_cols
]

X=pd.read_csv(
"./flood_project_workspace/Results/Spatial_Transfer/Input/X_transfer_scaled.csv"
)
y=pd.read_csv(
    "./flood_project_workspace/Results/Spatial_Transfer/Input/y_transfer.csv"
).values.ravel()

# ----------------------------------------------------------
# Load optimal thresholds
# ----------------------------------------------------------

thresholds=pd.read_csv(
"./flood_project_workspace/Results/Threshold_Correction/Corrected_Threshold_Results.csv"
)

# ----------------------------------------------------------
# Load models
# ----------------------------------------------------------

models={

"Optimized_XGBoost":
joblib.load(
"./flood_project_workspace/Results/ML_Models/Optuna_Optimization/Best_XGBoost.pkl"
),

"Optimized_CatBoost":
joblib.load(
"./flood_project_workspace/Results/ML_Models/Optuna_Optimization/Best_CatBoost.pkl"
)

}

# ----------------------------------------------------------
# Failure analysis
# ----------------------------------------------------------

for name,model in models.items():

    threshold=float(

        thresholds[
            thresholds["Model"]==name
        ]["Validation_Optimal_Threshold"].values[0]

    )

    prob=model.predict_proba(X)[:,1]

    pred=(prob>=threshold).astype(int)

    result=transfer[[
        "Date",
        "gauge_id",
        "Q",
        "Flood_Event"
    ]].copy()

    result["Probability"]=prob
    result["Prediction"]=pred

    result["FP"]=(
        (pred==1)&(y==0)
    ).astype(int)

    result["FN"]=(
        (pred==0)&(y==1)
    ).astype(int)

    result.to_csv(
        f"{output_dir}/{name}_Prediction_Details.csv",
        index=False
    )

    summary=result.groupby(
        "gauge_id"
    )[["FP","FN"]].sum()

    summary["Total_Error"]=summary["FP"]+summary["FN"]

    summary.to_csv(
        f"{output_dir}/{name}_Failure_By_Gauge.csv"
    )

    plt.figure(figsize=(10,5))

    summary["Total_Error"].plot(
        kind="bar"
    )

    plt.ylabel("Number of Errors", fontsize=14)

    plt.title(f"{name} Prediction Errors by Gauge")

    plt.tight_layout()

    plt.savefig(
        f"{output_dir}/{name}_Failure_By_Gauge.png",
        dpi=300
    )

    plt.close()

print("="*70)
print("CELL 8 COMPLETED")
print("="*70)

