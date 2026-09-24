# ============================================================
# Script 06 of 07 - Model interpretability analysis with SHAP
# (global + gauge-wise + success vs failure comparison)
#
# Requires: the output of script 04 (Spatial_Transfer/Input/...)
# Models are only loaded from GitHub - training is never repeated.
# ============================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_github_loader import BASE_DIR, ensure_models, pip_install, display  # noqa: F401

ensure_models()

from utils_github_loader import apply_journal_style
apply_journal_style()  # publication-style fonts for every figure in this script

# ==========================================================
# CELL 6!
# COMPREHENSIVE SHAP ANALYSIS
# GLOBAL + GAUGE-WISE + SUCCESS VS FAILURE COMPARISON
# ==========================================================


import os
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pip_install("catboost")
import catboost
print("="*80)
print("CELL 6 - COMPREHENSIVE TRANSFER SHAP ANALYSIS")
print("="*80)



# ==========================================================
# Paths
# ==========================================================

BASE="./flood_project_workspace/"


OUTPUT_DIR=os.path.join(
    BASE,
    "Spatial_Transfer/Predictions",
    "SHAP_Analysis"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)



# ==========================================================
# Load transfer data
# ==========================================================


transfer=pd.read_csv(
    "./flood_project_workspace/Results/Spatial_Transfer/Input/Spatial_Transfer_Final_Dataset.csv"
)



X_transfer=pd.read_csv(
    "./flood_project_workspace/Results/Spatial_Transfer/Input/X_transfer_scaled.csv"
)



print("Transfer dataset:")
print(transfer.shape)

print("Scaled X:")
print(X_transfer.shape)



# ==========================================================
# Features
# ==========================================================


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


print("\nNumber of features:",len(features))


# ensure same order

X_transfer.columns=features



# ==========================================================
# Load models
# ==========================================================


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


print("\nModels loaded")



# ==========================================================
# PART 1
# GLOBAL SHAP ANALYSIS
# ==========================================================


global_dir=os.path.join(
    OUTPUT_DIR,
    "Global"
)


os.makedirs(
    global_dir,
    exist_ok=True
)



global_rankings={}



for model_name, model in models.items():

    print("\nGLOBAL SHAP:",model_name)


    explainer=shap.TreeExplainer(
        model
    )


    shap_values=explainer.shap_values(
        X_transfer
    )


    mean_abs=np.abs(
        shap_values
    ).mean(axis=0)



    ranking=pd.DataFrame({

        "Feature":features,

        "Mean_ABS_SHAP":mean_abs

    })


    ranking=ranking.sort_values(
        "Mean_ABS_SHAP",
        ascending=False
    )


    global_rankings[model_name]=ranking



    ranking.to_csv(

        os.path.join(
            global_dir,
            f"{model_name}_Global_SHAP_Ranking.csv"
        ),

        index=False

    )



    plt.figure(figsize=(8,10))


    shap.summary_plot(

        shap_values,

        X_transfer,

        feature_names=features,

        show=False

    )


    plt.tight_layout()


    plt.savefig(

        os.path.join(
            global_dir,
            f"{model_name}_Global_SHAP.png"
        ),

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()



# ==========================================================
# PART 2
# GAUGE-WISE SHAP
# ==========================================================


gauge_dir=os.path.join(
    OUTPUT_DIR,
    "Gauge_Wise"
)


os.makedirs(
    gauge_dir,
    exist_ok=True
)



gauge_results=[]



gauges=transfer["gauge_id"].unique()



print("\nNumber of gauges:",len(gauges))



for gauge in gauges:


    print("\nProcessing gauge:",gauge)


    idx=transfer["gauge_id"]==gauge


    X_gauge=X_transfer.loc[idx].copy()



    gauge_folder=os.path.join(
        gauge_dir,
        str(gauge)
    )


    os.makedirs(
        gauge_folder,
        exist_ok=True
    )



    for model_name,model in models.items():


        print(
            "   ",
            model_name
        )


        explainer=shap.TreeExplainer(
            model
        )


        shap_values=explainer.shap_values(
            X_gauge
        )


        mean_abs=np.abs(
            shap_values
        ).mean(axis=0)



        ranking=pd.DataFrame({

            "Gauge_ID":gauge,

            "Feature":features,

            "Mean_ABS_SHAP":mean_abs

        })



        ranking=ranking.sort_values(
            "Mean_ABS_SHAP",
            ascending=False
        )



        ranking.to_csv(

            os.path.join(
                gauge_folder,
                f"{model_name}_SHAP_Ranking.csv"
            ),

            index=False

        )



        plt.figure(figsize=(8,10))


        shap.summary_plot(

            shap_values,

            X_gauge,

            feature_names=features,

            show=False

        )


        plt.tight_layout()


        plt.savefig(

            os.path.join(
                gauge_folder,
                f"{model_name}_SHAP_Summary.png"
            ),

            dpi=300,

            bbox_inches="tight"

        )


        plt.close()



        ranking["Model"]=model_name

        gauge_results.append(
            ranking
        )



gauge_shap_df=pd.concat(
    gauge_results,
    ignore_index=True
)



gauge_shap_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "Gauge_Wise_SHAP_All.csv"
    ),

    index=False

)



# ==========================================================
# PART 3
# SUCCESS VS FAILURE TRANSFER ANALYSIS
# ==========================================================


performance=pd.read_csv(

"./flood_project_workspace/Results/Spatial_Transfer/Predictions/Per_Gauge_Performance.csv"

)



# average models

perf_avg=performance.groupby(
    "Gauge_ID"
)[
[
"F1",
"Recall",
"PR_AUC"
]

].mean()



print("\nPerformance summary")
print(perf_avg)



# define groups

successful=perf_avg[
    perf_avg["Recall"]>0.3
].index



failed=perf_avg[
    perf_avg["Recall"]<0.3
].index



print("\nSuccessful gauges:")
print(list(successful))


print("\nFailed gauges:")
print(list(failed))



# use XGBoost global importance

xgb_shap=gauge_shap_df[

    gauge_shap_df["Model"]
    ==
    "Optimized_XGBoost"

]



success_mean=xgb_shap[
    xgb_shap["Gauge_ID"].isin(successful)

].groupby(
"Feature"
)[
"Mean_ABS_SHAP"
].mean()



failure_mean=xgb_shap[
    xgb_shap["Gauge_ID"].isin(failed)

].groupby(
"Feature"
)[
"Mean_ABS_SHAP"
].mean()



comparison=pd.DataFrame({

"Successful_Gauges_SHAP":
success_mean,

"Failed_Gauges_SHAP":
failure_mean

})



comparison["Difference"]=(

comparison["Failed_Gauges_SHAP"]

-

comparison["Successful_Gauges_SHAP"]

)



comparison=comparison.sort_values(
"Difference",
ascending=False
)



comparison.to_csv(

os.path.join(
OUTPUT_DIR,
"Successful_vs_Failed_SHAP_Comparison.csv"
),

index=True

)


# ============================================================
# Figure: SHAP importance difference (journal style)
# ============================================================

from utils_github_loader import apply_journal_style
apply_journal_style()

plot_df = comparison.dropna(subset=["Difference"]).sort_values("Difference", ascending=False)
top_n = 6
plot_df = pd.concat([plot_df.head(top_n), plot_df.tail(top_n)]).drop_duplicates()
plot_df = plot_df.sort_values("Difference", ascending=True)

colors = ["#4C72B0" if v < 0 else "#C44E52" for v in plot_df["Difference"]]

fig, ax = plt.subplots(figsize=(9, max(5, 0.5 * len(plot_df))))
bars = ax.barh(plot_df.index, plot_df["Difference"], color=colors)

for bar, value in zip(bars, plot_df["Difference"]):
    x_pos = value + (0.01 if value >= 0 else -0.01)
    ha = "left" if value >= 0 else "right"
    ax.text(x_pos, bar.get_y() + bar.get_height() / 2, f"{value:+.3f}",
            va="center", ha=ha, fontsize=12)

ax.axvline(0, color="black", linewidth=1)
ax.set_xlabel("SHAP Importance Difference\n(Limited-Detection Gauges - Successful Gauges)")
ax.set_ylabel("Feature")
ax.set_title("SHAP Importance Difference: Limited-Detection vs Successful Gauges")

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "SHAP_Importance_Difference.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close()



# ==========================================================
# PART 4
# Heatmap
# ==========================================================


plt.figure(figsize=(10,12))


heatmap_data=gauge_shap_df[

gauge_shap_df["Model"]
==
"Optimized_XGBoost"

].pivot_table(

index="Gauge_ID",

columns="Feature",

values="Mean_ABS_SHAP"

)



plt.imshow(
heatmap_data,
aspect="auto"
)


plt.colorbar()


plt.xticks(

range(len(heatmap_data.columns)),

heatmap_data.columns,

rotation=90,

fontsize=10

)


plt.yticks(

range(len(heatmap_data.index)),

heatmap_data.index

)


plt.title(
"Gauge-wise SHAP Importance - XGBoost"
)


plt.tight_layout()


plt.savefig(

os.path.join(
OUTPUT_DIR,
"Gauge_Wise_SHAP_Heatmap.png"
),

dpi=300

)


plt.close()



print("="*80)
print("SHAP ANALYSIS COMPLETED")
print("="*80)


print("\nSaved folder:")
print(OUTPUT_DIR)

# ==========================================================
# PART 4
# Heatmap
# ==========================================================

plt.figure(figsize=(10,12))


heatmap_data = gauge_shap_df[
    gauge_shap_df["Model"] == "Optimized_XGBoost"
].pivot_table(
    index="Gauge_ID",
    columns="Feature",
    values="Mean_ABS_SHAP"
)


plt.imshow(
    heatmap_data,
    aspect="auto"
)


# ----------------------------------------------------------
# Colorbar
# ----------------------------------------------------------

cbar = plt.colorbar()

cbar.ax.tick_params(
    labelsize=12
)

cbar.set_label(
    "Mean |SHAP value|",
    fontsize=14
)


# ----------------------------------------------------------
# X-axis
# ----------------------------------------------------------

plt.xticks(
    range(len(heatmap_data.columns)),
    heatmap_data.columns,
    rotation=90,
    fontsize=10
)

plt.xlabel(
    "Feature",
    fontsize=14
)


# ----------------------------------------------------------
# Y-axis
# ----------------------------------------------------------

plt.yticks(
    range(len(heatmap_data.index)),
    heatmap_data.index,
    fontsize=12
)

plt.ylabel(
    "Gauge ID",
    fontsize=14
)


# ----------------------------------------------------------
# Title
# ----------------------------------------------------------

plt.title(
    "Gauge-wise SHAP Importance - Optimized XGBoost",
    fontsize=16
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "Gauge_Wise_SHAP_Heatmap.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close()

performance=pd.read_csv(

"./flood_project_workspace/Results/Spatial_Transfer/Predictions/Per_Gauge_Performance.csv"

)



# average models

perf_avg=performance.groupby(
    "Gauge_ID"
)[
[
"F1",
"Recall",
"PR_AUC"
]

].mean()



print("\nPerformance summary")
print(perf_avg)



# define groups

successful=perf_avg[
    perf_avg["Recall"]>=0.3
].index



failed=perf_avg[
    perf_avg["Recall"]<0.3
].index



print("\nSuccessful gauges:")
print(list(successful))


print("\nFailed gauges:")
print(list(failed))



# use XGBoost global importance

xgb_shap=gauge_shap_df[

    gauge_shap_df["Model"]
    ==
    "Optimized_XGBoost"

]



success_mean=xgb_shap[
    xgb_shap["Gauge_ID"].isin(successful)

].groupby(
"Feature"
)[
"Mean_ABS_SHAP"
].mean()



failure_mean=xgb_shap[
    xgb_shap["Gauge_ID"].isin(failed)

].groupby(
"Feature"
)[
"Mean_ABS_SHAP"
].mean()



comparison=pd.DataFrame({

"Successful_Gauges_SHAP":
success_mean,

"Failed_Gauges_SHAP":
failure_mean

})



comparison["Difference"]=(

comparison["Failed_Gauges_SHAP"]

-

comparison["Successful_Gauges_SHAP"]

)



comparison=comparison.sort_values(
"Difference",
ascending=False
)



comparison.to_csv(

os.path.join(
OUTPUT_DIR,
"Successful_vs_Failed_SHAP_Comparison.csv"
),

index=True

)


# ============================================================
# Figure: SHAP importance difference (journal style)
# ============================================================

from utils_github_loader import apply_journal_style
apply_journal_style()

plot_df = comparison.dropna(subset=["Difference"]).sort_values("Difference", ascending=False)
top_n = 6
plot_df = pd.concat([plot_df.head(top_n), plot_df.tail(top_n)]).drop_duplicates()
plot_df = plot_df.sort_values("Difference", ascending=True)

colors = ["#4C72B0" if v < 0 else "#C44E52" for v in plot_df["Difference"]]

fig, ax = plt.subplots(figsize=(9, max(5, 0.5 * len(plot_df))))
bars = ax.barh(plot_df.index, plot_df["Difference"], color=colors)

for bar, value in zip(bars, plot_df["Difference"]):
    x_pos = value + (0.01 if value >= 0 else -0.01)
    ha = "left" if value >= 0 else "right"
    ax.text(x_pos, bar.get_y() + bar.get_height() / 2, f"{value:+.3f}",
            va="center", ha=ha, fontsize=12)

ax.axvline(0, color="black", linewidth=1)
ax.set_xlabel("SHAP Importance Difference\n(Limited-Detection Gauges - Successful Gauges)")
ax.set_ylabel("Feature")
ax.set_title("SHAP Importance Difference: Limited-Detection vs Successful Gauges")

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "SHAP_Importance_Difference.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close()



# ==========================================================
# PART 4
# Heatmap
# ==========================================================


plt.figure(figsize=(10,12))


heatmap_data=gauge_shap_df[

gauge_shap_df["Model"]
==
"Optimized_XGBoost"

].pivot_table(

index="Gauge_ID",

columns="Feature",

values="Mean_ABS_SHAP"

)



plt.imshow(
heatmap_data,
aspect="auto"
)


plt.colorbar()


plt.xticks(

range(len(heatmap_data.columns)),

heatmap_data.columns,

rotation=90,

fontsize=8

)


plt.yticks(

range(len(heatmap_data.index)),

heatmap_data.index

)


plt.title(
"Gauge-wise SHAP Importance - XGBoost"
)


plt.tight_layout()


plt.savefig(

os.path.join(
OUTPUT_DIR,
"Gauge_Wise_SHAP_Heatmap.png"
),

dpi=300

)


plt.close()



print("="*80)
print("SHAP ANALYSIS COMPLETED")
print("="*80)


print("\nSaved folder:")
print(OUTPUT_DIR)
