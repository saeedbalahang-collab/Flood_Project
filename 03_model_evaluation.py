
# ============================================================
# Script 03 of 07 - Model evaluation on the test set
# (comprehensive evaluation, feature importance, calibration,
#  threshold optimization on validation/test)
#
# Requires: the output of script 01 (X_test_scaled.csv, y_test.csv, ...)
# Every model (baseline + advanced + optimized) is downloaded from
# GitHub - training is never repeated.
# ============================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_github_loader import BASE_DIR, ensure_all_trained_models, pip_install  # noqa: F401

ensure_all_trained_models()

"""**CELL 24
# COMPREHENSIVE MODEL EVALUATION**
"""

pip_install("catboost")

# ========================

# CELL 24
# COMPREHENSIVE MODEL EVALUATION
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

from sklearn.calibration import calibration_curve

# ------------------------------------------------
print("="*70)
print("COMPREHENSIVE MODEL EVALUATION")
print("="*70)

base = "./flood_project_workspace/Results"

input_dir = os.path.join(base,"ML_Training_Input")
model_dir = os.path.join(base,"ML_Models")
output_dir = os.path.join(BASE_DIR, "Results", "Model_Evaluation")

os.makedirs(output_dir,exist_ok=True)

# ------------------------------------------------

X_test = pd.read_csv(os.path.join(input_dir,"X_test_scaled.csv"))
y_test = pd.read_csv(os.path.join(input_dir,"y_test.csv")).values.ravel()

# ------------------------------------------------

models = {}

# Rename models
rename_models = {
    "Best_XGBoost": "Optimized_XGBoost_model",
    "Best_CatBoost": "Optimized_Catboost_model"
}

for root, dirs, files in os.walk(model_dir):

    for f in files:

        if not f.endswith(".pkl"):
            continue

        # Exclude the LightGBM model
        if f == "LightGBM_Model.pkl":
            continue

        name = f.replace(".pkl", "")

        name = rename_models.get(name, name)

        models[name] = joblib.load(os.path.join(root, f))

print("-"*60)
print(f"Loaded {len(models)} model(s):")
for m in sorted(models.keys()):
    print(" -", m)

expected_base_names = [
    "Logistic_Regression", "Random_Forest", "Gradient_Boosting",
    "XGBoost_Model", "CatBoost_Model",
    "Optimized_XGBoost_model", "Optimized_Catboost_model",
]
missing = [m for m in expected_base_names if m not in models]
if missing:
    print("[WARN] The following expected models were NOT loaded "
          "(scroll up for a [WARN]/download error message, and make "
          "sure the file exists under this exact name in your GitHub "
          "repository - see ALL_TRAINED_MODELS in utils_github_loader.py):")
    for m in missing:
        print("   -", m)
print("-"*60)


# ------------------------------------------------

summary=[]

plt.figure(figsize=(8,6))

for name,model in models.items():

    prob=model.predict_proba(X_test)[:,1]

    pred=(prob>=0.5).astype(int)

    summary.append({

        "Model":name,
        "Accuracy":accuracy_score(y_test,pred),
        "Precision":precision_score(y_test,pred),
        "Recall":recall_score(y_test,pred),
        "F1":f1_score(y_test,pred),
        "ROC_AUC":roc_auc_score(y_test,prob),
        "PR_AUC":average_precision_score(y_test,prob),
        "MCC":matthews_corrcoef(y_test,pred),
        "BalancedAccuracy":balanced_accuracy_score(y_test,pred),
        "Kappa":cohen_kappa_score(y_test,pred)

    })

    fpr,tpr,_=roc_curve(y_test,prob)

    plt.plot(fpr,tpr,label=name)

plt.plot([0,1],[0,1],'k--')

plt.legend()

plt.xlabel("False Positive Rate", fontsize=14)

plt.ylabel("True Positive Rate", fontsize=14)

plt.tight_layout()

plt.savefig(os.path.join(output_dir,"ROC_Curves.png"),dpi=300)

plt.close()

# ------------------------------------------------
print(models)

plt.figure(figsize=(8,6))

for name,model in models.items():

    prob=model.predict_proba(X_test)[:,1]

    p,r,_=precision_recall_curve(y_test,prob)

    plt.plot(r,p,label=name)

plt.legend()

plt.xlabel("Recall", fontsize=14)

plt.ylabel("Precision", fontsize=14)

plt.tight_layout()

plt.savefig(os.path.join(output_dir,"Precision_Recall_Curves.png"),dpi=300)

plt.close()

# ------------------------------------------------

results=pd.DataFrame(summary)

results.sort_values("PR_AUC",ascending=False,inplace=True)

results.to_csv(os.path.join(output_dir,"Model_Performance_All.csv"),index=False)

# ------------------------------------------------

plt.figure(figsize=(10,5))

results.set_index("Model")["PR_AUC"].plot.bar()

plt.tight_layout()

plt.savefig(os.path.join(output_dir,"PR_AUC_Comparison.png"),dpi=300)

plt.close()

# ------------------------------------------------

print(results)

print("="*70)
print("CELL 24 COMPLETED")
print("="*70)

plt.rcParams.update({
    "font.weight": "normal",
    "axes.labelweight": "normal",
    "axes.titleweight": "normal",
    "legend.fontsize": 12
})

"""**CELL 25**
# MODEL EXPLAINABILITY AND FEATURE IMPORTANCE ANALYSIS
"""

# ======================================================================
# CELL 25
# MODEL EXPLAINABILITY AND FEATURE IMPORTANCE ANALYSIS
# ======================================================================

import os
import json
import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance


print("="*70)
print("MODEL EXPLAINABILITY ANALYSIS STARTED")
print("="*70)


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

BASE = "./flood_project_workspace/Results"

MODEL_PATH = (
    BASE +
    "/ML_Models/Optuna_Optimization/"
)

DATA_PATH = (
    BASE +
    "/ML_Training_Input/"
)

OUTPUT_PATH = (
    BASE +
    "/Explainability/"
)

os.makedirs(
    OUTPUT_PATH,
    exist_ok=True
)



# ----------------------------------------------------------------------
# Load test data
# ----------------------------------------------------------------------

print("-"*70)
print("Loading datasets")
print("-"*70)


X_test = pd.read_csv(
    DATA_PATH +
    "X_test_scaled.csv"
)


y_test = pd.read_csv(
    DATA_PATH +
    "y_test.csv"
)

test_df = pd.read_csv("./flood_project_workspace/Results/ML_Training_Input/Temporal_Test.csv"
)

if y_test.shape[1] == 1:
    y_test = y_test.iloc[:,0]


print("X_test:", X_test.shape)
print("y_test:", y_test.shape)



# ----------------------------------------------------------------------
# Load feature names
# ----------------------------------------------------------------------

drop_columns = [

    "Flood_Event",
    "Q",
    "gauge_id",
    "Date"

]


feature_columns = [

    c for c in test_df.columns

    if c not in drop_columns

]

feature_metadata = pd.DataFrame({

    "Feature":feature_columns

})


feature_metadata.to_csv(
    output_dir+"/Final_Feature_List.csv",
    index=False
)
features = feature_metadata


if features.shape[1] == 1:
    feature_names = features.iloc[:,0].tolist()
else:
    feature_names = features.columns.tolist()


# safety check

if len(feature_names) != X_test.shape[1]:

    feature_names = [
        f"Feature_{i}"
        for i in range(X_test.shape[1])
    ]


X_test.columns = feature_names


print("Features:", len(feature_names))



# ----------------------------------------------------------------------
# Load optimized models
# ----------------------------------------------------------------------

print("-"*70)
print("Loading optimized models")
print("-"*70)


xgb_model = joblib.load(
    MODEL_PATH +
    "Best_XGBoost.pkl"
)


cat_model = joblib.load(
    MODEL_PATH +
    "Best_CatBoost.pkl"
)


print("Models loaded successfully")



# ----------------------------------------------------------------------
# Create SHAP sample
# ----------------------------------------------------------------------

print("-"*70)
print("Creating SHAP sample")
print("-"*70)


np.random.seed(42)


sample_size = min(
    5000,
    len(X_test)
)


X_sample = X_test.sample(
    sample_size,
    random_state=42
)


print(
    "SHAP sample:",
    X_sample.shape
)



# ======================================================================
# XGBOOST EXPLAINABILITY
# ======================================================================

print("="*70)
print("XGBOOST EXPLAINABILITY")
print("="*70)



# Native importance

xgb_importance = pd.DataFrame({

    "Feature":
    feature_names,

    "Importance":
    xgb_model.feature_importances_

})


xgb_importance = xgb_importance.sort_values(
    "Importance",
    ascending=False
)


xgb_importance.to_csv(
    OUTPUT_PATH +
    "XGBoost_Feature_Importance.csv",
    index=False
)



# SHAP

explainer_xgb = shap.TreeExplainer(
    xgb_model
)


shap_values_xgb = explainer_xgb(
    X_sample
)


shap_df_xgb = pd.DataFrame({

    "Feature":
    feature_names,

    "Mean_ABS_SHAP":
    np.abs(
        shap_values_xgb.values
    ).mean(axis=0)

})


shap_df_xgb = shap_df_xgb.sort_values(
    "Mean_ABS_SHAP",
    ascending=False
)


shap_df_xgb.to_csv(
    OUTPUT_PATH +
    "XGBoost_SHAP_Feature_Ranking.csv",
    index=False
)



plt.figure(figsize=(10,8))

shap.summary_plot(
    shap_values_xgb,
    X_sample,
    show=False,
    max_display=20
)

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH +
    "XGBoost_SHAP_Summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()



# ======================================================================
# CATBOOST EXPLAINABILITY
# ======================================================================

print("="*70)
print("CATBOOST EXPLAINABILITY")
print("="*70)



cat_importance = pd.DataFrame({

    "Feature":
    feature_names,

    "Importance":
    cat_model.get_feature_importance()

})


cat_importance = cat_importance.sort_values(
    "Importance",
    ascending=False
)


cat_importance.to_csv(
    OUTPUT_PATH +
    "CatBoost_Feature_Importance.csv",
    index=False
)



# SHAP

explainer_cat = shap.TreeExplainer(
    cat_model
)


shap_values_cat = explainer_cat(
    X_sample
)


shap_df_cat = pd.DataFrame({

    "Feature":
    feature_names,

    "Mean_ABS_SHAP":
    np.abs(
        shap_values_cat.values
    ).mean(axis=0)

})


shap_df_cat = shap_df_cat.sort_values(
    "Mean_ABS_SHAP",
    ascending=False
)


shap_df_cat.to_csv(
    OUTPUT_PATH +
    "CatBoost_SHAP_Feature_Ranking.csv",
    index=False
)



plt.figure(figsize=(10,8))


shap.summary_plot(
    shap_values_cat,
    X_sample,
    show=False,
    max_display=20
)


plt.tight_layout()


plt.savefig(
    OUTPUT_PATH +
    "CatBoost_SHAP_Summary.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()



# ----------------------------------------------------------------------
# Combine ranking
# ----------------------------------------------------------------------

comparison = pd.merge(

    xgb_importance.rename(
        columns={
            "Importance":
            "XGB_Importance"
        }
    ),

    cat_importance.rename(
        columns={
            "Importance":
            "CatBoost_Importance"
        }
    ),

    on="Feature",
    how="outer"

)



comparison.to_csv(
    OUTPUT_PATH +
    "Feature_Importance_Comparison.csv",
    index=False
)



# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------

print("="*70)
print("FILES SAVED")
print("="*70)


for f in os.listdir(OUTPUT_PATH):
    print(f)


print("="*70)
print("CELL 28 COMPLETED SUCCESSFULLY")
print("="*70)

"""# **CELL 26 #CALIBRATION & RELIABILITY ANALYSIS**"""

# ======================================================================
# CALIBRATION & RELIABILITY ANALYSIS
# CELL 29F
# ======================================================================


import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
    average_precision_score
)

import joblib


print("="*70)
print("CALIBRATION ANALYSIS STARTED")
print("="*70)



# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

model_path = (
"./flood_project_workspace/Results/ML_Models/Optuna_Optimization"
)



output_path = (
"./flood_project_workspace/Calibration"
)


os.makedirs(
    output_path,
    exist_ok=True
)


X=pd.read_csv("./flood_project_workspace/Results/ML_Training_Input/X_test_scaled.csv")
y= pd.read_csv("./flood_project_workspace/Results/ML_Training_Input/y_test.csv")

print("Test shape")
print(X.shape)



# ---------------------------------------------------------
# Models
# ---------------------------------------------------------

models={

"Optimized_XGBoost":
joblib.load(
os.path.join(
model_path,
"Best_XGBoost.pkl"
)
),


"Optimized_CatBoost":
joblib.load(
os.path.join(
model_path,
"Best_CatBoost.pkl"
)
)

}

results=[]

print(models)
plt.figure(figsize=(6,6))

for name,model in models.items():

    print("-"*50)
    print(name)


    prob=model.predict_proba(X)[:,1]


    brier=brier_score_loss(
        y,
        prob
    )


    pr_auc=average_precision_score(
        y,
        prob
    )


    roc=roc_auc_score(
        y,
        prob
    )


    results.append({

        "Model":name,
        "PR_AUC":pr_auc,
        "ROC_AUC":roc,
        "Brier_Score":brier

    })


    # calibration curve

    frac_pos,mean_pred=calibration_curve(
        y,
        prob,
        n_bins=10
    )

    print(name, prob.min(), prob.max(), len(prob))
    #plt.figure(figsize=(6,6))


    plt.plot(
        mean_pred,
        frac_pos,
        marker="o",
        label=name
    )


plt.plot(
[0,1],
[0,1],
"--",
label="Perfect Calibration"
)


plt.xlabel(
"Mean predicted probability", fontsize =14
)

plt.ylabel(
"Observed frequency", fontsize = 14
)

plt.title(
"Calibration Curve", fontsize = 16
)

plt.legend()

plt.grid(alpha=0.3)


plt.tight_layout()


plt.savefig(
os.path.join(
output_path,
"Calibration_Curve.png"
),
dpi=300
)


plt.close()



# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------


calibration_df=pd.DataFrame(results)


calibration_df.to_csv(
os.path.join(
output_path,
"Calibration_Performance.csv"
),
index=False
)



print("="*70)
print("CALIBRATION RESULTS")
print("="*70)

print(calibration_df)



print("="*70)
print("CELL 26 COMPLETED")
print("="*70)

"""# **CELL 27: Corrected Threshold Optimization**"""

# ============================================================
# CELL 27: Corrected Threshold Optimization (validation set)
#             + Bootstrap Confidence Intervals + XGB-vs-CatBoost test
#
# Purpose:
#   1. Re-run F1-threshold optimization on the VALIDATION set
#      (Cell 30 incorrectly used the test set).
#   2. Apply the resulting threshold, fixed, to the test set only.
#   3. Bootstrap CIs for PR-AUC/ROC-AUC/F1/Brier on the test set.
#   4. Paired bootstrap test of XGBoost vs CatBoost PR-AUC difference.
# ============================================================

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss
)

RNG = np.random.default_rng(42)
N_BOOT = 2000

X_val = pd.read_csv("./flood_project_workspace/Results/ML_Training_Input/X_validation_scaled.csv")
y_val = pd.read_csv("./flood_project_workspace/Results/ML_Training_Input/y_validation.csv").values.ravel()
X_test = pd.read_csv("./flood_project_workspace/Results/ML_Training_Input/X_test_scaled.csv")
y_test = pd.read_csv("./flood_project_workspace/Results/ML_Training_Input/y_test.csv").values.ravel()

models = {
    "Optimized_XGBoost": joblib.load("./flood_project_workspace/Results/ML_Models/Optuna_Optimization/Best_XGBoost.pkl"),
    "Optimized_CatBoost": joblib.load("./flood_project_workspace/Results/ML_Models/Optuna_Optimization/Best_CatBoost.pkl"),
}

# --- Step 1-2: correct threshold optimization on VALIDATION, applied to TEST ---
thresholds = np.arange(0.01, 0.99, 0.01)
test_probs, corrected_results = {}, []

for name, model in models.items():
    val_prob = model.predict_proba(X_val)[:, 1]
    best_f1, best_t = -1, 0.5
    for t in thresholds:
        f1 = f1_score(y_val, (val_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    test_prob = model.predict_proba(X_test)[:, 1]
    test_probs[name] = test_prob
    pred = (test_prob >= best_t).astype(int)

    corrected_results.append({
        "Model": name,
        "Validation_Optimal_Threshold": best_t,
        "Test_Precision_at_threshold": precision_score(y_test, pred, zero_division=0),
        "Test_Recall_at_threshold": recall_score(y_test, pred, zero_division=0),
        "Test_F1_at_threshold": f1_score(y_test, pred, zero_division=0),
    })

corrected_df = pd.DataFrame(corrected_results)
corrected_df.to_csv("./flood_project_workspace/Results/Threshold_Correction/Corrected_Threshold_Results.csv", index=False)
print(corrected_df)

# --- Step 3: bootstrap CIs on the test set ---
n = len(y_test)
boot_records = []
for name, prob in test_probs.items():
    stats = {"PR_AUC": [], "ROC_AUC": [], "Brier": []}
    for _ in range(N_BOOT):
        idx = RNG.integers(0, n, n)
        yb, pb = y_test[idx], prob[idx]
        if yb.sum() == 0 or yb.sum() == n:
            continue
        stats["PR_AUC"].append(average_precision_score(yb, pb))
        stats["ROC_AUC"].append(roc_auc_score(yb, pb))
        stats["Brier"].append(brier_score_loss(yb, pb))
    for metric, vals in stats.items():
        lo, hi = np.percentile(vals, [2.5, 97.5])
        boot_records.append({"Model": name, "Metric": metric,
                              "Point_Estimate": np.mean(vals),
                              "CI_Lower": lo, "CI_Upper": hi})

boot_df = pd.DataFrame(boot_records)
boot_df.to_csv("./flood_project_workspace/Results/Threshold_Correction/Bootstrap_CIs.csv", index=False)
print(boot_df)

# --- Step 4: paired bootstrap test, XGBoost vs CatBoost PR-AUC ---
xgb_prob, cat_prob = test_probs["Optimized_XGBoost"], test_probs["Optimized_CatBoost"]
diffs = []
for _ in range(N_BOOT):
    idx = RNG.integers(0, n, n)
    yb = y_test[idx]
    if yb.sum() == 0 or yb.sum() == n:
        continue
    diffs.append(average_precision_score(yb, xgb_prob[idx]) - average_precision_score(yb, cat_prob[idx]))

diffs = np.array(diffs)
p_value = 2 * min((diffs > 0).mean(), (diffs < 0).mean())
print(f"XGBoost - CatBoost PR-AUC diff: mean={diffs.mean():.4f}, "
      f"95% CI=({np.percentile(diffs,2.5):.4f}, {np.percentile(diffs,97.5):.4f}), "
      f"two-sided bootstrap p={p_value:.3f}")

pd.DataFrame({"Diff": diffs}).to_csv(
    "./flood_project_workspace/Results/Threshold_Correction/XGB_vs_CatBoost_PRAUC_Bootstrap.csv", index=False
)




# ============================================================
# Step 5: Visualization
# ============================================================

import matplotlib.pyplot as plt


fig, axes = plt.subplots(
    2, 2,
    figsize=(12,10)
)


# ------------------------------------------------------------
# (a) Threshold optimization curves
# ------------------------------------------------------------

for name, model in models.items():

    val_prob = model.predict_proba(X_val)[:,1]

    f1_values=[]

    for t in thresholds:

        f1_values.append(
            f1_score(
                y_val,
                (val_prob>=t).astype(int),
                zero_division=0
            )
        )

    axes[0,0].plot(
        thresholds,
        f1_values,
        label=name
    )


axes[0,0].set_xlabel(
    "Probability Threshold"
)

axes[0,0].set_ylabel(
    "F1-score"
)

axes[0,0].set_title(
    "(a) Validation threshold optimization"
)

axes[0,0].legend()



# ------------------------------------------------------------
# (b) PR-AUC confidence intervals
# ------------------------------------------------------------

pr_models=[]
pr_values=[]
pr_low=[]
pr_high=[]


for _,row in boot_df[
    boot_df["Metric"]=="PR_AUC"
].iterrows():

    pr_models.append(row["Model"])
    pr_values.append(row["Point_Estimate"])
    pr_low.append(
        row["CI_Lower"]
    )
    pr_high.append(
        row["CI_Upper"]
    )


x=np.arange(len(pr_models))

axes[0,1].errorbar(
    x,
    pr_values,
    yerr=[
        np.array(pr_values)-np.array(pr_low),
        np.array(pr_high)-np.array(pr_values)
    ],
    fmt="o",
    capsize=5
)


axes[0,1].set_xticks(x)
axes[0,1].set_xticklabels(
    pr_models,
    rotation=45,
    ha="right"
)

axes[0,1].set_ylabel(
    "PR-AUC"
)

axes[0,1].set_title(
  "(b) Bootstrap PR-AUC uncertainty"
)



# ------------------------------------------------------------
# (c) ROC-AUC confidence intervals
# ------------------------------------------------------------

roc_models=[]
roc_values=[]
roc_low=[]
roc_high=[]


for _,row in boot_df[
    boot_df["Metric"]=="ROC_AUC"
].iterrows():

    roc_models.append(row["Model"])
    roc_values.append(row["Point_Estimate"])
    roc_low.append(row["CI_Lower"])
    roc_high.append(row["CI_Upper"])



x=np.arange(len(roc_models))


axes[1,0].errorbar(
    x,
    roc_values,
    yerr=[
        np.array(roc_values)-np.array(roc_low),
        np.array(roc_high)-np.array(roc_values)
    ],
    fmt="o",
    capsize=5
)


axes[1,0].set_xticks(x)

axes[1,0].set_xticklabels(
    roc_models,
    rotation=45,
    ha="right"
)

axes[1,0].set_ylabel(
    "ROC-AUC"
)

axes[1,0].set_title(
    "(c) Bootstrap ROC-AUC uncertainty"
)



# ------------------------------------------------------------
# (d) Paired bootstrap PR-AUC difference
# ------------------------------------------------------------


axes[1,1].hist(
    diffs,
    bins=40
)


axes[1,1].axvline(
    0,
    linestyle="--"
)


axes[1,1].set_xlabel(
    "PR-AUC Difference (XGBoost - CatBoost)"
)

axes[1,1].set_ylabel(
    "Frequency"
)

axes[1,1].set_title(
    "(d) Paired bootstrap comparison"
)



plt.tight_layout()


figure_path = (
"./flood_project_workspace/Results/Threshold_Correction/"
"Threshold_Bootstrap_Robustness_Analysis.png"
)


plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
"Figure saved:",
figure_path
)

# ============================================================
# Step 5: Visualization
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Output directory
# ------------------------------------------------------------

output_dir = "./flood_project_workspace/Results/Model_Evaluation"
os.makedirs(output_dir, exist_ok=True)


# ------------------------------------------------------------
# Global journal-style font settings
# ------------------------------------------------------------

LABEL_FONT_SIZE = 14
TICK_FONT_SIZE = 11
TITLE_FONT_SIZE = 13
LEGEND_FONT_SIZE = 12


# ============================================================
# (a) Threshold optimization curves
# ============================================================

fig, ax = plt.subplots(figsize=(7, 5.5))

for name, model in models.items():

    val_prob = model.predict_proba(X_val)[:, 1]

    f1_values = []

    for t in thresholds:

        f1_values.append(
            f1_score(
                y_val,
                (val_prob >= t).astype(int),
                zero_division=0
            )
        )

    ax.plot(
        thresholds,
        f1_values,
        linewidth=2,
        label=name
    )


ax.set_xlabel(
    "Probability Threshold",
    fontsize=LABEL_FONT_SIZE
)

ax.set_ylabel(
    "F1-score",
    fontsize=LABEL_FONT_SIZE
)

ax.set_title(
    "(a) Validation threshold optimization",
    fontsize=TITLE_FONT_SIZE,
    fontweight="bold"
)

ax.tick_params(
    axis="both",
    labelsize=TICK_FONT_SIZE
)

ax.legend(
    fontsize=LEGEND_FONT_SIZE,
    frameon=False
)

ax.grid(
    True,
    linestyle="--",
    alpha=0.3
)

fig.tight_layout()

figure_path = os.path.join(
    output_dir,
    "a_Threshold_Optimization.png"
)

fig.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

print("Figure (a) saved:", figure_path)


# ============================================================
# (b) PR-AUC confidence intervals
# ============================================================

pr_models = []
pr_values = []
pr_low = []
pr_high = []


for _, row in boot_df[
    boot_df["Metric"] == "PR_AUC"
].iterrows():

    pr_models.append(row["Model"])
    pr_values.append(row["Point_Estimate"])
    pr_low.append(row["CI_Lower"])
    pr_high.append(row["CI_Upper"])


x = np.arange(len(pr_models))


fig, ax = plt.subplots(figsize=(7, 5.5))

ax.errorbar(
    x,
    pr_values,
    yerr=[
        np.array(pr_values) - np.array(pr_low),
        np.array(pr_high) - np.array(pr_values)
    ],
    fmt="o",
    markersize=7,
    capsize=5,
    capthick=1.5,
    linewidth=1.5
)


ax.set_xticks(x)

ax.set_xticklabels(
    pr_models,
    rotation=45,
    ha="right",
    fontsize=TICK_FONT_SIZE
)

ax.set_ylabel(
    "PR-AUC",
    fontsize=LABEL_FONT_SIZE
)

ax.set_title(
    "(b) Bootstrap PR-AUC uncertainty",
    fontsize=TITLE_FONT_SIZE,
    fontweight="bold"
)

ax.tick_params(
    axis="y",
    labelsize=TICK_FONT_SIZE
)

ax.grid(
    True,
    axis="y",
    linestyle="--",
    alpha=0.3
)

fig.tight_layout()

figure_path = os.path.join(
    output_dir,
    "b_PR_AUC_Confidence_Intervals.png"
)

fig.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

print("Figure (b) saved:", figure_path)


# ============================================================
# (c) ROC-AUC confidence intervals
# ============================================================

roc_models = []
roc_values = []
roc_low = []
roc_high = []


for _, row in boot_df[
    boot_df["Metric"] == "ROC_AUC"
].iterrows():

    roc_models.append(row["Model"])
    roc_values.append(row["Point_Estimate"])
    roc_low.append(row["CI_Lower"])
    roc_high.append(row["CI_Upper"])


x = np.arange(len(roc_models))


fig, ax = plt.subplots(figsize=(7, 5.5))

ax.errorbar(
    x,
    roc_values,
    yerr=[
        np.array(roc_values) - np.array(roc_low),
        np.array(roc_high) - np.array(roc_values)
    ],
    fmt="o",
    markersize=7,
    capsize=5,
    capthick=1.5,
    linewidth=1.5
)


ax.set_xticks(x)

ax.set_xticklabels(
    roc_models,
    rotation=45,
    ha="right",
    fontsize=TICK_FONT_SIZE
)

ax.set_ylabel(
    "ROC-AUC",
    fontsize=LABEL_FONT_SIZE
)

ax.set_title(
    "(c) Bootstrap ROC-AUC uncertainty",
    fontsize=TITLE_FONT_SIZE,
    fontweight="bold"
)

ax.tick_params(
    axis="y",
    labelsize=TICK_FONT_SIZE
)

ax.grid(
    True,
    axis="y",
    linestyle="--",
    alpha=0.3
)

fig.tight_layout()

figure_path = os.path.join(
    output_dir,
    "c_ROC_AUC_Confidence_Intervals.png"
)

fig.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

print("Figure (c) saved:", figure_path)


# ============================================================
# (d) Paired bootstrap PR-AUC difference
# ============================================================

fig, ax = plt.subplots(figsize=(7, 5.5))

ax.hist(
    diffs,
    bins=40,
    edgecolor="black",
    linewidth=0.7
)


ax.axvline(
    0,
    linestyle="--",
    linewidth=1.5
)


ax.set_xlabel(
    "PR-AUC Difference (XGBoost - CatBoost)",
    fontsize=LABEL_FONT_SIZE
)

ax.set_ylabel(
    "Frequency",
    fontsize=LABEL_FONT_SIZE
)

ax.set_title(
    "(d) Paired bootstrap comparison",
    fontsize=TITLE_FONT_SIZE,
    fontweight="bold"
)

ax.tick_params(
    axis="both",
    labelsize=TICK_FONT_SIZE
)

ax.grid(
    True,
    axis="y",
    linestyle="--",
    alpha=0.3
)

fig.tight_layout()

figure_path = os.path.join(
    output_dir,
    "d_Paired_Bootstrap_PR_AUC_Difference.png"
)

fig.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

print("Figure (d) saved:", figure_path)


# ============================================================
# Final message
# ============================================================

print("\nAll figures were saved separately in:")
print(output_dir)

# ============================================================
# CELL 27: Corrected Threshold Optimization (validation set)
#             + Bootstrap Confidence Intervals + XGB-vs-CatBoost test
#
# Purpose:
#   1. Re-run F1-threshold optimization on the VALIDATION set
#      (Cell 30 incorrectly used the test set).
#   2. Apply the resulting threshold, fixed, to the test set only.
#   3. Bootstrap CIs for PR-AUC/ROC-AUC/F1/Brier on the test set.
#   4. Paired bootstrap test of XGBoost vs CatBoost PR-AUC difference.
# ============================================================

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss
)

RNG = np.random.default_rng(42)
N_BOOT = 2000

X_val = pd.read_csv("./flood_project_workspace/Results/ML_Training_Input/X_validation_scaled.csv")
y_val = pd.read_csv("./flood_project_workspace/Results/ML_Training_Input/y_validation.csv").values.ravel()
X_test = pd.read_csv("./flood_project_workspace/Results/ML_Training_Input/X_test_scaled.csv")
y_test = pd.read_csv("./flood_project_workspace/Results/ML_Training_Input/y_test.csv").values.ravel()

models = {
    "Optimized_XGBoost": joblib.load("./flood_project_workspace/Results/ML_Models/Optuna_Optimization/Best_XGBoost.pkl"),
    "Optimized_CatBoost": joblib.load("./flood_project_workspace/Results/ML_Models/Optuna_Optimization/Best_CatBoost.pkl"),
}

# --- Step 1-2: correct threshold optimization on VALIDATION, applied to TEST ---
thresholds = np.arange(0.01, 0.99, 0.01)
test_probs, corrected_results = {}, []

for name, model in models.items():
    val_prob = model.predict_proba(X_val)[:, 1]
    best_f1, best_t = -1, 0.5
    for t in thresholds:
        f1 = f1_score(y_val, (val_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    test_prob = model.predict_proba(X_test)[:, 1]
    test_probs[name] = test_prob
    pred = (test_prob >= best_t).astype(int)

    corrected_results.append({
        "Model": name,
        "Validation_Optimal_Threshold": best_t,
        "Test_Precision_at_threshold": precision_score(y_test, pred, zero_division=0),
        "Test_Recall_at_threshold": recall_score(y_test, pred, zero_division=0),
        "Test_F1_at_threshold": f1_score(y_test, pred, zero_division=0),
    })

corrected_df = pd.DataFrame(corrected_results)
corrected_df.to_csv("./flood_project_workspace/Corrected_Threshold_Results.csv", index=False)
print(corrected_df)

# --- Step 3: bootstrap CIs on the test set ---
n = len(y_test)
boot_records = []
for name, prob in test_probs.items():
    stats = {"PR_AUC": [], "ROC_AUC": [], "Brier": []}
    for _ in range(N_BOOT):
        idx = RNG.integers(0, n, n)
        yb, pb = y_test[idx], prob[idx]
        if yb.sum() == 0 or yb.sum() == n:
            continue
        stats["PR_AUC"].append(average_precision_score(yb, pb))
        stats["ROC_AUC"].append(roc_auc_score(yb, pb))
        stats["Brier"].append(brier_score_loss(yb, pb))
    for metric, vals in stats.items():
        lo, hi = np.percentile(vals, [2.5, 97.5])
        boot_records.append({"Model": name, "Metric": metric,
                              "Point_Estimate": np.mean(vals),
                              "CI_Lower": lo, "CI_Upper": hi})

boot_df = pd.DataFrame(boot_records)
boot_df.to_csv("./flood_project_workspace/Bootstrap_CIs.csv", index=False)
print(boot_df)

# --- Step 4: paired bootstrap test, XGBoost vs CatBoost PR-AUC ---
xgb_prob, cat_prob = test_probs["Optimized_XGBoost"], test_probs["Optimized_CatBoost"]
diffs = []
for _ in range(N_BOOT):
    idx = RNG.integers(0, n, n)
    yb = y_test[idx]
    if yb.sum() == 0 or yb.sum() == n:
        continue
    diffs.append(average_precision_score(yb, xgb_prob[idx]) - average_precision_score(yb, cat_prob[idx]))

diffs = np.array(diffs)
p_value = 2 * min((diffs > 0).mean(), (diffs < 0).mean())
print(f"XGBoost - CatBoost PR-AUC diff: mean={diffs.mean():.4f}, "
      f"95% CI=({np.percentile(diffs,2.5):.4f}, {np.percentile(diffs,97.5):.4f}), "
      f"two-sided bootstrap p={p_value:.3f}")

pd.DataFrame({"Diff": diffs}).to_csv(
    "./flood_project_workspace/XGB_vs_CatBoost_PRAUC_Bootstrap.csv", index=False
)

"""# **Fig 5b**"""

# ============================================================
# Validation Threshold Optimization Curve
# ============================================================

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

plt.figure(figsize=(8,5))

for name, model in models.items():

    val_prob = model.predict_proba(X_val)[:,1]

    f1_values = []

    for t in thresholds:

        f1_values.append(
            f1_score(
                y_val,
                (val_prob >= t).astype(int),
                zero_division=0
            )
        )

    plt.plot(
        thresholds,
        f1_values,
        label=name,
        linewidth=2
    )


plt.xlabel(
    "Probability Threshold",
    fontsize=12
)

plt.ylabel(
    "F1-score",
    fontsize=12
)

plt.title(
    "Validation Threshold Optimization",
    fontsize=13
)

plt.legend()
plt.grid(True)
plt.ylim(0,1)

plt.tight_layout()


# Save figure
figure_path = (
    "./flood_project_workspace/Results/Threshold_Correction/"
    "Validation_Threshold_Optimization.png"
)

plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Figure saved:", figure_path)


# ============================================================
# CELL 31A (moved here from script 04 - it evaluates extreme
# flood detection on the MAIN test set using the corrected
# thresholds produced above, so it belongs with the rest of
# the calibration/threshold-analysis content in this script,
# not with the spatial-transfer data preparation in script 04)
# CORRECTED EXTREME FLOOD EVENT DETECTION ANALYSIS
# ============================================================

import joblib

print("="*70)
print("CORRECTED EXTREME FLOOD EVENT ANALYSIS STARTED")
print("="*70)



# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

base="./flood_project_workspace/Results"


test_path = (
    f"{base}/ML_Training_Input/Temporal_Test.csv"
)


model_path = (
    f"{base}/ML_Models/Optuna_Optimization"
)


threshold_path = (
    './flood_project_workspace/Results/Threshold_Correction/Corrected_Threshold_Results.csv'
)


output_path = (
    f"{base}/Extreme_Event_Analysis/"
    "Corrected"
)


os.makedirs(
    output_path,
    exist_ok=True
)



# ----------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------

test=pd.read_csv(
    test_path
)


features=[
    c for c in test.columns
    if c not in
    [
        "Flood_Event",
        "Date",
        "gauge_id","Q"
    ]
]
#print(pd.read_csv('./flood_project_workspace/Results/ML_Training_Input/X_test_scaled.csv'))

scaler=joblib.load("./flood_project_workspace/Results/ML_Training_Input/StandardScaler.pkl")

#X= scaler().transform(test[features])
X = pd.read_csv('./flood_project_workspace/Results/ML_Training_Input/X_test_scaled.csv')

y=pd.read_csv('./flood_project_workspace/Results/ML_Training_Input/y_test.csv')



print("Total test samples:")
print(test.shape)



# ----------------------------------------------------------------------
# Define true extreme floods
# ----------------------------------------------------------------------

flood_only=test[
    test["Flood_Event"]==1
].copy()



q90=flood_only["Q"].quantile(
    0.90
)



extreme_events=flood_only[
    flood_only["Q"]>=q90
].copy()



X_extreme=extreme_events[features]

y_extreme=extreme_events["Flood_Event"]



print("-"*70)

print("Flood events:")
print(len(flood_only))


print("Extreme flood threshold Q90:")
print(q90)


print("Extreme flood events:")
print(len(extreme_events))



# ----------------------------------------------------------------------
# Load thresholds
# ----------------------------------------------------------------------

thresholds=pd.read_csv(
    threshold_path
)



models_extreme={

"Optimized_XGBoost":
joblib.load(
    f"{model_path}/Best_XGBoost.pkl"
),


"Optimized_CatBoost":
joblib.load(
    f"{model_path}/Best_CatBoost.pkl"
)

}



results=[]



# ----------------------------------------------------------------------
# Evaluate models
# ----------------------------------------------------------------------

for name,model in models_extreme.items():


    print("-"*60)
    print(name)


    threshold=float(
        thresholds[
            thresholds["Model"]==name
        ]
        ["Validation_Optimal_Threshold"]
        .values[0]
    )


    prob=model.predict_proba(
        X_extreme
    )[:,1]


    pred=(prob>=threshold).astype(int)



    detected=pred.sum()

    missed=len(pred)-detected



    detection_rate=(
        detected /
        len(pred)
    )



    results.append({

        "Model":name,

        "Extreme_Flood_Count":
        len(pred),

        "Detected_Extreme_Floods":
        detected,

        "Missed_Extreme_Floods":
        missed,

        "Detection_Rate":
        detection_rate,

        "Threshold":
        threshold

    })



# ----------------------------------------------------------------------
# Save results
# ----------------------------------------------------------------------

results_df=pd.DataFrame(results)



results_df.to_csv(
    f"{output_path}/Corrected_Extreme_Flood_Detection.csv",
    index=False
)



print("="*70)
print("CORRECTED EXTREME FLOOD RESULTS")
print("="*70)

print(results_df)



# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------

plt.figure(
    figsize=(7,5)
)


plt.bar(
    results_df["Model"],
    results_df["Detection_Rate"]
)


plt.ylabel(
    "Extreme Flood Detection Rate"
)


plt.title(
    "Detection Capability for Extreme Flood Events"
)


plt.xticks(
    rotation=20
)


plt.ylim(
    0,
    1
)


plt.tight_layout()


plt.savefig(
    f"{output_path}/Corrected_Extreme_Flood_Detection.png",
    dpi=300
)


plt.close()



print("="*70)
print("FILES SAVED")
print("="*70)

print(
    os.listdir(output_path)
)


print("="*70)
print("CELL 31A COMPLETED SUCCESSFULLY")
print("="*70)
