# ============================================================
# Script 02 of 07 - Model training - REFERENCE ONLY / for code transparency
#
# NOTE: as requested for this project, this script must NOT be
# re-run. The final models (Best_XGBoost.pkl / Best_CatBoost.pkl and
# every other model) were already trained and uploaded to GitHub.
# Scripts 03 through 07 only load these models - training is never
# repeated.
#
# This script is kept in the repository purely to document the full
# training procedure (baseline models, Optuna tuning for XGBoost /
# CatBoost, and saving the final models).
# Requirement to run this script (if ever needed): the output of
# script 01 (./flood_project_workspace/Results/ML_Training_Input/...)
# ============================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_github_loader import BASE_DIR, pip_install  # noqa: F401

# ============================================================
# CELL 21
# BASELINE MACHINE LEARNING MODELS TRAINING
# ============================================================


import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

import joblib


print("="*70)
print("BASELINE ML MODEL TRAINING STARTED")
print("="*70)


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

input_dir = "./flood_project_workspace/Results/ML_Training_Input"

output_dir = "./flood_project_workspace/Results/ML_Models/Baseline"

os.makedirs(output_dir, exist_ok=True)


# ------------------------------------------------------------
# Load datasets
# ------------------------------------------------------------

print("-"*70)
print("Loading training datasets")
print("-"*70)


X_train = pd.read_csv(
    f"{input_dir}/X_train_scaled.csv"
)

X_valid = pd.read_csv(
    f"{input_dir}/X_validation_scaled.csv"
)

X_test = pd.read_csv(
    f"{input_dir}/X_test_scaled.csv"
)


y_train = pd.read_csv(
    f"{input_dir}/y_train.csv"
).values.ravel()


y_valid = pd.read_csv(
    f"{input_dir}/y_validation.csv"
).values.ravel()


y_test = pd.read_csv(
    f"{input_dir}/y_test.csv"
).values.ravel()


print("Train:", X_train.shape)
print("Validation:", X_valid.shape)
print("Test:", X_test.shape)



# ------------------------------------------------------------
# Define models
# ------------------------------------------------------------

models = {


"Logistic_Regression":
LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
),


"Random_Forest":
RandomForestClassifier(
    #n_estimators=300,
    #max_depth=None,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
),


"Gradient_Boosting":
GradientBoostingClassifier(
    #n_estimators=200,
    #learning_rate=0.05,
    #max_depth=5,
    random_state=42
)

}


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

results=[]


for name, model in models.items():

    print("-"*70)
    print("Training:", name)
    print("-"*70)


    model.fit(
        X_train,
        y_train
    )


    # predictions

    pred = model.predict(
        X_test
    )


    prob = model.predict_proba(
        X_test
    )[:,1]


    # metrics

    metrics={

    "Model":name,

    "Accuracy":
    accuracy_score(y_test,pred),

    "Precision":
    precision_score(
        y_test,
        pred,
        zero_division=0
    ),

    "Recall":
    recall_score(
        y_test,
        pred,
        zero_division=0
    ),

    "F1":
    f1_score(
        y_test,
        pred,
        zero_division=0
    ),

    "ROC_AUC":
    roc_auc_score(
        y_test,
        prob
    ),

    "PR_AUC":
    average_precision_score(
        y_test,
        prob
    )

    }


    results.append(metrics)



    # save model

    joblib.dump(
        model,
        f"{output_dir}/{name}.pkl"
    )


    # save predictions

    pred_df=pd.DataFrame({

        "Observed":
        y_test,

        "Predicted":
        pred,

        "Probability":
        prob

    })


    pred_df.to_csv(
        f"{output_dir}/{name}_Predictions.csv",
        index=False
    )


    # confusion matrix

    cm = confusion_matrix(
        y_test,
        pred
    )


    pd.DataFrame(
        cm,
        columns=[
            "Predicted_0",
            "Predicted_1"
        ],
        index=[
            "Observed_0",
            "Observed_1"
        ]
    ).to_csv(
        f"{output_dir}/{name}_Confusion_Matrix.csv"
    )



# ------------------------------------------------------------
# Save comparison table
# ------------------------------------------------------------

results_df=pd.DataFrame(results)


results_df.to_csv(
    f"{output_dir}/Baseline_Model_Performance.csv",
    index=False
)



print("="*70)
print("BASELINE MODEL TRAINING COMPLETED")
print("="*70)


print(results_df)


print("-"*70)
print("FILES SAVED")
print("-"*70)

for f in os.listdir(output_dir):
    print(f)


print("="*70)
print("CELL 21 COMPLETED SUCCESSFULLY")
print("="*70)

"""C22"""

pip_install("catboost")

# ============================================================
# CELL 22
# ADVANCED BOOSTING MODEL TRAINING
# ============================================================

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


print("="*70)
print("ADVANCED BOOSTING MODEL TRAINING STARTED")
print("="*70)


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

input_dir = "./flood_project_workspace/Results/ML_Training_Input"

output_dir = "./flood_project_workspace/Results/ML_Models/Advanced"

os.makedirs(output_dir, exist_ok=True)


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

X_train = pd.read_csv(
    f"{input_dir}/X_train_scaled.csv"
)

X_valid = pd.read_csv(
    f"{input_dir}/X_validation_scaled.csv"
)

X_test = pd.read_csv(
    f"{input_dir}/X_test_scaled.csv"
)


y_train = pd.read_csv(
    f"{input_dir}/y_train.csv"
).values.ravel()


y_valid = pd.read_csv(
    f"{input_dir}/y_validation.csv"
).values.ravel()


y_test = pd.read_csv(
    f"{input_dir}/y_test.csv"
).values.ravel()



print("Datasets loaded")
print("--------------------------------")
print("Train:", X_train.shape)
print("Validation:", X_valid.shape)
print("Test:", X_test.shape)



# ------------------------------------------------------------
# Class imbalance handling
# ------------------------------------------------------------

positive_ratio = np.sum(y_train==0) / np.sum(y_train==1)


models = {

"XGBoost":

XGBClassifier(
    #n_estimators=400,
    #learning_rate=0.05,
    #max_depth=6,
    #subsample=0.8,
    #colsample_bytree=0.8,
    scale_pos_weight=positive_ratio,
    random_state=42,
    eval_metric="aucpr"
),


"LightGBM":

LGBMClassifier(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight="balanced",
    random_state=42
),


"CatBoost":

CatBoostClassifier(
    #iterations=400,
    #learning_rate=0.05,
    #depth=6,
    loss_function="Logloss",
    auto_class_weights="Balanced",
    verbose=0,
    random_seed=42
)

}



results=[]



# ------------------------------------------------------------
# Training loop
# ------------------------------------------------------------

for name, model in models.items():

    print("--------------------------------")
    print("Training:", name)
    print("--------------------------------")


    model.fit(
        X_train,
        y_train
    )


    pred_prob = model.predict_proba(
        X_test
    )[:,1]


    pred = (
        pred_prob >= 0.5
    ).astype(int)



    performance={

        "Model":name,

        "Accuracy":
        accuracy_score(y_test,pred),

        "Precision":
        precision_score(
            y_test,
            pred,
            zero_division=0
        ),

        "Recall":
        recall_score(
            y_test,
            pred,
            zero_division=0
        ),

        "F1":
        f1_score(
            y_test,
            pred,
            zero_division=0
        ),

        "ROC_AUC":
        roc_auc_score(
            y_test,
            pred_prob
        ),

        "PR_AUC":
        average_precision_score(
            y_test,
            pred_prob
        )
    }


    results.append(performance)



    model_dir=f"{output_dir}/{name}"

    os.makedirs(
        model_dir,
        exist_ok=True
    )


    joblib.dump(
        model,
        f"{model_dir}/{name}_Model.pkl"
    )


    pd.DataFrame({

        "Observed":y_test,
        "Probability":pred_prob,
        "Prediction":pred

    }).to_csv(
        f"{model_dir}/{name}_Predictions.csv",
        index=False
    )


    pd.DataFrame(
        confusion_matrix(
            y_test,
            pred
        )
    ).to_csv(
        f"{model_dir}/{name}_Confusion_Matrix.csv",
        index=False
    )



# ------------------------------------------------------------
# Save comparison
# ------------------------------------------------------------

comparison=pd.DataFrame(results)


comparison.to_csv(
    f"{output_dir}/Advanced_Model_Comparison.csv",
    index=False
)


print("="*70)
print("ADVANCED MODEL TRAINING COMPLETED")
print("="*70)

print(comparison)


print("="*70)
print("FILES SAVED")
print("="*70)

print(
f"""
{output_dir}

Advanced_Model_Comparison.csv

XGBoost/
LightGBM/
CatBoost/
"""
)

print("="*70)
print("CELL 22 COMPLETED SUCCESSFULLY")
print("="*70)

pip_install("optuna")

# ============================================================
# CELL 23
# OPTUNA HYPERPARAMETER OPTIMIZATION
# XGBOOST AND CATBOOST
# ============================================================

import os
import json
import pandas as pd
import numpy as np

import optuna

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score
)

import joblib


print("="*70)
print("OPTUNA BOOSTING MODEL OPTIMIZATION STARTED")
print("="*70)


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

input_dir = "./flood_project_workspace/Results/ML_Training_Input"

output_dir = "./flood_project_workspace/Results/ML_Models/Optuna_Optimization"

os.makedirs(output_dir, exist_ok=True)


# ------------------------------------------------------------
# Load datasets
# ------------------------------------------------------------

X_train = pd.read_csv(
    f"{input_dir}/X_train_scaled.csv"
)

X_valid = pd.read_csv(
    f"{input_dir}/X_validation_scaled.csv"
)


y_train = pd.read_csv(
    f"{input_dir}/y_train.csv"
).values.ravel()


y_valid = pd.read_csv(
    f"{input_dir}/y_validation.csv"
).values.ravel()



print("Datasets loaded")
print("-----------------------------")
print("Train:", X_train.shape)
print("Validation:", X_valid.shape)



# ============================================================
# XGBOOST OPTIMIZATION
# ============================================================


print("="*70)
print("Optimizing XGBoost")
print("="*70)


def xgb_objective(trial):

    params = {

        "n_estimators":
            trial.suggest_int(
                "n_estimators",
                10,
                2000
            ),

        "learning_rate":
            trial.suggest_float(
                "learning_rate",
                0.005,
                0.3,
                log=True
            ),

        "max_depth":
            trial.suggest_int(
                "max_depth",
                3,
                12
            ),

        "subsample":
            trial.suggest_float(
                "subsample",
                0.5,
                1.0
            ),

        "colsample_bytree":
            trial.suggest_float(
                "colsample_bytree",
                0.5,
                1.0
            ),

        "min_child_weight":
            trial.suggest_int(
                "min_child_weight",
                1,
                20
            ),

        "gamma":
            trial.suggest_float(
                "gamma",
                0,
                5
            ),

        "tree_method":
            "hist",

        "eval_metric":
            "aucpr",

        "random_state":
            42
    }


    model = XGBClassifier(**params)


    model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_valid,y_valid)
        ],
        verbose=False
    )


    pred = model.predict_proba(
        X_valid
    )[:,1]


    return average_precision_score(
        y_valid,
        pred
    )



study_xgb = optuna.create_study(
    direction="maximize"
)


study_xgb.optimize(
    xgb_objective,
    n_trials=30
)



best_xgb_params = study_xgb.best_params



with open(
    f"{output_dir}/Best_XGBoost_Params.json",
    "w"
) as f:

    json.dump(
        best_xgb_params,
        f,
        indent=4
    )


pd.DataFrame(
    study_xgb.trials_dataframe()
).to_csv(
    f"{output_dir}/Optuna_XGBoost_Trials.csv",
    index=False
)



# ============================================================
# CATBOOST OPTIMIZATION
# ============================================================


print("="*70)
print("Optimizing CatBoost")
print("="*70)



def cat_objective(trial):


    params={

        "iterations":
            trial.suggest_int(
                "iterations",
                100,
                2000
            ),

        "learning_rate":
            trial.suggest_float(
                "learning_rate",
                0.005,
                0.3,
                log=True
            ),

        "depth":
            trial.suggest_int(
                "depth",
                4,
                12
            ),

        "l2_leaf_reg":
            trial.suggest_float(
                "l2_leaf_reg",
                1,
                20
            ),

        "random_strength":
            trial.suggest_float(
                "random_strength",
                0,
                2
            ),

        "loss_function":
            "Logloss",

        "eval_metric":
            "AUC",

        "verbose":
            False,

        "random_seed":
            42
    }


    model = CatBoostClassifier(
        **params
    )


    model.fit(
        X_train,
        y_train,
        eval_set=(
            X_valid,
            y_valid
        ),
        verbose=False
    )


    pred=model.predict_proba(
        X_valid
    )[:,1]


    return average_precision_score(
        y_valid,
        pred
    )



study_cat = optuna.create_study(
    direction="maximize"
)


study_cat.optimize(
    cat_objective,
    n_trials=30
)



best_cat_params = study_cat.best_params



with open(
    f"{output_dir}/Best_CatBoost_Params.json",
    "w"
) as f:

    json.dump(
        best_cat_params,
        f,
        indent=4
    )



pd.DataFrame(
    study_cat.trials_dataframe()
).to_csv(
    f"{output_dir}/Optuna_CatBoost_Trials.csv",
    index=False
)



# ============================================================
# FINAL TRAINING USING BEST PARAMETERS
# ============================================================


print("="*70)
print("Training final optimized models")
print("="*70)



best_xgb = XGBClassifier(
    **best_xgb_params,
    tree_method="hist",
    random_state=42
)


best_xgb.fit(
    X_train,
    y_train
)



best_cat = CatBoostClassifier(
    **best_cat_params,
    verbose=False,
    random_seed=42
)


best_cat.fit(
    X_train,
    y_train
)



joblib.dump(
    best_xgb,
    f"{output_dir}/Best_XGBoost.pkl"
)


joblib.dump(
    best_cat,
    f"{output_dir}/Best_CatBoost.pkl"
)



print("="*70)
print("CELL 23 COMPLETED SUCCESSFULLY")
print("="*70)

import joblib
joblib.dump(
    best_xgb,
    f"{output_dir}/Best_XGBoost.pkl"
)


joblib.dump(
    best_cat,
    f"{output_dir}/Best_CatBoost.pkl"
)

