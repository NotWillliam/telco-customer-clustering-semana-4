"""Funciones auxiliares para el analisis XAI del proyecto Telco Churn."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score

def preparar_carpetas(base_dir="results"):
    results_dir = Path(base_dir)
    figures_dir = results_dir / "figures"
    metrics_dir = results_dir / "metrics"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    return results_dir, figures_dir, metrics_dir

def calcular_importancia_shap(shap_churn, feature_names):
    return pd.DataFrame({
        "variable": feature_names,
        "importancia_shap_media_abs": np.abs(shap_churn).mean(axis=0)
    }).sort_values("importancia_shap_media_abs", ascending=False)

def comparar_tecnicas_globales(rf, X_test, y_test, shap_global_df, feature_names):
    rf_importance_df = pd.DataFrame({"variable": feature_names, "importancia_rf": rf.feature_importances_})
    perm = permutation_importance(rf, X_test, y_test, n_repeats=5, random_state=42, scoring="roc_auc", n_jobs=1)
    perm_importance_df = pd.DataFrame({"variable": feature_names, "importancia_permutation": perm.importances_mean})
    return shap_global_df.merge(rf_importance_df, on="variable").merge(perm_importance_df, on="variable")
