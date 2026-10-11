import joblib
from pathlib import Path
import numpy as np
import pandas as pd
import shap
import lime.lime_tabular
import matplotlib.pyplot as plt
import io
import base64
import warnings

# On ignore les petits avertissements de SHAP pour ne pas polluer le terminal
warnings.filterwarnings("ignore")

def fig_to_base64(fig=None):
    """Convertit un graphique Matplotlib en image Base64 pour le web."""
    buf = io.BytesIO()
    if fig is None:
        plt.savefig(buf, format="png", bbox_inches="tight")
    else:
        fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def get_global_explanations(model_name: str) -> dict:
    ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
    X_train = joblib.load(ARTIFACTS_DIR / "X_train.joblib")
    X_test = joblib.load(ARTIFACTS_DIR / "X_test.joblib")
    
    # Correction du nom de fichier : on utilise directement le nom exact
    model = joblib.load(ARTIFACTS_DIR / f"{model_name}.joblib")

    # On prend un petit echantillon pour que le site charge vite (100 patients)
    X_sample = shap.sample(X_test, 100)
    X_train_sample = shap.sample(X_train, 100)
    
    # ---------------------------------------------------------
    # LA LOGIQUE INTELLIGENTE : TreeExplainer vs KernelExplainer
    # ---------------------------------------------------------
    model_lower = model_name.lower()
    
    if "xgboost" in model_lower or "forest" in model_lower or "tree" in model_lower:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        # Random Forest renvoie une liste (une pour classe 0, une pour classe 1). On prend la 1.
        if isinstance(shap_values, list):
            shap_values_to_plot = shap_values[1]
        else:
            shap_values_to_plot = shap_values
    else:
        # Pour KNN et Logistic Regression (Plus lent)
        explainer = shap.KernelExplainer(model.predict_proba, X_train_sample)
        shap_values = explainer.shap_values(X_sample)
        shap_values_to_plot = shap_values[1] if isinstance(shap_values, list) else shap_values

    # 1. Graphique SHAP Summary Plot
    plt.figure()
    shap.summary_plot(shap_values_to_plot, X_sample, show=False)
    summary_b64 = fig_to_base64()

    # 2. Variable la plus importante
    mean_shap = np.abs(shap_values_to_plot).mean(axis=0)
    top_feature_idx = np.argmax(mean_shap)
    top_feature_name = X_train.columns[top_feature_idx]

    # 3. Graphique SHAP Dependence Plot
    plt.figure()
    shap.dependence_plot(top_feature_idx, shap_values_to_plot, X_sample, feature_names=X_train.columns, show=False)
    dependence_b64 = fig_to_base64()

    return {
        "summary_plot": summary_b64,
        "dependence_plot": dependence_b64,
        "top_feature": top_feature_name
    }

def get_local_explanations(model_name: str, patient_index: int) -> dict:
    ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
    X_train = joblib.load(ARTIFACTS_DIR / "X_train.joblib")
    X_test = joblib.load(ARTIFACTS_DIR / "X_test.joblib")
    
    model = joblib.load(ARTIFACTS_DIR / f"{model_name}.joblib")

    patient = X_test.iloc[patient_index]
    proba = model.predict_proba(patient.values.reshape(1, -1))[0][1]

    # ---------------------------------------------------------
    # EXPLICATION LOCALE SHAP
    # ---------------------------------------------------------
    model_lower = model_name.lower()
    plt.figure()
    
    if "xgboost" in model_lower or "forest" in model_lower or "tree" in model_lower:
        explainer_shap = shap.Explainer(model)
        shap_val_local = explainer_shap(patient.to_frame().T)
        
        # Gestion des dimensions selon si c'est RF ou XGB
        if len(shap_val_local.shape) == 3:
            shap.plots.waterfall(shap_val_local[0, :, 1], show=False)
        else:
            shap.plots.waterfall(shap_val_local[0], show=False)
    else:
        # Pour KNN/LogReg, le Waterfall de KernelExplainer bug souvent. On utilise un Bar Plot classique.
        explainer_shap = shap.KernelExplainer(model.predict_proba, shap.sample(X_train, 50))
        shap_val_local = explainer_shap.shap_values(patient.to_frame().T)
        shap_values_to_plot = shap_val_local[1][0] if isinstance(shap_val_local, list) else shap_val_local[0]
        
        pd.Series(shap_values_to_plot, index=X_train.columns).sort_values().tail(10).plot(kind='barh', color='dodgerblue')
        plt.title("Importance des 10 variables principales (SHAP)")

    waterfall_b64 = fig_to_base64()

    # ---------------------------------------------------------
    # EXPLICATION LOCALE LIME
    # ---------------------------------------------------------
    explainer_lime = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=['Normal', 'Danger (<30j)'],
        mode='classification'
    )
    exp = explainer_lime.explain_instance(patient.values, model.predict_proba, num_features=5)
    lime_html = exp.as_html()

    # ---------------------------------------------------------
    # SYNTHÈSE TEXTUELLE AUTOMATIQUE (Cahier des charges)
    # ---------------------------------------------------------
    dict_lime = exp.as_list()
    aggravants = [feat[0] for feat in dict_lime if feat[1] > 0]
    protecteurs = [feat[0] for feat in dict_lime if feat[1] < 0]

    phrase = f"🩺 Le modèle prédit un risque de réadmission de {proba*100:.1f}%. \n\n"
    if aggravants:
        phrase += f" **Le risque est élevé principalement à cause de :** {', '.join(aggravants[:2])}. \n"
    if protecteurs:
        phrase += f" **À l'inverse, les facteurs qui rassurent le modèle sont :** {', '.join(protecteurs[:2])}."

    return {
        "proba": float(proba),
        "waterfall_plot": waterfall_b64,
        "lime_html": lime_html,
        "synthesis": phrase
    }