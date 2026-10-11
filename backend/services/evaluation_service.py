import joblib
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score, 
    confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
)

def get_comparison_data() -> dict:
    """Calcule les courbes et metriques globales pour tous les modeles entraines."""
    ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
    
    try:
        X_test = joblib.load(ARTIFACTS_DIR / "X_test.joblib")
        y_test = joblib.load(ARTIFACTS_DIR / "y_test.joblib")
    except FileNotFoundError:
        raise FileNotFoundError("Donnees de test introuvables. Executez le Module 2.")

    # On ignore les fichiers de donnees pour ne garder que les modeles
    ignored_files = ["X_train.joblib", "X_test.joblib", "y_train.joblib", "y_test.joblib", "scaler.joblib"]
    models_files = [f for f in ARTIFACTS_DIR.glob("*.joblib") if f.name not in ignored_files]
    
    if not models_files:
        raise ValueError("Aucun modele trouve. Veuillez entrainer au moins un modele au Module 3.")

    results = {}
    
    for mf in models_files:
        model_name = mf.stem  # Nom du fichier sans l'extension (ex: Random_Forest_100_10)
        model = joblib.load(mf)
        
        # Probabilite de la classe 1 (Danger)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Calcul Courbe ROC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        # Calcul Courbe Precision-Rappel (PR)
        prec, rec, pr_thresholds = precision_recall_curve(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)

        # RECHERCHE DU SEUIL OPTIMAL (Max F1-Score)
        f1_scores = []
        for p, r in zip(prec[:-1], rec[:-1]):
            if p + r == 0:
                f1_scores.append(0)
            else:
                f1_scores.append(2 * p * r / (p + r))
        
        best_idx = np.argmax(f1_scores)
        best_threshold = float(pr_thresholds[best_idx])

        # Metriques exactes au SEUIL OPTIMAL
        y_pred_opt = (y_proba >= best_threshold).astype(int)
        
        results[model_name] = {
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "optimal_threshold": float(best_threshold),
            "accuracy": float(accuracy_score(y_test, y_pred_opt)),
            "precision": float(precision_score(y_test, y_pred_opt, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred_opt, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred_opt, zero_division=0)),
            "fpr": fpr.tolist(), 
            "tpr": tpr.tolist(),
            "precision_curve": prec.tolist(),
            "recall_curve": rec.tolist()
        }
        
    return results

def get_threshold_metrics(model_name: str, threshold: float) -> dict:
    """Calcule la matrice de confusion en direct selon le seuil (slider)."""
    ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
    X_test = joblib.load(ARTIFACTS_DIR / "X_test.joblib")
    y_test = joblib.load(ARTIFACTS_DIR / "y_test.joblib")

    # Reconstruire le nom du fichier depuis le nom envoye par le frontend
    filename = f"{model_name}.joblib"
    model = joblib.load(ARTIFACTS_DIR / filename)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": cm.tolist()
    }