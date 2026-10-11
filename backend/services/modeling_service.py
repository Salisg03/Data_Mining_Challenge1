import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def train_model(model_name: str, params: dict) -> dict:
    ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
    
    try:
        X_train = joblib.load(ARTIFACTS_DIR / "X_train.joblib")
        y_train = joblib.load(ARTIFACTS_DIR / "y_train.joblib")
    except FileNotFoundError:
        raise Exception("Les données d'entraînement n'existent pas. Lancez le Module 2 d'abord.")

    if model_name == "Logistic Regression":
        model = LogisticRegression(C=params.get("C", 1.0), max_iter=1000, random_state=42)
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(max_depth=params.get("max_depth", None), random_state=42)
    elif model_name == "KNN":
        model = KNeighborsClassifier(n_neighbors=params.get("n_neighbors", 5))
    elif model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=params.get("n_estimators", 100), max_depth=params.get("max_depth", None), random_state=42)
    elif model_name == "XGBoost":
        model = XGBClassifier(n_estimators=params.get("n_estimators", 100), learning_rate=params.get("learning_rate", 0.1), max_depth=params.get("max_depth", 6), random_state=42, eval_metric='logloss')
    else:
        raise ValueError(f"Modèle non reconnu : {model_name}")

    model.fit(X_train, y_train)

    # NOUVEAU : On ajoute les valeurs des hyperparamètres dans le nom du fichier pour ne pas écraser !
    param_str = "_".join([str(v) for v in params.values()])
    model_filename = f"{model_name.replace(' ', '_')}_{param_str}.joblib"
    
    joblib.dump(model, ARTIFACTS_DIR / model_filename)

    return {
        "status": "success",
        "message": f"Modèle {model_name} (Paramètres: {param_str}) entraîné et sauvegardé !"
    }