import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler

from backend.services.data_loader_service import get_raw_data

def process_and_split_data(test_size: float) -> dict:
    """Pipeline complet de Data Science : Nettoyage, Under-Sampling, Encodage."""
    df = get_raw_data()
    
    
    # 1. CRÉATION DE LA CIBLE
    df['target'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
    df = df.drop(columns=['readmitted'])
    
    
    # 2. APPLICATION DES RÈGLES DE L'EDA (Nettoyage strict)
    # On remplace les ? par du vide
    df = df.replace('?', np.nan)
    
    # A. Suppression des colonnes vides à >80% et des IDs (Data Leakage)
    cols_to_drop = ['encounter_id', 'patient_nbr', 'weight', 'max_glu_serum', 'A1Cresult', 'payer_code']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # B. Remplissage intelligent des NaN restants
    df['medical_specialty'] = df['medical_specialty'].fillna('Missing')
    df['race'] = df['race'].fillna('Unknown')
    
    # 3. TRADUCTION DES IDs (Le Mapping)
    admission_type_map = {1: 'Emergency', 2: 'Urgent', 3: 'Elective'}
    discharge_map = {1: 'Home', 2: 'Hospital', 3: 'SNF', 11: 'Expired'} 
    admission_source_map = {7: 'Emergency Room', 1: 'Physician Referral'}

    df['admission_type_id'] = df['admission_type_id'].map(admission_type_map).fillna('Other')
    df['discharge_disposition_id'] = df['discharge_disposition_id'].map(discharge_map).fillna('Other')
    df['admission_source_id'] = df['admission_source_id'].map(admission_source_map).fillna('Other')

    
    # 4. GESTION DE LA HAUTE CARDINALITÉ
    # Conservation du Top 10 pour diag_1, suppression de diag_2 et diag_3
    top_10_diag = df['diag_1'].value_counts().index[:10]
    df['diag_1'] = df['diag_1'].apply(lambda x: x if x in top_10_diag else 'Other_Disease')
    df = df.drop(columns=['diag_2', 'diag_3'])

    # 5. ENCODAGE ET SÉPARATION
    X = df.drop(columns=['target'])
    y = df['target']
    
    # One-Hot Encoding
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    # 🚨 LA CORRECTION POUR XGBOOST : Nettoyer les noms des colonnes !
    X_encoded.columns = X_encoded.columns.str.replace('[', '_', regex=False)
    X_encoded.columns = X_encoded.columns.str.replace(']', '_', regex=False)
    X_encoded.columns = X_encoded.columns.str.replace('<', '_', regex=False)
    
    # Séparation (Train/Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=42, stratify=y 
    )
   
    # 6. RANDOM UNDER-SAMPLING (RÉÉQUILIBRAGE)
    # Au lieu de créer des faux patients, on réduit la classe majoritaire
    rus = RandomUnderSampler(random_state=42)
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)
    
    # 7. NORMALISATION 
  
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_resampled), columns=X_train_resampled.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    # 8. SAUVEGARDE DES ARTEFACTS
    ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    
    joblib.dump(X_train_scaled, ARTIFACTS_DIR / "X_train.joblib")
    joblib.dump(X_test_scaled, ARTIFACTS_DIR / "X_test.joblib")
    joblib.dump(y_train_resampled, ARTIFACTS_DIR / "y_train.joblib")
    joblib.dump(y_test, ARTIFACTS_DIR / "y_test.joblib")
    joblib.dump(scaler, ARTIFACTS_DIR / "scaler.joblib")
    
    return {
        "train_size_before": len(y_train),
        "train_size_after": len(y_train_resampled),
        "minority_class_before": int(y_train.sum()),
        "minority_class_after": int(y_train_resampled.sum()),
        "test_size": len(X_test_scaled),
        "features_count": len(X_train_scaled.columns)
    }