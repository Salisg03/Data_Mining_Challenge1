import pandas as pd
import numpy as np
from backend.services.data_loader_service import get_raw_data 

def get_missing_values_stats() -> dict:
    df = get_raw_data() # Charge 100% des données
    df_temp = df.replace('?', np.nan)
    missing_percent = (df_temp.isnull().sum() / len(df_temp)) * 100
    return missing_percent[missing_percent > 0].sort_values(ascending=False).to_dict()

def get_target_distribution() -> dict:
    df = get_raw_data()
    return df['readmitted'].value_counts().to_dict()

def get_data_types() -> dict:
    """Renvoie le type de chaque colonne (int, float, object)."""
    df = get_raw_data()
    # On convertit les types en string pour le JSON
    return df.dtypes.astype(str).to_dict() 

def get_basic_stats() -> dict:
    """Calcule la moyenne, std, min, max, et quartiles (pour les outliers)."""
    df = get_raw_data()
    # df.describe() fait tout le travail statistique !
    stats_df = df.describe()
    return stats_df.to_dict()

def get_numerical_correlations() -> dict:
    """Calcule la matrice de corrélation pour repérer les variables redondantes."""
    df = get_raw_data()
    # Garde uniquement les colonnes numériques
    numeric_df = df.select_dtypes(include=[np.number]) 
    return numeric_df.corr().to_dict()

def get_outliers_info() -> dict:
    """Détecte les outliers mathématiques via la méthode IQR (Interquartile Range)."""
    df = get_raw_data()
    
    # On prend uniquement les colonnes numériques qui nous intéressent
    num_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_diagnoses']
    outliers_report = {}
    
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Les limites pour être considéré comme "Normal"
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # On compte combien de patients dépassent ces limites
        outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        outliers_report[col] = int(outliers_count)
        
    return outliers_report

def get_bias_analysis() -> dict:
    """
    Analyse Éthique : Regarde le taux de réadmission par tranche d'âge 
    pour identifier d'éventuels biais algorithmiques.
    """
    df = get_raw_data()
    
    # On crée temporairement la cible binaire juste pour l'analyse
    df['target'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
    
    # On calcule le pourcentage de réadmission (<30j) pour chaque tranche d'âge
    # groupby('age')['target'].mean() donne la proportion de 1 par âge
    bias_df = df.groupby('age')['target'].mean() * 100 
    
    return bias_df.to_dict()