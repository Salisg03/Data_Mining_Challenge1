from ucimlrepo import fetch_ucirepo 
import pandas as pd

def load_diabetes_data():
    # Fetch dataset (l'ID de ce dataset sur UCI est 296)
    diabetes_130_us_hospitals_for_years_1999_2008 = fetch_ucirepo(id=296) 
    
    # Récupération des features (X) et de la target (y)
    X = diabetes_130_us_hospitals_for_years_1999_2008.data.features 
    y = diabetes_130_us_hospitals_for_years_1999_2008.data.targets 

    # On rassemble le tout dans un seul DataFrame pour l'exploration
    df = pd.concat([X, y], axis=1)

    # Aperçu de tes données
    print(df.head())
    print(df.info())
    return df