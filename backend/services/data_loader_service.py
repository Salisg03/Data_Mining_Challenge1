from pathlib import Path
import pandas as pd

def get_raw_data() -> pd.DataFrame:
    """Charge le dataset brut depuis le cache local."""
    BASE_DIR = Path(__file__).resolve().parent.parent 
    DATA_PATH = BASE_DIR / "data" / "diabetic_data.csv"

    try :
        df = pd.read_csv(DATA_PATH)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Erreur: Fichier introuvable à {DATA_PATH}.")