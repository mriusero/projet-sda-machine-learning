# preprocessing.py
import pandas as pd

from ..functions import ParticleFilter
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def clean_data(df):
    """Nettoie les données en supprimant les valeurs manquantes."""

    df.dropna()

    return df

def standardize_values(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Standardise les valeurs des colonnes spécifiées."""
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

def normalize_values(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Normalise les valeurs des colonnes spécifiées."""
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df


