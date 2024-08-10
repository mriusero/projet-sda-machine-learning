# preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def clean_data(df):

    df = df.dropna()

    return df

def standardize_values(df: pd.DataFrame, columns: list) -> pd.DataFrame:

    for col in columns:
        if col not in df.columns:
            raise ValueError(f"La colonne '{col}' n'existe pas dans le DataFrame")

    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])

    return df
def normalize_values(df: pd.DataFrame, columns: list) -> pd.DataFrame:

    for col in columns:
        if col not in df.columns:
            raise ValueError(f"La colonne '{col}' n'existe pas dans le DataFrame")

    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])

    return df


