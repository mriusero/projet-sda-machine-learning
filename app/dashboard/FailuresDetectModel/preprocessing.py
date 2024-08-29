# preprocessing.py
import pandas as pd

from ..functions import ParticleFilter
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import pandas as pd


def clean_data(df):
    """Nettoie les données en supprimant les valeurs manquantes."""
    def complete_empty_item_data(group):
        # Vérifier et remplir 'Failure mode' et 'Time to failure (months)' si nécessaire
        if 'Failure mode' in group.columns:
            if group['Failure mode'].isnull().all() and group['Time to failure (months)'].isnull().all():
                group['Failure mode'] = 'Fatigue crack'
                group['Time to failure (months)'] = group['time (months)'].max()

        # Initialiser les colonnes 'end_life' et 'has_zero_twice' pour tous les groupes
        if 'rul (months)' in group.columns:
            max_time = group['time (months)'].max()

            group['rul (months)'] = max_time - group['time (months)'] + 1

            # Traiter les valeurs dans 'rul (months)'
            group['end_life'] = (group['rul (months)'] <= 6).astype(int)
            count_zeros = (group['rul (months)'] == 0).sum()
            group['has_zero_twice'] = count_zeros >= 2
        else:
            # Définir des valeurs par défaut si 'rul (months)' est absent
            group['end_life'] = 0
            group['has_zero_twice'] = False

        return group

    df = df.rename(columns={'crack length (arbitary unit)': 'length_measured'})
    df = df.sort_values(by=['item_id', 'time (months)'])
    df['source'] = 0                                            # Original data tag

    df['crack_failure'] = (df['length_measured'] >= 0.85).astype(int)
    df = df.groupby('item_id').apply(complete_empty_item_data)

    #df['rul (months)'] = df.groupby('item_index').cumcount().add(1)

    df = df.dropna()

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


