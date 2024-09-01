# preprocessing.py
import streamlit as st
import numpy as np

from ..functions import ParticleFilter
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import pandas as pd


def clean_data(df):
    """Nettoie les données en supprimant les valeurs manquantes et en transformant la colonne 'Failure mode'."""

    def complete_empty_item_data(group):
        if 'source' not in group.columns:
            group['source'] = 0

        if 'label' not in group.columns:
            group['label'] = 0

        if 'true_rul' in group.columns:
            group['true_rul'] = group['true_rul'].astype(int)
        else:
            group['true_rul'] = group['label']

        if 'Failure mode' in group.columns:
            group['failure_month'] = np.where(group['Time to failure (months)'] == group['time (months)'], 1, 0)
            group.loc[group['failure_month'] != 1, ('Infant mortality', 'Control board failure', 'Fatigue crack')] = False

            if group['Failure mode'].isnull().all() and group['Time to failure (months)'].isnull().all():
                group['Failure mode'] = 'Fatigue crack'
                group['Time to failure (months)'] = group['time (months)'].max()

        if 'Time to failure (months)' in group.columns:
            group['Time to failure (months)'] = group['Time to failure (months)'].astype(int)


        if 'rul (months)' in group.columns:
            max_time = group['time (months)'].max()
            group['rul (months)'] = max_time - group['time (months)'] + 1

        #if 'failure_month' in group.columns:
        #    if group['failure_month'] == 0:
        #        group['Infant mortality'] = False
        #        group['Control board failure'] = False
        #        group['Fatigue crack'] = False




#        if 'Infant mortality' in group.columns:
#            group['deducted_rul'] = np.nan
#            # Créer les conditions booléennes
#            condition_others = (group['Infant mortality'] == True) | (group['Control board failure'] == True)
#            condition_fatigue_crack = group['Fatigue crack'] == True
#
#            # Calculer les totaux pour chaque condition
#            num_rul_others = condition_others.sum()
#            num_rul_fatigue_crack = condition_fatigue_crack.sum()
#
#            # Valeurs décroissantes pour 'deducted_rul'
#            start_rul_others = num_rul_others
#            start_rul_fatigue_crack = num_rul_fatigue_crack
#
#            # Assigner les valeurs décroissantes pour les autres conditions
#            if num_rul_others > 0:
#                group.loc[condition_others, 'deducted_rul'] = list(
#                    range(start_rul_others, start_rul_others - num_rul_others, -1))
#
#            if num_rul_fatigue_crack > 0:
#                group.loc[condition_fatigue_crack, 'deducted_rul'] = list(
#                    range(start_rul_fatigue_crack, start_rul_fatigue_crack - num_rul_fatigue_crack, -1))
#        else:
#            group['deducted_rul'] = np.nan

        return group

    # Convertir la colonne 'Failure mode' en variables indicatrices (1 et 0)
    if 'Failure mode' in df.columns:
        failure_mode_dummies = pd.get_dummies(df['Failure mode'], prefix=None)
        df = pd.concat([df, failure_mode_dummies], axis=1)

    df = df.groupby('item_id').apply(complete_empty_item_data)

    df = df.dropna()


    # Définir la liste complète des colonnes attendues dans tous les DataFrames
    all_possible_columns = ['source', 'item_id', 'time (months)', 'length_measured', 'label',
                                'Failure mode', 'Infant mortality', 'Control board failure', 'Fatigue crack',
                                'Time to failure (months)', 'rul (months)', 'true_rul', 'failure_month']
    for col in all_possible_columns:
        if col not in df.columns:
            df[col] = 0

    # Réorganiser les colonnes
    common_columns = ['source', 'item_id', 'time (months)', 'length_measured', 'label',
                            'Failure mode', 'Infant mortality', 'Control board failure', 'Fatigue crack',
                            'Time to failure (months)', 'rul (months)', 'true_rul', 'failure_month']
    all_columns = sorted(df.columns)  # Liste des colonnes après nettoyage
    remaining_columns = [col for col in all_columns if col not in common_columns]
    df = df[common_columns + remaining_columns]


    # Créer l'index unique
    df['unique_index'] = df.apply(lambda row: f"id_{(row['item_id'])}_&_mth_{(row['time (months)'])}", axis=1)
    # Réinitialiser l'index pour éviter le double index
    df.reset_index(drop=True, inplace=True)
    # Définir 'unique_index' comme index et supprimer l'ancien index
    df.set_index('unique_index', inplace=True)

    #st.dataframe(df)
    #st.write(f"{df.columns.to_list()} = {len(df.columns.to_list())} columns")

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


