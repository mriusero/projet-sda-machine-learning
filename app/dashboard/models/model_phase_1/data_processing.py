import numpy as np
import streamlit as st
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





def preprocess_data(train, test):
    """
    Création d'une Window pour chaque item.
    """
    unique_items = train['item_index'].unique()
    def prepare_item_data(item):
        train_item = train[train['item_index'] == item].copy()
        test_item = test[test['item_index'] == item].copy()

        if train_item.empty or test_item.empty:
            return None, None

        return train_item, test_item

    return unique_items, prepare_item_data