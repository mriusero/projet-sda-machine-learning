import numpy as np
import streamlit as st

def clean_data(df):
    """
    Nettoie les données d'un DataFrame.
    """

    df = df.dropna()
    ## Gestion des valeurs aberrantes
    ## Par exemple, en utilisant les percentiles pour détecter les valeurs aberrantes
    #for column in df.select_dtypes(include=[np.number]).columns:
    #    q1 = df[column].quantile(0.25)
    #    q3 = df[column].quantile(0.75)
    #    iqr = q3 - q1
    #    df = df[(df[column] >= (q1 - 1.5 * iqr)) & (df[column] <= (q3 + 1.5 * iqr))]

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