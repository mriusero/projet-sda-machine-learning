# features.py
import pandas as pd
import numpy as np

def add_features(df):
    # Exemple d'ajout de caractéristiques : par exemple, ajouter des interactions ou des transformations
    df['crack_length_squared'] = df['crack length (arbitary unit)'] ** 2
    return df
