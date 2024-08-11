# features.py
import pandas as pd
import numpy as np

def add_features(df):
    """Ajoute des fonctionnalités aux données."""
    df['crack_length_squared'] = df['crack length (arbitary unit)'] ** 2
    return df
