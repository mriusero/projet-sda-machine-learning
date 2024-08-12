# features.py
import pandas as pd
import numpy as np
from ..functions import ParticleFilter

def add_features(df, particles_filtery):
    """Ajoute des fonctionnalités aux données."""
    df['crack_length_squared'] = df['crack length (arbitary unit)'] ** 2

    if particles_filtery == True :
        pf = ParticleFilter()
        df = pf.filter(df, beta0_range=(-1, 1), beta1_range=(-0.1, 0.1), beta2_range=(0.1, 1))
        print(f'--> particles_filtery done !')

    return df
