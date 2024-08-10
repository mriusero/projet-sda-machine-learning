import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.seasonal import STL


def decompose_time_series(df, time_col, value_col, period=12):
    # Vérifiez que les colonnes existent dans le DataFrame
    if time_col not in df.columns or value_col not in df.columns:
        st.error(f"Les colonnes '{time_col}' ou '{value_col}' n'existent pas dans le DataFrame.")
        return

    # Convertir la colonne de temps en datetime si ce n'est pas déjà fait
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')

    # Triez le DataFrame par la colonne de temps
    df = df.sort_values(by=time_col)

    # Assurez-vous qu'il n'y a pas de valeurs manquantes
    df = df.dropna(subset=[time_col, value_col])

    # Mettre la colonne de temps en index
    df.set_index(time_col, inplace=True)

    # Effectuer la décomposition STL
    try:
        stl = STL(df[value_col], period=period)
        result = stl.fit()
    except Exception as e:
        st.error(f"Erreur lors de la décomposition STL: {e}")
        return

    # Tracer les résultats
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(f'Décomposition de la série temporelle : {value_col}', fontsize=16)

    # Tendance
    axes[0].plot(result.trend, label='Tendance', color='blue')
    axes[0].set_title('Tendance')
    axes[0].legend(loc='upper left')

    # Saison
    axes[1].plot(result.seasonal, label='Saisonnalité', color='green')
    axes[1].set_title('Saisonnalité')
    axes[1].legend(loc='upper left')

    # Résidu
    axes[2].plot(result.resid, label='Résidu', color='red')
    axes[2].set_title('Résidu')
    axes[2].legend(loc='upper left')

    # Série originale
    axes[3].plot(df[value_col], label='Série Originale', color='gray')
    axes[3].set_title('Série Originale')
    axes[3].legend(loc='upper left')

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Affichage avec Streamlit
    st.pyplot(fig)

# Exemple d'utilisation
# df = pd.DataFrame({
#     'date': pd.date_range(start='2023-01-01', periods=24, freq='M'),
#     'crack length (arbitary unit)': np.sin(np.linspace(0, 2 * np.pi, 24)) + np.random.normal(size=24)
# })

# decompose_time_series(df, 'date', 'crack length (arbitary unit)')
