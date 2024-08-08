import streamlit as st
import pandas as pd
from ..functions import load_data, merge_data, plot_distribution

def page_2():
    st.markdown('<div class="title">MACHINE LEARNING</div>', unsafe_allow_html=True)                            #TITLE
    st.markdown('<div class="header">#2 Exploration_</div>', unsafe_allow_html=True)    

    exploration = """
### 1) Exploration_

- **Résumé statistique** :
  - Utilisez `describe()` pour obtenir des statistiques descriptives (moyenne, écart type, min, max, quartiles).
  - Examinez la distribution des variables avec des histogrammes.
  - Visualisez les distributions avec des histogrammes, des boîtes à moustaches (box plots), et des densités.

- **Analyse des corrélations** :
  - Utilisez une matrice de corrélations pour identifier les relations entre les variables continues.

- **Identification des valeurs manquantes** :
  - Utilisez des méthodes comme `isnull().sum()` pour repérer les valeurs manquantes.

- **Analyse des valeurs aberrantes** :
  - Recherchez des valeurs extrêmes avec des box plots ou des scatter plots.

- **Examen des variables catégorielles** :
  - Comptez les occurrences de chaque catégorie pour vérifier leur distribution.
"""
    st.markdown(exploration)

    texte = """
### Input Data_
    """
    st.markdown(texte)

    st.markdown('## TRAIN')
    st.dataframe(pd.read_csv('./data/output/training/training_data.csv'))

    st.markdown('## PSEUDO_TEST')
    st.dataframe(pd.read_csv('./data/output/pseudo_testing/pseudo_testing_data.csv'))

    st.markdown('## PSEUDO_TEST_WITH_TRUTH')
    st.dataframe(pd.read_csv('./data/output/pseudo_testing/pseudo_testing_data_with_truth.csv'))

    st.markdown('## TEST')
    st.dataframe(pd.read_csv('./data/output/testing/testing_data_phase1.csv'))