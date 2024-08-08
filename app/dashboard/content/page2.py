import streamlit as st
import pandas as pd
from ..components import plot_correlation_matrix, plot_scatter
from ..functions import plot_distribution
def page_2(dataframes):
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
    col1, col2 = st.columns([2,5])
    with col1:
        st.dataframe(pd.read_csv('./data/output/training/training_data.csv'))
        plot_correlation_matrix(dataframes['train'])
    with col2:
        plot_distribution(dataframes['train'])
        plot_scatter(dataframes['train'], 'time (months)', 'crack length (arbitary unit)')


    st.markdown('## PSEUDO_TEST_WITH_TRUTH')
    col1, col2 = st.columns([2, 5])
    with col1:
        st.dataframe(pd.read_csv('./data/output/pseudo_testing/pseudo_testing_data_with_truth.csv'))
        plot_correlation_matrix(dataframes['pseudo_test_with_truth'])
    with col2:
        plot_distribution(dataframes['pseudo_test_with_truth'])
        plot_scatter(dataframes['pseudo_test_with_truth'], 'time (months)', 'crack length (arbitary unit)')


    st.markdown('## TEST')
    col1, col2 = st.columns([2, 5])
    with col1:
        st.dataframe(pd.read_csv('./data/output/testing/testing_data_phase1.csv'))
        plot_correlation_matrix(dataframes['test'])
    with col2:
        plot_distribution(dataframes['test'])
        plot_scatter(dataframes['test'], 'time (months)', 'crack length (arbitary unit)')

