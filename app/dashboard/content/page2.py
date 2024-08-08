import streamlit as st
import pandas as pd
from ..components import plot_correlation_matrix, plot_scatter, plot_scatter2, plot_dist2
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
    col1, col2 = st.columns([2,2])
    df = dataframes['train'].sort_values(by=['item_index', 'time (months)'])
    st.dataframe(df)
    with col1:
        plot_scatter(df, 'time (months)', 'crack length (arbitary unit)', 'Failure mode')
    with col2:
        plot_dist2(df, 'Failure mode')
    plot_distribution(df)



    st.markdown('## PSEUDO_TEST_WITH_TRUTH')
    col1, col2 = st.columns([2, 2])
    df = dataframes['pseudo_test_with_truth'].sort_values(by=['item_index', 'time (months)'])
    with col1:
        plot_scatter2(df, 'time (months)', 'crack length (arbitary unit)', 'label')
    with col2:
        plot_dist2(df, 'label')
    plot_distribution(df)


    st.markdown('## TEST')
    col1, col2 = st.columns([2, 2])
    df = dataframes['test'].sort_values(by=['item_index', 'time (months)'])
    with col1:
        plot_scatter(df, 'time (months)', 'crack length (arbitary unit)', 'item_index')
    with col2:
        plot_dist2(df, 'item_index')
    plot_distribution(df)
