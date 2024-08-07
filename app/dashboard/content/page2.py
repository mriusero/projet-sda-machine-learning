import streamlit as st
from ..functions import load_training_data

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

    training_data = load_training_data()

    col1, col2 = st.columns([2,3])
    with col1:
        st.markdown("#### Failure data_")
        st.dataframe(training_data['failure_data'])
    with col2:
        st.markdown("#### Degradation data_")
        st.dataframe(training_data['degradation_data'])


    st.markdown("#### Combined data_")
    st.dataframe(training_data['combined_data'])