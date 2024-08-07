import streamlit as st
import pandas as pd
from ..functions import load_training_data, merge_training_data, plot_distribution

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
    training_df = training_data['solution_data']
    failure_df = training_data['failure_data']
    degradation_df = training_data['degradation_data']

    merged_df = merge_training_data(training_data)

    col1, col2, col3 = st.columns([2,2,3])

    with col1:
            st.markdown("""
        #### #Solution data_

            <class 'pandas.core.frame.DataFrame'>
            RangeIndex: 50 entries, 0 to 49
            Data columns (total 3 columns):
            
             #   Column      Non-Null Count  Dtype  
            ---  ------      --------------  -----  
             0   item_index  50 non-null     object 
             1   label       50 non-null     int64  
             2   true_rul    50 non-null     float64
             
            dtypes: float64(1), int64(1), object(1)
            memory usage: 1.3+ KB

                """)
            st.dataframe(training_df.describe())

    with col2:
        st.markdown("""
    #### #Failure data_

        <class 'pandas.core.frame.DataFrame'>
        RangeIndex: 49 entries, 0 to 48
        Data columns (total 3 columns):
        
         #   Column                    Non-Null Count  Dtype 
        ---  ------                    --------------  ----- 
         0   item_id                   49 non-null     int64 
         1   Time to failure (months)  49 non-null     int64 
         2   Failure mode              49 non-null     object
         
        dtypes: int64(2), object(1)
        memory usage: 1.3+ KB
    
        """)
        st.dataframe(failure_df.describe())

    with col3:
        st.markdown("""
    #### #Degradation data_

        <class 'pandas.core.frame.DataFrame'>
        RangeIndex: 1805 entries, 0 to 1804
        Data columns (total 4 columns):
         #   Column                        Non-Null Count  Dtype  
        ---  ------                        --------------  -----  
         0   time (months)                 1805 non-null   int64  
         1   crack length (arbitary unit)  1805 non-null   float64
         2   rul (months)                  1805 non-null   int64  
         3   item_id                       1805 non-null   int64  
         
        dtypes: float64(1), int64(3)
        memory usage: 56.5 KB


        """)
        st.dataframe(degradation_df.describe())

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Merged data_")
        st.dataframe(merged_df)
    with col2:
        st.markdown("#### Statistics_")
        st.dataframe(merged_df.describe())

    st.markdown("#### Distribution_")
    plot_distribution(merged_df)

    st.markdown("#### Failure modes analysis_")
    arbitrary_df = failure_df.groupby('Failure mode').agg(
        Count=('Failure mode', 'size'),
        Min_Time_to_Failure=('Time to failure (months)', 'min'),
        Avg_Time_to_Failure=('Time to failure (months)', 'mean'),
        Max_Time_to_Failure=('Time to failure (months)', 'max')
    ).reset_index()
    st.dataframe(arbitrary_df)