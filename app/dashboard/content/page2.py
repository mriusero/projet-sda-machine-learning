import streamlit as st
import pandas as pd

from scipy import stats
import plotly.express as px

from ..functions import detect_outliers, dataframing_data


from ..FailuresDetectModel import clean_data

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def page_2():                          #TITLE
    st.markdown('<div class="header">#2 Cleaning_</div>', unsafe_allow_html=True)
    texte = f"""

        Here is the Cleaning phase

    """
    st.markdown(texte)

    dataframes = dataframing_data()

    col1, col2 = st.columns([1,2])

    with col1:
        st.markdown("""
    ## #Dataframes_
        """)
        for name, dataframe in dataframes.items():

            df = dataframe
            missing_percentage = df.isnull().mean() * 100

            st.markdown(f"### {name}")
            st.write(missing_percentage)

    with col2:
        st.markdown("""
    ## #Job done_
    
        def clean_data(df):
            Nettoie les données en supprimant les valeurs manquantes.
            def complete_empty_item_data(group):
                # Vérifier et remplir 'Failure mode' et 'Time to failure (months)' si nécessaire
                if 'Failure mode' in group.columns:
                    if group['Failure mode'].isnull().all() and group['Time to failure (months)'].isnull().all():
                        group['Failure mode'] = 'Fatigue crack'
                        group['Time to failure (months)'] = group['time (months)'].max()
        
                # Initialiser les colonnes 'end_life' et 'has_zero_twice' pour tous les groupes
                if 'rul (months)' in group.columns:
                    max_time = group['time (months)'].max()
        
                    group['rul (months)'] = max_time - group['time (months)'] + 1
        
                    # Traiter les valeurs dans 'rul (months)'
                    group['end_life'] = (group['rul (months)'] <= 6).astype(int)
                    count_zeros = (group['rul (months)'] == 0).sum()
                    group['has_zero_twice'] = count_zeros >= 2
                else:
                    # Définir des valeurs par défaut si 'rul (months)' est absent
                    group['end_life'] = 0
                    group['has_zero_twice'] = False
        
                return group
        
            df = df.rename(columns={'crack length (arbitary unit)': 'length_measured'})
            df = df.sort_values(by=['item_index', 'time (months)'])
            df['source'] = 0                                            # Original data tag
        
            df['crack_failure'] = (df['length_measured'] >= 0.85).astype(int)
            df = df.groupby('item_index').apply(complete_empty_item_data)
        
            #df['rul (months)'] = df.groupby('item_index').cumcount().add(1)
        
            df = df.dropna()
        
            return df 
        """)