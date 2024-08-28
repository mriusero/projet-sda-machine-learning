import pandas as pd
import streamlit as st
from ..FailuresDetectModel import clean_data
from ..functions import DataVisualizer, compare_dataframes
from ..FailuresDetectModel import FeatureAdder, run_statistical_test

def page_3():

    st.markdown('<div class="header">#3 Feature Engineering_</div>', unsafe_allow_html=True)
    texte = """

    Here is the Feature Engineering phase.

"""
    st.markdown(texte)

    df1 = pd.read_csv('./data/output/training/training_data.csv')
    df2 = st.session_state.data.get_the('train')

    # Obtenir les colonnes des deux DataFrames
    cols_df1 = set(df1.columns)
    cols_df2 = set(df2.columns)

    # Colonnes présentes dans df1 mais pas dans df2
    cols_only_in_df1 = cols_df1 - cols_df2
    st.write("Colonnes présentes uniquement dans le premier DataFrame :", cols_only_in_df1)

    # Colonnes présentes dans df2 mais pas dans df1
    cols_only_in_df2 = cols_df2 - cols_df1
    st.write("Colonnes présentes uniquement dans le second DataFrame :", cols_only_in_df2)
