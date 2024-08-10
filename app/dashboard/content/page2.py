import streamlit as st
import pandas as pd

from scipy import stats
import plotly.express as px

from ..functions import detect_outliers, dataframing_data
from ..components import plot_correlation_matrix, plot_scatter, plot_scatter2, plot_histogram

from ..FailuresDetectModel import clean_data

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def page_2():
    st.markdown('<div class="title">MACHINE LEARNING</div>', unsafe_allow_html=True)                            #TITLE
    st.markdown('<div class="header">#2 Cleaning_</div>', unsafe_allow_html=True)
    line = '_'*50
    texte = f"""

        Here is the Cleaning phase
    {line}
    """
    st.markdown(texte)

    dataframes = dataframing_data()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
    ## #Before Cleaning_
        """)
        for name, dataframe in dataframes.items():

            df = dataframe
            missing_percentage = df.isnull().mean() * 100

            st.markdown(f"### {name}")
            st.write(missing_percentage)


    with col2:
        st.markdown("""
    ## #After Cleaning_
        """)
        for name, dataframe in dataframes.items():

            df = clean_data(dataframe.copy())
            missing_percentage = df.isnull().mean() * 100

            st.markdown(f"### {name}")
            st.write(missing_percentage)
