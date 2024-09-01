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
