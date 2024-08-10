import streamlit as st
import pandas as pd

from ..functions import decompose_time_series, create_features, plot_covariance_matrix, check_homoscedasticity, dataframing_data
from ..components import plot_correlation_matrix, plot_scatter, plot_scatter2, plot_histogram


def page_2():
    st.markdown('<div class="title">MACHINE LEARNING</div>', unsafe_allow_html=True)                            #TITLE
    st.markdown('<div class="header">#2 Feature Engineering_</div>', unsafe_allow_html=True)
    texte = """

        Here is the Feature Engineering phase.

    """
    st.markdown(texte)
    dataframes = dataframing_data()

    st.dataframe(dataframes['train'])




