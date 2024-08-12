import streamlit as st
import pandas as pd
from ..functions.utils import dataframing_data
from ..FailuresDetectModel.main import handle_scenarios

def page_6():
    st.markdown('<div class="title">MACHINE LEARNING</div>', unsafe_allow_html=True)  # TITLE
    st.markdown('<div class="header">#6 Prediction_</div>', unsafe_allow_html=True)
    texte = """
    Here is the Prediction phase.
    """
    st.markdown(texte)

    dataframes = dataframing_data()



    handle_scenarios(dataframes)
