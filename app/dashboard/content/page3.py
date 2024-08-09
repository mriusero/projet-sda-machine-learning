import streamlit as st
from ..models import handle_scenarios


def page_3(dataframes):
    st.markdown('<div class="title">MACHINE LEARNING</div>', unsafe_allow_html=True)                            #TITLE
    st.markdown('<div class="header">#3 RUL Prediction_</div>', unsafe_allow_html=True)
    texte = """

    Here is the Machine Learning phase.

"""
    st.markdown(texte)    

    handle_scenarios(dataframes)
    
