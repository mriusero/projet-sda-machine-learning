import streamlit as st
from ..models import handle_scenarios
from ..functions import dataframing_data


def page_3():
    st.markdown('<div class="title">MACHINE LEARNING</div>', unsafe_allow_html=True)                            #TITLE
    st.markdown('<div class="header">#3 Training_</div>', unsafe_allow_html=True)
    texte = """

    Here is the training phase.

"""
    st.markdown(texte)

    dataframes = dataframing_data()
    handle_scenarios(dataframes)
    
