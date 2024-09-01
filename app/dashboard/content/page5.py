import streamlit as st
from ..FailuresDetectModel.main_phase_I import handle_models_phase_I


def page_5():
    st.markdown('<div class="header">#5 Prediction phase I</div>', unsafe_allow_html=True)
    texte = """
    Here is the prediction for phase I   
    
    """
    st.markdown(texte)

    handle_models_phase_I()

