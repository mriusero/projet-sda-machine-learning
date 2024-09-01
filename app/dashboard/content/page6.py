import streamlit as st
import pandas as pd
from ..functions.utils import dataframing_data
from ..FailuresDetectModel.main_phase_II import handle_models_phase_II

def page_6():
    st.markdown('<div class="header">#6 Prediction phase II</div>', unsafe_allow_html=True)
    texte = """
    Here is the Prediction for the phase II.
    """
    st.markdown(texte)

    #st.write('### Train DataFrame')
    #st.dataframe(st.session_state.data.get_the('train'))
    #st.write('### Pseudo-test DataFrame')
    #st.dataframe(st.session_state.data.get_the('pseudo_test'))

    handle_models_phase_II()

