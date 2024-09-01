import streamlit as st
import pandas as pd
from ..functions.utils import dataframing_data, load_scenarios
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

    scenarios_data, global_df = load_scenarios()
    st.write('Scenarios loaded:')
    st.write(scenarios_data)
    st.write(scenarios_data['scenario_0'])
    st.dataframe(global_df)

    handle_models_phase_II()

    scenarios = st.session_state.data.df['scenarios']
    st.dataframe(scenarios)