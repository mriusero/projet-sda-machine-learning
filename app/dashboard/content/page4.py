import streamlit as st
import pandas as pd
from ..FailuresDetectModel import clean_data, run_statistical_test
from ..functions import dataframing_data, display_variable_types, compare_dataframes
from ..FailuresDetectModel import standardize_values, normalize_values
def page_4():
    st.markdown('<div class="header">#4 Statistics_</div>', unsafe_allow_html=True)
    texte = """

    Here is the Statistics phase.

----------------------------------------------

    'length_filtered', 'length_measured'

    'beta0', 'beta1', 'beta2', 
   
    'static_std_length_measured', 
    'static_max_length_measured',
    'static_mean_length_measured', 
    'static_min_length_measured',
    'rolling_mean_length_measured', 
    'rolling_max_length_measured', 
    'rolling_std_length_measured', 
    'rolling_min_length_measured'
    
    'static_std_time (months)',
    'static_mean_time (months)', 
    'static_max_time (months)', 
    'static_min_time (months)',
    'rolling_std_time (months)',    
    'rolling_min_time (months)', 
    'rolling_max_time (months)',  
    'rolling_mean_time (months)', 
    
    'static_std_length_filtered', 
    'static_mean_length_filtered', 
    'static_max_length_filtered', 
    'static_min_length_filtered', 
    'rolling_max_length_filtered', 
    'rolling_mean_length_filtered',
    'rolling_min_length_filtered', 
    'rolling_std_length_filtered', 
     
    'crack_failure', 
    'has_zero_twice', 
    'end_life', 
    
"""
    st.markdown(texte)

    train_df = st.session_state.data.get_the('train')

    st.markdown('## #Train_')
    col1, col2 = st.columns([1,2])

    with col1:
        st.dataframe(train_df)
        st.markdown('### #Variables types_')
        variable_types_df = display_variable_types(train_df)
        st.dataframe(variable_types_df)
    with col2:
        st.session_state.data.plot_pairplot('train', hue='Failure mode')
        df = pd.read_csv('./data/output/training/training_data.csv')
        run_statistical_test(df, 'normality', 'time (months)')
        run_statistical_test(df, 'normality', 'crack length (arbitary unit)')
        run_statistical_test(df, 'normality', 'rul (months)')
        run_statistical_test(df, 'normality', 'Time to failure (months)')

        st.session_state.data.boxplot('train', 'Failure mode', 'Time to failure (months)')

    col1, col2 = st.columns(2)
    with col1 :
        st.markdown("### #Length Measured")
        st.session_state.data.decompose_time_series('train', 'time (months)', 'length_measured')
    with col2 :
        st.markdown("### #Length Filtered")
        st.session_state.data.decompose_time_series('train', 'time (months)', 'length_filtered')
