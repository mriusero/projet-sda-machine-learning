import streamlit as st
import pandas as pd
import numpy as np

from ..functions import dataframing_data, load_data

def page_1():
    st.markdown('<div class="header">#1 Exploration_</div>', unsafe_allow_html=True)

    st.markdown('## #Evolution of crack length over time_')
    col1, col2 = st.columns([2, 2])

    qualitative_pal = 'Safe'
    continuous_pal = 'Viridis'
    cyclical_pal = 'IceFire'
    sequential_pal = 'Inferno'

    with col1:
        st.session_state.data.plot_scatter_with_color(df_key='train',
                                                      x_col='time (months)',
                                                      y_col='length_measured',
                                                      color_col='Failure mode',
                                                      palette_type='qualitative',
                                                      palette_name=qualitative_pal)

        st.session_state.data.plot_histogram_with_color(df_key='train',
                                                      x_col='time (months)',
                                                      y_col='length_measured',
                                                      color_col='Failure mode',
                                                      palette_type='qualitative',
                                                      palette_name=qualitative_pal)

    with col2:
        st.session_state.data.plot_scatter_with_color(df_key='train',
                                                      x_col='time (months)',
                                                      y_col='length_filtered',
                                                      color_col='Failure mode',
                                                      palette_type='qualitative',
                                                      palette_name=qualitative_pal)

        st.session_state.data.plot_histogram_with_color(df_key='train',
                                                      x_col='time (months)',
                                                      y_col='length_filtered',
                                                      color_col='Failure mode',
                                                      palette_type='qualitative',
                                                      palette_name=qualitative_pal)

    st.markdown("## #PSEUDO_TEST_WITH_TRUTH_")
    st.markdown('## #End of life_')
    col1, col2 = st.columns([2, 2])
    with col1:
        st.session_state.data.plot_scatter_with_color('pseudo_test_with_truth', 'time (months)', 'length_measured', 'label',
                                                      palette_type='continuous',
                                                      palette_name=continuous_pal)
        st.session_state.data.plot_scatter_with_color('pseudo_test_with_truth', 'time (months)', 'length_measured', 'true_rul',
                                                      palette_type='continuous',
                                                      palette_name=continuous_pal)
        st.session_state.data.plot_histogram_with_color('pseudo_test_with_truth', 'time (months)', 'length_measured', 'label',
                                                        palette_type='qualitative',
                                                        palette_name=qualitative_pal)
        st.session_state.data.plot_histogram_with_color('pseudo_test_with_truth', 'time (months)', 'length_measured', 'true_rul',
                                                        palette_type='qualitative',
                                                        palette_name=qualitative_pal)

    with col2:
        st.session_state.data.plot_scatter_with_color('pseudo_test_with_truth', 'time (months)', 'length_filtered', 'label',
                                                      palette_type='continuous',
                                                      palette_name=continuous_pal)
        st.session_state.data.plot_scatter_with_color('pseudo_test_with_truth', 'time (months)', 'length_filtered', 'true_rul',
                                                      palette_type='continuous',
                                                      palette_name=continuous_pal)
        st.session_state.data.plot_histogram_with_color('pseudo_test_with_truth', 'time (months)', 'length_filtered', 'label',
                                                      palette_type='qualitative',
                                                      palette_name=qualitative_pal)
        st.session_state.data.plot_histogram_with_color('pseudo_test_with_truth', 'time (months)', 'length_filtered', 'true_rul',
                                                      palette_type='qualitative',
                                                      palette_name=qualitative_pal)

    st.write("Statistiques Descriptives :")  # Calcul des statistiques descriptives
    df = pd.read_csv('./data/output/training/training_data.csv')
    statistics = df.copy().describe(include=[np.number])
    st.dataframe(statistics)

    grouped_stats_df = df.copy().groupby('Failure mode').describe()
    st.dataframe(grouped_stats_df)



