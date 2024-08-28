import streamlit as st
import pandas as pd
import numpy as np

from ..functions import dataframing_data, load_data

def page_1():
    st.markdown('<div class="header">#1 Exploration_</div>', unsafe_allow_html=True)

    dataframes = dataframing_data()

    texte = """
## #TRAIN_
        """
    st.markdown(texte)
    df = dataframes['train'].sort_values(by=['item_index', 'time (months)'])

    col1, col2 = st.columns([2, 2])

    with col1:
        st.session_state.data.plot_scatter_with_color(df_key='train',
                                                      x_col='time (months)',
                                                      y_col='length_measured',
                                                      color_col='Failure mode')

        st.session_state.data.plot_multiple_histogram(df_key='train',
                                                      x_col='time (months)',
                                                      y_col='length_measured',
                                                      color_col='Failure mode')

    with col2:
        st.session_state.data.plot_scatter_with_color(df_key='train',
                                                      x_col='time (months)',
                                                      y_col='length_filtered',
                                                      color_col='Failure mode')

        st.session_state.data.plot_multiple_histogram(df_key='train',
                                                      x_col='time (months)',
                                                      y_col='length_filtered',
                                                      color_col='Failure mode')

    st.session_state.data.plot_correlation_matrix(df_key='train')
    st.session_state.data.plot_multiple_histogram('train', 'rul (months)', 'length_filtered', 'Failure mode')

    texte = """
## #PSEUDO_TEST_WITH_TRUTH_
            """
    st.markdown(texte)
    col1, col2 = st.columns([2, 2])
    with col1:
        st.session_state.data.plot_multiple_scatter('pseudo_test_with_truth', 'time (months)', 'length_measured', 'rolling_max_time (months)')
        st.session_state.data.plot_multiple_scatter('pseudo_test_with_truth', 'time (months)', 'length_measured', 'label')
        #plot_histogram(df, 'RUL_inf_6')
    with col2:
        st.session_state.data.plot_multiple_histogram('pseudo_test_with_truth', 'time (months)', 'length_measured', 'label')

        plot_scatter2(df, 'time (months)', 'crack length (arbitary unit)', 'label')
        plot_histogram(df, 'label')

    st.write("Statistiques Descriptives :")  # Calcul des statistiques descriptives
    statistics = df.describe(include='all')
    st.dataframe(statistics)

    texte = """
## TEST_
                """
    st.markdown(texte)
    df = dataframes['test'].sort_values(by=['item_index', 'time (months)'])

    plot_scatter(df, 'time (months)', 'crack length (arbitary unit)', 'item_index')
