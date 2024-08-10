import streamlit as st
import pandas as pd
from ..components import plot_correlation_matrix, plot_scatter, plot_scatter2, plot_histogram
import plotly.express as px
from ..functions import decompose_time_series, create_features, plot_covariance_matrix, check_homoscedasticity


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def page_2():
    st.markdown('<div class="title">MACHINE LEARNING</div>', unsafe_allow_html=True)                            #TITLE
    st.markdown('<div class="header">#2 Feature Engineering_</div>', unsafe_allow_html=True)
    texte = """

        Here is the Feature Engineering phase.

    """
    st.markdown(texte)
    df = pd.read_csv(
        '/Users/mariusayrault/GitHub/Sorb-Data-Analytics/projet-sda-machine-learning/app/data/output/training/training_data.csv')

    st.markdown("""
    ## #Mise à l'échelle_
                    """)
    # Colonnes à normaliser et standardiser
    standardize_columns = ['time (months)', 'crack length (arbitary unit)', 'rul (months)', 'Time to failure (months)']
    normalize_columns = ['time (months)', 'crack length (arbitary unit)', 'rul (months)', 'Time to failure (months)']

    st.markdown("""
        ## Comparaison Avant et Après Standardisation
        """)

    # Création du scaler pour la standardisation
    scaler_standard = StandardScaler()
    df_standardized = df.copy()
    df_standardized[standardize_columns] = scaler_standard.fit_transform(df_standardized[standardize_columns])

    # Créer deux colonnes pour les visualisations
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Avant Standardisation")
        for col in standardize_columns:
            fig_before = px.histogram(df, x=col, nbins=10,
                                      title=f'Distribution de {col} Avant Standardisation',
                                      labels={col: col})
            fig_before.update_layout(bargap=0.2)  # Ajuster l'écart entre les barres
            st.plotly_chart(fig_before)

    with col2:
        st.markdown("### Après Standardisation")
        for col in standardize_columns:
            fig_after = px.histogram(df_standardized, x=col, nbins=10,
                                     title=f'Distribution de {col} Après Standardisation',
                                     labels={col: col})
            fig_after.update_layout(bargap=0.2)  # Ajuster l'écart entre les barres
            st.plotly_chart(fig_after)

    st.markdown("""
        ## Comparaison Avant et Après Normalisation
        """)

    # Création du scaler pour la normalisation
    scaler_normalize = MinMaxScaler()
    df_normalized = df.copy()
    df_normalized[normalize_columns] = scaler_normalize.fit_transform(df_normalized[normalize_columns])

    # Créer deux colonnes pour les visualisations
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Avant Normalisation")
        for col in normalize_columns:
            fig_before = px.histogram(df, x=col, nbins=10,
                                      title=f'Distribution de {col} Avant Normalisation',
                                      labels={col: col})
            fig_before.update_layout(bargap=0.2)  # Ajuster l'écart entre les barres
            st.plotly_chart(fig_before)

    with col2:
        st.markdown("### Après Normalisation")
        for col in normalize_columns:
            fig_after = px.histogram(df_normalized, x=col, nbins=10,
                                     title=f'Distribution de {col} Après Normalisation',
                                     labels={col: col})
            fig_after.update_layout(bargap=0.2)  # Ajuster l'écart entre les barres
            st.plotly_chart(fig_after)

    st.markdown("""
    ## #Statistics_
            """)

    st.markdown("""
        ## Statistiques Descriptives
        """)
    st.write("Statistiques Descriptives :")  # Calcul des statistiques descriptives
    statistics = df.describe(include='all')
    st.dataframe(statistics)

    decompose_time_series(df.copy(), 'time (months)', 'crack length (arbitary unit)')
    df = create_features(df, 'time (months)', 'crack length (arbitary unit)', 'item_index', period=12)
    st.dataframe(df)
    plot_correlation_matrix(df)
    plot_covariance_matrix(df, exclude_columns=['item_index', 'Failure mode'])
    check_homoscedasticity(df, target_col='crack length (arbitary unit)',
                           exclude_columns=['item_index', 'Failure mode'])



