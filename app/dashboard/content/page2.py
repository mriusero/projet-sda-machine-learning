import streamlit as st
import pandas as pd

from scipy import stats
import plotly.express as px

from ..functions import decompose_time_series, create_features, plot_covariance_matrix, check_homoscedasticity, dataframing_data
from ..components import plot_correlation_matrix, plot_scatter, plot_scatter2, plot_histogram

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def page_2():
    st.markdown('<div class="title">MACHINE LEARNING</div>', unsafe_allow_html=True)                            #TITLE
    st.markdown('<div class="header">#2 Cleaning_</div>', unsafe_allow_html=True)
    line = '_'*50
    texte = f"""

        Here is the Cleaning phase
    {line}
    """
    st.markdown(texte)
    dataframes = dataframing_data()

    df = dataframes['train']

    st.markdown("""
        ## #Cleaning_
                """)
    col1, col2 = st.columns([1, 2])

    with col1:
        missing_percentage = (df.isna().sum() / len(df)) * 100
        st.write("Pourcentage de valeurs manquantes par colonne :")
        st.write(missing_percentage)

        df['z_score'] = stats.zscore(df[['crack length (arbitary unit)']])
        anomalies_z = df[df['z_score'].abs() > 3]
        st.write("Anomalies détectées avec Z-score :")
        st.dataframe(anomalies_z)

        Q1 = df['crack length (arbitary unit)'].quantile(0.25)
        Q3 = df['crack length (arbitary unit)'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        anomalies_iqr = df[
            (df['crack length (arbitary unit)'] < lower_bound) | (df['crack length (arbitary unit)'] > upper_bound)]
        st.write("Anomalies détectées avec IQR :")
        st.dataframe(anomalies_iqr)

    with col2:
        rows_with_missing = df[df.isna().any(axis=1)]
        st.write("Lignes valeurs manquantes (Before cleaning) :")
        st.dataframe(rows_with_missing)

        df = df.dropna()

        rows_with_missing = df[df.isna().any(axis=1)]
        st.write("Lignes valeurs manquantes (After cleaning) :")
        st.dataframe(rows_with_missing)

    st.markdown("""
        ## #Isolation Forest_
                    """)
    col1, col2 = st.columns([3, 1])
    with col1:
        # Créez un modèle Isolation Forest
        iso_forest = IsolationForest(
            contamination=0.01)  # Ajustez le paramètre de contamination en fonction de vos besoins
        # Entraînez le modèle et prédisez les anomalies
        df['iso_forest'] = iso_forest.fit_predict(df[['crack length (arbitary unit)']])
        # Filtrez les anomalies détectées
        anomalies1 = df[df['iso_forest'] == -1]
        # Affichez les anomalies détectées
        st.write("Anomalies détectées avec Isolation Forest :")
        st.dataframe(anomalies1)

        # Calculez le nombre de valeurs uniques pour 'item_index' dans les anomalies détectées
    with col2:
        unique_item_count = anomalies1['item_index'].nunique()
        st.write(f"'item_index' unique détectés: {unique_item_count}")

        num_rows_anomalies = anomalies1.shape[0]
        st.write(f"'index' unique détectés: {num_rows_anomalies}")

    st.markdown("""
        ## #OneClassSVM_
                    """)
    col1, col2 = st.columns([3, 1])
    with col1:
        oc_svm = OneClassSVM(gamma='auto', nu=0.01)  # Ajustez les paramètres en fonction de vos besoins
        df['OneClass'] = oc_svm.fit_predict(df[['crack length (arbitary unit)']])
        anomalies2 = df[df['OneClass'] == -1]
        st.write("Anomalies détectées avec One-Class SVM :")
        st.dataframe(anomalies2)

    with col2:
        unique_item_count = anomalies2['item_index'].nunique()
        st.write(f"'item_index' unique détectés: {unique_item_count}")

        num_rows_anomalies = anomalies2.shape[0]
        st.write(f"'index' unique détectés: {num_rows_anomalies}")

    # Marquer les anomalies dans le DataFrame initial
    df['detected'] = df.index.isin(anomalies1.index).astype(int)
    df['detected'] = df.index.isin(anomalies2.index).astype(int)

    st.markdown("""
        ## #Before Cleaning_
                        """)
    plot_scatter2(df, 'time (months)', 'crack length (arbitary unit)', 'detected')

    df_cleaned = df[df['detected'] == 0].drop(columns=['z_score', 'detected', 'iso_forest', 'OneClass'])
    df = df_cleaned.copy()

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


