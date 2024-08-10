import streamlit as st
from ..components import plot_correlation_matrix, plot_scatter, plot_histogram, plot_scatter2
import pandas as pd
from scipy import stats
from ..models.model_phase_1.data_processing import clean_data
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

def page_1(dataframes):
    st.markdown('<div class="title">MACHINE LEARNING</div>', unsafe_allow_html=True)                            #TITLE
    st.markdown('<div class="header">#1 Exploration & Cleaning_</div>', unsafe_allow_html=True)

    texte = """
# Exploration_
        """
    st.markdown(texte)

    st.markdown('## TRAIN')
    col1, col2 = st.columns([2, 2])
    df = dataframes['train'].sort_values(by=['item_index', 'time (months)'])
    with col1:
        plot_scatter(df, 'time (months)', 'crack length (arbitary unit)', 'Failure mode')
        plot_correlation_matrix(df)
    with col2:
        plot_histogram(df, 'Failure mode')

    st.markdown('## PSEUDO_TEST_WITH_TRUTH')
    col1, col2 = st.columns([2, 2])
    df = dataframes['pseudo_test_with_truth'].sort_values(by=['item_index', 'time (months)'])
    with col1:
        plot_scatter2(df, 'time (months)', 'crack length (arbitary unit)', 'label')
        st.write("Statistiques Descriptives :")  # Calcul des statistiques descriptives
        statistics = df.describe(include='all')
        st.dataframe(statistics)
    with col2:
        plot_histogram(df, 'label')

    st.markdown('## TEST')
    df = dataframes['test'].sort_values(by=['item_index', 'time (months)'])
    plot_scatter(df, 'time (months)', 'crack length (arbitary unit)', 'item_index')

    texte = """
# Cleaning_
        """
    st.markdown(texte)
    training_description = """

        Here is the Cleaning phase!

    """
    st.markdown(training_description)

    df = pd.read_csv(
        '/Users/mariusayrault/GitHub/Sorb-Data-Analytics/projet-sda-machine-learning/app/data/output/training/training_data.csv')

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

        df = clean_data(df)

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

