import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import io


def plot_distribution(df):
    # Fonction pour détecter les types de variables
    def detect_variable_type(df):
        variable_types = {}
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                variable_types[column] = 'Numeric'
            elif pd.api.types.is_categorical_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
                variable_types[column] = 'Categorical'
            else:
                variable_types[column] = 'Other'
        return variable_types

    # Fonction pour créer un graphique pour une variable donnée
    def create_plot(column, var_type):
        plt.figure(figsize=(5, 3))
        if var_type == 'Numeric':
            sns.histplot(df[column], bins=30, kde=True)
            plt.title(f'{column}')
        elif var_type == 'Categorical':
            sns.countplot(data=df, x=column)
            plt.title(f'{column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.tight_layout()
        # Sauvegarde du graphique dans un buffer pour l'afficher dans Streamlit
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return buf

    # Détection des types de variables
    variable_types = detect_variable_type(df)

    # Affichage des graphiques dans Streamlit


    num_vars = len(variable_types)

    cols = st.columns(3)  # Crée une grille avec 2 colonnes
    for i, (column, var_type) in enumerate(variable_types.items()):
        # Déterminer la colonne de la grille où le graphique doit être affiché
        col_index = i % 3
        with cols[col_index]:

            buf = create_plot(column, var_type)
            st.image(buf)

