import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
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
        if var_type == 'Numeric':
            fig = px.histogram(df, x=column, nbins=30, title=f'{column}',
                               marginal='rug',  # Ajoute une rug plot pour une meilleure vue d'ensemble
                               labels={column: 'Frequency'})
        elif var_type == 'Categorical':
            fig = px.bar(df, x=column, title=f'{column}',
                         labels={column: 'Frequency'})
        else:
            fig = go.Figure()  # Un graphique vide pour les types non gérés

        fig.update_layout(
            xaxis_title=column,
            yaxis_title='Frequency',
            xaxis_tickangle=-45
        )
        return fig

    # Détection des types de variables
    variable_types = detect_variable_type(df)

    # Affichage des graphiques dans Streamlit
    num_vars = len(variable_types)
    cols = st.columns(6)  # Crée une grille avec 3 colonnes

    for i, (column, var_type) in enumerate(variable_types.items()):
        # Déterminer la colonne de la grille où le graphique doit être affiché
        col_index = i % 6
        with cols[col_index]:
            fig = create_plot(column, var_type)
            st.plotly_chart(fig)

