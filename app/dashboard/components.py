import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import plotly.express as px




def plot_correlation_matrix(df):
    """
    Cette fonction prend un DataFrame en entrée et affiche une heatmap de la matrice de corrélation
    entre les variables numériques du DataFrame.

    :param df: DataFrame avec des variables numériques pour lesquelles calculer la corrélation.
    """
    numeric_df = df.select_dtypes(include=[float, int])
    corr_matrix = numeric_df.corr()
    fig = plt.figure(figsize=(4, 2))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1, center=0)
    plt.title('Matrice de Corrélation')
    st.pyplot(fig)


def plot_scatter(df, x_col, y_col, color_col):
    if x_col and y_col and color_col:
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                         title=f'Nuage de points pour {x_col} vs {y_col}',
                         color_continuous_scale=px.colors.cyclical.IceFire)  # Optionnel : choisir une palette de couleurs
        st.plotly_chart(fig)

def plot_scatter2(df, x_col, y_col, color_col):
    if x_col and y_col and color_col:
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                         title=f'Nuage de points pour {x_col} vs {y_col}',
                         color_discrete_map={0: 'blue', 1: 'red'})  # Définir les couleurs pour les valeurs 0 et 1
        st.plotly_chart(fig)


def plot_dist2(df, color):
    fig = px.histogram(df, x="time (months)", y="crack length (arbitary unit)", color=color, marginal="box",
                       hover_data=df.columns)
    st.plotly_chart(fig)