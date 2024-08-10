import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go



def plot_correlation_matrix(df):
    numeric_df = df.select_dtypes(include=[float, int])
    corr_matrix = numeric_df.corr()

    # Créer une trace pour la matrice de corrélation
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        colorbar=dict(title='Correlation'),
        text=corr_matrix.round(2).astype(str).values,  # Texte des annotations
        texttemplate='%{text}',  # Formater le texte
        textfont=dict(size=12, color='white'),  # Couleur et taille du texte
        hoverinfo='text',
        showscale=True
    ))

    # Mettre à jour la disposition pour le mode sombre
    fig.update_layout(
        title='Matrice de Corrélation',
        xaxis_title='Variables',
        yaxis_title='Variables',
        xaxis=dict(ticks='', side='bottom', title_standoff=0),
        yaxis=dict(ticks='', side='left', title_standoff=0),
        plot_bgcolor='#0E1117',
        paper_bgcolor='#0E1117',
        font=dict(color='white'),
        title_font=dict(color='white'),
        coloraxis_colorbar=dict(title='Correlation', tickvals=[-1, 0, 1], ticktext=['-1', '0', '1'])
    )

    st.plotly_chart(fig, use_container_width=True)


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


def plot_histogram(df, color):
    fig = px.histogram(df, x="time (months)", y="crack length (arbitary unit)", color=color, marginal="box",
                       hover_data=df.columns)
    st.plotly_chart(fig)

    fig = px.histogram(df, x='crack length (arbitary unit)', nbins=30,
                           color=color,
                           labels={'crack length (arbitary unit)': 'Crack length (arbitary unit)'},
                           title='Histogramme de la longueur de fissure (unité arbitraire) par mode de défaillance')

    fig.update_traces(opacity=0.7)  # Ajuster la transparence des barres
    fig.update_layout(
        xaxis_title='Crack length (arbitary unit)',
        yaxis_title='Frequency',
        title_text='Histogramme pour détecter les anomalies',
        title_x=0.5,
        barmode='stack'  # Empiler les barres pour une meilleure visibilité
    )
    st.plotly_chart(fig)