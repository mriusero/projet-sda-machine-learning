import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import STL
import seaborn as sns
import pandas as pd
from ..FailuresDetectModel import clean_data, FeatureAdder
from ..functions import load_data, dataframing_data


class DataVisualizer:
    def __init__(self):
        self.dataframes = dataframing_data()
        self.feature_adder = FeatureAdder(min_sequence_length=2)
        self.df = {
            'train': self.preprocessing(self.dataframes['train']),
            'pseudo_test': self.preprocessing(self.dataframes['pseudo_test']),
            'pseudo_test_with_truth': self.preprocessing(self.dataframes['pseudo_test_with_truth']),
            'test': self.preprocessing(self.dataframes['test'])
        }

    @st.cache_data
    def preprocessing(_self, df):
        return _self.feature_adder.add_features(clean_data(df), particles_filtery=True)

    def get_dataframes(self):
        return self.dataframes

    def get_the(self, df_key):
        return self.df[df_key]

    def get_color_palette(self):
        return {
            'blue': '#1f77b4',
            'orange': '#ff7f0e',
            'green': '#2ca02c',
            'red': '#d62728',
            'purple': '#9467bd',
            'cyclical': px.colors.cyclical.IceFire,
            'pastel': px.colors.qualitative.Bold
        }

    ## --- Scatter ---
    def plot_scatter_with_color(self, df_key, x_col, y_col, color_col):
        df = self.df[df_key]
        color_palette = self.get_color_palette()
        if x_col and y_col and color_col:
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                             title=f'Nuage de points pour {x_col} vs {y_col}',
                             color_continuous_scale=color_palette['cyclical'])
            st.plotly_chart(fig)

    def plot_multiple_scatter(self, df_key, x_col, y_col, color_col):
        df = self.df[df_key]
        color_palette = self.get_color_palette()

        if x_col and y_col and color_col:
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                             title=f'Nuage de points pour {x_col} vs {y_col}',
                             color_discrete_map={0: color_palette['pastel'][0], 1: color_palette['pastel'][1]})
            st.plotly_chart(fig)

    def scatter_plot(self, df_key, x_column, y_column, color=None):
        color = color or self.get_color_palette()['blue']
        df = self.df[df_key]
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x=x_column, y=y_column, color=color)
        plt.title(f'Scatter Plot of {x_column} vs {y_column}')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        st.pyplot(plt.gcf())
        plt.close()

    ## --- Histogram ---
    def plot_distribution_histogram(self, df_key, column_name, color='blue'):
        color = color or self.get_color_palette()['blue']
        df = self.df[df_key]

        fig = px.histogram(df, x=column_name, marginal="kde", color=color)

        fig.update_layout(
            title=f'Distribution of {column_name}',
            title_font_color='white',
            xaxis_title=column_name,
            xaxis_title_font_color='white',
            yaxis_title='Frequency',
            yaxis_title_font_color='white',
            plot_bgcolor='black',
            paper_bgcolor='black',
            font_color='white'
        )

        st.plotly_chart(fig, use_container_width=True)

    def plot_multiple_histogram(self, df_key, x_col, y_col, color_col):
        df = self.df[df_key]
        color_palette = self.get_color_palette()

        fig = px.histogram(df, x=x_col, y=y_col, color=color_col, marginal="box",
                           hover_data=df.columns)
        st.plotly_chart(fig)

        fig = px.histogram(df, x=x_col, nbins=30,
                           color=color_col,
                           labels={'crack length (arbitary unit)': 'Crack length (arbitary unit)'},
                           title='Histogramme de la longueur de fissure (unité arbitraire) par mode de défaillance')

        fig.update_traces(opacity=0.7)
        fig.update_layout(
            xaxis_title='Crack length (arbitary unit)',
            yaxis_title='Frequency',
            title_text='Histogramme pour détecter les anomalies',
            title_x=0.5,
            barmode='stack'
        )
        st.plotly_chart(fig)

    ## --- Specific ---
    def plot_correlation_matrix(self, df_key):
        df = self.df[df_key]
        color_palette = self.get_color_palette()

        numeric_df = df.select_dtypes(include=[float, int])
        corr_matrix = numeric_df.corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='Viridis',
            zmin=-1,
            zmax=1,
            colorbar=dict(title='Correlation'),
            text=corr_matrix.round(2).astype(str).values,
            texttemplate='%{text}',
            textfont=dict(size=12, color='white'),
            hoverinfo='text',
            showscale=True
        ))

        fig.update_layout(
            title='Matrice de Corrélation',
            xaxis_title='Variables',
            yaxis_title='Variables',
            xaxis=dict(ticks='', side='bottom', title_standoff=0),
            yaxis=dict(ticks='', side='left', title_standoff=0),
            plot_bgcolor='#313234',
            paper_bgcolor='#313234',
            font=dict(color='white'),
            title_font=dict(color='white'),
            coloraxis_colorbar=dict(title='Correlation', tickvals=[-1, 0, 1], ticktext=['-1', '0', '1'])
        )

        st.plotly_chart(fig, use_container_width=True)

    def plot_pairplot(self, df_key, hue=None, palette='Set2'):
        """
        Tracer un pair plot à partir d'un DataFrame Pandas.
        Paramètres:
        - data: DataFrame contenant les données.
        - hue: Nom de la colonne pour colorer les points selon une variable catégorielle.
        - palette: Palette de couleurs à utiliser pour le graphique.
        """
        data = pd.read_csv('./data/output/training/training_data.csv')
        # Créer le pair plot
        fig = sns.pairplot(data, hue=hue, palette=palette)
        # Afficher le graphique
        st.pyplot(fig)

    def boxplot(self, df_key, x_col, y_col):
        """
        Trace un graphique avec plusieurs boxplots de x_col en fonction de y_col.

        Paramètres:
        - df_key : La clé du DataFrame à utiliser.
        - x_col : La colonne du DataFrame à utiliser pour l'axe x.
        - y_col : La colonne du DataFrame à utiliser pour l'axe y.
        """
        data = self.df[df_key]

        fig = plt.figure(figsize=(10, 6))  # Définir la taille de la figure
        sns.boxplot(x=x_col, y=y_col, data=data, palette="Set2")  # Créer le boxplot avec seaborn

        plt.xlabel(x_col)  # Label pour l'axe x
        plt.ylabel(y_col)  # Label pour l'axe y
        plt.title(f'Boxplot de {x_col} en fonction de {y_col}')  # Titre du graphique

        st.pyplot(fig)                               # Afficher le graphique

    def decompose_time_series(self, df_key, time_col, value_col, period=12):
        df = self.df[df_key]

        if time_col not in df.columns or value_col not in df.columns:
            st.error(f"Les colonnes '{time_col}' ou '{value_col}' n'existent pas dans le DataFrame.")
            return

        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.sort_values(by=time_col)

        df = df.dropna(subset=[time_col, value_col])

        df.set_index(time_col, inplace=True)

        # Effectuer la décomposition STL
        try:
            stl = STL(df[value_col], period=period)
            result = stl.fit()
        except Exception as e:
            st.error(f"Erreur lors de la décomposition STL: {e}")
            return

        # Tracer les résultats
        fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
        fig.suptitle(f'Décomposition de la série temporelle : {value_col}', fontsize=16)

        # Tendance
        axes[0].plot(result.trend, label='Tendance', color='blue')
        axes[0].set_title('Tendance')
        axes[0].legend(loc='upper left')

        # Saison
        axes[1].plot(result.seasonal, label='Saisonnalité', color='green')
        axes[1].set_title('Saisonnalité')
        axes[1].legend(loc='upper left')

        # Résidu
        axes[2].plot(result.resid, label='Résidu', color='red')
        axes[2].set_title('Résidu')
        axes[2].legend(loc='upper left')

        # Série originale
        axes[3].plot(df[value_col], label='Série Originale', color='gray')
        axes[3].set_title('Série Originale')
        axes[3].legend(loc='upper left')

        plt.tight_layout(rect=[0, 0, 1, 0.97])

        st.pyplot(fig)