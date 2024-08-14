import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
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