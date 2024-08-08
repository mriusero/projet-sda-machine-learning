import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import plotly.express as px

def plot_crack_length_by_item_with_failures(df):
    specific_failure_modes = ['Fatigue crack', 'Control board failure', 'Infant mortality']
    cols = st.columns(len(specific_failure_modes))

    for idx, mode in enumerate(specific_failure_modes):
        mode_df = df[df['Failure mode'] == mode]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title(f'Crack Length Evolution by Item for Failure Mode: {mode}')

        item_ids = mode_df['item_index'].unique()

        for item_id in item_ids:
            item_df = mode_df[mode_df['item_index'] == item_id]
            ax.plot(item_df['time (months)'], item_df['crack length (arbitary unit)'], label=f'Item {item_id}')
            failure_data = item_df[['Time to failure (months)', 'Failure mode']].dropna().drop_duplicates()

            if not failure_data.empty:
                time_to_failure = failure_data['Time to failure (months)'].values[0]
                ax.annotate(
                    f'{mode}',
                    xy=(time_to_failure, item_df['crack length (arbitary unit)'].max()),
                    xytext=(time_to_failure + 1, item_df['crack length (arbitary unit)'].max() * 0.8),
                    arrowprops=dict(facecolor='red', shrink=0.05),
                    fontsize=9
                )

        ax.set_xlabel('Time (months)')
        ax.set_ylabel('Crack Length (arbitrary unit)')
        ax.legend()
        ax.grid(True)

        with cols[idx]:
            st.pyplot(fig)

        plt.close(fig)


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


def plot_scatter(df, x_col, y_col):
    if x_col and y_col:
        fig = px.scatter(df, x=x_col, y=y_col, title=f'Nuage de points pour {x_col} vs {y_col}')
        st.plotly_chart(fig)

