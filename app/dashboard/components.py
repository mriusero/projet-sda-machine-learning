import matplotlib.pyplot as plt

def plot_crack_length_by_item_with_failures(df):
    """
    Trace la courbe de l'évolution de la longueur de fissure en fonction du temps pour chaque item_id unique,
    et ajoute des annotations pour les moments de panne et le mode de défaillance.

    """
    fig = plt.figure(figsize=(12, 14))

    item_ids = df['item_index'].unique()   # Obtenir les item_ids uniques

    for item_id in item_ids:

        item_df = df[df['item_index'] == item_id]      # Filtrer les données pour l'item_id courant

        plt.plot(item_df['time (months)'], item_df['crack length (arbitary unit)'], label=f'Item {item_id}')        # Tracer la courbe pour cet item_id

        failure_data = item_df[['Time to failure (months)', 'Failure mode']].dropna().drop_duplicates()     # Trouver les données de panne pour cet item_id
        if not failure_data.empty:
            time_to_failure = failure_data['Time to failure (months)'].values[0]
            failure_mode = failure_data['Failure mode'].values[0]

            # Ajouter une annotation pour le moment de la panne
            plt.annotate(
                f'{failure_mode}',
                xy=(time_to_failure, item_df['crack length (arbitary unit)'].max()),
                xytext=(time_to_failure + 1, item_df['crack length (arbitary unit)'].max() * 0.8),
                arrowprops=dict(facecolor='red', shrink=0.05),
                fontsize=9
            )

    # Ajouter des labels et un titre
    plt.xlabel('Time (months)')
    plt.ylabel('Crack Length (arbitrary unit)')
    plt.title('Crack Length Evolution by Item with Failure Annotations')
    plt.legend()
    plt.grid(True)

    return fig



