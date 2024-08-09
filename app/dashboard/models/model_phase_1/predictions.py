import numpy as np
import pandas as pd

def predict_RUL(test_item, models, window_size=3):
    model_rul, model_ttf, model_failure_mode, le_failure_mode = models

    def make_predictions(data):
        X = data[['time (months)', 'crack length (arbitary unit)']]
        rul = model_rul.predict(X) + 6
        ttf = model_ttf.predict(X) + 6
        failure_mode_encoded = model_failure_mode.predict(X)
        failure_mode = le_failure_mode.inverse_transform(failure_mode_encoded)
        return rul, ttf, failure_mode

    # Prédictions pour les données de test
    rul, ttf, failure_mode = make_predictions(test_item)
    test_item['predicted_rul'] = rul
    test_item['predicted_ttf'] = ttf
    test_item['predicted_failure_mode'] = failure_mode
    test_item['classification'] = (ttf <= test_item['time (months)'].max() + 6).astype(int)
    test_item['source'] = 0  # Données d'origine

    # Préparation des données pour les mois futurs
    last_time = test_item['time (months)'].max()
    future_months = range(last_time + 1, last_time + 7)
    future_data = pd.DataFrame({
        'item_index': test_item['item_index'].iloc[0],
        'time (months)': future_months
    })

    # Fonction pour calculer la moyenne mobile glissante
    def calculate_moving_average(last_values, new_value, window_size):
        last_values.append(new_value)
        if len(last_values) > window_size:
            last_values.pop(0)
        return np.mean(last_values)

    # Prévoir les longueurs de fissure pour les mois futurs
    last_crack_lengths = test_item['crack length (arbitary unit)'].tolist()
    future_crack_length = []

    for t in future_months:
        if len(last_crack_lengths) >= window_size:
            mean_crack_length = calculate_moving_average(last_crack_lengths, last_crack_lengths[-1], window_size)
        else:
            mean_crack_length = np.mean(last_crack_lengths)
        future_crack_length.append(mean_crack_length)
        last_crack_lengths.append(mean_crack_length)  # Ajouter la prédiction pour le mois futur

    future_data['crack length (arbitary unit)'] = future_crack_length

    # Prédictions pour les mois futurs
    future_rul, future_ttf, future_failure_mode = make_predictions(future_data)
    future_data['predicted_rul'] = future_rul
    future_data['predicted_ttf'] = future_ttf
    future_data['predicted_failure_mode'] = future_failure_mode
    future_data['classification'] = (future_ttf <= last_time + 6).astype(int)
    future_data['source'] = 1  # Données prédites

    return pd.concat([test_item, future_data], ignore_index=True)
