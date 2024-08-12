import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Input
from .models_base import ModelBase
from ...components import plot_scatter2

class LSTMModelV2(ModelBase):
    def __init__(self, min_sequence_length, forecast_months):
        super().__init__()
        self.min_sequence_length = min_sequence_length
        self.forecast_months = forecast_months
        self.model = None                           # Model initialisation to None

    def train(self, X_train, y_train):
        input_shape = (self.min_sequence_length, 10)  # 10 caractéristiques par séquence
        self.model = Sequential()
        self.model.add(Input(shape=input_shape))  # Utiliser Input pour définir la forme
        self.model.add(Masking(mask_value=0.0))
        self.model.add(LSTM(50, return_sequences=False))
        self.model.add(Dense(2 * self.forecast_months))  # Deux sorties pour chaque mois de prévision
        self.model.compile(optimizer='adam', loss='mse')

        print(np.isnan(X_train).any(), np.isinf(X_train).any())
        print(np.isnan(y_train).any(), np.isinf(y_train).any())

        self.model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)


    def predict(self, X_test):
        if self.model is None:
            raise ValueError("Le modèle n'a pas été entraîné.")
        return self.model.predict(X_test)

    def add_features(self, df):
        # Définir la taille de la fenêtre pour les statistiques
        window_size = self.min_sequence_length

        # Calculer les caractéristiques supplémentaires avec des fenêtres mobiles
        def calculate_rolling_features(series):
            return {
                'mean': series.rolling(window=window_size, min_periods=1).mean(),
                'std': series.rolling(window=window_size, min_periods=1).std(),
                'max': series.rolling(window=window_size, min_periods=1).max(),
                'min': series.rolling(window=window_size, min_periods=1).min()
            }

        def replace_nan(series):
            #columns_mean = series.columns.mean()
            return series.fillna(series.mean())

        # Calcul des caractéristiques et remplacement des NaN
        for length_type in ['length_measured', 'length_filtered']:
            for stat in ['mean', 'std', 'max', 'min']:
                col_name = f'rolling_{stat}_{length_type}'
                df[col_name] = df.groupby('item_index')[length_type].transform(
                    lambda x: calculate_rolling_features(x)[stat]
                )
                df[col_name] = df.groupby('item_index')[col_name].transform(
                    lambda x: replace_nan(x)
                )

        return df

    def prepare_sequences(self, df):
        df = self.add_features(df)  # Ajouter les nouvelles caractéristiques

        item_indices = df['item_index'].unique()
        sequences = []
        targets = []

        for item_index in item_indices:
            item_data = df[df['item_index'] == item_index].sort_values(by='time (months)')
            times = item_data['time (months)'].values
            lengths_filtered = item_data['length_filtered'].values
            lengths_measured = item_data['length_measured'].values

            # Nouvelles caractéristiques
            rolling_means_filtered = item_data['rolling_mean_length_filtered'].values
            rolling_stds_filtered = item_data['rolling_std_length_filtered'].values
            rolling_maxs_filtered = item_data['rolling_max_length_filtered'].values
            rolling_mins_filtered = item_data['rolling_min_length_filtered'].values

            rolling_means_measured = item_data['rolling_mean_length_measured'].values
            rolling_stds_measured = item_data['rolling_std_length_measured'].values
            rolling_maxs_measured = item_data['rolling_max_length_measured'].values
            rolling_mins_measured = item_data['rolling_min_length_measured'].values

            print(f"item_index: {item_index}, Length of data: {len(times)}")

            sequence_length = self.min_sequence_length
            for i in range(len(times) - sequence_length - self.forecast_months + 1):
                seq = np.column_stack((
                    times[i:i + sequence_length],
                    lengths_filtered[i:i + sequence_length],
                    rolling_means_filtered[i:i + sequence_length],
                    rolling_stds_filtered[i:i + sequence_length],
                    rolling_maxs_filtered[i:i + sequence_length],
                    rolling_mins_filtered[i:i + sequence_length],
                    rolling_means_measured[i:i + sequence_length],
                    rolling_stds_measured[i:i + sequence_length],
                    rolling_maxs_measured[i:i + sequence_length],
                    rolling_mins_measured[i:i + sequence_length]
                ))
                sequences.append(seq)

                target = np.column_stack((
                    lengths_filtered[i + sequence_length:i + sequence_length + self.forecast_months],
                    lengths_measured[i + sequence_length:i + sequence_length + self.forecast_months]
                ))
                targets.append(target)

        if len(sequences) == 0:
            raise ValueError("Aucune séquence valide n'a été créée avec les données fournies.")

        sequences_padded = pad_sequences(sequences, maxlen=self.min_sequence_length, padding='post', dtype='float32')
        targets = np.array(targets).reshape(-1, 2 * self.forecast_months)

        return np.array(sequences_padded), np.array(targets)

    def predict_futures_values(self, df):
        if self.model is None:
            raise ValueError("Le modèle n'a pas été entraîné.")

        def extract_features(item_data):
            item_data = self.add_features(item_data)
            features = {
                'times': item_data['time (months)'].values,
                'length_filtered': item_data['length_filtered'].values,
                'length_measured': item_data['length_measured'].values,
                'rolling_means_filtered': item_data['rolling_mean_length_filtered'].values,
                'rolling_stds_filtered': item_data['rolling_std_length_filtered'].values,
                'rolling_maxs_filtered': item_data['rolling_max_length_filtered'].values,
                'rolling_mins_filtered': item_data['rolling_min_length_filtered'].values,
                'rolling_means_measured': item_data['rolling_mean_length_measured'].values,
                'rolling_stds_measured': item_data['rolling_std_length_measured'].values,
                'rolling_maxs_measured': item_data['rolling_max_length_measured'].values,
                'rolling_mins_measured': item_data['rolling_min_length_measured'].values
            }
            return features

        def prepare_sequence(features):
            last_sequence = np.column_stack((
                features['times'][-self.min_sequence_length:],
                features['length_filtered'][-self.min_sequence_length:],
                features['rolling_means_filtered'][-self.min_sequence_length:],
                features['rolling_stds_filtered'][-self.min_sequence_length:],
                features['rolling_maxs_filtered'][-self.min_sequence_length:],
                features['rolling_mins_filtered'][-self.min_sequence_length:],
                features['rolling_means_measured'][-self.min_sequence_length:],
                features['rolling_stds_measured'][-self.min_sequence_length:],
                features['rolling_maxs_measured'][-self.min_sequence_length:],
                features['rolling_mins_measured'][-self.min_sequence_length:]
            ))
            return pad_sequences([last_sequence], maxlen=self.min_sequence_length, padding='post', dtype='float32')

        item_indices = df['item_index'].unique()
        all_predictions = []

        for item_index in item_indices:
            item_data = df[df['item_index'] == item_index].sort_values(by='time (months)')
            features = extract_features(item_data)
            last_sequence_padded = prepare_sequence(features)
            prediction = self.model.predict(last_sequence_padded)

            # Séparer les prédictions pour length_filtered et length_measured
            pred_lengths_filtered = prediction[0][:self.forecast_months]
            pred_lengths_measured = prediction[0][self.forecast_months:]
            combined_predictions = np.column_stack((pred_lengths_filtered, pred_lengths_measured))

            all_predictions.append(combined_predictions)

        return all_predictions

    def add_predictions_to_data(self, scenario, df, predictions):
        def prepare_initial_data(item_data, item_index, source):
            times = item_data['time (months)'].values
            lengths_filtered = item_data['length_filtered'].values
            lengths_measured = item_data['length_measured'].values

            item_data = self.add_features(item_data)
            features = {
                'rolling_means_filtered': item_data['rolling_mean_length_filtered'].values,
                'rolling_stds_filtered': item_data['rolling_std_length_filtered'].values,
                'rolling_maxs_filtered': item_data['rolling_max_length_filtered'].values,
                'rolling_mins_filtered': item_data['rolling_min_length_filtered'].values,
                'rolling_means_measured': item_data['rolling_mean_length_measured'].values,
                'rolling_stds_measured': item_data['rolling_std_length_measured'].values,
                'rolling_maxs_measured': item_data['rolling_max_length_measured'].values,
                'rolling_mins_measured': item_data['rolling_min_length_measured'].values
            }

            data_dict = {
                'item_index': item_index,
                'time (months)': times,
                'length_filtered': lengths_filtered,
                'length_measured': lengths_measured,
                'source': source
            }
            data_dict.update(features)

            if scenario == 'Scenario2':
                data_dict.update({
                    'label': item_data['label'].values,
                    'true_rul': item_data['true_rul'].values
                })
            return pd.DataFrame(data_dict)

        item_indices = df['item_index'].unique()
        extended_data = []

        for idx, item_index in enumerate(item_indices):
            item_data = df[df['item_index'] == item_index].sort_values(by='time (months)')
            max_time = np.max(item_data['time (months)'].values)
            forecast_length = len(predictions[idx])
            future_times = np.arange(np.ceil(max_time + 1), np.ceil(max_time + 1) + forecast_length)

            future_lengths_filtered = predictions[idx][:, 0]
            future_lengths_measured = predictions[idx][:, 1]

            initial_data = prepare_initial_data(item_data, item_index, source=0)
            forecast_data = pd.DataFrame({
                'item_index': item_index,
                'time (months)': future_times,
                'length_filtered': future_lengths_filtered,
                'length_measured': future_lengths_measured,
                'source': 1
            })

            extended_data.append(pd.concat([initial_data, forecast_data]))

        if not extended_data:
            raise ValueError("Aucune donnée étendue n'a été créée avec les prévisions fournies.")

        df_extended = pd.concat(extended_data).reset_index(drop=True)
        df_extended['crack_failure'] = (df_extended['length_measured'] >= 0.85).astype(int)

        return df_extended

    def save_predictions(self, df, output_path):
        file_path = f"{output_path}/lstm_predictions.csv"
        df.to_csv(file_path, index=False)

        return f"Predictions saved successfully : {output_path}"

    def display_results(self, df):
        col1, col2 = st.columns(2)
        with col1:
            plot_scatter2(df, 'time (months)', 'length_measured', 'source')
            plot_scatter2(df, 'time (months)', 'length_measured', 'item_index')
            plot_scatter2(df, 'time (months)', 'length_measured', 'crack_failure')
        with col2:
            plot_scatter2(df, 'time (months)', 'length_filtered', 'source')
            plot_scatter2(df, 'time (months)', 'length_filtered', 'item_index')
            plot_scatter2(df, 'time (months)', 'length_filtered', 'crack_failure')
