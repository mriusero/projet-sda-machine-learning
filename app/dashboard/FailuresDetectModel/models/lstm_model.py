import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
from .models_base import ModelBase
from ...components import plot_scatter2

class LSTMModel(ModelBase):
    def __init__(self, min_sequence_length, forecast_months):
        super().__init__()
        self.min_sequence_length = min_sequence_length
        self.forecast_months = forecast_months
        self.model = None                           # Model initialisation to None

    def train(self, X_train, y_train):
        self.model = Sequential()
        self.model.add(Masking(mask_value=0.0, input_shape=(self.min_sequence_length, 2)))
        self.model.add(LSTM(50, return_sequences=False))
        self.model.add(Dense(2 * self.forecast_months))  # Two outputs for each forecast month
        self.model.compile(optimizer='adam', loss='mse')
        self.model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

    def predict(self, X_test):
        if self.model is None:
            raise ValueError("Le modèle n'a pas été entraîné.")
        return self.model.predict(X_test)

    def prepare_sequences(self, df):
        item_indices = df['item_index'].unique()
        sequences = []
        targets = []

        for item_index in item_indices:
            item_data = df[df['item_index'] == item_index].sort_values(by='time (months)')
            times = item_data['time (months)'].values
            lengths_filtered = item_data['length_filtered'].values
            lengths_measured = item_data['length_measured'].values

            print(f"item_index: {item_index}, Length of data: {len(times)}")

            sequence_length = self.min_sequence_length
            for i in range(len(times) - sequence_length - self.forecast_months + 1):
                seq = np.column_stack((times[i:i + sequence_length], lengths_filtered[i:i + sequence_length]))
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

        item_indices = df['item_index'].unique()
        all_predictions = []

        for item_index in item_indices:
            item_data = df[df['item_index'] == item_index].sort_values(by='time (months)')
            times = item_data['time (months)'].values
            lengths_filtered = item_data['length_filtered'].values
            lengths_measured = item_data['length_measured'].values

            last_sequence = np.column_stack(
                (times[-self.min_sequence_length:], lengths_filtered[-self.min_sequence_length:]))
            last_sequence_padded = pad_sequences([last_sequence], maxlen=self.min_sequence_length, padding='post',
                                                 dtype='float32')
            prediction = self.model.predict(last_sequence_padded)

            # Reshape predictions to separate lengths_filtered and lengths_measured
            pred_lengths_filtered = prediction[0][:self.forecast_months]
            pred_lengths_measured = prediction[0][self.forecast_months:]
            combined_predictions = np.column_stack((pred_lengths_filtered, pred_lengths_measured))

            all_predictions.append(combined_predictions)

        return all_predictions

    def add_predictions_to_data(self, scenario, df, predictions):
        extended_data = []

        for idx, item_index in enumerate(df['item_index'].unique()):
            item_data = df[df['item_index'] == item_index].sort_values(by='time (months)')
            times = item_data['time (months)'].values

            lengths_filtered = item_data['length_filtered'].values
            lengths_measured = item_data['length_measured'].values

            max_time = np.max(times)
            forecast_length = len(predictions[idx])
            future_start_month = int(np.ceil(max_time + 1))
            future_times = np.arange(future_start_month, future_start_month + forecast_length)
            future_lengths_filtered = predictions[idx][:, 0]
            future_lengths_measured = predictions[idx][:, 1]

            if scenario[:] == 'Scenario2':
                label = item_data['label'].values
                true_rul = item_data['true_rul'].values

                initial_data = pd.DataFrame({
                    'item_index': item_index,
                    'time (months)': times,
                    'label': label,
                    'true_rul': true_rul,
                    'length_filtered': lengths_filtered,
                    'length_measured': lengths_measured,
                    'source': 0
                })
            else:
                initial_data = pd.DataFrame({
                    'item_index': item_index,
                    'time (months)': times,
                    'length_filtered': lengths_filtered,
                    'length_measured': lengths_measured,
                    'source': 0
                })

            forecast_data = pd.DataFrame({
                'item_index': item_index,
                'time (months)': future_times,
                'length_filtered': future_lengths_filtered,
                'length_measured': future_lengths_measured,
                'source': 1
            })

            extended_data.append(pd.concat([initial_data, forecast_data]))

        if len(extended_data) == 0:
            raise ValueError("Aucune donnée étendue n'a été créée avec les prévisions fournies.")

        df_extended = pd.concat(extended_data).reset_index(drop=True)
        df_extended.loc[df_extended['length_measured'] >= 0.85, 'crack_failure'] = 1
        df_extended.loc[df_extended['length_measured'] < 0.85, 'crack_failure'] = 0

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
