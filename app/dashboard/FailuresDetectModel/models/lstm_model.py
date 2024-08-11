import numpy as np
import pandas as pd

from ...components import plot_scatter2

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
from .models_base import ModelBase

class LSTMModel(ModelBase):
    def __init__(self, min_sequence_length, forecast_months):
        super().__init__()
        self.min_sequence_length = min_sequence_length
        self.forecast_months = forecast_months
        self.model = None                                       # Model initialisation to None

    def train(self, X_train, y_train):
        self.model = Sequential()
        self.model.add(Masking(mask_value=0.0, input_shape=(self.min_sequence_length, 2)))
        self.model.add(LSTM(50, return_sequences=False))
        self.model.add(Dense(self.forecast_months))
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
            crack_lengths = item_data['crack length (arbitary unit)'].values

            print(f"item_index: {item_index}, Length of data: {len(times)}")

            sequence_length = self.min_sequence_length
            for i in range(len(times) - sequence_length - self.forecast_months + 1):
                seq = np.column_stack((times[i:i + sequence_length], crack_lengths[i:i + sequence_length]))
                sequences.append(seq)
                target = crack_lengths[i + sequence_length:i + sequence_length + self.forecast_months]
                targets.append(target)

        if len(sequences) == 0:
            raise ValueError("Aucune séquence valide n'a été créée avec les données fournies.")

        sequences_padded = pad_sequences(sequences, maxlen=self.min_sequence_length, padding='post', dtype='float32')
        targets = np.array(targets)

        return np.array(sequences_padded), np.array(targets)

    def predict_futures_values(self, df):
        if self.model is None:
            raise ValueError("Le modèle n'a pas été entraîné.")

        item_indices = df['item_index'].unique()
        all_predictions = []

        for item_index in item_indices:
            item_data = df[df['item_index'] == item_index].sort_values(by='time (months)')
            times = item_data['time (months)'].values
            crack_lengths = item_data['crack length (arbitary unit)'].values

            last_sequence = np.column_stack((times[-self.min_sequence_length:], crack_lengths[-self.min_sequence_length:]))
            last_sequence_padded = pad_sequences([last_sequence], maxlen=self.min_sequence_length, padding='post',
                                                 dtype='float32')
            prediction = self.model.predict(last_sequence_padded)
            all_predictions.append(prediction[0])

        return all_predictions

    def add_predictions_to_data(self, scenario, df, predictions):
        extended_data = []

        for idx, item_index in enumerate(df['item_index'].unique()):
            item_data = df[df['item_index'] == item_index].sort_values(by='time (months)')
            times = item_data['time (months)'].values

            crack_lengths = item_data['crack length (arbitary unit)'].values

            max_time = np.max(times)
            forecast_length = len(predictions[idx])
            future_start_month = int(np.ceil(max_time + 1))
            future_times = np.arange(future_start_month, future_start_month + forecast_length)
            future_crack_lengths = predictions[idx]

            if scenario[:] == 'Scenario2':
                label = item_data['label'].values
                true_rul = item_data['true_rul'].values

                initial_data = pd.DataFrame({
                    'item_index': item_index,
                    'time (months)': times,
                    'label': label,
                    'true_rul': true_rul,
                    'crack length (arbitary unit)': crack_lengths,
                    'source': 0
                })
            else:
                initial_data = pd.DataFrame({
                    'item_index': item_index,
                    'time (months)': times,
                    'crack length (arbitary unit)': crack_lengths,
                    'source': 0
                })

            forecast_data = pd.DataFrame({
                'item_index': item_index,
                'time (months)': future_times,
                'crack length (arbitary unit)': future_crack_lengths,
                'source': 1
            })

            extended_data.append(pd.concat([initial_data, forecast_data]))

        if len(extended_data) == 0:
            raise ValueError("Aucune donnée étendue n'a été créée avec les prévisions fournies.")

        df_extended = pd.concat(extended_data).reset_index(drop=True)
        df_extended.loc[df_extended['crack length (arbitary unit)'] >= 0.85, 'crack_failure'] = 1
        df_extended.loc[df_extended['crack length (arbitary unit)'] < 0.85, 'crack_failure'] = 0

        return df_extended

    def save_predictions(self, df, output_path):
        file_path = f"{output_path}/lstm_predictions.csv"
        df.to_csv(file_path, index=False)

        return f"Predictions saved successfully : {output_path}"

    def display_results(self, df):
        plot_scatter2(df, 'time (months)', 'crack length (arbitary unit)', 'source')
        plot_scatter2(df, 'time (months)', 'crack length (arbitary unit)', 'item_index')
        plot_scatter2(df, 'time (months)', 'crack length (arbitary unit)', 'crack_failure')