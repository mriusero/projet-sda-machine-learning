import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Masking, Input
from tensorflow.keras.optimizers import Adam
from .models_base import ModelBase
from ...components import plot_scatter2
from ...functions import load_failures


class LSTMModelV3(ModelBase):
    def __init__(self, min_sequence_length, forecast_months):
        super().__init__()
        self.min_sequence_length = min_sequence_length
        self.forecast_months = forecast_months
        self.failure_mode_encoder = LabelEncoder()
        self.model = None  # Model initialisation to None

    def fit_failure_mode_encoder(self, failures_df):
        failures_df['Failure mode'].fillna('UNKNOWN', inplace=True)
        failure_modes = failures_df['Failure mode']
        self.failure_mode_encoder.fit(failure_modes)

    def encode_failure_mode(self, failure_modes):
        return self.failure_mode_encoder.transform(failure_modes)

    def decode_failure_mode(self, encoded_modes):
        return self.failure_mode_encoder.inverse_transform(encoded_modes)

    def train(self, X_train, y_train, y_failure_modes):
        input_shape = (self.min_sequence_length, 10)  # 10 caractéristiques par séquence
        num_classes = len(self.failure_mode_encoder.classes_)  # Nombre de classes pour Failure mode

        # Define input layer
        inputs = Input(shape=input_shape)

        # Define LSTM layer
        x = Masking(mask_value=0.0)(inputs)
        x = LSTM(50, return_sequences=False)(x)

        # Define outputs
        forecast_output = Dense(2 * self.forecast_months, name='forecast_output')(x)
        classification_output = Dense(num_classes, activation='softmax', name='classification_output')(x)

        # Create model
        self.model = Model(inputs=inputs, outputs=[forecast_output, classification_output])

        # Compile model
        self.model.compile(optimizer='adam',
                           loss={'forecast_output': 'mse', 'classification_output': 'sparse_categorical_crossentropy'},
                           metrics={'forecast_output': 'mae', 'classification_output': 'accuracy'})

        # Print for debugging
        print(np.isnan(X_train).any(), np.isinf(X_train).any())
        print(np.isnan(y_train).any(), np.isinf(y_train).any())
        print(np.isnan(y_failure_modes).any(), np.isinf(y_failure_modes).any())

        # Train model
        self.model.fit(X_train, {'forecast_output': y_train, 'classification_output': y_failure_modes},
                       epochs=50, batch_size=32, validation_split=0.2)

    def predict(self, X_test):
        if self.model is None:
            raise ValueError("Le modèle n'a pas été entraîné.")

        # Prédictions de valeurs temporelles et de Failure mode
        predictions = self.model.predict(X_test)

        forecast_predictions = predictions[0]  # Prédictions des valeurs temporelles
        classification_predictions = np.argmax(predictions[1], axis=1)  # Prédictions de Failure mode

        return forecast_predictions, classification_predictions

    def add_features(self, df):

        to_recalcul = ['rolling_means_filtered',
                       'rolling_stds_filtered',
                       'rolling_maxs_filtered',
                       'rolling_mins_filtered',
                       'rolling_means_measured',
                       'rolling_stds_measured',
                       'rolling_maxs_measured',
                       'rolling_mins_measured']

        to_delete = [col for col in to_recalcul if col in df.columns]
        df = df.drop(columns=to_delete, errors='ignore')

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

    def prepare_train_sequences(self, df):
        df = self.add_features(df)  # Ajouter les nouvelles caractéristiques

        item_indices = df['item_index'].unique()
        sequences = []
        targets = []
        failure_modes = []


        failures_df = load_failures()
        self.fit_failure_mode_encoder(failures_df)  # Appel unique ici
        failure_mode_dict = failures_df.set_index('item_index')['Failure mode'].to_dict()


        for item_index in item_indices:

            item_data = df[df['item_index'] == item_index].sort_values(by='time (months)')

            item_data['Failure mode'] = failure_mode_dict.get(item_index, None)
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

                # Encode failure mode
                item_data = item_data.dropna(subset=['Failure mode'])
                mode = item_data['Failure mode'].iloc[i + sequence_length]
                if pd.isna(mode):
                    mode = 'UNKNOWN'  # Remplacer les NaN par 'UNKNOWN'
                failure_modes.append(self.encode_failure_mode([mode])[0])

        if len(sequences) == 0:
            raise ValueError("Aucune séquence valide n'a été créée avec les données fournies.")

        sequences_padded = np.array(pad_sequences(sequences, maxlen=self.min_sequence_length, padding='post', dtype='float32'))
        targets = np.array(targets).reshape(-1, 2 * self.forecast_months)
        failure_modes = np.array(failure_modes)
        print(f"Failures_mode encoded :{failure_modes}")

        return sequences_padded, targets, failure_modes

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

        def prepare_test_sequence(features):
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
        all_failure_mode_predictions = []

        for item_index in item_indices:
            item_data = df[df['item_index'] == item_index].sort_values(by='time (months)')
            features = extract_features(item_data)
            last_sequence_padded = prepare_test_sequence(features)
            prediction = self.model.predict(last_sequence_padded)

            # Vérifier la structure des prédictions
            print("Prediction shape:", prediction[0].shape, prediction[1].shape)

            # Séparer les prédictions pour length_filtered, length_measured et Failure mode
            forecast_output = prediction[0]  # Prédictions des valeurs temporelles
            pred_lengths_filtered = forecast_output[:, :self.forecast_months]
            pred_lengths_measured = forecast_output[:, self.forecast_months:]
            pred_failure_mode = np.argmax(prediction[1], axis=1)  # Prédictions de Failure mode

            # Vérifier les tailles des tableaux avant la concaténation
            print("pred_lengths_filtered shape:", pred_lengths_filtered.shape)
            print("pred_lengths_measured shape:", pred_lengths_measured.shape)

            combined_predictions = np.column_stack((pred_lengths_filtered.flatten(), pred_lengths_measured.flatten()))
            all_predictions.append(combined_predictions)
            all_failure_mode_predictions.append(pred_failure_mode)

        return all_predictions, all_failure_mode_predictions

    def add_predictions_to_data(self, scenario, df, predictions, failure_mode_predictions):
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
                'Failure mode': None,
                'source': source
            }
            data_dict.update(features)

            if scenario[:] == 'Scenario2':
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
            future_failure_modes = self.decode_failure_mode(failure_mode_predictions[idx])
            extended_failure_modes = np.repeat(future_failure_modes, len(future_times))

            print(f"Length of future_times: {len(future_times)}")
            print(f"Length of future_lengths_filtered: {len(future_lengths_filtered)}")
            print(f"Length of future_lengths_measured: {len(future_lengths_measured)}")
            print(f"Length of future_failure_modes: {len(future_failure_modes)}")

            initial_data = prepare_initial_data(item_data, item_index, source=0)
            forecast_data = pd.DataFrame({
                'item_index': item_index,
                'time (months)': future_times,
                'length_filtered': future_lengths_filtered,
                'length_measured': future_lengths_measured,
                'Failure mode': extended_failure_modes,
                'source': 1
            })

            extended_data.append(pd.concat([initial_data, forecast_data]))

        if not extended_data:
            raise ValueError("Aucune donnée étendue n'a été créée avec les prévisions fournies.")

        df_extended = pd.concat(extended_data).reset_index(drop=True)
        df_extended['crack_failure'] = (df_extended['length_measured'] >= 0.85).astype(int)
        df_extended = self.add_features(df_extended)

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

        plot_scatter2(df, 'time (months)', 'length_measured', 'Failure mode')