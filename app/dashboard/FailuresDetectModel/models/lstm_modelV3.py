import numpy as np
import pandas as pd
import streamlit as st

from .models_base import ModelBase
from ...components import plot_scatter2
from ...functions import load_failures
from ..features import FeatureAdder

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Masking, Input

class LSTMModelV3(ModelBase):
    def __init__(self, min_sequence_length, forecast_months):
        super().__init__()
        self.min_sequence_length = min_sequence_length
        self.forecast_months = forecast_months
        self.failure_mode_encoder = LabelEncoder()
        self.model = None                                   # Model initialisation to None

    def fit_failure_mode_encoder(self, failures_df):
        failures_df.fillna({'Failure mode': 'UNKNOWN'}, inplace=True)
        failure_modes = failures_df['Failure mode']
        return self.failure_mode_encoder.fit(failure_modes)

    def encode_failure_mode(self, failure_modes):
        return self.failure_mode_encoder.transform(failure_modes)

    def decode_failure_mode(self, encoded_modes):
        return self.failure_mode_encoder.inverse_transform(encoded_modes)

    def train(self, X_train, y_train, y_failure_modes):
        input_shape = (self.min_sequence_length, 10)                    # 10 features per sequence
        num_classes = len(self.failure_mode_encoder.classes_)           # Number of classes for Failure mode

        inputs = Input(shape=input_shape)                               # Define input layer

        x = Masking(mask_value=0.0)(inputs)                             # Define LSTM layer
        x = LSTM(50, return_sequences=False)(x)

        forecast_output = Dense(2 * self.forecast_months, name='forecast_output')(x)        # Define outputs
        classification_output = Dense(num_classes, activation='softmax', name='classification_output')(x)

        self.model = Model(inputs=inputs, outputs=[forecast_output, classification_output])     # Create model

        self.model.compile(optimizer='adam',        # Compile model
                           loss={'forecast_output': 'mse', 'classification_output': 'sparse_categorical_crossentropy'},
                           metrics={'forecast_output': 'mae', 'classification_output': 'accuracy'})

        print(np.isnan(X_train).any(), np.isinf(X_train).any())         # Print for debugging
        print(np.isnan(y_train).any(), np.isinf(y_train).any())
        print(np.isnan(y_failure_modes).any(), np.isinf(y_failure_modes).any())

        num_epochs = 50                                                 # Training progress
        with st.spinner('Training the model...'):
            progress_bar = st.progress(0)
            for epoch in range(num_epochs):
                self.model.fit(X_train, {'forecast_output': y_train, 'classification_output': y_failure_modes},
                                        epochs=1, batch_size=32, validation_split=0.2)
                progress_bar.progress((epoch + 1) / num_epochs)
            progress_bar.empty()

    def predict(self, X_test):
        if self.model is None:
            raise ValueError("The model has not been trained.")

        with st.spinner('Generating predictions...'):           # Predictions of time series values and Failure mode
            progress_bar = st.progress(0)
            num_batches = len(X_test) // 32
            predictions = []
            for i in range(num_batches):
                batch = X_test[i*32:(i+1)*32]
                batch_predictions = self.model.predict(batch)
                predictions.append(batch_predictions)
                progress_bar.progress((i + 1) / num_batches)
            predictions = np.concatenate(predictions, axis=0)
            progress_bar.empty()
        st.success('Predictions generated successfully!')

        forecast_predictions = predictions[0]                                   # Time series predictions
        classification_predictions = np.argmax(predictions[1], axis=1)          # Failure mode predictions

        return forecast_predictions, classification_predictions

    def prepare_train_sequences(self, df):

        item_indices = df['item_index'].unique()
        sequences = []
        targets = []
        failure_modes = []

        failures_df = load_failures()
        self.fit_failure_mode_encoder(failures_df)                                          # Unique call here
        failure_mode_dict = failures_df.set_index('item_index')['Failure mode'].to_dict()

        for item_index in item_indices:

            item_data = df[df['item_index'] == item_index].sort_values(by='time (months)')
            item_data['Failure mode'] = failure_mode_dict.get(item_index, None)

            times = item_data['time (months)'].values
            lengths_filtered = item_data['length_filtered'].values
            lengths_measured = item_data['length_measured'].values
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

                item_data = item_data.dropna(subset=['Failure mode'])               # Encode failure mode
                mode = item_data['Failure mode'].iloc[i + sequence_length]
                if pd.isna(mode):
                    mode = 'UNKNOWN'                                                # Replace NaNs with 'UNKNOWN'
                failure_modes.append(self.encode_failure_mode([mode])[0])

        if len(sequences) == 0:
            raise ValueError("No valid sequence was created with the provided data.")

        sequences_padded = np.array(pad_sequences(sequences, maxlen=self.min_sequence_length, padding='post', dtype='float32'))
        targets = np.array(targets).reshape(-1, 2 * self.forecast_months)
        failure_modes = np.array(failure_modes)
        print(f"Encoded failure modes: {failure_modes}")

        return sequences_padded, targets, failure_modes

    def predict_futures_values(self, df):
        if self.model is None:
            raise ValueError("The model has not been trained.")
        def extract_features(item_data):
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

        with st.spinner('Calculating future values...'):

            progress_bar = st.progress(0)
            num_items = len(item_indices)

            for idx, item_index in enumerate(item_indices):

                item_data = df[df['item_index'] == item_index].sort_values(by='time (months)')
                features = extract_features(item_data)
                last_sequence_padded = prepare_test_sequence(features)
                prediction = self.model.predict(last_sequence_padded)

                print("Prediction shape:", prediction[0].shape, prediction[1].shape)    # Check prediction structure

                forecast_output = prediction[0]                                         # Time series predictions
                pred_lengths_filtered = forecast_output[:, :self.forecast_months]
                pred_lengths_measured = forecast_output[:, self.forecast_months:]
                pred_failure_mode = np.argmax(prediction[1], axis=1)                    # Failure mode predictions

                print("pred_lengths_filtered shape:", pred_lengths_filtered.shape)  # Check array sizes for concat
                print("pred_lengths_measured shape:", pred_lengths_measured.shape)

                combined_predictions = np.column_stack((pred_lengths_filtered.flatten(), pred_lengths_measured.flatten()))
                all_predictions.append(combined_predictions)
                all_failure_mode_predictions.append(pred_failure_mode)

                progress_bar.progress((idx + 1) / num_items)

            progress_bar.empty()

        st.success('Future values calculation completed!')
        return all_predictions, all_failure_mode_predictions

    def add_predictions_to_data(self, scenario, df, predictions, failure_mode_predictions):
        def prepare_initial_data(item_data, item_index, source):

            times = item_data['time (months)'].values
            lengths_filtered = item_data['length_filtered'].values
            lengths_measured = item_data['length_measured'].values

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
                'Failure mode (lstm)': 'ToDefine',
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
                'Failure mode (lstm)': extended_failure_modes,
                'source': 1
            })
            extended_data.append(pd.concat([initial_data, forecast_data]))

        if not extended_data:
            raise ValueError("No extended data was created with the provided predictions.")

        df_extended = pd.concat(extended_data).reset_index(drop=True)
        df_extended['crack_failure'] = ((df_extended['length_measured'] >= 0.85) | (df_extended['length_filtered'] >= 0.85)).astype(int)

        feature_adder = FeatureAdder(min_sequence_length=self.min_sequence_length)
        df_extended = feature_adder.add_features(df_extended, particles_filtery=False)

        return df_extended

    def save_predictions(self, df, output_path):
        file_path = f"{output_path}/lstm_predictions.csv"
        df.to_csv(file_path, index=False)

        return f"Predictions saved successfully: {output_path}"

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

        plot_scatter2(df, 'time (months)', 'length_measured', 'Failure mode (lstm)')

        # st.dataframe(extended_df)