from ..functions import ParticleFilter
import streamlit as st
import pandas as pd
from statsmodels.tsa.seasonal import STL

class FeatureAdder:
    def __init__(self, min_sequence_length):
        self.min_sequence_length = min_sequence_length

    def add_features(self, df, particles_filtery):

        """Feature Engineering"""

        def calculate_rolling_features(series, window_size):
            return {
                'mean': series.rolling(window=window_size, min_periods=1).mean(),
                'std': series.rolling(window=window_size, min_periods=1).std(),
                'max': series.rolling(window=window_size, min_periods=1).max(),
                'min': series.rolling(window=window_size, min_periods=1).min()
            }
        def calculate_static_features(series):
            return {
                'mean': series.mean(),
                'std': series.std(),
                'max': series.max(),
                'min': series.min()
            }

        def replace_nan_by_mean(series):
            return series.fillna(series.mean())

        def replace_nan_by_one(series):
            return series.fillna(1)

        def particles_filtering(df):
            pf = ParticleFilter()
            df = pf.filter(df, beta0_range=(-1, 1), beta1_range=(-0.1, 0.1), beta2_range=(0.1, 1))
            return df

        if particles_filtery == True:
            to_recall = [
                'length_filtered', 'beta0', 'beta1', 'beta2',
                'rolling_means_filtered', 'rolling_stds_filtered',
                'rolling_maxs_filtered', 'rolling_mins_filtered',
                'rolling_means_measured', 'rolling_stds_measured',
                'rolling_maxs_measured', 'rolling_mins_measured'
            ]
            existing_columns = [col for col in to_recall if col in df.columns]
            df = df.drop(columns=existing_columns)
            df = particles_filtering(df)

        else:
            to_recall = [
                'rolling_means_filtered', 'rolling_stds_filtered', 'rolling_maxs_filtered', 'rolling_mins_filtered',
                'rolling_means_measured', 'rolling_stds_measured', 'rolling_maxs_measured', 'rolling_mins_measured'
            ]
            existing_columns = [col for col in to_recall if col in df.columns]
            df = df.drop(columns=existing_columns)

        def add_shifts_and_ratios(df, columns, max_shift=3):
            for col in columns:
                # Ajouter des colonnes décalées
                for shift in range(1, max_shift + 1):
                    df[f'{col}_shift_{shift}'] = df.groupby('item_id')[col].shift(shift)

                # Ajouter des ratios entre les décalages
                for shift in range(1, max_shift):
                    df[f'{col}_ratio_{shift}-{shift + 1}'] = (
                            df[f'{col}_shift_{shift}'] / (df[f'{col}_shift_{shift + 1}'] + 1e-9)
                    # Ajout d'une petite valeur pour éviter la division par zéro
                    )

            return df

        def decompose_time_series(df, time_col, value_col, period=12):
            # Vérifier la présence des colonnes
            if time_col not in df.columns or value_col not in df.columns:
                print(f"Les colonnes '{time_col}' ou '{value_col}' n'existent pas dans le DataFrame.")
                return df  # Retourner le DataFrame initial en cas d'erreur

            # Créer une copie du DataFrame pour les calculs
            df_copy = df.copy()

            # Définir la date de départ
            start_date = pd.Timestamp('2024-01-01')
            df_copy[time_col] = df_copy[time_col].apply(lambda x: start_date + pd.DateOffset(months=int(x)))

            # Trier et préparer le DataFrame
            df_copy = df_copy.sort_values(by=time_col)
            df_copy = df_copy.dropna(subset=[time_col, value_col])
            #df_copy.set_index(time_col, inplace=True)

            # Effectuer la décomposition STL
            try:
                stl = STL(df_copy[value_col], period=period)
                result = stl.fit()
            except Exception as e:
                print(f"Erreur lors de la décomposition STL: {e}")
                return df  # Retourner le DataFrame initial en cas d'erreur

            # Ajouter les résultats au DataFrame original
            df['Trend'] = result.trend
            df['Seasonal'] = result.seasonal
            df['Residual'] = result.resid

            return df


        for col in ['time (months)', 'length_measured', 'length_filtered']:
            for stat in ['mean', 'std', 'max', 'min']:
                col_name = f'rolling_{stat}_{col}'
                df[col_name] = df.groupby('item_id')[col].transform(
                    lambda x: calculate_rolling_features(x, len(x))[stat]
                )
                df[col_name] = df.groupby('item_id')[col_name].transform(
                    lambda x: replace_nan_by_mean(x)
                )
                df[col_name] = df.groupby('item_id')[col_name].transform(
                    lambda x: replace_nan_by_one(x)
                )
                col_name = f'static_{stat}_{col}'
                df[col_name] = df.groupby('item_id')[col].transform(
                    lambda x: calculate_static_features(x)[stat]
                )
                df[col_name] = df.groupby('item_id')[col_name].transform(
                    lambda x: replace_nan_by_mean(x)
                )
                df[col_name] = df.groupby('item_id')[col_name].transform(
                    lambda x: replace_nan_by_one(x)
                )

        to_fill_0 = ['time (months)', 'length_measured', 'length_filtered']
        df[to_fill_0] = df[to_fill_0].fillna(0)

        # Liste des colonnes pour lesquelles on souhaite créer des caractéristiques décalées et des ratios
        shift_columns = ['length_measured', 'length_filtered']
        df = add_shifts_and_ratios(df, shift_columns)

        df = decompose_time_series(df, 'time (months)', 'length_filtered')

        if 'rul (months)' in df.columns:
            df['label'] = (df['rul (months)'] <= 6).astype(int)
            df = df.sort_values(by=["item_id", "time (months)"], ascending=[True, True])
        else:
            print("La colonne 'rul (months)' n'est pas présente dans le DataFrame.")

        return df
