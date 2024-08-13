from ..functions import ParticleFilter

class FeatureAdder:
    def __init__(self, min_sequence_length):
        self.min_sequence_length = min_sequence_length

    def add_features(self, df, particles_filtery):

        """Feature Engineering"""

        df['crack_failure'] = (df['length_measured'] >= 0.85).astype(int)

        def calculate_rolling_features(series, window_size):
            return {
                'mean': series.rolling(window=window_size, min_periods=1).mean(),
                'std': series.rolling(window=window_size, min_periods=1).std(),
                'max': series.rolling(window=window_size, min_periods=1).max(),
                'min': series.rolling(window=window_size, min_periods=1).min()
            }

        def replace_nan(series):
            return series.fillna(series.mean())

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
                'rolling_means_filtered', 'rolling_stds_filtered',
                'rolling_maxs_filtered', 'rolling_mins_filtered',
                'rolling_means_measured', 'rolling_stds_measured',
                'rolling_maxs_measured', 'rolling_mins_measured'
            ]
            existing_columns = [col for col in to_recall if col in df.columns]
            df = df.drop(columns=existing_columns)

        for length_type in ['length_measured', 'length_filtered']:
            for stat in ['mean', 'std', 'max', 'min']:
                col_name = f'rolling_{stat}_{length_type}'
                df[col_name] = df.groupby('item_index')[length_type].transform(
                    lambda x: calculate_rolling_features(x, len(x))[stat]
                )
                df[col_name] = df.groupby('item_index')[col_name].transform(
                    lambda x: replace_nan(x)
                )

        to_fill_0 = ['length_measured', 'length_filtered']
        df[to_fill_0] = df[to_fill_0].fillna(0)

        return df
