import numpy as np
import pandas as pd
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
import optuna
from sklearn.preprocessing import LabelEncoder
from sksurv.util import Surv
import pickle
import os
import streamlit as st
from ..display import DisplayData
from sklearn.metrics import mean_absolute_error, brier_score_loss
import matplotlib.pyplot as plt
from ..validation import generate_submission_file, calculate_score

class GradientBoostingSurvivalModel:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.best_params = None

    def fit_label_encoder(self, df, columns):
        """Fit label encoders for categorical columns."""
        for column in columns:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            self.label_encoders[column] = le

    def encode_labels(self, df, columns):
        """Encode categorical columns."""
        for column in columns:
            le = self.label_encoders.get(column)
            if le is not None:
                df[column] = df[column].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
        return df

    def align_columns(self, df, reference_df):
        """Ensure both DataFrames have the same columns and in the same order."""
        reference_columns = list(reference_df.columns)
        df = df.reindex(columns=reference_columns, fill_value=0)  # Fill missing columns with a default value
        return df

    def prepare_data(self, df, reference_df=None):
        """Prepare data for training and prediction."""
        if reference_df is not None:
            df = self.align_columns(df, reference_df)

        categorical_columns = ['Failure mode']
        self.fit_label_encoder(df, categorical_columns)
        df_encoded = self.encode_labels(df, categorical_columns)

        df['label'] = df['label'].astype(bool)

        X = pd.DataFrame(df_encoded, columns=df_encoded.drop(columns=['item_id', 'label', 'time (months)']).columns)
        y = Surv.from_dataframe('label', 'time (months)', df_encoded)

        return X, y

    def train(self, X_train, y_train):
        """Train the Gradient Boosting Survival Analysis model."""
        print(f"Training data shape: {X_train.shape}")
        print(f"Training labels shape: {y_train.shape}")

        # Initialize model with hyperparameters if provided, else use defaults
        if self.best_params is None:
            self.best_params = {
                'learning_rate': 0.1,
                'n_estimators': 100,
                'max_depth': 3,
                'subsample': 1.0,
                'min_samples_split': 2,
                'min_samples_leaf': 1
            }

        self.model = GradientBoostingSurvivalAnalysis(
            learning_rate=self.best_params.get('learning_rate', 0.1),
            n_estimators=self.best_params.get('n_estimators', 100),
            max_depth=self.best_params.get('max_depth', 3),
            subsample=self.best_params.get('subsample', 1.0),
            min_samples_split=self.best_params.get('min_samples_split', 2),
            min_samples_leaf=self.best_params.get('min_samples_leaf', 1)
        )

        self.model.fit(X_train, y_train)

    def predict(self, X_test, original_df):
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("The model has not been trained.")

        surv_funcs = self.model.predict_survival_function(X_test)

        survival_prob_6_months = np.array([surv_func(6) for surv_func in surv_funcs])
        survival_prob_1_month = np.array([surv_func(1) for surv_func in surv_funcs])

        predictions_6_months = np.where(survival_prob_6_months <= 0.5, 1, 0)  # 1 if survival probability <= 0.5, else 0
        predictions_1_month = np.where(survival_prob_1_month <= 0.5, 1, 0)  # 1 if survival probability <= 0.5, else 0

        predictions_df = original_df.copy()
        predictions_df['predicted_survival_6_months'] = survival_prob_6_months
        predictions_df['predicted_survival_1_month'] = survival_prob_1_month

        return predictions_df

    def optimize_hyperparameters(self, X_train, y_train):
        """Optimize hyperparameters using Optuna."""

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5, log=True),
                'max_depth': trial.suggest_int('max_depth', 2, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20)
            }

            model = GradientBoostingSurvivalAnalysis(**params)
            model.fit(X_train, y_train)

            surv_funcs = model.predict_survival_function(X_train)

            median_times = []
            for surv_func in surv_funcs:
                times = surv_func.x
                survival_probabilities = surv_func.y

                if any(survival_probabilities <= 0.5):
                    median_time = times[np.where(survival_probabilities <= 0.5)[0][0]]
                else:
                    median_time = times[-1]
                median_times.append(median_time)
            y_pred_median = np.array(median_times)

            c_index_result = concordance_index_censored(y_train['label'], y_train['time (months)'], y_pred_median)
            c_index = c_index_result[0]

            return c_index

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=10 * len(X_train.columns))

        self.best_params = study.best_params
        self._save_hyperparameters()
        print(f"Best hyperparameters: {self.best_params}")

    def _save_hyperparameters(self):
        """Save the best hyperparameters to a file."""
        with open('data/output/submission/gradient_boosting/best_hyperparameters.pkl', 'wb') as f:
            pickle.dump(self.best_params, f)

    def _load_hyperparameters(self):
        """Load the best hyperparameters from a file."""
        if os.path.exists('data/output/submission/gradient_boosting/best_hyperparameters.pkl'):
            with open('data/output/submission/gradient_boosting/best_hyperparameters.pkl', 'rb') as f:
                self.best_params = pickle.load(f)
        else:
            self.best_params = None

    def evaluate_model_performance(self, pseudo_test_with_truth_df, predictions_df):
        """Evaluate the performance of the model based on true RUL and predicted RUL."""

        predictions_df = predictions_df.sort_values(['item_id', 'time (months)'], ascending=True)
        pseudo_test_with_truth_df = pseudo_test_with_truth_df.sort_values(['item_id', 'time (months)'], ascending=True)

        # Calculate MAE for RUL & predicted_survival_6_months
        true_rul = pseudo_test_with_truth_df['true_rul']
        predicted_rul_6_months = predictions_df['predicted_survival_6_months']
        mae_6_months = mean_absolute_error(true_rul, predicted_rul_6_months)

        st.write(f"MAE for 6-month survival predictions: {mae_6_months}")

        return mae_6_months

    def plot_feature_importance(self, X_train):
        """Plot feature importance of the trained model."""
        if self.model is None:
            raise ValueError("The model has not been trained.")

        feature_importances = self.model.feature_importances_
        features = X_train.columns
        indices = np.argsort(feature_importances)[::-1]

        fig = plt.figure(figsize=(12, 8))
        plt.title("Feature Importances")
        plt.bar(range(X_train.shape[1]), feature_importances[indices], align="center")
        plt.xticks(range(X_train.shape[1]), features[indices], rotation=90)
        plt.xlim([-1, X_train.shape[1]])
        st.pyplot(fig)

    def save_predictions(self, output_path, predictions_df, step):
        """Save predictions to a CSV file with 'item_id' included."""
        file_path = f"{output_path}/gbsa_predictions_{step}.csv"
        predictions_df.to_csv(file_path, index=False)

    def display_results(self, df):
        """Display results in Streamlit."""
        display = DisplayData(df)

        col1, col2 = st.columns(2)
        with col1:
            display.plot_discrete_scatter(df, 'time (months)', 'length_measured', 'predicted_survival_6_months')
            display.plot_discrete_scatter(df, 'time (months)', 'length_measured', 'predicted_survival_1_month')
        with col2:
            display.plot_discrete_scatter(df, 'time (months)', 'length_filtered', 'predicted_survival_6_months')
            display.plot_discrete_scatter(df, 'time (months)', 'length_filtered', 'predicted_survival_1_month')

    def run_full_pipeline(self, train_df, pseudo_test_with_truth_df, test_df, optimize):
        model_name = 'GradientBoostingSurvivalModel'
        output_path = 'data/output/submission/gradient_boosting'
        st.markdown(f"## {model_name}")

        # Data preparation
        X_train, y_train = self.prepare_data(train_df, reference_df=train_df)
        X_val, y_val = self.prepare_data(pseudo_test_with_truth_df.drop(columns=['true_rul']),
                                         reference_df=train_df)

        self._load_hyperparameters()

        # Optimize hyperparameters if needed
        if self.best_params is None:
            self.optimize_hyperparameters(X_train, y_train)
            st.success("Hyperparameters have been optimized and saved")
        elif self.best_params is not None and optimize is True:
            self.optimize_hyperparameters(X_train, y_train)
            st.success("Hyperparameters have been optimized and saved")
        elif self.best_params is not None and optimize is False:
            st.info("Hyperparameters already optimized")

        # Train the model
        self.train(X_train, y_train)

        # Display model characteristics
        st.write(f"X_train contains: {len(X_train.columns)} variables")
        self.plot_feature_importance(X_train)

        # Cross-validation
        predictions_val = self.predict(X_val, pseudo_test_with_truth_df)
        self.save_predictions(output_path, predictions_val, step='cross-val')
        generate_submission_file(model_name, output_path, step='cross-val')
        score = calculate_score(output_path, step='cross-val')
        st.write(f"Cross-validation score for {model_name}: {score}")

        # Evaluate performance
        mae_6_months = self.evaluate_model_performance(pseudo_test_with_truth_df, predictions_val)

        # Final test
        X_test, _ = self.prepare_data(test_df, reference_df=train_df)
        predictions_test = self.predict(X_test, test_df)
        self.save_predictions(output_path, predictions_test, step='final-test')
        generate_submission_file(model_name, output_path, step='final-test')
        final_score = calculate_score(output_path, step='final-test')
        st.write(f"The final score for {model_name} is {final_score}")

        # Display results
        self.display_results(predictions_test.sort_values(['item_id', 'time (months)'], ascending=True))
