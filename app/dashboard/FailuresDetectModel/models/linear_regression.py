import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from ..display import DisplayData

class RULPredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.is_fitted = False

    def preprocess_data(self, df):
        df['item_index'] = df['item_index'].astype('category').cat.codes

        X = df[['item_index', 'time (months)', 'length_measured', 'length_filtered',
                'rolling_mean_length_filtered', 'rolling_std_length_filtered', 'rolling_max_length_filtered',
                'rolling_min_length_filtered',
                'rolling_mean_length_measured', 'rolling_std_length_measured', 'rolling_max_length_measured',
                'rolling_min_length_measured']]
        if 'rul (months)' in df.columns:
            y = df['rul (months)']
        elif 'true_rul' in df.columns:
            y = df['true_rul']
        else:
            raise ValueError("Aucune colonne appropriée ('rul (months)' ou 'true_rul') trouvée dans le DataFrame.")
        return X, y

    def fit(self, df):
        X, y = self.preprocess_data(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        self.is_fitted = True

        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        st.write(f'Mean Squared Error: {mse}')

    def predict(self, new_data):
        if not self.is_fitted:
            raise RuntimeError("Le modèle doit être entraîné avant la prédiction.")

        new_data['item_index'] = new_data['item_index'].astype('category').cat.codes

        predictions = self.model.predict(new_data[['item_index', 'time (months)', 'length_measured', 'length_filtered',
                                                   'rolling_mean_length_filtered', 'rolling_std_length_filtered',
                                                   'rolling_max_length_filtered', 'rolling_min_length_filtered',
                                                   'rolling_mean_length_measured', 'rolling_std_length_measured',
                                                   'rolling_max_length_measured', 'rolling_min_length_measured']])

        new_data['predicted_rul (months)'] = predictions
        return new_data

    def save_predictions(self, df, output_path):
        file_path = f"{output_path}/lr_predictions.csv"
        df.to_csv(file_path, index=False)
        return f"Predictions saved successfully: {file_path}"

    def display_results(self, df):
        if 'predicted_rul (months)' not in df.columns:
            df = self.predict(df)

        df['residual'] = df['true_rul'] - df['predicted_rul (months)']

        display_data = DisplayData(df)
        col1, col2 = st.columns(2)

        with col1:
            # Scatter plot des valeurs réelles vs prédites
            fig = display_data.plot_scatter_real_vs_predicted(df['true_rul'], df['predicted_rul (months)'])
            st.plotly_chart(fig)

        with col2:
            # Residuals vs Predicted
            fig = display_data.plot_residuals_vs_predicted(df['true_rul'], df['predicted_rul (months)'])
            st.plotly_chart(fig)

        col1, col2 = st.columns(2)

        with col1:
            # Histogramme des résidus
            fig = display_data.plot_histogram_of_residuals(df['residual'])
            st.plotly_chart(fig)

        with col2:
            # QQ plot des résidus
            fig = display_data.plot_qq_diagram(df['residual'])
            st.plotly_chart(fig)

        col1, col2 = st.columns(2)
        with col1:            # Plot de la matrice de corrélation
            st.subheader("Matrice de Corrélation")
            fig = display_data.plot_correlation_matrix()
            st.plotly_chart(fig)

        with col2:              # Plot de la courbe d'apprentissage (ici, on doit avoir accès au modèle et aux données)
            if self.is_fitted:
                X, y = self.preprocess_data(df)
                fig = display_data.plot_learning_curve(self.model, X, y)
                st.plotly_chart(fig)
            else:
                st.warning("Le modèle n'est pas encore entraîné. La courbe d'apprentissage ne peut pas être affichée.")