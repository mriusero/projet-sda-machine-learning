import pandas as pd
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

from ...components import plot_scatter2

def clean_data(df):
    """
    Nettoie les données d'un DataFrame.

    :param df: DataFrame à nettoyer
    :return: DataFrame nettoyé
    """
    # Gestion des valeurs manquantes
    # Imputation des valeurs manquantes pour les colonnes numériques avec la moyenne
    for column in df.select_dtypes(include=[np.number]).columns:
        #df[column].fillna(df[column].mean(), inplace=True)
        df[column].dropna()

    # Gestion des valeurs aberrantes
    # Par exemple, en utilisant les percentiles pour détecter les valeurs aberrantes
    for column in df.select_dtypes(include=[np.number]).columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        df = df[(df[column] >= (q1 - 1.5 * iqr)) & (df[column] <= (q3 + 1.5 * iqr))]

    return df


# Définition de la fonction logistique
def logistic_function(t, beta0, beta1, beta2):
    return beta2 / (1 + np.exp(-(beta0 + beta1 * t)))

def fit_logistic(t, y):
    # Ajuster la fonction logistique sur les données d'une machine spécifique
    popt, _ = curve_fit(logistic_function, t, y, maxfev=10000)
    return popt


def train_and_predict(train_df, test_df, is_classification=True):
    results = []

    # Traiter chaque machine individuellement
    for item_index in train_df['item_index'].unique():
        # Filtrer les données pour la machine spécifique
        train_machine_df = train_df[train_df['item_index'] == item_index]
        test_machine_df = test_df[test_df['item_index'] == item_index]

        if train_machine_df.empty or test_machine_df.empty:
            continue

        # Préparer les données
        X_train = train_machine_df[['time (months)', 'crack length (arbitary unit)']]
        y_train = train_machine_df['crack length (arbitary unit)']

        # Ajuster la fonction logistique pour cette machine
        beta0, beta1, beta2 = fit_logistic(X_train['time (months)'], y_train)

        # Créer un DataFrame pour les prédictions futures
        future_times = []
        future_predictions = []

        # Pour chaque ligne du DataFrame de test, générer des prédictions pour les 6 mois suivants
        for current_time in test_machine_df['time (months)']:
            future_time_range = np.arange(current_time + 1, current_time + 7)  # Mois suivant jusqu'à 6 mois
            predictions = logistic_function(future_time_range, beta0, beta1, beta2)

            # Ajouter les résultats pour cette période
            future_times.extend(future_time_range)
            future_predictions.extend(predictions)

        # Ajouter les résultats pour cette machine
        machine_results = pd.DataFrame({
            'item_index': item_index,
            'time (months)': future_times,
            'predicted_crack_length': future_predictions
        })
        results.append(machine_results)

    # Combiner les résultats de toutes les machines
    results_df = pd.concat(results).reset_index(drop=True)

    # Ajouter les étiquettes si nécessaire
    if is_classification:
        threshold = 0.85
        results_df['predicted_label'] = (results_df['predicted_crack_length'] > threshold).astype(int)

    return results_df


def handle_scenarios(dataframes):
    train = clean_data(dataframes['train'])
    pseudo_test = clean_data(dataframes['pseudo_test'])
    pseudo_test_with_truth = clean_data(dataframes['pseudo_test_with_truth'])
    testing_data_phase1_df = clean_data(dataframes['test'])

    # Scenario 1: training_data.csv + pseudo_testing_data.csv
    st.markdown("## Scenario 1: pseudo_testing_data")
    scenario1_train_df = train
    scenario1_test_df = pseudo_test
    scenario1_predictions = train_and_predict(scenario1_train_df, scenario1_test_df)
    generate_submission_file(scenario1_predictions, '/Users/mariusayrault/GitHub/Sorb-Data-Analytics/projet-sda-machine-learning/app/data/output/submission/submission_scenario1.csv')

    col1, col2 = st.columns([2,4])
    with col1:
        st.dataframe(scenario1_predictions)
    with col2:
        plot_scatter2(scenario1_predictions, 'time (months)', 'predicted_crack_length', 'item_index')


    # Scenario 2: training_data.csv + pseudo_testing_data_with_truth.csv
    st.markdown("## Scenario 2: pseudo_testing_data_with_truth")
    scenario2_train_df = train
    scenario2_test_df = pseudo_test_with_truth
    scenario2_predictions = train_and_predict(scenario2_train_df, scenario2_test_df)
    st.dataframe(scenario2_predictions)
    generate_submission_file(scenario2_predictions, '/Users/mariusayrault/GitHub/Sorb-Data-Analytics/projet-sda-machine-learning/app/data/output/submission/submission_scenario2.csv')

    #true_rul = pseudo_test_with_truth['true_rul']       # Calculer et afficher les erreurs pour le Scenario 2
    #mse = mean_squared_error(true_rul, scenario2_predictions)
    #st.markdown("Mean Squared Error for Scenario 2:", mse)

    # Scenario 3: training_data.csv + testing_data_phase1.csv
    st.markdown("## Scenario 3: testing_data_phase1")
    scenario3_train_df = train
    scenario3_test_df = testing_data_phase1_df
    scenario3_predictions = train_and_predict(scenario3_train_df, scenario3_test_df)
    st.dataframe(scenario3_predictions)
    generate_submission_file(scenario3_predictions, '/Users/mariusayrault/GitHub/Sorb-Data-Analytics/projet-sda-machine-learning/app/data/output/submission/submission_scenario3.csv')



def generate_submission_file(predictions, output_path):
    template = pd.read_csv('/Users/mariusayrault/GitHub/Sorb-Data-Analytics/projet-sda-machine-learning/app/data/output/submission/submission_template.csv')
    predictions['item_index'] = predictions['item_index'].apply(lambda x: f'item_{x}')
    submission = pd.merge(template, predictions, on='item_index', how='left')
    submission['predicted_rul'] = np.where(submission['predicted_crack_length'] > 0.85, 0, 1)
    submission = submission.groupby('item_index', as_index=False).agg({
        'predicted_rul': 'mean'  # ou 'max', 'min', ou une autre méthode d'agrégation selon le besoin
    })
    submission['predicted_rul'] = submission['predicted_rul'].apply(lambda x: 1 if x == 1 else 0)
    submission.to_csv(output_path, index=False)

def calculate_score(solution_path, submission_path):
    solution = pd.read_csv(solution_path)
    submission = pd.read_csv(submission_path)

    reward = 5
    penalty_false_positive = -1 / 6
    penalty_false_negative = -10

    merged = pd.merge(solution, submission, on='item_index', how='left')

    rewards_penalties = []
    for _, row in merged.iterrows():
        sol_label = row['label']
        sub_label = row['predicted_rul']
        true_rul = row['true_rul']

        if sol_label == sub_label:
            rewards_penalties.append(reward)
        elif sol_label == 1 and sub_label == 0:
            rewards_penalties.append(penalty_false_negative)
        elif sol_label == 0 and sub_label == 1:
            rewards_penalties.append(penalty_false_positive * true_rul)
        else:
            rewards_penalties.append(0)  # Pas de récompense ou pénalité si les labels ne correspondent pas

    return sum(rewards_penalties)


