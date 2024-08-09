import pandas as pd
import streamlit as st
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_recall_fscore_support

from .data_processing import clean_data, preprocess_data
from .model_training import train_models
from .predictions import predict_RUL
from .performance import evaluate_models
from ...components import plot_scatter2

def handle_scenarios(dataframes):

    keys = ['train', 'pseudo_test', 'pseudo_test_with_truth', 'test']
    cleaned_dfs = {key: clean_data(dataframes[key]) for key in keys}

    scenarios = [       # Définir les scénarios avec leurs noms et chemins de fichiers
        {
            'name': 'Scenario1',
            'test_df': cleaned_dfs['pseudo_test'],
            'output_path': '/Users/mariusayrault/GitHub/Sorb-Data-Analytics/projet-sda-machine-learning/app/data/output/submission/scenario1/'
        },
        {
            'name': 'Scenario2',
            'test_df': cleaned_dfs['pseudo_test_with_truth'],
            'output_path': '/Users/mariusayrault/GitHub/Sorb-Data-Analytics/projet-sda-machine-learning/app/data/output/submission/scenario2/'
        },
        {
            'name': 'Scenario3',
            'test_df': cleaned_dfs['test'],
            'output_path': '/Users/mariusayrault/GitHub/Sorb-Data-Analytics/projet-sda-machine-learning/app/data/output/submission/scenario3/'
        }
    ]
    for scenario in scenarios:                      # Traitement de chaque scénario
        st.markdown(f"## {scenario['name']}")
        scenario_predictions = process_predictions(scenario['name'], cleaned_dfs['train'], scenario['test_df'], scenario['output_path'])
        #col1, col2 = st.columns(2)
        #with col1:
        st.dataframe(scenario_predictions)
        #with col2:
        plot_scatter2(scenario_predictions, 'crack length (arbitary unit)', 'time (months)', 'source')


def process_predictions(scenario, train, test, folder_path):
    results = []
    unique_items, prepare_item_data = preprocess_data(train, test)  # Prétraitement des données

    all_y_true_rul = []
    all_y_pred_rul = []
    all_y_true_ttf = []
    all_y_pred_ttf = []
    all_y_true_failure_mode = []
    all_y_pred_failure_mode = []

    for item in unique_items:  # Pour chaque item, entraîner et prédire
        train_item, test_item = prepare_item_data(item)

        if train_item is None or test_item is None:  # Si aucune donnée pour cet item, passer à l'item suivant
            continue

        models = train_models(train_item)  # Entraînement des modèles
        result_df = predict_RUL(test_item, models)  # Prédictions et extension

        # Collecter les valeurs réelles et prédites pour l'évaluation
        if 'time (months)' in test_item.columns and 'predicted_rul' in result_df.columns:
            y_true_rul = test_item['time (months)']
            y_pred_rul = result_df['predicted_rul']
            all_y_true_rul.extend(y_true_rul)
            all_y_pred_rul.extend(y_pred_rul)

        if 'Time to failure (months)' in test_item.columns and 'predicted_ttf' in result_df.columns:
            y_true_ttf = test_item['Time to failure (months)']
            y_pred_ttf = result_df['predicted_ttf']
            all_y_true_ttf.extend(y_true_ttf)
            all_y_pred_ttf.extend(y_pred_ttf)

        if 'Failure mode' in test_item.columns and 'predicted_failure_mode' in result_df.columns:
            y_true_failure_mode = test_item['Failure mode']
            y_pred_failure_mode = result_df['predicted_failure_mode']
            all_y_true_failure_mode.extend(y_true_failure_mode)
            all_y_pred_failure_mode.extend(y_pred_failure_mode)

        results.append(result_df)

    scenario_predictions = pd.concat(results, ignore_index=True)  # Concatenation des résultats pour tous les items

#    if scenario[:] == 'Scenario2':
#
#        print(f"## {scenario[:]}")
#        if all_y_true_rul and all_y_pred_rul:
#            mse_rul = mean_squared_error(all_y_true_rul, all_y_pred_rul)
#            mae_rul = mean_absolute_error(all_y_true_rul, all_y_pred_rul)
#            st.markdown(f"**RUL Metrics:**")
#            st.markdown(f"- Mean Squared Error (MSE): {mse_rul:.2f}")
#            st.markdown(f"- Mean Absolute Error (MAE): {mae_rul:.2f}")
#
#        if all_y_true_ttf and all_y_pred_ttf:
#            mse_ttf = mean_squared_error(all_y_true_ttf, all_y_pred_ttf)
#            mae_ttf = mean_absolute_error(all_y_true_ttf, all_y_pred_ttf)
#            st.markdown(f"**TTF Metrics:**")
#            st.markdown(f"- Mean Squared Error (MSE): {mse_ttf:.2f}")
#            st.markdown(f"- Mean Absolute Error (MAE): {mae_ttf:.2f}")
#
#        if all_y_true_failure_mode and all_y_pred_failure_mode:
#            accuracy_failure_mode = accuracy_score(all_y_true_failure_mode, all_y_pred_failure_mode)
#            st.markdown(f"**Failure Mode Metrics:**")
#            st.markdown(f"- Accuracy: {accuracy_failure_mode:.2f}")

    return scenario_predictions

def generate_submission_file(predictions, output_path):
    template = pd.read_csv('./app/data/output/submission/template/submission_template.csv')
    predictions['item_index'] = predictions['item_index'].apply(lambda x: f'item_{x}')
    submission = pd.merge(template, predictions, on='item_index', how='left')
    #submission['predicted_rul'] = np.where(submission['predicted_crack_length'] > 0.85, 0, 1)

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


