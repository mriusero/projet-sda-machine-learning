# main.py
import pandas as pd
import streamlit as st
from .preprocessing import clean_data
from .features import add_features
from ..FailuresDetectModel import LSTMModel
from .validation import generate_submission_file, calculate_score

def instance_model(model_name, min_sequence_length, forecast_months):
    model_classes = {
        'LSTMModel': lambda: LSTMModel(min_sequence_length, forecast_months),
    }
    if model_name not in model_classes:
        raise ValueError(f"Modèle '{model_name}' non supporté.")

    model_instance = model_classes[model_name]()
    return model_instance

def process_predictions(scenario, train_df, test_df, output_path):

    #----LSTDM_Model---------------------------------------------------------------------
    min_sequence_length = 2
    forecast_months = 6
    model = instance_model('LSTMModel', min_sequence_length, forecast_months)
    X_train, y_train = model.prepare_sequences(train_df)
    model.train(X_train, y_train)
    predictions = model.predict_futures_values(test_df)
    extended_df = model.add_predictions_to_data(scenario, test_df, predictions)
    model.display_results(extended_df)
    model.save_predictions(extended_df, output_path)

    # ----End Scenario ---------------------------------------------------------------------
    generate_submission_file(output_path)
    score = calculate_score(output_path)
    st.write(f"Le score est de {score}")
    st.dataframe(extended_df)

    return extended_df


def handle_scenarios(dataframes):
    keys = ['train', 'pseudo_test', 'pseudo_test_with_truth', 'test']
    cleaned_dfs = {key: clean_data(dataframes[key]) for key in keys}
    featured_dfs = {key: add_features(cleaned_dfs[key].copy()) for key in keys}

    scenarios = [
        {
            'name': 'Scenario1',
            'test_df': featured_dfs['pseudo_test'],
            'output_path': '/Users/mariusayrault/GitHub/Sorb-Data-Analytics/projet-sda-machine-learning/app/data/output/submission/scenario1/'
        },
        {
            'name': 'Scenario2',
            'test_df': featured_dfs['pseudo_test_with_truth'],
            'output_path': '/Users/mariusayrault/GitHub/Sorb-Data-Analytics/projet-sda-machine-learning/app/data/output/submission/scenario2/'
        },
        {
            'name': 'Scenario3',
            'test_df': featured_dfs['test'],
            'output_path': '/Users/mariusayrault/GitHub/Sorb-Data-Analytics/projet-sda-machine-learning/app/data/output/submission/scenario3/'
        }
    ]

    selected_scenario_name = st.selectbox(
        'Choisissez le scénario à exécuter',
        options=[scenario['name'] for scenario in scenarios]
    )

    if st.button('Exécuter le scénario'):
        selected_scenario = next(scenario for scenario in scenarios if scenario['name'] == selected_scenario_name)
        st.markdown(f"## {selected_scenario['name']}")

        predictions_df = process_predictions(
            selected_scenario['name'],
            featured_dfs['train'],
            selected_scenario['test_df'],
            selected_scenario['output_path']
        )
