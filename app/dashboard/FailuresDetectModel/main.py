# main.py
import streamlit as st
from .preprocessing import clean_data
from .features import add_features
from ..FailuresDetectModel import LSTMModel, LSTMModelV2, LSTMModelV3
from .validation import generate_submission_file, calculate_score
from sklearn.preprocessing import LabelEncoder

def instance_model(model_name):
    model_classes = {
        'LSTMModel': lambda: LSTMModel(
            min_sequence_length=2,
            forecast_months=6),
        'LSTMModelV2': lambda: LSTMModelV2(
            min_sequence_length=2,
            forecast_months=6),
        'LSTMModelV3': lambda: LSTMModelV3(
            min_sequence_length=2,
            forecast_months=6)
    }
    if model_name not in model_classes:
        raise ValueError(f"Modèle '{model_name}' non supporté.")

    model_instance = model_classes[model_name]()
    return model_instance

def process_predictions(scenario, train_df, test_df, output_path):

    #----LSTM_Model---------------------------------------------------------------------
    #model = instance_model('LSTMModel')
    #X_train, y_train = model.prepare_sequences(train_df)
    #model.train(X_train, y_train)
    #predictions = model.predict_futures_values(test_df)
    #extended_df = model.add_predictions_to_data(scenario, test_df, predictions)
    #model.display_results(extended_df)
    #model.save_predictions(extended_df, output_path)

    # ----LSTM Model V3---------------------------------------------------------------------
    model = instance_model('LSTMModelV3')
    X_train, y_train, y_failure_modes = model.prepare_train_sequences(train_df)
    model.train(X_train, y_train, y_failure_modes)
    predictions, failure_mode_predictions = model.predict_futures_values(test_df)
    extended_df = model.add_predictions_to_data(scenario, test_df, predictions, failure_mode_predictions)
    model.display_results(extended_df)
    model.save_predictions(extended_df, output_path)

    # Ajouter les prédictions aux données et sauvegarder
    #df_with_predictions = model.add_predictions_to_data(test_df, predictions)
    #model.save_predictions(df_with_predictions, 'chemin/vers/output')

    # ----End Scenario ---------------------------------------------------------------------
    generate_submission_file(output_path)
    score = calculate_score(output_path)
    st.write(f"Le score est de {score}")
    st.dataframe(extended_df)

    return extended_df


def handle_scenarios(dataframes):

    keys = ['train', 'pseudo_test', 'pseudo_test_with_truth', 'test']
    df = {key: (dataframes[key]) for key in keys}

    scenarios = [
        {
            'name': 'Scenario1',
            'test_df': df['pseudo_test'],
            'output_path': '/Users/mariusayrault/GitHub/Sorb-Data-Analytics/projet-sda-machine-learning/app/data/output/submission/scenario1/'
        },
        {
            'name': 'Scenario2',
            'test_df': df['pseudo_test_with_truth'],
            'output_path': '/Users/mariusayrault/GitHub/Sorb-Data-Analytics/projet-sda-machine-learning/app/data/output/submission/scenario2/'
        },
        {
            'name': 'Scenario3',
            'test_df': df['test'],
            'output_path': '/Users/mariusayrault/GitHub/Sorb-Data-Analytics/projet-sda-machine-learning/app/data/output/submission/scenario3/'
        }
    ]

    selected_scenario_name = st.selectbox(
        'Choisissez le scénario à exécuter',
        options=[scenario['name'] for scenario in scenarios]
    )
    if st.button('Exécuter le scénario'):
        selected_scenario = next(scenario for scenario in scenarios if scenario['name'] == selected_scenario_name)
        preprocessed = lambda x: add_features(clean_data(x), particles_filtery=True)

        st.markdown(f"## {selected_scenario['name']}")
        predictions_df = process_predictions(
            selected_scenario['name'],
            preprocessed(df['train']),
            preprocessed(selected_scenario['test_df']),
            selected_scenario['output_path'],
        )
