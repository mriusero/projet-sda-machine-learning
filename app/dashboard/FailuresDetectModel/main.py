# main.py

import streamlit as st
from ..FailuresDetectModel import (RandomForestClassifierModel, LSTMModel)
from .validation import generate_submission_file, calculate_score

def instance_model(model_name):
    model_classes = {
        'LSTMModel': lambda: LSTMModel(min_sequence_length=2, forecast_months=6),
        'RandomForestClassifierModel': lambda: RandomForestClassifierModel(),
    }
    if model_name not in model_classes:
        raise ValueError(f"Modèle '{model_name}' non supporté.")
    return model_classes[model_name]()

@st.cache_resource
def ml_pipeline(train_df, pseudo_test_with_truth_df, test_df):
    """
    Fonction qui exécute les trois phases du pipeline de machine learning pour un modèle donné.

    Args:
        model_name (str):                          Nom du modèle.
        train_df (DataFrame):                      Jeu de données d'entraînement.
        pseudo_test_with_truth_df (DataFrame):     Jeu de données de test pour la validation croisée.
        test_df (DataFrame):                       Jeu de données de test final pour les prédictions.
        output_path (str):                         Chemin de sortie pour sauvegarder les résultats.
    """
    st.write("")
    # --------- Random Forest Classifier Model ------------
    model_name = 'RandomForestClassifierModel'
    output_path = 'data/output/submission/random_forest'
    st.markdown(f"## {model_name}")
    rf_model = instance_model(model_name)

    X_, y_ = rf_model.prepare_data(train_df)
    rf_model.train(X_, y_)

    X_, y_ = rf_model.prepare_data(pseudo_test_with_truth_df)
    predictions = rf_model.predict(X_)
    rf_model.save_predictions(output_path, predictions, step='cross-val')
    generate_submission_file(model_name, output_path, step='cross-val')
    score = calculate_score(output_path, step='cross-val')
    st.write(f"Score de cross validation pour {model_name}: {score}")

    X_test, _ = rf_model.prepare_data(test_df, target_col=None)
    predictions = rf_model.predict(X_test)
    rf_model.save_predictions(output_path, predictions, step='final-test')
    generate_submission_file(model_name, output_path, step='final-test')
    final_score = calculate_score(output_path, step='final-test')
    st.write(f"Le score final pour {model_name} est de {final_score}")
    st.dataframe(predictions)


    # --------- LSTM Model ------------
    model_name = 'LSTMModel'
    output_path = 'data/output/submission/lstm'
    st.markdown(f"## {model_name}")
    lstm_model = instance_model(model_name)

    X_, y_ = lstm_model.prepare_train_sequences(train_df)
    lstm_model.train(X_, y_)

    all_predictions = lstm_model.predict_futures_values(pseudo_test_with_truth_df)
    lstm_predictions = lstm_model.add_predictions_to_data(pseudo_test_with_truth_df, all_predictions)
    lstm_model.save_predictions(output_path, lstm_predictions, step='cross-val')
    generate_submission_file(model_name, output_path, step='cross-val')
    score = calculate_score(output_path, step='cross-val')
    st.write(f"Score de cross validation pour {model_name}: {score}")

    all_predictions = lstm_model.predict_futures_values(test_df)
    lstm_predictions = lstm_model.add_predictions_to_data(test_df, all_predictions)
    lstm_model.save_predictions(output_path, lstm_predictions, step='final-test')
    generate_submission_file(model_name, output_path, step='final-test')
    final_score = calculate_score(output_path, step='final-test')
    st.write(f"Le score final pour {model_name} est de {final_score}")

    lstm_model.display_results(lstm_predictions)


    #--------- Random Forest Classifier Model ------------
    model_name = 'RandomForestClassifierModel'
    output_path = 'data/output/submission/random_forest'
    st.markdown(f"## {model_name}")
    rf_model2 = instance_model(model_name)

    X_, y_ = rf_model2.prepare_data(train_df)
    rf_model2.train(X_, y_)

    X_, y_ = rf_model2.prepare_data(pseudo_test_with_truth_df)
    predictions = rf_model2.predict(X_)
    rf_model2.save_predictions(output_path, predictions, step='cross-val')
    generate_submission_file(model_name, output_path, step='cross-val')
    score = calculate_score(output_path, step='cross-val')
    st.write(f"Score de cross validation pour {model_name}: {score}")

    X_test, _ = rf_model2.prepare_data(lstm_predictions, target_col=None)
    predictions = rf_model2.predict(X_test)
    rf_model2.save_predictions(output_path, predictions, step='final-test')
    generate_submission_file(model_name, output_path, step='final-test')
    final_score = calculate_score(output_path, step='final-test')
    st.write(f"Le score final pour {model_name} est de {final_score}")
    st.dataframe(predictions)





def handle_models():
    """
    Fonction de gestion des modèles qui exécute le pipeline pour chaque modèle.
    """
    train_df = st.session_state.data.df['train']                                         # Dataset configs
    pseudo_test_with_truth_df = st.session_state.data.df['pseudo_test_with_truth']
    test_df = st.session_state.data.df['test']

    models_to_run = ['RandomForestClassifierModel', 'LSTMModel']  # Models to execute

    if st.button('Run predictions'):
        ml_pipeline(train_df, pseudo_test_with_truth_df, test_df)
