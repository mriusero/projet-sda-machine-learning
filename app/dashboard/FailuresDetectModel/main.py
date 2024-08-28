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
def ml_pipeline(model_name, train_df, pseudo_test_with_truth_df, test_df, output_path):
    """
    Fonction qui exécute les trois phases du pipeline de machine learning pour un modèle donné.

    Args:
        model_name (str):                          Nom du modèle.
        train_df (DataFrame):                      Jeu de données d'entraînement.
        pseudo_test_with_truth_df (DataFrame):     Jeu de données de test pour la validation croisée.
        test_df (DataFrame):                       Jeu de données de test final pour les prédictions.
        output_path (str):                         Chemin de sortie pour sauvegarder les résultats.
    """
    model = instance_model(model_name)    # Instancier le modèle

    # -- TRAINING --
    if model_name == 'RandomForestClassifierModel' and hasattr(model, 'prepare_data'):
        X_, y_ = model.prepare_data(train_df)
    elif model_name == 'LSTMModel' and hasattr(model, 'prepare_train_sequences'):
        X_, y_ = model.prepare_train_sequences(train_df)
    else:
        raise SystemError(f'No appropriated class structuration for {model_name} training')
    model.train(X_, y_)
    st.success(f"Model trained successfully!")

    # -- CROSS VALIDATION --
    if model_name == 'RandomForestClassifierModel' and hasattr(model, 'prepare_data'):
        X_, y_ = model.prepare_data(pseudo_test_with_truth_df)
        predictions = model.predict(X_)
        model.save_predictions(predictions, output_path, step='cross-val')
    elif model_name == 'LSTMModel' and hasattr(model, 'predict_futures_values'):
        all_predictions = model.predict_futures_values(pseudo_test_with_truth_df)
        predictions = model.add_predictions_to_data(pseudo_test_with_truth_df, all_predictions)
        model.save_predictions(predictions, output_path, step='cross-val')
    else:
        raise SystemError(f'No appropriated class structuration for {model_name} Cross validation')
    generate_submission_file(model_name, output_path, step='cross-val')
    score = calculate_score(output_path, step='cross-val')
    st.write(f"Score de cross validation pour {model_name}: {score}")

    # -- PREDICTION --
    if model_name == 'RandomForestClassifierModel' and hasattr(model, 'prepare_data'):
        X_test, _ = model.prepare_data(test_df, target_col=None)
        predictions = model.predict(X_test)
        model.save_predictions(predictions, output_path, step='final-test')

    elif model_name == 'LSTMModel' and hasattr(model, 'predict_futures_values'):
        all_predictions = model.predict_futures_values(test_df)
        predictions = model.add_predictions_to_data(test_df, all_predictions)
        model.save_predictions(predictions, output_path, step='final-test')
    else:
        raise SystemError(f'No appropriated class structuration for {model_name} final testing')
    generate_submission_file(model_name, output_path, step='final-test')
    final_score = calculate_score(output_path, step='final-test')
    st.write(f"Le score final pour {model_name} est de {final_score}")

    # model.display_results(predictions)

def handle_models():
    """
    Fonction de gestion des modèles qui exécute le pipeline pour chaque modèle.
    """
    train_df = st.session_state.data.df['train']                                         # Dataset configs
    pseudo_test_with_truth_df = st.session_state.data.df['pseudo_test_with_truth']
    test_df = st.session_state.data.df['test']

    output_paths = {
        'LSTMModel': 'data/output/submission/lstm',
        'RandomForestClassifierModel': 'data/output/submission/random_forest'
    }
    models_to_run = ['RandomForestClassifierModel', 'LSTMModel']  # Models to execute

    if st.button('Run predictions'):
        for model_name in models_to_run:
            st.markdown(f"## #{model_name} predictions_")
            ml_pipeline(model_name, train_df, pseudo_test_with_truth_df, test_df, output_paths[model_name])
