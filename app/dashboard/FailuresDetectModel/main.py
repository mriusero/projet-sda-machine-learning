# main.py
import streamlit as st
from ..FailuresDetectModel import (RandomForestClassifierModel, LSTMModel, GradientBoostingSurvivalModel)

def instance_model(model_name):
    model_classes = {
        'LSTMModel': lambda: LSTMModel(min_sequence_length=2, forecast_months=6),
        'RandomForestClassifierModel': lambda: RandomForestClassifierModel(),
        'GradientBoostingSurvivalModel': lambda: GradientBoostingSurvivalModel()
    }
    if model_name not in model_classes:
        raise ValueError(f"Model '{model_name}' is not supported.")
    return model_classes[model_name]()

@st.cache_resource
def ml_pipeline(selected_models, train_df, pseudo_test_with_truth_df, test_df, optimize):
    """
    Function that executes the three phases of the machine learning pipeline for the selected models.

    Args:
        selected_models (list):                    List of models to run.
        train_df (DataFrame):                      Training dataset.
        pseudo_test_with_truth_df (DataFrame):     Test dataset for cross-validation.
        test_df (DataFrame):                       Final test dataset for predictions.
        optimize (bool):                           Whether to optimize hyperparameters.
    """
    if 'RandomForestClassifierModel' in selected_models:
        # --------- Random Forest Classifier Model ------------
        rf_model = instance_model('RandomForestClassifierModel')
        rf_model.run_full_pipeline(train_df, pseudo_test_with_truth_df, test_df)

    if 'LSTMModel' in selected_models:
        # --------- LSTM Model ------------
        lstm_model = instance_model('LSTMModel')
        lstm_model.run_full_pipeline(train_df, pseudo_test_with_truth_df, test_df)

    if 'GradientBoostingSurvivalModel' in selected_models:
        # --------- Gradient Boosting Survival Model ------------
        gb_model = instance_model('GradientBoostingSurvivalModel')
        gb_model.run_full_pipeline(train_df, pseudo_test_with_truth_df, test_df, optimize)

def handle_models():
    """
    Model management function that runs the pipeline for each selected model.
    """
    train_df = st.session_state.data.df['train']
    pseudo_test_with_truth_df = st.session_state.data.df['pseudo_test_with_truth']
    test_df = st.session_state.data.df['test']

    selected_models = st.multiselect(
        'Select models to run:',
        ['RandomForestClassifierModel', 'LSTMModel', 'GradientBoostingSurvivalModel'],
        default=['RandomForestClassifierModel', 'LSTMModel', 'GradientBoostingSurvivalModel']
    )

    if not selected_models:
        st.warning("Please select at least one model to run.")
        return

    if st.button('Run predictions'):
        ml_pipeline(selected_models, train_df, pseudo_test_with_truth_df, test_df, optimize=False)

    if st.button('Optimize Hyperparameters'):
        ml_pipeline(selected_models, train_df, pseudo_test_with_truth_df, test_df, optimize=True)