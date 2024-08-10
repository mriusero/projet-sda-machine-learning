# main.py
import pandas as pd
import streamlit as st
from .models_base import RandomForestModel, LogisticRegressionModel
from .preprocessing import clean_data
from .features import add_features
from .predictions import make_predictions, save_predictions
from .validation import calculate_score

def process_predictions(scenario_name, train_df, test_df, output_path):
    # Prétraitement des données
    train_df = clean_data(train_df)
    test_df = clean_data(test_df)

    train_df = add_features(train_df)
    test_df = add_features(test_df)

    # Instancier plusieurs modèles
    models = []
    model_configs = []

    models.append(RandomForestModel())
    models.append(LogisticRegressionModel())
    model_configs.append(models[0].get_column_config())
    model_configs.append(models[1].get_column_config())

    # Ajoutez d'autres modèles et configurations si nécessaire
    predictions_dict = {}

    # Entraîner les modèles
    for model, config in zip(models, model_configs):
        X_train = train_df[config['X']]
        y_train = train_df[config['y']]
        X_test = test_df[config['X']]
        y_test = test_df[config['y']] if 'label' in test_df.columns else None

        model.train(X_train, y_train)
        predictions = model.predict(X_test)
        predictions_dict[model.__class__.__name__] = predictions

        # Évaluer les performances de chaque modèle (si vérité terrain disponible)
        if y_test is not None:
            score = calculate_score(y_test, predictions, test_df['true_rul'])
            st.write(f"Score for {model.__class__.__name__} in {scenario_name}: {score}")

    # Sauvegarder les prédictions
    save_predictions(predictions_dict, output_path)

    return predictions_dict


def handle_scenarios(dataframes):
    keys = ['train', 'pseudo_test', 'pseudo_test_with_truth', 'test']
    cleaned_dfs = {key: clean_data(dataframes[key]) for key in keys}

    scenarios = [  # Définir les scénarios avec leurs noms et chemins de fichiers
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
    for scenario in scenarios:  # Traitement de chaque scénario
        st.markdown(f"## {scenario['name']}")
        scenario_predictions = process_predictions(scenario['name'], cleaned_dfs['train'], scenario['test_df'],
                                                   scenario['output_path'])
        st.dataframe(pd.DataFrame(scenario_predictions))
        # plot_scatter2(scenario_predictions, 'crack length (arbitary unit)', 'time (months)', 'source')
