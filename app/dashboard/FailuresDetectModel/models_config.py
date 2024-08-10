# models_config.py
MODEL_COLUMN_CONFIG = {
    'RandomForestModel': {
        'X': ['crack length (arbitary unit)', 'time (months)'],
        'y': 'label'
    },
    'LogisticRegressionModel': {
        'X': ['crack length (arbitary unit)', 'time (months)'],
        'y': 'label'
    }
    # Ajoutez d'autres modèles et leurs colonnes si nécessaire
}