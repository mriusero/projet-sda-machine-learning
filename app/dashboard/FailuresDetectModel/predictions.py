# predictions.py
import pandas as pd

def make_predictions(models, X_test):
    # Prédictions pour chaque modèle
    predictions = {model.__class__.__name__: model.predict(X_test) for model in models}
    return predictions

def save_predictions(predictions_dict, output_path):
    # Sauvegarder les prédictions de tous les modèles
    for model_name, preds in predictions_dict.items():
        pd.DataFrame(preds, columns=['predictions']).to_csv(f"{output_path}/{model_name}_predictions.csv", index=False)
