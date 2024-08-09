from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_recall_fscore_support
import numpy as np


def evaluate_models(y_true_rul, y_pred_rul, y_true_ttf, y_pred_ttf, y_true_failure_mode, y_pred_failure_mode):
    # Evaluation des modèles de régression
    mse_rul = mean_squared_error(y_true_rul, y_pred_rul)
    rmse_rul = np.sqrt(mse_rul)
    mae_rul = mean_absolute_error(y_true_rul, y_pred_rul)

    mse_ttf = mean_squared_error(y_true_ttf, y_pred_ttf)
    rmse_ttf = np.sqrt(mse_ttf)
    mae_ttf = mean_absolute_error(y_true_ttf, y_pred_ttf)

    # Evaluation du modèle de classification
    accuracy, precision, recall, fscore, _ = precision_recall_fscore_support(y_true_failure_mode, y_pred_failure_mode,
                                                                             average='weighted')

    # Retourner les résultats sous forme de dictionnaire
    return {
        'RUL': {
            'MSE': mse_rul,
            'RMSE': rmse_rul,
            'MAE': mae_rul
        },
        'TTF': {
            'MSE': mse_ttf,
            'RMSE': rmse_ttf,
            'MAE': mae_ttf
        },
        'Failure Mode': {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F-score': fscore
        }
    }