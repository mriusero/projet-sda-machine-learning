# validation.py
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

def cross_validate(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv)
    return scores.mean()

def calculate_score(true_labels, predictions, true_rul):
    score = 0
    for true_label, pred, rul in zip(true_labels, predictions, true_rul):
        if true_label == pred:
            score += 2
        elif true_label == 1 and pred == 0:
            score -= 4
        elif true_label == 0 and pred == 1:
            score -= (1/60) * rul
    return score

def generate_submission_file(predictions, output_path):
    template = pd.read_csv('./app/data/output/submission/template/submission_template.csv')
    predictions['item_index'] = predictions['item_index'].apply(lambda x: f'item_{x}')
    submission = pd.merge(template, predictions, on='item_index', how='left')
    #submission['predicted_rul'] = np.where(submission['predicted_crack_length'] > 0.85, 0, 1)

    submission.to_csv(output_path, index=False)