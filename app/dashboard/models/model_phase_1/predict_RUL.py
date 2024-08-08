import pandas as pd
import streamlit as st
import numpy as np
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def load_and_prepare_data(data_path):
    df = pd.read_csv(data_path)
    df['Time to failure (months)'].fillna(df['Time to failure (months)'].mean(), inplace=True)
    df['Failure mode'].fillna(df['Failure mode'].mode()[0], inplace=True)
    df = pd.get_dummies(df, columns=['Failure mode'])
    df = df.sort_values(by=['item_index', 'Time to failure (months)'])
    return df

def create_datasets(df):
    df['target_crack_length'] = df.groupby('item_index')['crack length (arbitary unit)'].shift(-6)
    df = df.dropna(subset=['target_crack_length'])
    return df

def create_temporal_features(df):
    df['crack_length_lag1'] = df.groupby('item_index')['crack length (arbitary unit)'].shift(1)
    df['crack_length_lag2'] = df.groupby('item_index')['crack length (arbitary unit)'].shift(2)
    df['rolling_mean_3'] = df.groupby('item_index')['crack length (arbitary unit)'].rolling(window=3).mean().reset_index(level=0, drop=True)
    df['rolling_std_3'] = df.groupby('item_index')['crack length (arbitary unit)'].rolling(window=3).std().reset_index(level=0, drop=True)
    df = df.dropna()
    return df

def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def predict_crack_length(model, X):
    return model.predict(X)

def generate_submission_file(submission_template_path, predictions, output_path):
    submission = pd.read_csv(submission_template_path)
    submission = pd.merge(submission, predictions, on='item_index', how='left')
    submission['predicted_rul'] = np.where(submission['predicted_crack_length'] > 0.85, 0, 1)
    submission = submission.groupby('item_index', as_index=False).agg({
        'predicted_rul': 'mean'  # ou 'max', 'min', ou une autre méthode d'agrégation selon le besoin
    })
    submission['predicted_rul'] = submission['predicted_rul'].apply(lambda x: 1 if x == 1 else 0)
    submission.to_csv(output_path, index=False)

def calculate_score(solution_path, submission_path):
    solution = pd.read_csv(solution_path)
    submission = pd.read_csv(submission_path)

    reward = 5
    penalty_false_positive = -1 / 6
    penalty_false_negative = -10

    merged = pd.merge(solution, submission, on='item_index', how='left')

    rewards_penalties = []
    for _, row in merged.iterrows():
        sol_label = row['label']
        sub_label = row['predicted_rul']
        true_rul = row['true_rul']

        if sol_label == sub_label:
            rewards_penalties.append(reward)
        elif sol_label == 1 and sub_label == 0:
            rewards_penalties.append(penalty_false_negative)
        elif sol_label == 0 and sub_label == 1:
            rewards_penalties.append(penalty_false_positive * true_rul)
        else:
            rewards_penalties.append(0)  # Pas de récompense ou pénalité si les labels ne correspondent pas

    return sum(rewards_penalties)


def plot_learning_curves(items_data, num_cols=5):
    """
    Trace les courbes d'apprentissage pour plusieurs items de régression dans une grille sur Streamlit.

    Parameters:
    - items_data : liste de tuples contenant les données de chaque item (X_train, y_train, X_test, y_test)
    - num_cols : nombre de graphiques par ligne
    """
    num_items = len(items_data)
    num_rows = int(np.ceil(num_items / num_cols))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 4), sharex=True, sharey=True)

    for i, (X_train, y_train, X_test, y_test) in tqdm(enumerate(items_data), total=num_items, desc="Création des graphiques"):
        train_errors, test_errors = [], []

        for m in range(1, len(X_train) + 1):
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train[:m], y_train[:m])
            y_train_predict = model.predict(X_train[:m])
            y_test_predict = model.predict(X_test)
            train_errors.append(mean_absolute_error(y_train[:m], y_train_predict))
            test_errors.append(mean_absolute_error(y_test, y_test_predict))

        row, col = divmod(i, num_cols)
        ax = axes[row, col] if num_rows > 1 else axes[col]
        ax.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="Ensemble d'entraînement")
        ax.plot(np.sqrt(test_errors), "b-", linewidth=3, label="Ensemble de test")
        ax.set_xlabel("Nombre d'échantillons")
        ax.set_ylabel("Erreur Quadratique Moyenne")
        ax.set_title(f"Item {i}")
        ax.grid(True)
        ax.legend()

    # Suppression des sous-graphiques inutilisés
    if num_items % num_cols != 0:
        for j in range(num_items, num_rows * num_cols):
            fig.delaxes(axes.flatten()[j])

    plt.tight_layout()
    st.pyplot(fig)  # Affiche le graphique dans Streamlit


def RULprediction():
    data_path = './data/output/train_split.csv'
    submission_template_path = './data/input/testing/sent_to_student/group_0/Sample_submission.csv'
    solution_path = './data/input/training/pseudo_testing_data_with_truth/Solution.csv'
    submission_output_path = './data/output/submission.csv'

    df = load_and_prepare_data(data_path)
    df = create_datasets(df)
    df = create_temporal_features(df)

    df['item_index'] = df['item_index'].apply(lambda x: int(re.search(r'\d+', x).group()))

    items_data = []

    for item_index, group in df.groupby('item_index'):
        X = group.drop(columns=['target_crack_length', 'true_rul'])
        y = group['target_crack_length']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        items_data.append((X_train, y_train, X_test, y_test))

    # Plot learning curves for all items
    plot_learning_curves(items_data)

    predictions = []

    for item_index, group in tqdm(df.groupby('item_index'), total=df['item_index'].nunique(),desc="Génération des prédictions"):
        X = group.drop(columns=['target_crack_length', 'true_rul'])
        y = group['target_crack_length']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = train_model(X_train, y_train)
        group_predictions = group[['item_index']].copy()
        group_predictions['predicted_crack_length'] = predict_crack_length(model, X)
        predictions.append(group_predictions)

    all_predictions = pd.concat(predictions)
    all_predictions['item_index'] = all_predictions['item_index'].apply(lambda x: f'item_{x}')

    generate_submission_file(submission_template_path, all_predictions, submission_output_path)
    score = calculate_score(solution_path, submission_output_path)

    st.markdown("Fichier de soumission généré : 'submission.csv'")
    st.markdown(f"Score de la soumission : {score}")



