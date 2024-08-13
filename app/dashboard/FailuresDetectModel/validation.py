# validation.py
import pandas as pd

def generate_submission_file(output_path):

    template = pd.read_csv('../app/data/output/submission/template/submission_template.csv')

    # --- LSTM Model V3 ---
    lstm_results = pd.read_csv(f"{output_path}/lstm_predictions.csv")
    lstm_results = lstm_results[['item_index', 'crack_failure', 'Failure mode (lstm)']]

    submission_df = template.copy()
    submission_df['predicted_rul'] = 0

    for index, row in lstm_results.iterrows():
        item_index = row['item_index']
        crack_failure = row['crack_failure']
        if crack_failure == 1:
            submission_df.loc[submission_df['item_index'] == item_index, 'label'] = 1

    for index, row in lstm_results.iterrows():
        item_index = row['item_index']
        failure = row['Failure mode (lstm)']
        #if failure == 'Crack failure':
            #submission_df.loc[submission_df['item_index'] == item_index, 'label'] = 1
        if failure == 'Control board failure':
            submission_df.loc[submission_df['item_index'] == item_index, 'label'] = 1
        if failure == 'Infant Mortality':
            submission_df.loc[submission_df['item_index'] == item_index, 'label'] = 1

    # --- Random Forest Classifier Model ---
    rf_results = pd.read_csv(f"{output_path}/rf_predictions.csv")

    for index, row in rf_results.iterrows():
        item_index = row['item_index']
        failure = row['Failure mode (rf)']
        #if failure == 'Crack failure':
            #submission_df.loc[submission_df['item_index'] == item_index, 'label'] = 1
        if failure == 'Control board failure':
            submission_df.loc[submission_df['item_index'] == item_index, 'label'] = 1
        if failure == 'Infant Mortality':
            submission_df.loc[submission_df['item_index'] == item_index, 'label'] = 1

    return submission_df.to_csv(f"{output_path}/submission.csv", index=False)

def calculate_score(output_path):

    solution = pd.read_csv('../app/data/input/training_data/pseudo_testing_data_with_truth/Solution.csv')
    submission = pd.read_csv(f"{output_path}/submission.csv")

    reward = 2
    penalty_false_positive = -1 / 60
    penalty_false_negative = -4

    # Compare labels and calculate rewards/penalties
    rewards_penalties = []
    for _, (sol_label, sub_label, true_rul) in enumerate(
            zip(solution['label'], submission['label'], solution['true_rul'])):
        if sol_label == sub_label:
            rewards_penalties.append(reward)
        elif sol_label == 1 and sub_label == 0:
            rewards_penalties.append(penalty_false_negative)
        elif sol_label == 0 and sub_label == 1:
            rewards_penalties.append(penalty_false_positive * true_rul)
        else:
            rewards_penalties.append(0)  # No reward or penalty if labels don't match

    return sum(rewards_penalties)