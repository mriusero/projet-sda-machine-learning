import pandas as pd
import os
import re
def load_data():

    failure_data_path = './data/input/training_data/failure_data.csv'           #Failure data
    failure_data = pd.read_csv(failure_data_path)

    degradation_data_path = './data/input/training_data/degradation_data'       #Degradation data
    degradation_dfs = []
    for filename in os.listdir(degradation_data_path):
        if filename.endswith('.csv'):
            item_id = int(filename.split('_')[1].split('.')[0])
            file_path = os.path.join(degradation_data_path, filename)
            df = pd.read_csv(file_path)
            df['item_id'] = item_id
            degradation_dfs.append(df)
    degradation_data = pd.concat(degradation_dfs, ignore_index=True)

    pseudo_testing_data_path = './data/input/training_data/pseudo_testing_data'     #Pseudo testing
    pseudo_testing_dfs = []
    for filename in os.listdir(pseudo_testing_data_path):
        if filename.endswith('.csv'):
            item_id = int(filename.split('_')[1].split('.')[0])
            file_path = os.path.join(pseudo_testing_data_path, filename)
            df = pd.read_csv(file_path)
            df['item_id'] = item_id
            pseudo_testing_dfs.append(df)
    pseudo_testing_data = pd.concat(pseudo_testing_dfs, ignore_index=True)

    pseudo_testing_data_with_truth_path = './data/input/training_data/pseudo_testing_data_with_truth'  # Pseudo testing with truth
    pseudo_testing_data_with_truth_dfs = []
    for filename in os.listdir(pseudo_testing_data_with_truth_path):
        if filename.endswith('.csv'):
            match = re.search(r'(\d+)', filename)
            if match:
                item_id = int(match.group(1))
                file_path = os.path.join(pseudo_testing_data_with_truth_path, filename)
                df = pd.read_csv(file_path)
                df['item_id'] = item_id
                pseudo_testing_data_with_truth_dfs.append(df)
    pseudo_testing_data_with_truth = pd.concat(pseudo_testing_dfs, ignore_index=True)

    solution_data_path = './data/input/training_data/pseudo_testing_data_with_truth/Solution.csv'   # Solution
    solution_data = pd.read_csv(solution_data_path)

    testing_data_path = './data/input/testing_data/phase1'  # Testing data
    testing_dfs = []
    for filename in os.listdir(testing_data_path):
        if filename.endswith('.csv'):
            item_id = int(filename.split('_')[2].split('.')[0])
            file_path = os.path.join(testing_data_path, filename)
            df = pd.read_csv(file_path)
            df['item_id'] = item_id
            testing_dfs.append(df)
    testing_data = pd.concat(testing_dfs, ignore_index=True)

    return {
        'failure_data': failure_data,
        'degradation_data': degradation_data,
        'pseudo_testing_data': pseudo_testing_data,
        'pseudo_testing_data_with_truth': pseudo_testing_data_with_truth,
        'solution_data': solution_data,
        'testing_data': testing_data
    }

def merge_data(training_data):

    df1 = pd.merge(training_data['degradation_data'], training_data['failure_data'], on='item_id', how='left')
    df1.rename(columns={'item_id': 'item_index'}, inplace=True)
    df1['item_index'] = df1['item_index'].apply(lambda x: f'item_{x}')
    df1.to_csv('./data/output/training/training_data.csv', index=False)
    train = df1.copy()

    df2 = pd.merge(training_data['pseudo_testing_data'], training_data['failure_data'], on='item_id', how='left')
    df2.rename(columns={'item_id': 'item_index'}, inplace=True)
    df2['item_index'] = df2['item_index'].apply(lambda x: f'item_{x}')
    df2.to_csv('./data/output/pseudo_testing/pseudo_testing_data.csv', index=False)
    pseudo_test = df2.copy()

    df3 = pd.merge(training_data['pseudo_testing_data_with_truth'], training_data['failure_data'], on='item_id', how='left')
    df3.rename(columns={'item_id': 'item_index'}, inplace=True)
    df3['item_index'] = df3['item_index'].apply(lambda x: f'item_{x}')
    df3 = pd.merge(df3, training_data['solution_data'], on='item_index', how='left')
    df3.to_csv('./data/output/pseudo_testing/pseudo_testing_data_with_truth.csv', index=False)
    pseudo_test_with_truth = df3.copy()

    df4 = pd.merge(training_data['testing_data'], training_data['failure_data'], on='item_id', how='left')
    df4.rename(columns={'item_id': 'item_index'}, inplace=True)
    df4['item_index'] = df2['item_index'].apply(lambda x: f'{x}')
    df4.to_csv('./data/output/testing/testing_data_phase1.csv', index=False)
    test = df4.copy()

    return train, pseudo_test, pseudo_test_with_truth, test


