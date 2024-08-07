import pandas as pd
import os
def load_training_data():

    degradation_data_path = './data/input/training_data/degradation_data'
    degradation_dfs = []
    for filename in os.listdir(degradation_data_path):
        if filename.endswith('.csv'):
            item_id = int(filename.split('_')[1].split('.')[0])
            file_path = os.path.join(degradation_data_path, filename)
            df = pd.read_csv(file_path)
            df['item_id'] = item_id
            degradation_dfs.append(df)
    degradation_data = pd.concat(degradation_dfs, ignore_index=True)

    failure_data_path = './data/input/training_data/failure_data.csv'
    failure_data = pd.read_csv(failure_data_path)

    solution_data_path = './data/input/training_data/pseudo_testing_data_with_truth/Solution.csv'
    solution_data = pd.read_csv(solution_data_path)
    
    return {
        'degradation_data': degradation_data,
        'failure_data': failure_data,
        'solution_data': solution_data,
    }

def merge_training_data(training_data):
    df = pd.merge(training_data['degradation_data'], training_data['failure_data'], on='item_id', how='left')
    df.rename(columns={'item_id': 'item_index'}, inplace=True)
    df['item_index'] = df['item_index'].apply(lambda x: f'item_{x}')
    df = pd.merge(df, training_data['solution_data'], on='item_index', how='left')
    return df

