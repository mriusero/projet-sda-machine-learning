import pandas as pd
import os
import re
import glob
from datetime import datetime
import gc

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

    data = {
        'failure_data': failure_data,
        'degradation_data': degradation_data,
        'pseudo_testing_data': pseudo_testing_data,
        'pseudo_testing_data_with_truth': pseudo_testing_data_with_truth,
        'solution_data': solution_data,
        'testing_data': testing_data
    }

    return merge_data(data)

def merge_data(training_data):

    df1 = pd.merge(training_data['degradation_data'], training_data['failure_data'], on='item_id', how='left')
    df1.rename(columns={'item_id': 'item_index'}, inplace=True)
    df1['item_index'] = df1['item_index'].apply(lambda x: f'item_{x}')
    df1.to_csv('./data/output/training/training_data.csv', index=False)

    df2 = training_data['pseudo_testing_data'].copy()
    df2.rename(columns={'item_id': 'item_index'}, inplace=True)
    df2['item_index'] = df2['item_index'].apply(lambda x: f'item_{x}')
    df2.to_csv('./data/output/pseudo_testing/pseudo_testing_data.csv', index=False)

    df3 = training_data['pseudo_testing_data_with_truth'].copy()
    df3.rename(columns={'item_id': 'item_index'}, inplace=True)
    df3['item_index'] = df3['item_index'].apply(lambda x: f'item_{x}')
    df3 = pd.merge(df3, training_data['solution_data'], on='item_index', how='left')
    df3.to_csv('./data/output/pseudo_testing/pseudo_testing_data_with_truth.csv', index=False)

    df4 = training_data['testing_data'].copy()
    df4.rename(columns={'item_id': 'item_index'}, inplace=True)
    df4['item_index'] = df2['item_index'].apply(lambda x: f'{x}')
    df4.to_csv('./data/output/testing/testing_data_phase1.csv', index=False)

    update_message = 'New random data generated'
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    gc.collect()
    os.system('clear')

    return f"[{timestamp}] {update_message}"

def dataframing_data():
    paths = {
        'train': './data/output/training/training_data.csv',
        'pseudo_test': './data/output/pseudo_testing/pseudo_testing_data.csv',
        'pseudo_test_with_truth': './data/output/pseudo_testing/pseudo_testing_data_with_truth.csv',
        'test': './data/output/testing/testing_data_phase1.csv'
    }
    dataframes = {
        'train': pd.read_csv(paths['train']),
        'pseudo_test': pd.read_csv(paths['pseudo_test']),
        'pseudo_test_with_truth': pd.read_csv(paths['pseudo_test_with_truth']),
        'test': pd.read_csv(paths['test'])
    }
    return dataframes

def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    outliers_list = []
    for column in df.select_dtypes(include=['number']).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        outlier_mask = (df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))
        outliers = df[outlier_mask].copy()
        if not outliers.empty:
            outliers['outlier_column'] = column
            outliers['outlier_type'] = 'High'  # Possible extension: could also mark 'Low' for lower outliers
            outliers_list.append(outliers)

    if outliers_list:
        all_outliers_df = pd.concat(outliers_list)
        return all_outliers_df
    else:
        return pd.DataFrame(columns=df.columns.tolist() + ['outlier_column', 'outlier_type'])

def combine_submissions_for_scenario(folder_path):

    file_paths = glob.glob(os.path.join(folder_path, '*.csv'))
    dfs = [pd.read_csv(file) for file in file_paths]
    combined_df = pd.concat(dfs)
    final_df = combined_df.groupby('item_index').agg({'predicted_rul': 'max'}).reset_index()
    output_file = os.path.join(folder_path, 'Submission.csv')
    final_df.to_csv(output_file, index=False)