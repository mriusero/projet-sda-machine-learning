import pandas as pd
import os
def load_training_data():
    
    failure_data_path = './data/input/training_data/failure_data.csv'
    degradation_data_path = './data/input/training_data/degradation_data'
        
    failure_data = pd.read_csv(failure_data_path)       
        
    degradation_dfs = []        # Liste pour stocker les DataFrames de dégradation

    for filename in os.listdir(degradation_data_path):      # Lire chaque fichier de dégradation et les ajouter à la liste
        if filename.endswith('.csv'):
            item_id = int(filename.split('_')[1].split('.')[0])
            file_path = os.path.join(degradation_data_path, filename)
            df = pd.read_csv(file_path)
            df['item_id'] = item_id
            degradation_dfs.append(df)

    degradation_data = pd.concat(degradation_dfs)   # Concaténer tous les DataFrames de dégradation en un seul DataFrame
    
    combined_data = pd.merge(degradation_data, failure_data, on='item_id', how='left')      # Fusionner les données de dégradation avec les données de failure_data sur 'item_id'
    
    return {                                            # Retourner les DataFrames dans un dictionnaire
        'failure_data': failure_data,
        'degradation_data': degradation_data,
        'combined_data': combined_data
    }


