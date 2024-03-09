import pandas as pd
import numpy as np
from IPython.display import display, HTML
import os
import pickle

def read_convert_date(path, dt_col):
    df = pd.read_csv(path)
    df[dt_col] = pd.to_datetime(df[dt_col])
    df.set_index(dt_col, inplace=True)
    return df
    
def print_bold(string):
    if isinstance(string, np.ndarray):
        string = f"<b style='font-size:20px;'>{', '.join(string)}</b>"
    elif isinstance(string, str):
        string = f"<b style='font-size:20px;'>{string}</b>"
    return(display(HTML(string)))

def create_pkl(dfs, pkl_name):
    v = 1
    pkl = pkl_name + '.pkl'
    while os.path.exists(pkl):
        v+=1
        pkl = pkl_name + '_' + str(v) + '.pkl'
    with open(pkl,'wb') as f:
        pickle.dump(dfs, f)
    pass

def load_pkl(pkl_name):
    pkl_file = pkl_name + '.pkl'
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    print(f"Pickle file ({pkl_file}) loaded successfully.")
    return data
    
def load_latest_pkl(pkl_name):
    pkl_files = [file for file in os.listdir('.') if file.endswith('.pkl') and file.startswith(pkl_name)]
    if not pkl_files:
        return None, None

    version_file_map = {}
    for file in pkl_files:
        if file == pkl_name + '.pkl':
            version = 1
        else:
            version_parts = file.split(pkl_name + '_')[1].split('.pkl')[0]
            version = int(version_parts) if version_parts.isdigit() else 1
        version_file_map[version] = file

    latest_version = max(version_file_map.keys())
    latest_pkl_file = version_file_map[latest_version]

    with open(latest_pkl_file, 'rb') as f:
        latest_data = pickle.load(f)

    if latest_data is not None:
        print(f"Latest pickle file ({latest_pkl_file}) loaded successfully.")
    else:
        print("No pickle files found.")
    return latest_data

def create_folder(path):
    v = 1
    while os.path.exists(path):
        v+=1
        if path[-2:-1] == '_':
            path = path[:-1] + str(v)
        else:
            path = path + '_' + str(v)
    os.makedirs(path)
    return path
    
def folder_version(base_folder_name, search_directory='.'):
    all_items = os.listdir(search_directory)
    versioned_folders = [item for item in all_items if os.path.isdir(os.path.join(search_directory, item)) and item.startswith(base_folder_name)]
    
    version_folder_map = {}
    for folder in versioned_folders:
        if folder == base_folder_name:
            version = 1
        else:
            parts = folder.split('_')
            version = int(parts[-1]) if parts[-1].isdigit() else 1
        version_folder_map[version] = folder

    if version_folder_map:
        latest_version = max(version_folder_map.keys())
        latest_version_folder = version_folder_map[latest_version]
        return os.path.join(search_directory, latest_version_folder)
    else:
        return None




