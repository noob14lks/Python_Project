import numpy as np
import os

def load_all_features(folder_path):
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npz')]
    datasets = []
    for fp in file_paths:
        data = np.load(fp)
        X = data['features']
        y = data['labels']
        datasets.append((X, y))
    return datasets, file_paths
