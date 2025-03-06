import pandas as pd
import numpy as np  
import glob


def get_files(test_path):
    files = []
    for file in glob.glob(f"{test_path}/test_*.csv"):
        index = int(file.split('_')[-1].split('.')[0])
        files.append((file, index))
    return sorted(files, key = lambda x: x[-1])


def evaluate(model, test_path, length):
    files = get_files(test_path)
    y = np.zeros((len(files), length))
    for file, index in files:
        data = pd.read_csv(file)[:length, :]
        yhat, _ = model.predict(data,  quantiles=(0.025, 0.5, 0.975))
        y[index] = yhat[0].mean(axis=0)
    return y 


def get_data(npy_file, sep):
    data = np.load(npy_file)    
    return data[:, :sep], data[:, sep:]
