import numpy as np

def create_rolling_windows(series, window):
    X = []
    for i in range(len(series) - window):
        X.append(series[i:i + window])
    return np.array(X)
