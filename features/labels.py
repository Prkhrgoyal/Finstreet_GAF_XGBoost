import numpy as np

def generate_directional_labels(close_prices):
    returns = np.diff(close_prices)
    return (returns > 0).astype(int)
