import numpy as np

def sharpe_ratio(returns):
    if returns.std() == 0:
        return 0
    return (returns.mean() / returns.std()) * (252 ** 0.5)

def max_drawdown(cum_returns):
    peak = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - peak) / peak
    return drawdown.min()
