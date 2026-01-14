import numpy as np
from config import THRESHOLD
from backtest.metrics import sharpe_ratio, max_drawdown

def walk_forward(model, X, y, prices):
    preds, rets = [], []

    for i in range(len(X) - 1):
        model.fit(X[:i+1], y[:i+1])
        prob = model.predict_proba(X[i+1].reshape(1, -1))[0][1]
        preds.append(prob)

        if prob > THRESHOLD:
            ret = (prices[i+2] - prices[i+1]) / prices[i+1]
        else:
            ret = 0
        rets.append(ret)

    rets = np.array(rets)
    cum = np.cumprod(1 + rets)

    return {
        "Sharpe": sharpe_ratio(rets),
        "MaxDrawdown": max_drawdown(cum),
        "TotalReturn": cum[-1] - 1,
        "Trades": sum(p > THRESHOLD for p in preds)
    }
