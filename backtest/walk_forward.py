import numpy as np
from backtest.metrics import sharpe_ratio, max_drawdown
from config import THRESHOLD


def walk_forward(model, X, y, prices):
    """
    Expanding-window walk-forward backtest.
    Trains only on past data and skips iterations
    where class diversity is insufficient.
    """

    preds = []
    rets = []

    for i in range(len(X) - 2):

        # --- training labels up to time i ---
        y_train = y[: i + 1]

        # --- CRITICAL FIX: need both classes ---
        if len(np.unique(y_train)) < 2:
            preds.append(0.0)
            rets.append(0.0)
            continue

        # --- train on past only ---
        model.fit(X[: i + 1], y_train)

        # --- predict probability for next step ---
        proba = model.predict_proba(
            X[i + 1].reshape(1, -1)
        )[0, 1]

        preds.append(proba)

        # --- simple directional trading rule ---
        if proba > THRESHOLD:
            ret = (prices[i + 2] - prices[i + 1]) / prices[i + 1]
        else:
            ret = 0.0

        rets.append(ret)

    rets = np.array(rets)

    # --- equity curve ---
    equity = np.cumprod(1 + rets)

    # --- metrics ---
    sharpe = sharpe_ratio(rets)
    mdd = max_drawdown(equity)
    total_return = equity[-1] - 1 if len(equity) > 0 else 0.0
    trades = int(np.sum(rets != 0.0))

    return {
        "Sharpe": sharpe,
        "MaxDrawdown": mdd,
        "TotalReturn": total_return,
        "Trades": trades,
    }
