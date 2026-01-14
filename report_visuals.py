import numpy as np
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField

# ---- IMPORT FROM YOUR EXISTING CODE ----
from main import load_data
from backtest.walk_forward import walk_forward
from features.gaf import gaf_transform
from models.model import get_model
from config import WINDOW_SIZE


# ---------------- PLOTTING FUNCTIONS ----------------

def plot_equity_curve(rets):
    equity = np.cumprod(1 + rets)
    plt.figure(figsize=(10, 4))
    plt.plot(equity)
    plt.title("Equity Curve (Walk-Forward Backtest)")
    plt.xlabel("Time Steps")
    plt.ylabel("Normalized Equity")
    plt.grid(True)
    plt.show()


def plot_drawdown(rets):
    equity = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak

    plt.figure(figsize=(10, 4))
    plt.plot(drawdown, color="red")
    plt.title("Drawdown Curve")
    plt.xlabel("Time Steps")
    plt.ylabel("Drawdown")
    plt.grid(True)
    plt.show()


def plot_trade_distribution(rets):
    trade_rets = rets[rets != 0]
    plt.figure(figsize=(6, 4))
    plt.hist(trade_rets, bins=20, edgecolor="black")
    plt.title("Distribution of Trade Returns")
    plt.xlabel("Return")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()


def plot_gaf_example(close, index=100):
    """
    Plot a single GAF image for report illustration.
    """

    window = close[index : index + WINDOW_SIZE]

    # normalize to [0, 1]
    window = (window - window.min()) / (window.max() - window.min())

    gaf = GramianAngularField(method="summation")
    gaf_image = gaf.fit_transform(window.reshape(1, -1))[0]

    plt.figure(figsize=(4, 4))
    plt.imshow(gaf_image, cmap="rainbow", origin="lower")
    plt.title("Gramian Angular Field Representation")
    plt.colorbar()
    plt.show()


# ---------------- MAIN DRIVER ----------------

def main():
    # ---- LOAD DATA (same as main.py) ----
    df = load_data("data/ircon.csv")
    close = df["close"].values

    # ---- CREATE WINDOWS (same logic as training) ----
    windows = []
    prices = []

    for i in range(len(close) - WINDOW_SIZE - 1):
        windows.append(close[i : i + WINDOW_SIZE])
        prices.append(close[i + WINDOW_SIZE])

    windows = np.array(windows)
    prices = np.array(prices)

    # ---- GAF TRANSFORMATION ----
    X = gaf_transform(windows)

    # ---- LABELS (same directional logic) ----
    y = (prices[1:] > prices[:-1]).astype(int)
    X = X[:-1]
    prices = prices[:-1]

    model = get_model()

    # ---- RUN BACKTEST ----
    results = walk_forward(model, X, y, prices)
    rets = results["Returns"]

    # ---- GENERATE REPORT VISUALS ----
    plot_equity_curve(rets)
    plot_drawdown(rets)
    plot_trade_distribution(rets)
    plot_gaf_example(close)


if __name__ == "__main__":
    main()
