from utils.data_loader import load_data
from features.windowing import create_rolling_windows
from features.labels import generate_directional_labels
from features.gaf import gaf_transform
from models.xgb_model import build_model
from backtest.walk_forward import walk_forward
from config import WINDOW_SIZE

def main():
    df = load_data("data/ircon.csv")
    close = df['close'].values

    windows = create_rolling_windows(close, WINDOW_SIZE)
    labels = generate_directional_labels(close[WINDOW_SIZE:])

    X = gaf_transform(windows[:-1])
    y = labels[:len(X)]
    prices = close[WINDOW_SIZE:]

    model = build_model()
    results = walk_forward(model, X, y, prices)

    print("Backtest Results")
    for k, v in results.items():
    if isinstance(v, (int, float)):
        print(f"{k}: {v:.4f}")
    else:
        print(f"{k}: [array of length {len(v)}]")

if __name__ == "__main__":
    main()
