Finstreet – GAF + XGBoost Directional Trading Model
Overview

This repository contains the final submission for the Finstreet case competition, implementing a directional trading strategy using:

Gramian Angular Field (GAF) image encoding of price windows

XGBoost classifier for next-period direction prediction

Walk-forward backtesting to simulate realistic trading performance

The design explicitly focuses on small-data robustness, avoiding deep neural networks and look-ahead bias while maintaining explainability and strong empirical performance.

Problem Statement

Given limited historical price data, build a model that:

Generates long / short directional signals

Avoids data leakage

Can be backtested realistically

Is suitable for eventual live deployment (FYERS API integration ready)

Model Rationale (High Level)
Component	Justification
GAF Encoding	Converts 1D price windows into 2D texture-rich representations, preserving temporal structure
XGBoost	Strong performance on small datasets, handles non-linearity, robust to noise
Directional Labels	More stable than point return prediction under volatility
Walk-Forward Backtest	Prevents look-ahead bias and simulates real trading
Repository Structure
Finstreet_GAF_XGBoost/
│
├── main.py                  # Entry point (end-to-end pipeline)
├── requirements.txt         # Python dependencies
│
├── data/
│   └── ircon.csv             # OHLC price data (sample)
│
├── config.py                # Global parameters (window size, etc.)
│
├── features/
│   ├── windowing.py          # Rolling window creation
│   ├── labels.py             # Directional label generation
│   └── gaf.py                # GAF transformation
│
├── models/
│   └── xgb_model.py          # XGBoost classifier
│
├── backtest/
│   └── walk_forward.py       # Walk-forward training & evaluation
│
└── utils/
    └── data_loader.py        # CSV data loading utilities

Environment Setup
1. Clone the Repository
git clone https://github.com/Prkhrgoyal/Finstreet.git
cd Finstreet_GAF_XGBoost

2. Create Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows

3. Install Dependencies
pip install -r requirements.txt

How to Run the Project


Run the Pipeline

python main.py

What Happens When You Run main.py

Loads historical price data

Creates rolling price windows

Converts each window into a GAF image

Generates directional labels (up / down)

Trains XGBoost in a walk-forward manner

Produces:

Predicted signals

Strategy returns

Backtest performance metrics

This structure ensures:

No future data leakage

Realistic evaluation

Reproducibility

Backtesting Methodology

Walk-forward training

Model retrained incrementally

Prediction made only on unseen future data

Strategy returns computed from predicted directions

This mimics real trading conditions more accurately than static train-test splits.