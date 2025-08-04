## Project Overview
This project implements a deep reinforcement learning framework for high-frequency cryptocurrency trading. It features a sliding window-based signal processing pipeline that computes real-time technical indicators (e.g., EMA, MACD, RSI, KDJ) from hourly BTC and DOGE data. A PPO agent with LSTM is used to handle temporal features and market sentiment data, trained with a customized reward function to learn effective trading strategies. The system supports full-cycle backtesting, evaluation, and live deployment simulation.

---

## 📂 File Overview

### 🧠 `RL_brain.py`
Defines the PPO agent:
- Uses a shared MLP + LSTM structure for both policy and value estimation.
- Incorporates `Dropout` and `LayerNorm` for better generalization.
- Contains `pi()` and `v()` functions for policy and value output.
- Implements `train_net()` with GAE (Generalized Advantage Estimation) and PPO-Clip loss.

### 🧪 `stock_env.py`
A custom OpenAI-Gym-style environment for trading:
- Simulates buy/sell/hold operations with realistic fees and cash handling.
- Uses a rolling window of technical + sentiment indicators as state.
- Computes reward based on price movements and trading decisions.
- Outputs visualizations for trade signals and cumulative profit.

### 🏃 `run_this(ppo).py`
Main script for:
- Preprocessing training/testing data
- Running PPO training loop
- Periodically saving models, testing performance, and visualizing results
- Outputs: `.pkl` models, `.npy` profit data, and `.png` plots

---

## 🎯 Features

- ✅ LSTM-enhanced PPO for time-aware decisions
- ✅ Technical + sentiment features
- ✅ Visualization of trades, rewards, and profits
- ✅ Custom trading environment
- ✅ Supports GPU acceleration with PyTorch

---

## 📈 Example Output

- Cumulative profit comparison (account vs market)
- Trade signal visualization (`^` = buy, `v` = sell)
- Reward trends during training/testing

---

## 🧰 How to Run

### 1. Prepare CSV Data
Make sure the CSV file `BTC_20241031_with_sentiments_2.csv` includes:
- OHLCV (Open, High, Low, Close, Volume)
- Technical indicators (MACD, RSI, MA, etc.)
- Sentiment indicators (Regulatory Impact, Overall Sentiment, etc.)

### 2. Train & Evaluate
```bash
python run_this(ppo).py
```

Models and result plots will be saved to:
- `model_OHLCV/`
- `Reward/`
- `Trade/`

---

## 📌 Requirements

- Python 3.7+
- PyTorch
- numpy, pandas, matplotlib
- scikit-learn

Install with:
```bash
pip install -r requirements.txt
```

