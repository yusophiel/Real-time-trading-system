## Project Overview
This repository is a section of the reinforcement learning module of a larger cryptocurrency trading system that integrates multi-source sentiment analysis from large language models (LLMs) with deep reinforcement learning. In the full system, LLM-generated sentiment signals are filtered using a “Trust-the-Majority” strategy and dynamically weighted based on the Ebbinghaus Forgetting Curve to reflect their decaying market impact.

**My contribution focuses on the design and implementation of the downstream reinforcement learning framework.** Specifically, I developed a high-frequency trading module using the Proximal Policy Optimization (PPO) algorithm in PyTorch, augmented with an LSTM-based architecture to capture temporal dependencies in both market data and sentiment signals. 
- The agent observes a state space constructed from a sliding window of technical indicators (e.g., EMA, MACD, RSI, KDJ) and sentiment scores derived from LLM outputs.
- To guide the agent’s learning, I designed a composite reward function that incorporates return on investment (ROI) as the primary signal, while introducing a volatility penalty to control excessive risk-taking behavior. The module also implements data normalization (Z-score + outlier truncation), dynamic action sampling, and action interpretation for buy/hold/sell decisions.
- I was also responsible for the full training loop, backtesting pipeline, and performance visualization tools. This includes model checkpointing, Sharpe ratio evaluation, trade signal plotting, and simulation-based validation on unseen market data.

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

