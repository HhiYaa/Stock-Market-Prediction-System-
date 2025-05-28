# Stock-Market-Prediction-System-
# 📈 Stock Market Prediction System

This repository contains a machine learning pipeline that predicts stock price movement (up or down) based on financial indicators such as moving averages, RSI, and volume. It achieves ~80% accuracy on historical test data.

## 🚀 Features

- Fetches real-time stock data using public APIs.
- Processes and engineers features including SMA, EMA, RSI, and Volume.
- Uses a Scikit-learn ML pipeline (Random Forest Classifier).
- Automated daily data update support.
- Saves and reloads model with `joblib`.

## 🧠 ML Pipeline

- Preprocessing: Handles missing values, scales data.
- Feature Engineering: SMA(14), EMA(14), RSI(14), Volume.
- Model: Random Forest Classifier (can swap with others easily).
- Evaluation: Accuracy, precision, recall, and F1 score.

## 📦 Setup

```bash
git clone https://github.com/yourusername/stock-prediction-system.git
cd stock-prediction-system
pip install -r requirements.txt
