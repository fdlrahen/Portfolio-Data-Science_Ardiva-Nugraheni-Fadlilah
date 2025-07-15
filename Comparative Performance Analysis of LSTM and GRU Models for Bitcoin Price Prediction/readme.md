# 📈 Bitcoin Forecasting: LSTM vs GRU

This project compares the performance of Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) networks for forecasting Bitcoin prices based on historical time-series data.

## 🎯 Objective
Evaluate the temporal generalization capacity of LSTM and GRU for crypto forecasting using key metrics like Accuracy, Precision, Recall, and F1 Score.

## 📊 Data
- Source: Historical Bitcoin prices
- Preprocessing:
  - Normalization
  - Sliding window sequence generation

## 🧪 Modeling
- LSTM Model:
  - Layers: Input → LSTM → Dense
  - Loss: MSE
- GRU Model:
  - Layers: Input → GRU → Dense
  - Loss: MSE
- Evaluation via classification metrics (price up/down prediction)

## 📈 Visualizations
- Grouped bar charts for performance metrics
- Time-series plot of predictions vs actual prices
- Confusion matrix per model

## 📦 Dependencies
- PyTorch or TensorFlow
- Scikit-learn
- Matplotlib / Seaborn
- Pandas / NumPy
