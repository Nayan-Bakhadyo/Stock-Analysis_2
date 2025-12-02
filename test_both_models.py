"""Quick test to verify LSTM and xLSTM are both learning properly"""

import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path
import config
from ml_predictor import MLStockPredictor
from xlstm_stock_forecaster import prepare_multi_horizon_data, xLSTMStockForecaster, xLSTMForecasterTrainer
import torch
from torch.utils.data import TensorDataset, DataLoader

print("="*70)
print("TESTING: LSTM vs xLSTM - Are they learning?")
print("="*70)

# Load data
db_path = Path(config.DATA_DIR) / 'nepse_stocks.db'
conn = sqlite3.connect(db_path)
df = pd.read_sql_query('SELECT date, open, high, low, close, volume FROM price_history WHERE symbol = "PFL" ORDER BY date DESC LIMIT 500', conn)
conn.close()

df = df.iloc[::-1].reset_index(drop=True)
df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
df['date'] = pd.to_datetime(df['date'])

# Add capitalized for LSTM
df['Date'] = df['date']
df['Open'] = df['open']
df['High'] = df['high']
df['Low'] = df['low']
df['Close'] = df['close']
df['Volume'] = df['volume']

current_price = df['close'].iloc[-1]
print(f"\nCurrent PFL price: ${current_price:.2f}")
print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

# Test LSTM
print("\n" + "-"*70)
print("TEST 1: LSTM (TensorFlow)")
print("-"*70)
predictor = MLStockPredictor(lookback_days=60)
print("Training for 10 epochs...")
metrics = predictor.train_model(df, epochs=10, batch_size=32, validation_split=0.2)
print(f"Val MAE: {metrics.get('val_mae', 0):.4f}")

predictions_dict = predictor.predict_future_prices(df, days=[1, 3, 7])
print("\nLSTM Predictions:")
lstm_preds = []
for day in [1, 3, 7]:
    day_key = f'day_{day}'
    if 'days' in predictions_dict and day_key in predictions_dict['days']:
        pred = predictions_dict['days'][day_key]['predicted_price']
    else:
        pred = predictions_dict.get(day, current_price)
    change = ((pred - current_price) / current_price) * 100
    print(f"  {day}-day: ${pred:.2f} ({change:+.2f}%)")
    lstm_preds.append(pred)

# Check if model is learning (predictions should vary, not all be the same)
lstm_pred_variance = np.std(lstm_preds)
lstm_learning = lstm_pred_variance > 1.0  # Standard deviation > $1
print(f"\n{'✓' if lstm_learning else '✗'} LSTM is {'LEARNING (variance=${:.2f})'.format(lstm_pred_variance) if lstm_learning else 'NOT LEARNING (stuck at current price)'}")

# Test xLSTM
print("\n" + "-"*70)
print("TEST 2: xLSTM (PyTorch)")
print("-"*70)

X_train, y_train_dict, X_test, y_test_dict, scaler = prepare_multi_horizon_data(
    df, horizons=[1, 3, 7], lookback=60, train_split=0.8
)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = xLSTMStockForecaster(
    input_size=1,
    hidden_size=128,
    num_blocks=3,
    horizons=[1, 3, 7],
).to(device)

print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
print(f"Device: {device}")
print("Training for 10 epochs...")

trainer = xLSTMForecasterTrainer(model, device=device)

# Create dataloaders
train_targets = torch.stack([torch.FloatTensor(y_train_dict[h]) for h in [1, 3, 7]], dim=1)
X_train_tensor = torch.FloatTensor(X_train)
train_dataset = TensorDataset(X_train_tensor, train_targets)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_targets = torch.stack([torch.FloatTensor(y_test_dict[h]) for h in [1, 3, 7]], dim=1)
X_test_tensor = torch.FloatTensor(X_test)
val_dataset = TensorDataset(X_test_tensor, val_targets)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

trainer.fit(train_loader, val_loader, epochs=10, verbose=False)

# Predict on last sample
predictions = trainer.predict(X_test_tensor)
print("\nxLSTM Predictions (last test sample):")
for h in [1, 3, 7]:
    pred_norm = predictions[h][-1, 0]  # Last prediction
    pred_price = scaler.inverse_transform([[pred_norm]])[0, 0]
    change = ((pred_price - current_price) / current_price) * 100
    print(f"  {h}-day: ${pred_price:.2f} ({change:+.2f}%)")

xlstm_learning = abs(pred_price - current_price) > 1.0
print(f"\n{'✓' if xlstm_learning else '✗'} xLSTM is {'LEARNING' if xlstm_learning else 'NOT LEARNING'}")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"LSTM:  {'✓ Working' if lstm_learning else '✗ NOT Learning - outputs current price'}")
print(f"xLSTM: {'✓ Working' if xlstm_learning else '✗ NOT Learning'}")
print("="*70)
