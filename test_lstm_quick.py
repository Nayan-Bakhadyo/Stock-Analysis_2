from ml_predictor import MLStockPredictor
import pandas as pd
import sqlite3
from pathlib import Path
import config

# Load PFL data
db_path = Path(config.DATA_DIR) / 'nepse_stocks.db'
conn = sqlite3.connect(db_path)
df = pd.read_sql_query('SELECT date, open, high, low, close, volume FROM price_history WHERE symbol = "PFL" ORDER BY date DESC LIMIT 500', conn)
conn.close()

df = df.iloc[::-1].reset_index(drop=True)
df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
df['Date'] = pd.to_datetime(df['Date'])

print(f'Data loaded: {len(df)} days')
print(f'Current price: ${df["Close"].iloc[-1]:.2f}')
print(f'Price range: ${df["Close"].min():.2f} - ${df["Close"].max():.2f}')

# Quick LSTM test
predictor = MLStockPredictor(lookback_days=60)
print('\nTraining LSTM (5 epochs)...')
metrics = predictor.train_model(df, epochs=5, batch_size=32, validation_split=0.2)
print(f'Val MAE: {metrics.get("val_mae", 0):.4f}')

# Predict
predictions_dict = predictor.predict_future_prices(df, days=list(range(1, 8)))
print(f'\nPredictions (current price: ${df["Close"].iloc[-1]:.2f}):')
for day in range(1, 8):
    price = predictions_dict.get(day, 0)
    change = ((price - df["Close"].iloc[-1]) / df["Close"].iloc[-1]) * 100
    print(f'  Day {day}: ${price:.2f} ({change:+.2f}%)')

print(f'\nLSTM is {"WORKING" if predictions_dict.get(1, 0) != df["Close"].iloc[-1] else "NOT LEARNING (outputs current price)"}')
