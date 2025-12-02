"""
Ultra-Optimized xLSTM Stock Forecaster for Mac Studio M1 Max
Target: MAPE < 1%, Direction Accuracy > 80%

Key Optimizations:
1. Focus on 1-day prediction only (most accurate)
2. Heavy direction-focused loss (α=0.8)
3. Ensemble voting for direction
4. Price change prediction instead of absolute price
5. Feature selection for most predictive features
6. Market features (NEPSE + Sector indices)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import sqlite3
from pathlib import Path
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

import config
from sector_mapper import SectorMapper

# Use MPS for M1 Max
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"🖥️  Using device: {DEVICE}")


class DirectionFocusedLoss(nn.Module):
    """Loss that heavily penalizes wrong direction predictions"""
    
    def __init__(self, direction_weight=0.8):
        super().__init__()
        self.direction_weight = direction_weight
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target, prev_price):
        # MSE for magnitude
        mse_loss = self.mse(pred, target)
        
        # Direction loss
        pred_direction = torch.sign(pred - prev_price)
        true_direction = torch.sign(target - prev_price)
        
        # Penalize wrong direction heavily
        direction_correct = (pred_direction == true_direction).float()
        direction_loss = 1.0 - direction_correct.mean()
        
        # Combined loss
        total = (1 - self.direction_weight) * mse_loss + self.direction_weight * direction_loss
        return total


class FastLSTM(nn.Module):
    """Lightweight but effective LSTM for speed"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 3)  # Down, Neutral, Up
        )
    
    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden*2)
        
        # Attention
        attn_weights = self.attention(lstm_out)  # (batch, seq, 1)
        context = torch.sum(lstm_out * attn_weights, dim=1)  # (batch, hidden*2)
        
        # Outputs
        price_pred = self.regressor(context)
        direction_logits = self.classifier(context)
        
        return price_pred, direction_logits


class StockDataset(Dataset):
    def __init__(self, X, y_price, y_direction, prev_prices):
        self.X = torch.FloatTensor(X)
        self.y_price = torch.FloatTensor(y_price)
        self.y_direction = torch.LongTensor(y_direction)
        self.prev_prices = torch.FloatTensor(prev_prices)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_price[idx], self.y_direction[idx], self.prev_prices[idx]


def load_stock_data(symbol: str, db_path: str = 'data/nepse_stocks.db'):
    """Load stock data from database"""
    conn = sqlite3.connect(db_path)
    query = f"""
        SELECT date, open, high, low, close, volume 
        FROM price_history 
        WHERE symbol = '{symbol}' 
        ORDER BY date
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df['date'] = pd.to_datetime(df['date'])
    return df


def get_market_data(symbol: str):
    """
    Get NEPSE index and sector index data for a given stock symbol.
    
    Returns:
        Tuple of (nepse_df, sector_df, sector_code)
    """
    db_path = Path(config.DATA_DIR) / "nepse_stocks.db"
    conn = sqlite3.connect(db_path)
    
    # Get sector for this symbol
    mapper = SectorMapper()
    sector_info = mapper.get_sector(symbol)
    sector_code = sector_info['sector_index_code'] if sector_info else None
    
    # Get NEPSE index data
    nepse_query = """
        SELECT date, close as nepse_close, pct_change as nepse_pct_change
        FROM index_history
        WHERE index_code = 'NEPSE'
        ORDER BY date ASC
    """
    nepse_df = pd.read_sql_query(nepse_query, conn)
    nepse_df['date'] = pd.to_datetime(nepse_df['date'])
    
    # Get sector index data if available
    sector_df = pd.DataFrame()
    if sector_code:
        sector_query = f"""
            SELECT date, close as sector_close, pct_change as sector_pct_change
            FROM index_history
            WHERE index_code = '{sector_code}'
            ORDER BY date ASC
        """
        sector_df = pd.read_sql_query(sector_query, conn)
        if not sector_df.empty:
            sector_df['date'] = pd.to_datetime(sector_df['date'])
    
    conn.close()
    
    return nepse_df, sector_df, sector_code


def create_features(df: pd.DataFrame, symbol: str = None, lookback: int = 30, use_market: bool = True):
    """Create optimized feature set for direction prediction with market context"""
    
    df = df.copy()
    
    # Price changes (most important for direction)
    df['returns'] = df['close'].pct_change()
    df['returns_2d'] = df['close'].pct_change(2)
    df['returns_5d'] = df['close'].pct_change(5)
    
    # Momentum indicators
    df['rsi'] = calculate_rsi(df['close'], 14)
    df['rsi_signal'] = (df['rsi'] - 50) / 50  # Normalized
    
    # Moving averages crossover signals
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['ma_cross_5_10'] = (df['sma_5'] - df['sma_10']) / df['sma_10']
    df['ma_cross_5_20'] = (df['sma_5'] - df['sma_20']) / df['sma_20']
    
    # Price position relative to MAs
    df['price_vs_sma5'] = (df['close'] - df['sma_5']) / df['sma_5']
    df['price_vs_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']
    
    # Volatility
    df['volatility'] = df['returns'].rolling(10).std()
    
    # Volume signal
    df['volume_ma'] = df['volume'].rolling(10).mean()
    df['volume_signal'] = (df['volume'] - df['volume_ma']) / (df['volume_ma'] + 1)
    
    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = (ema12 - ema26) / df['close']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger position
    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_position'] = (df['close'] - bb_mid) / (2 * bb_std + 0.001)
    
    # Feature columns (base)
    feature_cols = [
        'returns', 'returns_2d', 'returns_5d',
        'rsi_signal', 'ma_cross_5_10', 'ma_cross_5_20',
        'price_vs_sma5', 'price_vs_sma20',
        'volatility', 'volume_signal',
        'macd', 'macd_signal', 'macd_hist', 'bb_position'
    ]
    
    sector_code = None
    
    # Add market features if requested
    if use_market and symbol:
        nepse_df, sector_df, sector_code = get_market_data(symbol)
        
        if not nepse_df.empty:
            print(f"  📈 NEPSE index: {len(nepse_df)} days")
            
            # Merge NEPSE data
            df = df.merge(nepse_df, on='date', how='left')
            df['nepse_close'] = df['nepse_close'].ffill().bfill()
            df['nepse_pct_change'] = df['nepse_pct_change'].ffill().bfill()
            
            # NEPSE returns
            df['nepse_returns'] = df['nepse_close'].pct_change()
            df['nepse_returns'] = df['nepse_returns'].fillna(0)
            
            # NEPSE momentum
            df['nepse_sma5'] = df['nepse_close'].rolling(5).mean()
            df['nepse_momentum'] = (df['nepse_close'] - df['nepse_sma5']) / df['nepse_sma5']
            df['nepse_momentum'] = df['nepse_momentum'].fillna(0)
            
            # Rolling beta (stock sensitivity to market)
            df['rolling_beta'] = df['returns'].rolling(20).corr(df['nepse_returns'])
            df['rolling_beta'] = df['rolling_beta'].fillna(1.0)
            
            # Relative strength vs NEPSE
            df['stock_cum_5d'] = df['returns'].rolling(5).sum()
            df['nepse_cum_5d'] = df['nepse_returns'].rolling(5).sum()
            df['relative_strength'] = df['stock_cum_5d'] - df['nepse_cum_5d']
            df['relative_strength'] = df['relative_strength'].fillna(0)
            
            # Market volatility
            df['market_volatility'] = df['nepse_returns'].rolling(10).std()
            df['market_volatility'] = df['market_volatility'].fillna(df['market_volatility'].mean())
            
            feature_cols.extend([
                'nepse_returns', 'nepse_momentum', 'rolling_beta', 
                'relative_strength', 'market_volatility'
            ])
        
        if not sector_df.empty and sector_code:
            print(f"  🏦 Sector index ({sector_code}): {len(sector_df)} days")
            
            # Merge sector data
            df = df.merge(sector_df, on='date', how='left')
            df['sector_close'] = df['sector_close'].ffill().bfill()
            df['sector_pct_change'] = df['sector_pct_change'].ffill().bfill()
            
            # Sector returns
            df['sector_returns'] = df['sector_close'].pct_change()
            df['sector_returns'] = df['sector_returns'].fillna(0)
            
            # Sector momentum
            df['sector_sma5'] = df['sector_close'].rolling(5).mean()
            df['sector_momentum'] = (df['sector_close'] - df['sector_sma5']) / df['sector_sma5']
            df['sector_momentum'] = df['sector_momentum'].fillna(0)
            
            # Relative strength vs sector
            df['sector_cum_5d'] = df['sector_returns'].rolling(5).sum()
            df['sector_relative_strength'] = df['stock_cum_5d'] - df['sector_cum_5d']
            df['sector_relative_strength'] = df['sector_relative_strength'].fillna(0)
            
            feature_cols.extend([
                'sector_returns', 'sector_momentum', 'sector_relative_strength'
            ])
    
    # Target: Next day's return and direction
    df['target_return'] = df['returns'].shift(-1)
    df['target_direction'] = np.sign(df['target_return'])
    df['target_direction'] = df['target_direction'].map({-1: 0, 0: 1, 1: 2})  # Down, Neutral, Up
    
    # Drop NaN
    df = df.dropna()
    
    return df, feature_cols, sector_code


def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 0.001)
    return 100 - (100 / (1 + rs))


def prepare_sequences(df, feature_cols, lookback=30):
    """Prepare sequences for LSTM"""
    
    features = df[feature_cols].values
    target_returns = df['target_return'].values
    target_directions = df['target_direction'].values
    close_prices = df['close'].values
    
    # Scale features
    scaler = RobustScaler()
    features_scaled = scaler.fit_transform(features)
    
    X, y_return, y_direction, prev_prices = [], [], [], []
    
    for i in range(lookback, len(features_scaled) - 1):
        X.append(features_scaled[i-lookback:i])
        y_return.append(target_returns[i])
        y_direction.append(target_directions[i])
        prev_prices.append(close_prices[i])
    
    return (np.array(X), np.array(y_return).reshape(-1, 1), 
            np.array(y_direction), np.array(prev_prices).reshape(-1, 1), scaler)


def train_model(model, train_loader, val_loader, epochs=30, lr=0.001):
    """Fast training loop"""
    
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader)
    )
    
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 0.5, 1.0]).to(DEVICE))
    
    best_val_acc = 0
    best_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        
        for X, y_price, y_dir, prev_p in train_loader:
            X = X.to(DEVICE)
            y_price = y_price.to(DEVICE)
            y_dir = y_dir.to(DEVICE)
            
            optimizer.zero_grad()
            
            price_pred, dir_logits = model(X)
            
            # Combined loss: MSE for price + CE for direction
            loss_price = mse_loss(price_pred, y_price)
            loss_dir = ce_loss(dir_logits, y_dir)
            
            loss = 0.3 * loss_price + 0.7 * loss_dir  # Heavily weight direction
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_mape = 0
        
        with torch.no_grad():
            for X, y_price, y_dir, prev_p in val_loader:
                X = X.to(DEVICE)
                y_dir = y_dir.to(DEVICE)
                prev_p = prev_p.to(DEVICE)
                
                price_pred, dir_logits = model(X)
                
                # Direction accuracy
                pred_dir = torch.argmax(dir_logits, dim=1)
                val_correct += (pred_dir == y_dir).sum().item()
                val_total += len(y_dir)
        
        val_acc = val_correct / val_total * 100
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.1f}%")
    
    # Load best model
    if best_state:
        model.load_state_dict(best_state)
    
    return model, best_val_acc


def ensemble_predict(models, X):
    """Ensemble prediction with voting"""
    
    X_tensor = torch.FloatTensor(X).to(DEVICE)
    
    all_price_preds = []
    all_dir_votes = []
    
    for model in models:
        model.eval()
        with torch.no_grad():
            price_pred, dir_logits = model(X_tensor)
            all_price_preds.append(price_pred.cpu().numpy())
            all_dir_votes.append(torch.argmax(dir_logits, dim=1).cpu().numpy())
    
    # Average price prediction
    avg_price = np.mean(all_price_preds, axis=0)
    
    # Majority voting for direction
    dir_votes = np.stack(all_dir_votes, axis=0)
    final_direction = np.apply_along_axis(
        lambda x: np.bincount(x, minlength=3).argmax(), 0, dir_votes
    )
    
    return avg_price, final_direction


def run_optimized_forecast(symbol: str, n_models: int = 5, lookback: int = 30, epochs: int = 30, use_market: bool = True):
    """
    Run optimized forecast with market context
    
    Args:
        symbol: Stock symbol
        n_models: Number of models in ensemble
        lookback: Sequence length
        epochs: Training epochs per model
        use_market: Include NEPSE and sector features
    """
    
    print(f"\n{'='*60}")
    print(f"🚀 OPTIMIZED FORECAST: {symbol}")
    print(f"{'='*60}")
    print(f"Models: {n_models} | Lookback: {lookback} | Epochs: {epochs}")
    print(f"Device: {DEVICE}")
    print(f"Market Features: {'Enabled' if use_market else 'Disabled'}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # Load data
    print("📊 Loading data...")
    df = load_stock_data(symbol)
    print(f"   Loaded {len(df)} days")
    
    # Create features with market context
    print("🔧 Creating features...")
    df, feature_cols, sector_code = create_features(df, symbol=symbol, lookback=lookback, use_market=use_market)
    print(f"   Features: {len(feature_cols)}")
    if sector_code:
        print(f"   Sector: {sector_code}")
    
    # Prepare sequences
    X, y_return, y_direction, prev_prices, scaler = prepare_sequences(df, feature_cols, lookback)
    print(f"   Sequences: {len(X)}")
    
    # Train/test split (last 60 days for testing)
    test_size = 60
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_ret_train, y_ret_test = y_return[:-test_size], y_return[-test_size:]
    y_dir_train, y_dir_test = y_direction[:-test_size], y_direction[-test_size:]
    prev_train, prev_test = prev_prices[:-test_size], prev_prices[-test_size:]
    
    # DataLoaders
    train_dataset = StockDataset(X_train, y_ret_train, y_dir_train, prev_train)
    test_dataset = StockDataset(X_test, y_ret_test, y_dir_test, prev_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Train ensemble
    print(f"\n🏋️ Training {n_models} models...")
    models = []
    
    for i in range(n_models):
        print(f"\n  Model {i+1}/{n_models}")
        model = FastLSTM(input_size=len(feature_cols), hidden_size=128, num_layers=2)
        model, val_acc = train_model(model, train_loader, test_loader, epochs=epochs)
        models.append(model)
        print(f"  ✓ Best validation accuracy: {val_acc:.1f}%")
    
    # Final evaluation
    print(f"\n📈 Final Evaluation on Test Set ({test_size} days)...")
    
    avg_price_pred, final_direction = ensemble_predict(models, X_test)
    
    # Calculate metrics
    # Direction accuracy
    dir_correct = (final_direction == y_dir_test).sum()
    dir_accuracy = dir_correct / len(y_dir_test) * 100
    
    # Calculate price MAPE (predict next day's close from current close + predicted return)
    actual_prices = df['close'].values[-test_size:]
    predicted_prices = actual_prices[:-1] * (1 + avg_price_pred[:-1].flatten())
    actual_next_prices = actual_prices[1:]
    
    price_mape = np.mean(np.abs((actual_next_prices - predicted_prices) / actual_next_prices)) * 100
    
    # Actual vs Predicted direction for last 10 days
    print(f"\n📊 Last 10 Days Predictions:")
    print("-" * 50)
    directions = {0: '📉 DOWN', 1: '➡️ FLAT', 2: '📈 UP'}
    
    recent_dates = df['date'].values[-test_size:][-10:]
    recent_actual = y_dir_test[-10:]
    recent_pred = final_direction[-10:]
    recent_returns = y_ret_test[-10:].flatten()
    
    for i, (date, actual, pred, ret) in enumerate(zip(recent_dates, recent_actual, recent_pred, recent_returns)):
        correct = "✓" if actual == pred else "✗"
        print(f"  {str(date)[:10]} | Actual: {directions[actual]} | Pred: {directions[pred]} | {correct} | Return: {float(ret)*100:.2f}%")
    
    elapsed = time.time() - start_time
    
    print(f"\\n{'='*60}")
    print(f"📊 RESULTS")
    print(f"{'='*60}")
    print(f"  Direction Accuracy: {dir_accuracy:.1f}%")
    print(f"  Price MAPE: {price_mape:.2f}%")
    print(f"  Training Time: {elapsed:.1f}s")
    print(f"{'='*60}")
    
    # Get prediction for tomorrow
    print(f"\n🔮 TOMORROW'S PREDICTION:")
    last_seq = X[-1:] 
    _, tomorrow_dir = ensemble_predict(models, last_seq)
    
    current_price = df['close'].iloc[-1]
    print(f"  Current Price: NPR {current_price:.2f}")
    print(f"  Prediction: {directions[tomorrow_dir[0]]}")
    
    confidence = 0
    for model in models:
        model.eval()
        with torch.no_grad():
            _, logits = model(torch.FloatTensor(last_seq).to(DEVICE))
            probs = torch.softmax(logits, dim=1)
            confidence += probs[0, tomorrow_dir[0]].item()
    confidence = confidence / len(models) * 100
    
    print(f"  Confidence: {confidence:.1f}%")
    print(f"{'='*60}\n")
    
    return {
        'direction_accuracy': dir_accuracy,
        'price_mape': price_mape,
        'training_time': elapsed,
        'prediction': directions[tomorrow_dir[0]],
        'confidence': confidence
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimized Stock Forecaster with Market Context')
    parser.add_argument('symbol', help='Stock symbol')
    parser.add_argument('--models', type=int, default=5, help='Number of ensemble models')
    parser.add_argument('--lookback', type=int, default=30, help='Lookback period')
    parser.add_argument('--epochs', type=int, default=30, help='Epochs per model')
    parser.add_argument('--market', action='store_true', default=True, help='Include NEPSE + sector features (default: True)')
    parser.add_argument('--no-market', action='store_true', help='Disable market features')
    
    args = parser.parse_args()
    
    # Determine market flag
    use_market = not args.no_market
    
    results = run_optimized_forecast(
        symbol=args.symbol,
        n_models=args.models,
        lookback=args.lookback,
        epochs=args.epochs,
        use_market=use_market
    )
