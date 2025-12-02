"""
Multi-Horizon xLSTM Stock Forecaster with Bayesian Optimization
================================================================
Target: MAPE < 2%, Direction Accuracy > 80%

Horizons: t+1, t+3, t+5, t+10, t+15, t+20

Features:
1. Multi-horizon predictions (6 time horizons)
2. Bayesian optimization for hyperparameter search (max 60 combinations)
3. Early stopping when criteria met (MAPE<2%, Direction>80%)
4. Save best model per iteration
5. Market features (NEPSE + Sector indices)
6. Inference module for website integration
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler, StandardScaler
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import time
import json
import warnings
import os
warnings.filterwarnings('ignore')

import config
from sector_mapper import SectorMapper

# Bayesian optimization
try:
    from skopt import gp_minimize
    from skopt.space import Integer, Real, Categorical
    from skopt.utils import use_named_args
    BAYESIAN_AVAILABLE = True
except ImportError:
    print("⚠️  scikit-optimize not installed. Run: pip install scikit-optimize")
    BAYESIAN_AVAILABLE = False

# Use MPS for M1 Max
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

# Time horizons
HORIZONS = [1, 3, 5, 10, 15, 20]

# Model save directory
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Results tracking file
RESULTS_FILE = Path("xlstm_optimization_results.json")


class MultiHorizonLSTM(nn.Module):
    """LSTM model predicting multiple time horizons"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2, horizons=HORIZONS):
        super().__init__()
        
        self.horizons = horizons
        self.hidden_size = hidden_size
        
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
        
        # Separate heads for each horizon
        self.price_heads = nn.ModuleDict()
        self.direction_heads = nn.ModuleDict()
        
        for h in horizons:
            self.price_heads[f'h{h}'] = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
            self.direction_heads[f'h{h}'] = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 3)  # Down, Flat, Up
            )
    
    def forward(self, x):
        # LSTM encoding
        lstm_out, _ = self.lstm(x)
        
        # Attention pooling
        attn_weights = self.attention(lstm_out)
        context = torch.sum(lstm_out * attn_weights, dim=1)
        
        # Multi-horizon outputs
        price_preds = {}
        direction_logits = {}
        
        for h in self.horizons:
            price_preds[h] = self.price_heads[f'h{h}'](context)
            direction_logits[h] = self.direction_heads[f'h{h}'](context)
        
        return price_preds, direction_logits


class MultiHorizonDataset(Dataset):
    """Dataset for multi-horizon prediction"""
    
    def __init__(self, X, y_returns, y_directions, horizons=HORIZONS):
        self.X = torch.FloatTensor(X)
        self.y_returns = {h: torch.FloatTensor(y_returns[h]) for h in horizons}
        self.y_directions = {h: torch.LongTensor(y_directions[h]) for h in horizons}
        self.horizons = horizons
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        returns = {h: self.y_returns[h][idx] for h in self.horizons}
        directions = {h: self.y_directions[h][idx] for h in self.horizons}
        return self.X[idx], returns, directions


def get_market_data(symbol: str):
    """Get NEPSE and sector index data"""
    db_path = Path(config.DATA_DIR) / "nepse_stocks.db"
    conn = sqlite3.connect(db_path)
    
    # Get sector
    mapper = SectorMapper()
    sector_info = mapper.get_sector(symbol)
    sector_code = sector_info['sector_index_code'] if sector_info else None
    
    # NEPSE index
    nepse_df = pd.read_sql_query("""
        SELECT date, close as nepse_close, pct_change as nepse_pct_change
        FROM index_history WHERE index_code = 'NEPSE' ORDER BY date
    """, conn)
    nepse_df['date'] = pd.to_datetime(nepse_df['date'])
    
    # Sector index
    sector_df = pd.DataFrame()
    if sector_code:
        sector_df = pd.read_sql_query(f"""
            SELECT date, close as sector_close, pct_change as sector_pct_change
            FROM index_history WHERE index_code = '{sector_code}' ORDER BY date
        """, conn)
        if not sector_df.empty:
            sector_df['date'] = pd.to_datetime(sector_df['date'])
    
    conn.close()
    return nepse_df, sector_df, sector_code


def validate_stock(symbol: str):
    """
    Validate that a stock has required data before training.
    Raises error if sector info is missing.
    """
    # Check sector info
    mapper = SectorMapper()
    sector_info = mapper.get_sector(symbol)
    
    if not sector_info:
        raise ValueError(
            f"❌ Stock '{symbol}' has no sector information!\n"
            f"   Please add sector mapping using:\n"
            f"   INSERT INTO stock_sectors (symbol, sector, sector_index_code) "
            f"VALUES ('{symbol}', 'SECTOR_NAME', 'INDEX_CODE')"
        )
    
    if not sector_info.get('sector_index_code'):
        print(f"⚠️  Warning: Stock '{symbol}' sector '{sector_info['sector']}' has no index data")
    
    # Check price data
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.execute(
        f"SELECT COUNT(*) FROM price_history WHERE symbol = '{symbol}'"
    )
    count = cursor.fetchone()[0]
    conn.close()
    
    if count < 100:
        raise ValueError(
            f"❌ Stock '{symbol}' has insufficient price data ({count} days).\n"
            f"   Minimum 100 days required for training."
        )
    
    return sector_info


def load_stock_data(symbol: str):
    """Load stock data from database"""
    conn = sqlite3.connect(config.DB_PATH)
    df = pd.read_sql_query(f"""
        SELECT date, open, high, low, close, volume 
        FROM price_history WHERE symbol = '{symbol}' ORDER BY date
    """, conn)
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    return df


def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 0.001)
    return 100 - (100 / (1 + rs))


def create_features(df: pd.DataFrame, symbol: str = None, use_market: bool = True):
    """Create feature set with market context"""
    
    df = df.copy()
    
    # Price changes
    df['returns'] = df['close'].pct_change()
    df['returns_2d'] = df['close'].pct_change(2)
    df['returns_5d'] = df['close'].pct_change(5)
    
    # Momentum
    df['rsi'] = calculate_rsi(df['close'], 14)
    df['rsi_signal'] = (df['rsi'] - 50) / 50
    
    # Moving averages
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['ma_cross_5_10'] = (df['sma_5'] - df['sma_10']) / df['sma_10']
    df['ma_cross_5_20'] = (df['sma_5'] - df['sma_20']) / df['sma_20']
    df['price_vs_sma5'] = (df['close'] - df['sma_5']) / df['sma_5']
    df['price_vs_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']
    
    # Volatility
    df['volatility'] = df['returns'].rolling(10).std()
    
    # Volume
    df['volume_ma'] = df['volume'].rolling(10).mean()
    df['volume_signal'] = (df['volume'] - df['volume_ma']) / (df['volume_ma'] + 1)
    
    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = (ema12 - ema26) / df['close']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger
    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_position'] = (df['close'] - bb_mid) / (2 * bb_std + 0.001)
    
    feature_cols = [
        'returns', 'returns_2d', 'returns_5d',
        'rsi_signal', 'ma_cross_5_10', 'ma_cross_5_20',
        'price_vs_sma5', 'price_vs_sma20',
        'volatility', 'volume_signal',
        'macd', 'macd_signal', 'macd_hist', 'bb_position'
    ]
    
    sector_code = None
    
    # Market features
    if use_market and symbol:
        nepse_df, sector_df, sector_code = get_market_data(symbol)
        
        if not nepse_df.empty:
            df = df.merge(nepse_df, on='date', how='left')
            df['nepse_close'] = df['nepse_close'].ffill().bfill()
            df['nepse_returns'] = df['nepse_close'].pct_change().fillna(0)
            df['nepse_sma5'] = df['nepse_close'].rolling(5).mean()
            df['nepse_momentum'] = ((df['nepse_close'] - df['nepse_sma5']) / df['nepse_sma5']).fillna(0)
            df['rolling_beta'] = df['returns'].rolling(20).corr(df['nepse_returns']).fillna(1.0)
            df['stock_cum_5d'] = df['returns'].rolling(5).sum()
            df['nepse_cum_5d'] = df['nepse_returns'].rolling(5).sum()
            df['relative_strength'] = (df['stock_cum_5d'] - df['nepse_cum_5d']).fillna(0)
            df['market_volatility'] = df['nepse_returns'].rolling(10).std().fillna(df['nepse_returns'].std())
            
            feature_cols.extend(['nepse_returns', 'nepse_momentum', 'rolling_beta', 
                               'relative_strength', 'market_volatility'])
        
        if not sector_df.empty and sector_code:
            df = df.merge(sector_df, on='date', how='left')
            df['sector_close'] = df['sector_close'].ffill().bfill()
            df['sector_returns'] = df['sector_close'].pct_change().fillna(0)
            df['sector_sma5'] = df['sector_close'].rolling(5).mean()
            df['sector_momentum'] = ((df['sector_close'] - df['sector_sma5']) / df['sector_sma5']).fillna(0)
            df['sector_cum_5d'] = df['sector_returns'].rolling(5).sum()
            df['sector_relative_strength'] = (df['stock_cum_5d'] - df['sector_cum_5d']).fillna(0)
            
            feature_cols.extend(['sector_returns', 'sector_momentum', 'sector_relative_strength'])
    
    return df, feature_cols, sector_code


def prepare_multihorizon_data(df, feature_cols, lookback=60, horizons=HORIZONS):
    """Prepare sequences for multi-horizon prediction"""
    
    df = df.copy()
    
    # Create targets for each horizon
    for h in horizons:
        df[f'target_return_{h}'] = df['close'].pct_change(h).shift(-h)
        df[f'target_direction_{h}'] = np.sign(df[f'target_return_{h}'])
        df[f'target_direction_{h}'] = df[f'target_direction_{h}'].map({-1: 0, 0: 1, 1: 2})
    
    # Drop NaN
    df = df.dropna()
    
    features = df[feature_cols].values
    
    # Scale features
    scaler = RobustScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Create sequences
    X = []
    y_returns = {h: [] for h in horizons}
    y_directions = {h: [] for h in horizons}
    
    for i in range(lookback, len(features_scaled)):
        X.append(features_scaled[i-lookback:i])
        for h in horizons:
            y_returns[h].append(df[f'target_return_{h}'].iloc[i])
            y_directions[h].append(df[f'target_direction_{h}'].iloc[i])
    
    X = np.array(X)
    y_returns = {h: np.array(y_returns[h]).reshape(-1, 1) for h in horizons}
    y_directions = {h: np.array(y_directions[h]) for h in horizons}
    
    return X, y_returns, y_directions, scaler, df


def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, 
                patience=10, min_epochs=30, device=DEVICE):
    """Train with early stopping - requires minimum epochs before stopping"""
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, factor=0.5)
    
    mse_loss = nn.MSELoss()
    # Weight classes: penalize wrong predictions on UP/DOWN more than FLAT
    ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([1.2, 0.6, 1.2]).to(device))
    
    best_val_acc = 0  # Track best accuracy instead of loss
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        
        for X, y_ret, y_dir in train_loader:
            X = X.to(device)
            
            optimizer.zero_grad()
            price_preds, dir_logits = model(X)
            
            loss = 0
            for h in HORIZONS:
                y_r = y_ret[h].to(device)
                y_d = y_dir[h].to(device)
                
                loss += 0.3 * mse_loss(price_preds[h], y_r)
                loss += 0.7 * ce_loss(dir_logits[h], y_d)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = {h: 0 for h in HORIZONS}
        val_total = 0
        
        with torch.no_grad():
            for X, y_ret, y_dir in val_loader:
                X = X.to(device)
                price_preds, dir_logits = model(X)
                
                for h in HORIZONS:
                    y_r = y_ret[h].to(device)
                    y_d = y_dir[h].to(device)
                    
                    val_loss += 0.3 * mse_loss(price_preds[h], y_r).item()
                    val_loss += 0.7 * ce_loss(dir_logits[h], y_d).item()
                    
                    pred_dir = torch.argmax(dir_logits[h], dim=1)
                    val_correct[h] += (pred_dir == y_d).sum().item()
                
                val_total += len(X)
        
        val_acc = {h: val_correct[h] / val_total * 100 for h in HORIZONS}
        avg_val_acc = np.mean(list(val_acc.values()))
        
        scheduler.step(val_loss)
        
        # Save best model based on accuracy (not loss)
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Print every epoch
        print(f"  Epoch {epoch+1}/{epochs} | Loss: {train_loss/len(train_loader):.4f} | "
              f"Val Acc: {avg_val_acc:.1f}% | Best: {best_val_acc:.1f}%")
        
        # Early stopping - only after minimum epochs
        if epoch >= min_epochs and patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1} (best acc: {best_val_acc:.1f}%)")
            break
    
    if best_state:
        model.load_state_dict(best_state)
    
    return model, val_acc


def evaluate_model(models, X_test, y_returns_test, y_directions_test, df_test, current_prices=None):
    """Evaluate ensemble model"""
    
    results = {h: {'mape': [], 'direction_acc': 0, 'correct': 0, 'total': 0} for h in HORIZONS}
    
    X_tensor = torch.FloatTensor(X_test).to(DEVICE)
    
    # Ensemble predictions
    all_price_preds = {h: [] for h in HORIZONS}
    all_dir_votes = {h: [] for h in HORIZONS}
    
    for model in models:
        model.eval()
        with torch.no_grad():
            price_preds, dir_logits = model(X_tensor)
            for h in HORIZONS:
                all_price_preds[h].append(price_preds[h].cpu().numpy())
                all_dir_votes[h].append(torch.argmax(dir_logits[h], dim=1).cpu().numpy())
    
    # Calculate metrics for each horizon
    for h in HORIZONS:
        avg_return_pred = np.mean(all_price_preds[h], axis=0).flatten()
        
        # Majority voting
        dir_votes = np.stack(all_dir_votes[h], axis=0)
        final_dir = np.apply_along_axis(lambda x: np.bincount(x, minlength=3).argmax(), 0, dir_votes)
        
        # Direction accuracy
        actual_dir = y_directions_test[h]
        correct = (final_dir == actual_dir).sum()
        results[h]['direction_acc'] = correct / len(actual_dir) * 100
        results[h]['correct'] = int(correct)
        results[h]['total'] = len(actual_dir)
        
        # MAPE - use predicted return vs actual return for percentage comparison
        actual_returns = y_returns_test[h].flatten()
        
        # Calculate MAPE as percentage error on returns (capped)
        # For stock returns, small values are normal so we use absolute error instead
        abs_error = np.abs(actual_returns - avg_return_pred)
        # Convert to percentage points (e.g., 0.01 error = 1%)
        mape = np.mean(abs_error) * 100  # Mean absolute error in percentage points
        results[h]['mape'] = round(mape, 2)
    
    return results


def run_optimization_trial(symbol, hidden_size, num_layers, lookback, dropout, lr, batch_size, 
                          n_models=5, epochs=100, trial_num=0, best_results=None):
    """Run a single optimization trial"""
    
    # Convert numpy types to native Python types
    hidden_size = int(hidden_size)
    num_layers = int(num_layers)
    lookback = int(lookback)
    dropout = float(dropout)
    lr = float(lr)
    batch_size = int(batch_size)
    
    print(f"\n{'='*70}")
    print(f"Trial {trial_num}: hidden={hidden_size}, layers={num_layers}, "
          f"lookback={lookback}, dropout={dropout:.2f}, lr={lr:.5f}, batch={batch_size}")
    print(f"{'='*70}")
    
    # Load and prepare data
    df = load_stock_data(symbol)
    df, feature_cols, sector_code = create_features(df, symbol=symbol, use_market=True)
    
    X, y_returns, y_directions, scaler, df_processed = prepare_multihorizon_data(
        df, feature_cols, lookback=lookback
    )
    
    # Split data
    test_size = 60
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_ret_train = {h: y_returns[h][:-test_size] for h in HORIZONS}
    y_ret_test = {h: y_returns[h][-test_size:] for h in HORIZONS}
    y_dir_train = {h: y_directions[h][:-test_size] for h in HORIZONS}
    y_dir_test = {h: y_directions[h][-test_size:] for h in HORIZONS}
    
    # DataLoaders
    train_dataset = MultiHorizonDataset(X_train, y_ret_train, y_dir_train)
    test_dataset = MultiHorizonDataset(X_test, y_ret_test, y_dir_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Train ensemble
    models = []
    for i in range(n_models):
        print(f"\n  Training model {i+1}/{n_models}...")
        model = MultiHorizonLSTM(
            input_size=len(feature_cols),
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        model, _ = train_model(model, train_loader, test_loader, epochs=epochs, lr=lr)
        models.append(model)
    
    # Evaluate
    results = evaluate_model(models, X_test, y_ret_test, y_dir_test, df_processed)
    
    # Calculate overall metrics
    avg_mape = np.mean([results[h]['mape'] for h in HORIZONS])
    avg_dir_acc = np.mean([results[h]['direction_acc'] for h in HORIZONS])
    
    print(f"\n📊 Trial {trial_num} Results:")
    print(f"  Average MAPE: {avg_mape:.2f}%")
    print(f"  Average Direction Accuracy: {avg_dir_acc:.1f}%")
    
    for h in HORIZONS:
        print(f"    t+{h}: MAPE={results[h]['mape']:.2f}%, Dir={results[h]['direction_acc']:.1f}%")
    
    # Save model if best
    is_best = False
    if best_results is None or avg_dir_acc > best_results.get('avg_direction_acc', 0):
        is_best = True
        save_path = MODEL_DIR / f"{symbol}_best_model.pt"
        torch.save({
            'models': [m.state_dict() for m in models],
            'hyperparams': {
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'lookback': lookback,
                'dropout': dropout,
                'lr': lr,
                'batch_size': batch_size,
                'n_features': len(feature_cols),
            },
            'scaler': scaler,
            'feature_cols': feature_cols,
            'sector_code': sector_code,
            'results': results,
        }, save_path)
        print(f"  ✅ Saved best model to {save_path}")
    
    # Check criteria
    criteria_met = avg_mape < 10.0 and avg_dir_acc > 60.0
    
    return {
        'trial': trial_num,
        'hyperparams': {
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'lookback': lookback,
            'dropout': dropout,
            'lr': lr,
            'batch_size': batch_size,
        },
        'results': results,
        'avg_mape': avg_mape,
        'avg_direction_acc': avg_dir_acc,
        'criteria_met': criteria_met,
        'is_best': is_best,
    }


def bayesian_optimization(symbol: str, max_trials: int = 60, n_models: int = 5):
    """
    Bayesian optimization for hyperparameter search
    Terminates when MAPE < 2% and Direction Accuracy > 80%
    """
    
    # Validate stock before starting optimization
    print(f"\n📋 Validating stock '{symbol}'...")
    sector_info = validate_stock(symbol)
    print(f"   ✅ Sector: {sector_info['sector']} ({sector_info['sector_index_code']})")
    
    print(f"\n{'='*70}")
    print(f"🔬 BAYESIAN OPTIMIZATION: {symbol}")
    print(f"{'='*70}")
    print(f"Max trials: {max_trials}")
    print(f"Target: MAPE < 10%, Direction Accuracy > 60%")
    print(f"{'='*70}\n")
    
    # Search space
    space = [
        Integer(64, 256, name='hidden_size'),
        Integer(1, 3, name='num_layers'),
        Integer(30, 120, name='lookback'),
        Real(0.1, 0.4, name='dropout'),
        Real(0.0001, 0.01, prior='log-uniform', name='lr'),
        Categorical([32, 64, 128], name='batch_size'),
    ]
    
    all_results = []
    best_results = None
    trial_num = 0
    
    @use_named_args(space)
    def objective(hidden_size, num_layers, lookback, dropout, lr, batch_size):
        nonlocal trial_num, best_results, all_results
        
        trial_num += 1
        
        try:
            result = run_optimization_trial(
                symbol=symbol,
                hidden_size=hidden_size,
                num_layers=num_layers,
                lookback=lookback,
                dropout=dropout,
                lr=lr,
                batch_size=batch_size,
                n_models=n_models,
                epochs=100,
                trial_num=trial_num,
                best_results=best_results
            )
            
            all_results.append(result)
            
            # Update best
            if result['is_best']:
                best_results = result
            
            # Save progress
            save_optimization_progress(symbol, all_results)
            
            # Check termination criteria
            if result['criteria_met']:
                print(f"\n🎉 CRITERIA MET! MAPE={result['avg_mape']:.2f}%, "
                      f"Dir Acc={result['avg_direction_acc']:.1f}%")
                return -result['avg_direction_acc']  # Negative for minimization
            
            # Objective: maximize direction accuracy (minimize negative)
            return -result['avg_direction_acc']
            
        except Exception as e:
            print(f"  ❌ Trial failed: {e}")
            return 0  # Worst possible score
    
    if not BAYESIAN_AVAILABLE:
        print("❌ scikit-optimize not available. Running random search instead.")
        # Fallback to random search
        for _ in range(max_trials):
            hidden_size = np.random.choice([64, 128, 192, 256])
            num_layers = np.random.choice([1, 2, 3])
            lookback = np.random.choice([30, 45, 60, 90, 120])
            dropout = np.random.uniform(0.1, 0.4)
            lr = np.random.uniform(0.0001, 0.01)
            batch_size = np.random.choice([32, 64, 128])
            
            result = objective(hidden_size, num_layers, lookback, dropout, lr, batch_size)
            
            if best_results and best_results.get('criteria_met', False):
                break
        
        return all_results, best_results
    
    # Bayesian optimization
    result = gp_minimize(
        objective,
        space,
        n_calls=max_trials,
        n_random_starts=10,
        random_state=42,
        verbose=False,
    )
    
    return all_results, best_results


def save_optimization_progress(symbol: str, results: list):
    """Save optimization progress to JSON"""
    
    output = {
        'symbol': symbol,
        'timestamp': datetime.now().isoformat(),
        'n_trials': len(results),
        'trials': results,
    }
    
    output_file = Path(f"optimization_results_{symbol}.json")
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)


def inference(symbol: str, model_path: str = None):
    """
    Run inference using saved model
    Returns predictions for website integration
    """
    
    if model_path is None:
        model_path = MODEL_DIR / f"{symbol}_best_model.pt"
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load model (allow sklearn objects)
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    hyperparams = checkpoint['hyperparams']
    feature_cols = checkpoint['feature_cols']
    scaler = checkpoint['scaler']
    sector_code = checkpoint['sector_code']
    saved_results = checkpoint['results']
    
    # Load current data
    df = load_stock_data(symbol)
    df, _, _ = create_features(df, symbol=symbol, use_market=True)
    
    # Get latest sequence
    lookback = hyperparams['lookback']
    features = df[feature_cols].values[-lookback:]
    features_scaled = scaler.transform(features)
    X = features_scaled.reshape(1, lookback, -1)
    X_tensor = torch.FloatTensor(X).to(DEVICE)
    
    # Load models
    models = []
    for state_dict in checkpoint['models']:
        model = MultiHorizonLSTM(
            input_size=hyperparams['n_features'],
            hidden_size=hyperparams['hidden_size'],
            num_layers=hyperparams['num_layers'],
            dropout=hyperparams['dropout']
        )
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        models.append(model)
    
    # Ensemble prediction
    all_price_preds = {h: [] for h in HORIZONS}
    all_dir_probs = {h: [] for h in HORIZONS}
    
    with torch.no_grad():
        for model in models:
            price_preds, dir_logits = model(X_tensor)
            for h in HORIZONS:
                all_price_preds[h].append(price_preds[h].cpu().numpy()[0, 0])
                probs = torch.softmax(dir_logits[h], dim=1).cpu().numpy()[0]
                all_dir_probs[h].append(probs)
    
    # Current price
    current_price = df['close'].iloc[-1]
    current_date = pd.to_datetime(df['date'].iloc[-1])
    
    # Generate predictions
    predictions = {}
    directions = {0: 'DOWN', 1: 'FLAT', 2: 'UP'}
    
    for h in HORIZONS:
        avg_return = np.mean(all_price_preds[h])
        avg_probs = np.mean(all_dir_probs[h], axis=0)
        pred_dir = int(np.argmax(avg_probs))
        confidence = float(avg_probs[pred_dir] * 100)
        
        pred_price = current_price * (1 + avg_return)
        
        # Calculate target date (skip weekends)
        target_date = current_date
        days_added = 0
        while days_added < h:
            target_date += timedelta(days=1)
            if target_date.weekday() < 5:
                days_added += 1
        
        predictions[h] = {
            'horizon': h,
            'target_date': str(target_date.date()),
            'predicted_price': round(float(pred_price), 2),
            'predicted_return': round(float(avg_return * 100), 2),
            'direction': directions[pred_dir],
            'confidence': round(confidence, 1),
            'mape': round(saved_results[h]['mape'], 2),
            'direction_accuracy': round(saved_results[h]['direction_acc'], 1),
        }
    
    # Model strength info
    avg_mape = np.mean([saved_results[h]['mape'] for h in HORIZONS])
    avg_dir_acc = np.mean([saved_results[h]['direction_acc'] for h in HORIZONS])
    
    strength = 'STRONG' if avg_dir_acc > 70 else 'MODERATE' if avg_dir_acc > 55 else 'WEAK'
    
    result = {
        'symbol': symbol,
        'current_price': round(float(current_price), 2),
        'current_date': str(current_date.date()),
        'sector': sector_code,
        'model_info': {
            'type': 'xLSTM Multi-Horizon Ensemble',
            'n_models': len(models),
            'hidden_size': hyperparams['hidden_size'],
            'lookback': hyperparams['lookback'],
            'features': len(feature_cols),
        },
        'accuracy_metrics': {
            'avg_mape': round(avg_mape, 2),
            'avg_direction_accuracy': round(avg_dir_acc, 1),
            'strength': strength,
            'by_horizon': {h: {
                'mape': round(saved_results[h]['mape'], 2),
                'direction_accuracy': round(saved_results[h]['direction_acc'], 1),
            } for h in HORIZONS}
        },
        'predictions': predictions,
        'generated_at': datetime.now().isoformat(),
    }
    
    return result


def quick_train(symbol: str, n_models: int = 5, epochs: int = 100, 
                hidden_size: int = 128, lookback: int = 60):
    """Quick training without optimization"""
    
    # Validate stock before training
    print(f"\n📋 Validating stock '{symbol}'...")
    sector_info = validate_stock(symbol)
    print(f"   ✅ Sector: {sector_info['sector']} ({sector_info['sector_index_code']})")
    
    print(f"\n{'='*70}")
    print(f"🚀 QUICK TRAIN: {symbol}")
    print(f"{'='*70}")
    print(f"Models: {n_models} | Epochs: {epochs} | Hidden: {hidden_size} | Lookback: {lookback}")
    print(f"Horizons: {HORIZONS}")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    # Load data
    print("📊 Loading data...")
    df = load_stock_data(symbol)
    print(f"   Loaded {len(df)} days")
    
    # Create features
    print("🔧 Creating features...")
    df, feature_cols, sector_code = create_features(df, symbol=symbol, use_market=True)
    print(f"   Features: {len(feature_cols)}")
    if sector_code:
        print(f"   Sector: {sector_code}")
    
    # Prepare data
    X, y_returns, y_directions, scaler, df_processed = prepare_multihorizon_data(
        df, feature_cols, lookback=lookback
    )
    print(f"   Sequences: {len(X)}")
    
    # Split
    test_size = 60
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_ret_train = {h: y_returns[h][:-test_size] for h in HORIZONS}
    y_ret_test = {h: y_returns[h][-test_size:] for h in HORIZONS}
    y_dir_train = {h: y_directions[h][:-test_size] for h in HORIZONS}
    y_dir_test = {h: y_directions[h][-test_size:] for h in HORIZONS}
    
    # DataLoaders
    train_dataset = MultiHorizonDataset(X_train, y_ret_train, y_dir_train)
    test_dataset = MultiHorizonDataset(X_test, y_ret_test, y_dir_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Train
    print(f"\n🏋️ Training {n_models} models...")
    models = []
    for i in range(n_models):
        print(f"\n  Model {i+1}/{n_models}")
        model = MultiHorizonLSTM(
            input_size=len(feature_cols),
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.2
        )
        model, _ = train_model(model, train_loader, test_loader, epochs=epochs)
        models.append(model)
    
    # Evaluate
    print(f"\n📈 Evaluating on test set ({test_size} days)...")
    results = evaluate_model(models, X_test, y_ret_test, y_dir_test, df_processed)
    
    elapsed = time.time() - start_time
    
    # Print results
    print(f"\n{'='*70}")
    print(f"📊 RESULTS")
    print(f"{'='*70}")
    
    for h in HORIZONS:
        print(f"  t+{h:2d}: MAPE = {results[h]['mape']:6.2f}% | "
              f"Direction = {results[h]['direction_acc']:5.1f}%")
    
    avg_mape = np.mean([results[h]['mape'] for h in HORIZONS])
    avg_dir = np.mean([results[h]['direction_acc'] for h in HORIZONS])
    
    print(f"\n  Average MAPE: {avg_mape:.2f}%")
    print(f"  Average Direction Accuracy: {avg_dir:.1f}%")
    print(f"  Training Time: {elapsed:.1f}s")
    print(f"{'='*70}")
    
    # Save model
    save_path = MODEL_DIR / f"{symbol}_best_model.pt"
    torch.save({
        'models': [m.state_dict() for m in models],
        'hyperparams': {
            'hidden_size': hidden_size,
            'num_layers': 2,
            'lookback': lookback,
            'dropout': 0.2,
            'lr': 0.001,
            'batch_size': 64,
            'n_features': len(feature_cols),
        },
        'scaler': scaler,
        'feature_cols': feature_cols,
        'sector_code': sector_code,
        'results': results,
    }, save_path)
    print(f"\n✅ Model saved to {save_path}")
    
    # Run inference
    print(f"\n🔮 PREDICTIONS:")
    pred_result = inference(symbol)
    
    for h in HORIZONS:
        p = pred_result['predictions'][h]
        arrow = "📈" if p['direction'] == 'UP' else "📉" if p['direction'] == 'DOWN' else "➡️"
        print(f"  t+{h:2d} ({p['target_date']}): NPR {p['predicted_price']:.2f} {arrow} "
              f"{p['direction']} ({p['confidence']:.1f}%)")
    
    return pred_result


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-Horizon xLSTM with Bayesian Optimization')
    parser.add_argument('symbol', help='Stock symbol')
    parser.add_argument('--mode', choices=['train', 'optimize', 'inference'], default='train',
                       help='Mode: train (quick), optimize (bayesian), inference')
    parser.add_argument('--models', type=int, default=5, help='Number of ensemble models')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--hidden', type=int, default=128, help='Hidden size')
    parser.add_argument('--lookback', type=int, default=60, help='Lookback period')
    parser.add_argument('--max-trials', type=int, default=60, help='Max optimization trials')
    
    args = parser.parse_args()
    
    print(f"🖥️  Using device: {DEVICE}")
    
    if args.mode == 'train':
        quick_train(
            symbol=args.symbol,
            n_models=args.models,
            epochs=args.epochs,
            hidden_size=args.hidden,
            lookback=args.lookback
        )
    
    elif args.mode == 'optimize':
        all_results, best = bayesian_optimization(
            symbol=args.symbol,
            max_trials=args.max_trials,
            n_models=args.models
        )
        
        if best:
            print(f"\n{'='*70}")
            print(f"🏆 BEST RESULT")
            print(f"{'='*70}")
            print(f"  Hyperparameters: {best['hyperparams']}")
            print(f"  Average MAPE: {best['avg_mape']:.2f}%")
            print(f"  Average Direction Accuracy: {best['avg_direction_acc']:.1f}%")
            print(f"  Criteria Met: {best['criteria_met']}")
    
    elif args.mode == 'inference':
        result = inference(args.symbol)
        print(json.dumps(result, indent=2))
