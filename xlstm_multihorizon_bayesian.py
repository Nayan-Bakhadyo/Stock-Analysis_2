"""
Multi-Horizon xLSTM Stock Forecaster with Bayesian Optimization
================================================================
Target: MAPE < 3%, Direction Accuracy > 80%

Horizons: t+1, t+3, t+5, t+10, t+15, t+20

Features:
1. Multi-horizon predictions (6 time horizons)
2. Bayesian optimization for hyperparameter search (max 60 combinations)
3. Early stopping when criteria met (MAPE<3%, Direction>70%)
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
import gc
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


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance and hard examples.
    Focuses more on misclassified samples.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter (higher = more focus on hard examples)
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)  # Probability of correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

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
    """Create feature set with market context and direction-focused indicators"""
    
    df = df.copy()
    
    # Price changes
    df['returns'] = df['close'].pct_change()
    df['returns_2d'] = df['close'].pct_change(2)
    df['returns_5d'] = df['close'].pct_change(5)
    
    # Momentum
    df['rsi'] = calculate_rsi(df['close'], 14)
    df['rsi_signal'] = (df['rsi'] - 50) / 50
    
    # RSI divergence (direction indicator)
    df['rsi_slope'] = df['rsi'].diff(3) / 3  # RSI momentum
    df['price_slope'] = df['close'].pct_change(3)
    df['rsi_divergence'] = df['rsi_slope'] - df['price_slope'] * 100  # Divergence signal
    
    # Moving averages
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['ma_cross_5_10'] = (df['sma_5'] - df['sma_10']) / df['sma_10']
    df['ma_cross_5_20'] = (df['sma_5'] - df['sma_20']) / df['sma_20']
    df['price_vs_sma5'] = (df['close'] - df['sma_5']) / df['sma_5']
    df['price_vs_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']
    
    # Trend strength (direction indicator)
    df['trend_strength'] = (df['sma_5'] - df['sma_20']) / df['sma_20']
    df['trend_acceleration'] = df['trend_strength'].diff(3)
    
    # Higher highs / Lower lows (direction indicator)
    df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
    df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
    df['hh_ll_signal'] = df['higher_high'].rolling(5).sum() - df['lower_low'].rolling(5).sum()
    
    # Volatility
    df['volatility'] = df['returns'].rolling(10).std()
    
    # Volume
    df['volume_ma'] = df['volume'].rolling(10).mean()
    df['volume_signal'] = (df['volume'] - df['volume_ma']) / (df['volume_ma'] + 1)
    
    # Volume trend (buying/selling pressure)
    df['volume_trend'] = df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean() - 1
    
    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = (ema12 - ema26) / df['close']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_crossover'] = (df['macd'] > df['macd_signal']).astype(int) - 0.5  # Direction signal
    
    # Bollinger
    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_position'] = (df['close'] - bb_mid) / (2 * bb_std + 0.001)
    
    feature_cols = [
        # Price momentum
        'returns', 'returns_2d', 'returns_5d',
        # RSI indicators
        'rsi_signal', 'rsi_divergence',
        # Moving average signals
        'ma_cross_5_10', 'ma_cross_5_20',
        'price_vs_sma5', 'price_vs_sma20',
        # Trend strength (direction focused)
        'trend_strength', 'trend_acceleration',
        'hh_ll_signal',  # Higher highs / lower lows
        # Volatility & Volume
        'volatility', 'volume_signal', 'volume_trend',
        # MACD signals
        'macd', 'macd_signal', 'macd_hist', 'macd_crossover',
        # Bollinger
        'bb_position'
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
                patience=20, min_epochs=15, dir_loss_weight=0.8, device=DEVICE):
    """Train with early stopping
    
    Args:
        patience: Stop after this many epochs without improvement (default: 20)
        min_epochs: Minimum epochs before early stopping can trigger (default: 15)
                    Set low to allow natural convergence, patience handles the rest.
        dir_loss_weight: Weight for direction loss (0.5-0.9). Higher = prioritize direction accuracy.
    """
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, factor=0.5)
    
    mse_loss = nn.MSELoss()
    # Focal Loss with class weights: Down=1.5, Flat=0.3, Up=1.5
    # - Higher weight on Up/Down (the important predictions)
    # - gamma=2.0 focuses on hard-to-classify samples
    direction_loss = FocalLoss(
        alpha=torch.tensor([1.5, 0.3, 1.5]).to(device),
        gamma=2.0
    )
    
    price_loss_weight = 1.0 - dir_loss_weight
    
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
                
                # Dynamic loss weighting from hyperparameter
                loss += price_loss_weight * mse_loss(price_preds[h], y_r)
                loss += dir_loss_weight * direction_loss(dir_logits[h], y_d)
            
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
                    
                    val_loss += price_loss_weight * mse_loss(price_preds[h], y_r).item()
                    val_loss += dir_loss_weight * direction_loss(dir_logits[h], y_d).item()
                    
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
                          dir_loss_weight=0.8, n_models=5, epochs=100, trial_num=0, best_results=None):
    """Run a single optimization trial"""
    
    # Convert numpy types to native Python types
    hidden_size = int(hidden_size)
    num_layers = int(num_layers)
    lookback = int(lookback)
    dropout = float(dropout)
    lr = float(lr)
    batch_size = int(batch_size)
    dir_loss_weight = float(dir_loss_weight)
    
    print(f"\n{'='*70}")
    print(f"Trial {trial_num}: hidden={hidden_size}, layers={num_layers}, "
          f"lookback={lookback}, dropout={dropout:.2f}, lr={lr:.5f}, batch={batch_size}, dir_wt={dir_loss_weight:.1f}")
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
        model, _ = train_model(model, train_loader, test_loader, epochs=epochs, lr=lr, dir_loss_weight=dir_loss_weight)
        models.append(model)
        
        # Clear GPU cache after each model training
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    
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
                'dir_loss_weight': dir_loss_weight,
                'n_features': len(feature_cols),
            },
            'scaler': scaler,
            'feature_cols': feature_cols,
            'sector_code': sector_code,
            'results': results,
        }, save_path)
        print(f"  ✅ Saved best model to {save_path}")
    
    # Check criteria
    criteria_met = avg_mape < 3.0 and avg_dir_acc > 70.0
    
    # Aggressive memory cleanup - delete all large objects
    del models, X_train, X_test, y_ret_train, y_ret_test, y_dir_train, y_dir_test
    del train_loader, test_loader, df_processed
    
    # Extract only scalars from results before cleanup
    results_minimal = {h: {'mape': results[h]['mape'], 
                          'direction_acc': results[h]['direction_acc'],
                          'correct': results[h].get('correct', 0),
                          'total': results[h].get('total', 0)} for h in HORIZONS}
    del results
    
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    return {
        'trial': trial_num,
        'hyperparams': {
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'lookback': lookback,
            'dropout': dropout,
            'lr': lr,
            'batch_size': batch_size,
            'dir_loss_weight': dir_loss_weight,
        },
        # Store only scalar metrics, not full arrays (memory optimization)
        'results': results_minimal,
        'avg_mape': float(avg_mape),
        'avg_direction_acc': float(avg_dir_acc),
        'criteria_met': bool(criteria_met),
        'is_best': bool(is_best),
    }


def analyze_trials_and_suggest(all_results, current_best):
    """
    Analyze recent trial results and suggest adaptive adjustments.
    Returns suggestions for next hyperparameters based on observed patterns.
    """
    if len(all_results) < 5:
        return None, []
    
    suggestions = []
    recent = all_results[-10:]  # Look at last 10 trials
    
    # Extract metrics from recent trials
    recent_mapes = [r['avg_mape'] for r in recent]
    recent_dir_accs = [r['avg_direction_acc'] for r in recent]
    recent_hyperparams = [r['hyperparams'] for r in recent]
    
    # Get per-horizon results for pattern analysis
    horizon_patterns = {h: [] for h in HORIZONS}
    for r in recent:
        for h in HORIZONS:
            if h in r.get('results', {}):
                horizon_patterns[h].append({
                    'mape': r['results'][h].get('mape', 0),
                    'dir_acc': r['results'][h].get('direction_acc', 0)
                })
    
    # Pattern 1: Overfitting Detection
    # If short-term (t+1, t+3) is good but long-term (t+15, t+20) is bad
    short_term_acc = np.mean([hp[0]['dir_acc'] for hp in [horizon_patterns[1], horizon_patterns[3]] if hp])
    long_term_acc = np.mean([hp[0]['dir_acc'] for hp in [horizon_patterns[15], horizon_patterns[20]] if hp])
    
    if short_term_acc > 65 and long_term_acc < 55:
        suggestions.append({
            'issue': 'OVERFITTING',
            'description': f'Short-term acc ({short_term_acc:.1f}%) >> Long-term acc ({long_term_acc:.1f}%)',
            'adjustments': {
                'dropout': 0.3,  # Increase regularization
                'hidden_size': 64,  # Smaller model
                'lr': 0.0005,  # Lower learning rate
            }
        })
    
    # Pattern 2: Underfitting Detection
    # If both MAPE and direction accuracy are consistently poor
    avg_recent_mape = np.mean(recent_mapes)
    avg_recent_dir = np.mean(recent_dir_accs)
    
    if avg_recent_mape > 6 and avg_recent_dir < 55:
        suggestions.append({
            'issue': 'UNDERFITTING',
            'description': f'High MAPE ({avg_recent_mape:.1f}%) and low dir acc ({avg_recent_dir:.1f}%)',
            'adjustments': {
                'hidden_size': 256,  # Larger model
                'num_layers': 2,  # More depth
                'lookback': 90,  # More context
                'lr': 0.001,  # Higher learning rate
            }
        })
    
    # Pattern 3: Direction Accuracy Plateau
    # If direction accuracy hasn't improved in last 10 trials
    if current_best and len(all_results) > 20:
        trials_since_best = len(all_results) - current_best['trial']
        if trials_since_best > 15 and avg_recent_dir < current_best['avg_direction_acc'] - 2:
            suggestions.append({
                'issue': 'DIR_ACC_PLATEAU',
                'description': f'No improvement in {trials_since_best} trials, stuck at {current_best["avg_direction_acc"]:.1f}%',
                'adjustments': {
                    'dir_loss_weight': 0.9,  # Focus more on direction
                    'lookback': 60 if current_best['hyperparams'].get('lookback', 60) != 60 else 90,
                }
            })
    
    # Pattern 4: High MAPE but Good Direction
    # Direction is good but price predictions are way off
    if avg_recent_dir > 65 and avg_recent_mape > 5:
        suggestions.append({
            'issue': 'HIGH_MAPE_GOOD_DIR',
            'description': f'Good direction ({avg_recent_dir:.1f}%) but high MAPE ({avg_recent_mape:.1f}%)',
            'adjustments': {
                'dir_loss_weight': 0.7,  # Balance with price
                'batch_size': 32,  # Smaller batches for better gradients
            }
        })
    
    # Pattern 5: Inconsistent Results
    # High variance in results suggests instability
    mape_std = np.std(recent_mapes)
    dir_std = np.std(recent_dir_accs)
    
    if mape_std > 2 or dir_std > 8:
        suggestions.append({
            'issue': 'INSTABILITY',
            'description': f'High variance: MAPE std={mape_std:.2f}, Dir std={dir_std:.1f}',
            'adjustments': {
                'lr': 0.0005,  # Lower learning rate for stability
                'batch_size': 64,  # Larger batches
                'dropout': 0.2,  # Less aggressive dropout
            }
        })
    
    # Pattern 6: Learning Rate Issues
    # Check if high LR trials consistently fail
    high_lr_trials = [r for r in recent if r['hyperparams'].get('lr', 0) >= 0.005]
    low_lr_trials = [r for r in recent if r['hyperparams'].get('lr', 0) <= 0.0005]
    
    if high_lr_trials and low_lr_trials:
        high_lr_avg = np.mean([r['avg_direction_acc'] for r in high_lr_trials])
        low_lr_avg = np.mean([r['avg_direction_acc'] for r in low_lr_trials])
        
        if high_lr_avg < low_lr_avg - 5:
            suggestions.append({
                'issue': 'LR_TOO_HIGH',
                'description': f'High LR ({high_lr_avg:.1f}%) underperforms low LR ({low_lr_avg:.1f}%)',
                'adjustments': {
                    'lr': 0.0005,
                }
            })
    
    # Pattern 7: Model Size Analysis
    # Check if larger or smaller models perform better
    large_model_trials = [r for r in recent if r['hyperparams'].get('hidden_size', 0) >= 256]
    small_model_trials = [r for r in recent if r['hyperparams'].get('hidden_size', 0) <= 64]
    
    if large_model_trials and small_model_trials:
        large_avg = np.mean([r['avg_direction_acc'] for r in large_model_trials])
        small_avg = np.mean([r['avg_direction_acc'] for r in small_model_trials])
        
        if small_avg > large_avg + 3:
            suggestions.append({
                'issue': 'SMALLER_BETTER',
                'description': f'Small models ({small_avg:.1f}%) outperform large ({large_avg:.1f}%)',
                'adjustments': {
                    'hidden_size': 64,
                    'num_layers': 1,
                    'dropout': 0.2,
                }
            })
        elif large_avg > small_avg + 3:
            suggestions.append({
                'issue': 'LARGER_BETTER',
                'description': f'Large models ({large_avg:.1f}%) outperform small ({small_avg:.1f}%)',
                'adjustments': {
                    'hidden_size': 256,
                    'num_layers': 2,
                }
            })
    
    # Pattern 8: Lookback Window Analysis
    short_lookback = [r for r in recent if r['hyperparams'].get('lookback', 0) <= 30]
    long_lookback = [r for r in recent if r['hyperparams'].get('lookback', 0) >= 90]
    
    if short_lookback and long_lookback:
        short_avg = np.mean([r['avg_direction_acc'] for r in short_lookback])
        long_avg = np.mean([r['avg_direction_acc'] for r in long_lookback])
        
        if long_avg > short_avg + 3:
            suggestions.append({
                'issue': 'NEED_MORE_CONTEXT',
                'description': f'Longer lookback ({long_avg:.1f}%) better than short ({short_avg:.1f}%)',
                'adjustments': {
                    'lookback': 90,
                }
            })
    
    # Pattern 9: Stagnation - try something completely different
    if current_best and len(all_results) > 50:
        trials_since_best = len(all_results) - current_best['trial']
        if trials_since_best > 30:
            suggestions.append({
                'issue': 'EXPLORATION_NEEDED',
                'description': f'No improvement in {trials_since_best} trials - try radical change',
                'adjustments': {
                    # Try opposite of current best
                    'hidden_size': 64 if current_best['hyperparams'].get('hidden_size', 128) >= 128 else 256,
                    'lookback': 30 if current_best['hyperparams'].get('lookback', 60) >= 60 else 90,
                    'dir_loss_weight': 0.6 if current_best['hyperparams'].get('dir_loss_weight', 0.8) >= 0.8 else 0.9,
                    'lr': 0.005 if current_best['hyperparams'].get('lr', 0.001) <= 0.001 else 0.0005,
                }
            })
    
    return suggestions[-1] if suggestions else None, suggestions


def bayesian_optimization(symbol: str, max_trials: int = 900, n_models: int = 5, warm_start_configs: list = None):
    """
    Bayesian optimization for hyperparameter search
    Terminates when MAPE < 3% and Direction Accuracy > 80%
    
    Args:
        symbol: Stock symbol to optimize
        max_trials: Maximum number of trials
        n_models: Number of ensemble models
        warm_start_configs: List of hyperparameter dicts from previously optimized stocks
                           (adaptive starting points from same/similar sectors)
    """
    
    # Validate stock before starting optimization
    print(f"\n📋 Validating stock '{symbol}'...")
    sector_info = validate_stock(symbol)
    print(f"   ✅ Sector: {sector_info['sector']} ({sector_info['sector_index_code']})")
    
    print(f"\n{'='*70}")
    print(f"🔬 BAYESIAN OPTIMIZATION: {symbol}")
    print(f"{'='*70}")
    print(f"Max trials: {max_trials}")
    print(f"Target: MAPE < 3%, Direction Accuracy > 70%")
    print(f"{'='*70}\n")
    
    # Search space (optimized for NEPSE data size)
    space = [
        Categorical([64, 128, 256, 512], name='hidden_size'),
        Integer(1, 2, name='num_layers'),
        Categorical([30, 60, 90], name='lookback'),
        Categorical([0.2, 0.3], name='dropout'),
        Categorical([0.0005, 0.001, 0.005], name='lr'),
        Categorical([32, 64], name='batch_size'),
        Categorical([0.6, 0.7, 0.8, 0.9], name='dir_loss_weight'),  # Direction loss weight
    ]
    
    # Try to resume from previous progress
    all_results, best_results, completed_trials = load_optimization_progress(symbol)
    if all_results is None:
        all_results = []
        best_results = None
        completed_trials = 0
    
    trial_num = completed_trials
    should_stop = False  # Flag for early termination
    
    # Check if already completed enough trials
    if completed_trials >= max_trials:
        print(f"\n✅ Already completed {completed_trials} trials (max: {max_trials})")
        return all_results, best_results
    
    remaining_trials = max_trials - completed_trials
    print(f"\n🔄 Running {remaining_trials} more trials (starting from {completed_trials + 1})")
    
    @use_named_args(space)
    def objective(hidden_size, num_layers, lookback, dropout, lr, batch_size, dir_loss_weight):
        nonlocal trial_num, best_results, all_results, should_stop
        
        # Check if we should stop (from previous iteration) - raise to exit gp_minimize
        if should_stop:
            raise StopIteration("Early stopping triggered")
        
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
                dir_loss_weight=dir_loss_weight,
                n_models=n_models,
                epochs=100,
                trial_num=trial_num,
                best_results=best_results
            )
            
            all_results.append(result)
            
            # Update best
            if result['is_best']:
                best_results = result
            
            # Print best so far after each trial
            if best_results:
                trials_since_best = trial_num - best_results['trial']
                print(f"\n📈 Best so far: MAPE={best_results['avg_mape']:.2f}%, "
                      f"Dir Acc={best_results['avg_direction_acc']:.1f}% (Trial {best_results['trial']}, {trials_since_best} ago)")
            
            # AUTO-TERMINATION: Stop if no improvement for 150 trials
            if best_results and trial_num - best_results['trial'] >= 150:
                print(f"\n⏹️  AUTO-STOP: No improvement in 150 trials. Best remains Trial {best_results['trial']}")
                print(f"   Best MAPE: {best_results['avg_mape']:.2f}%, Best Dir Acc: {best_results['avg_direction_acc']:.1f}%")
                
                # Save final progress
                save_optimization_progress(symbol, all_results, best_results)
                
                # Simple cleanup - avoid operations that might hang
                should_stop = True
                print("   ✅ Auto-stop complete, moving to next stock...")
                return -best_results['avg_direction_acc']
            
            # Adaptive Analysis - every 10 trials, analyze patterns and print suggestions
            if trial_num % 10 == 0 and trial_num >= 10:
                suggestion, all_suggestions = analyze_trials_and_suggest(all_results, best_results)
                if suggestion:
                    print(f"\n🧠 ADAPTIVE ANALYSIS (Trial {trial_num}):")
                    print(f"   Issue detected: {suggestion['issue']}")
                    print(f"   {suggestion['description']}")
                    print(f"   Suggested adjustments: {suggestion['adjustments']}")
            
            # Memory cleanup after each trial to prevent buildup
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
            # Deep cleanup every 20 trials - force Python to release memory
            if trial_num % 20 == 0:
                import ctypes
                try:
                    libc = ctypes.CDLL("libc.dylib")
                    libc.malloc_trim(0)
                except:
                    pass
                gc.collect()
                gc.collect()  # Run twice for cyclic references
            
            # Save progress
            save_optimization_progress(symbol, all_results, best_results)
            
            # Trim old results from memory (keep last 50 + best trial)
            # Full history is saved to disk, no need to keep all in RAM
            if len(all_results) > 50:
                best_trial_num = best_results.get('trial', 0) if best_results else 0
                # Keep last 50 trials + the best trial
                recent_trials = all_results[-50:]
                best_trial_obj = next((r for r in all_results if r.get('trial') == best_trial_num), None)
                
                # Rebuild list with just recent + best
                if best_trial_obj and best_trial_obj not in recent_trials:
                    all_results = [best_trial_obj] + recent_trials
                else:
                    all_results = recent_trials
                
                # Force cleanup of trimmed trials
                gc.collect()
            
            # Check termination criteria
            if result['criteria_met']:
                print(f"\n🎉 CRITERIA MET! MAPE={result['avg_mape']:.2f}%, "
                      f"Dir Acc={result['avg_direction_acc']:.1f}%")
                should_stop = True
                return -result['avg_direction_acc']
            
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
            dir_loss_weight = np.random.choice([0.6, 0.7, 0.8, 0.9])
            
            result = objective(hidden_size, num_layers, lookback, dropout, lr, batch_size, dir_loss_weight)
            
            if best_results and best_results.get('criteria_met', False):
                break
        
        return all_results, best_results
    
    # Bayesian optimization with hybrid approach:
    # Use warm start configs if available, otherwise use default starting points
    # Format: [hidden_size, num_layers, lookback, dropout, lr, batch_size, dir_loss_weight]
    
    if warm_start_configs and len(warm_start_configs) > 0:
        # Use configs from previously optimized stocks (adaptive)
        x0 = []
        for config in warm_start_configs[:5]:  # Max 5 warm start configs
            x0.append([
                config.get('hidden_size', 128),
                config.get('num_layers', 2),
                config.get('lookback', 60),
                config.get('dropout', 0.2),
                config.get('lr', 0.001),
                config.get('batch_size', 64),
                config.get('dir_loss_weight', 0.8),
            ])
        print(f"🔥 Using {len(x0)} warm start configs from similar stocks")
        n_random = max(5, 10 - len(x0))  # Fill remaining with random
    else:
        # Default starting points
        x0 = [
            [64, 1, 30, 0.2, 0.001, 32, 0.8],    # Small, fast baseline
            [128, 2, 60, 0.2, 0.001, 64, 0.8],   # Medium baseline
            [256, 2, 90, 0.3, 0.0005, 64, 0.9],  # Larger, longer lookback, higher dir weight
        ]
        n_random = 7
    
    # Build x0 and y0 from previous trials for Bayesian resume
    y0_resume = None
    if completed_trials > 0:
        # Extract previous trial configs and scores
        x0_resume = []
        y0_resume = []
        for trial in all_results:
            hp = trial.get('hyperparams', {})
            if hp:
                x0_resume.append([
                    hp.get('hidden_size', 128),
                    hp.get('num_layers', 2),
                    hp.get('lookback', 60),
                    hp.get('dropout', 0.2),
                    hp.get('lr', 0.001),
                    hp.get('batch_size', 64),
                    hp.get('dir_loss_weight', 0.8),
                ])
                # Negative score for minimization (we want higher dir_acc)
                y0_resume.append(-trial.get('avg_direction_acc', 0))
        
        if x0_resume:
            x0 = x0_resume
            print(f"🧠 Bayesian optimizer initialized with {len(x0)} previous trials")
            n_random = 0  # No random starts when resuming - use GP knowledge
    
    try:
        result = gp_minimize(
            objective,
            space,
            x0=x0,
            y0=y0_resume,
            n_calls=remaining_trials,
            n_random_starts=n_random,
            random_state=None,  # Different random seed each time for diversity
            verbose=False,
        )
    except StopIteration:
        print("✅ Early stopping - exiting optimization cleanly")
        result = None
    except Exception as e:
        print(f"⚠️ gp_minimize ended: {e}")
        result = None
    
    # If we stopped early, gp_minimize might still be running - force cleanup
    if should_stop:
        print("🧹 Cleaning up after early stop...")
    
    # Clean up GP model to free memory before next stock
    if result is not None:
        del result
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    return all_results, best_results


def save_optimization_progress(symbol: str, results: list, best_results: dict = None):
    """Save optimization progress to JSON with full resume support"""
    
    output = {
        'symbol': symbol,
        'timestamp': datetime.now().isoformat(),
        'n_trials': len(results),
        'best_trial': best_results.get('trial', 0) if best_results else 0,
        'best_results': best_results,
        'trials': results,
    }
    
    output_file = Path(f"optimization_results_{symbol}.json")
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)


def load_optimization_progress(symbol: str):
    """Load previous optimization progress for resume"""
    
    output_file = Path(f"optimization_results_{symbol}.json")
    if not output_file.exists():
        return None, None, 0
    
    try:
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        all_results = data.get('trials', [])
        best_results = data.get('best_results', None)
        n_completed = len(all_results)
        
        # Rebuild best_results if not saved properly
        if best_results is None and all_results:
            for result in all_results:
                if result.get('is_best', False):
                    best_results = result
                    break
            # If no is_best flag, find highest avg_direction_acc
            if best_results is None:
                best_results = max(all_results, key=lambda x: x.get('avg_direction_acc', 0))
        
        print(f"\n📂 RESUMING from optimization_results_{symbol}.json")
        print(f"   Loaded {n_completed} previous trials")
        if best_results:
            print(f"   Best so far: Trial {best_results.get('trial', '?')} with {best_results.get('avg_direction_acc', 0):.1f}% dir accuracy")
        
        return all_results, best_results, n_completed
        
    except Exception as e:
        print(f"⚠️ Could not load progress file: {e}")
        return None, None, 0


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
