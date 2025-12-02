"""
Multi-Horizon Stock Price Forecaster using xLSTM
Based on "xLSTM: Extended Long Short-Term Memory" (Beck et al. 2024)

Optimized for: 1, 3, 5, 10, 15, and 21-day ahead predictions
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
)
from typing import Dict, List, Tuple
import json
from pathlib import Path
import sqlite3
from datetime import datetime, timedelta
import config


class MultiHorizonDataset(Dataset):
    """Dataset for multi-horizon forecasting"""
    
    def __init__(self, X: np.ndarray, y: Dict[int, np.ndarray]):
        """
        Args:
            X: Input sequences (samples, timesteps, features)
            y: Dict of targets {horizon: (samples, 1)}
        """
        self.X = torch.FloatTensor(X)
        self.y = {h: torch.FloatTensor(targets) for h, targets in y.items()}
        self.horizons = sorted(y.keys())
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        targets = [self.y[h][idx] for h in self.horizons]
        return self.X[idx], torch.cat(targets, dim=0)


class xLSTMStockForecaster(nn.Module):
    """
    Multi-horizon stock price forecaster using xLSTM
    
    Architecture optimized per paper recommendations:
    - Larger hidden dimensions (256-512) for better capacity
    - Multiple xLSTM blocks (4-7) for deep hierarchy
    - Mixed sLSTM + mLSTM for balance
    - Separate heads for each forecast horizon
    """
    
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 512,
        num_blocks: int = 7,
        num_heads: int = 8,
        dropout: float = 0.1,
        horizons: List[int] = [1, 3, 5, 10, 15, 21],
    ):
        """
        Args:
            input_size: Number of features per timestep
            hidden_size: Hidden dimension (paper recommends 256-1024)
            num_blocks: Number of xLSTM blocks (paper uses 4-48)
            num_heads: Number of attention heads (4-16)
            dropout: Dropout rate (paper uses 0.0-0.2)
            horizons: Forecast horizons in days
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.horizons = sorted(horizons)
        
        print(f"Building xLSTM forecaster:")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Num blocks: {num_blocks}")
        print(f"  Num heads: {num_heads}")
        print(f"  Horizons: {horizons}")
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)
        
        # Build xLSTM blocks (alternating sLSTM and mLSTM per paper)
        blocks = []
        for i in range(num_blocks):
            if i % 2 == 0:
                # sLSTM: Better for long-range dependencies (odd blocks)
                block_config = sLSTMBlockConfig(
                    slstm=sLSTMLayerConfig(
                        backend="vanilla",
                        num_heads=num_heads,
                        conv1d_kernel_size=4,
                        bias_init="powerlaw_blockdependent",
                    ),
                )
            else:
                # mLSTM: Better for storage capacity (even blocks)
                block_config = mLSTMBlockConfig(
                    mlstm=mLSTMLayerConfig(
                        num_heads=num_heads,
                        conv1d_kernel_size=4,
                    ),
                )
            blocks.append(block_config)
        
        # Create xLSTM stack
        xlstm_config = xLSTMBlockStackConfig(
            mlstm_block=blocks[1] if len(blocks) > 1 else None,
            slstm_block=blocks[0],
            context_length=252,  # ~1 year of trading days
            num_blocks=num_blocks,
            embedding_dim=hidden_size,
            add_post_blocks_norm=True,
            bias=False,
            dropout=dropout,
        )
        
        self.xlstm_stack = xLSTMBlockStack(xlstm_config)
        
        # Separate prediction heads for each horizon
        # Paper suggests task-specific heads improve performance
        self.horizon_heads = nn.ModuleDict()
        for horizon in self.horizons:
            self.horizon_heads[str(horizon)] = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, hidden_size // 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 4, 1),
            )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: (batch, seq_len, input_size)
        
        Returns:
            Dict of predictions {horizon: (batch, 1)}
        """
        # Input projection
        x = self.input_proj(x)  # (batch, seq_len, hidden)
        x = self.input_norm(x)
        
        # xLSTM processing
        x = self.xlstm_stack(x)  # (batch, seq_len, hidden)
        
        # Take last timestep
        x = x[:, -1, :]  # (batch, hidden)
        
        # Multi-horizon predictions
        predictions = {}
        for horizon in self.horizons:
            pred = self.horizon_heads[str(horizon)](x)
            predictions[horizon] = pred
        
        return predictions


class DirectionalLoss(nn.Module):
    """
    Combined loss that penalizes both magnitude errors AND wrong direction predictions.
    
    Loss = MSE + α * DirectionPenalty
    
    Where DirectionPenalty = sigmoid(-pred * target) when signs differ
    This ensures the model learns to get direction right, not just magnitude.
    """
    def __init__(self, alpha: float = 0.3, delta: float = 1.0):
        super().__init__()
        self.alpha = alpha  # Weight for direction penalty
        self.huber = nn.HuberLoss(delta=delta)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Magnitude loss (Huber for robustness)
        magnitude_loss = self.huber(pred, target)
        
        # Direction penalty: penalize when pred and target have different signs
        # Using soft penalty: higher when signs differ and magnitudes are large
        pred_sign = torch.sign(pred)
        target_sign = torch.sign(target)
        
        # Direction is wrong when signs differ (sign product is negative)
        sign_product = pred_sign * target_sign
        direction_wrong = (sign_product < 0).float()
        
        # Scale penalty by magnitude of prediction (larger wrong predictions = bigger penalty)
        direction_penalty = direction_wrong * torch.abs(pred - target)
        
        # Combine losses
        total_loss = magnitude_loss + self.alpha * direction_penalty.mean()
        
        return total_loss


class xLSTMForecasterTrainer:
    """Training wrapper with optimal settings from paper"""
    
    def __init__(
        self,
        model: xLSTMStockForecaster,
        device: str = 'mps',
        learning_rate: float = 0.0001,  # Paper uses small LR with Adam
        weight_decay: float = 0.01,  # Paper recommends weight decay
        direction_weight: float = 0.5,  # Weight for directional loss (0.5 for better direction accuracy)
    ):
        self.model = model.to(device)
        self.device = device
        
        # Paper uses AdamW with weight decay
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        # Different loss weights for different horizons (closer predictions weighted more)
        self.horizon_weights = {
            1: 3.0, 3: 2.5, 5: 2.0,
            10: 1.5, 15: 1.0, 21: 0.8
        }
        
        # Use directional loss for better direction accuracy
        self.criterion = DirectionalLoss(alpha=direction_weight, delta=1.0)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )
        
        self.history = {
            'train_loss': [], 'val_loss': [],
            'horizon_metrics': {h: [] for h in model.horizons}
        }
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            predictions = self.model(X_batch)
            
            # Compute weighted loss across horizons
            loss = 0
            start_idx = 0
            for horizon in self.model.horizons:
                pred = predictions[horizon]
                target = y_batch[:, start_idx:start_idx+1]
                horizon_loss = self.criterion(pred, target)
                
                # Weight by horizon importance
                weight = self.horizon_weights.get(horizon, 1.0)
                loss += weight * horizon_loss
                start_idx += 1
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader: DataLoader) -> Tuple[float, Dict[int, float]]:
        """Validate and compute per-horizon metrics"""
        self.model.eval()
        total_loss = 0
        horizon_errors = {h: [] for h in self.model.horizons}
        
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                predictions = self.model(X_batch)
                
                loss = 0
                start_idx = 0
                for horizon in self.model.horizons:
                    pred = predictions[horizon]
                    target = y_batch[:, start_idx:start_idx+1]
                    horizon_loss = self.criterion(pred, target)
                    
                    weight = self.horizon_weights.get(horizon, 1.0)
                    loss += weight * horizon_loss
                    
                    # Track MAE per horizon
                    mae = torch.abs(pred - target).mean().item()
                    horizon_errors[horizon].append(mae)
                    
                    start_idx += 1
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        horizon_mae = {h: np.mean(errors) for h, errors in horizon_errors.items()}
        
        return avg_loss, horizon_mae
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        early_stopping_patience: int = 20,
        verbose: bool = True,
    ):
        """Train with early stopping"""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, horizon_mae = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            for h, mae in horizon_mae.items():
                self.history['horizon_metrics'][h].append(mae)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"\nEpoch {epoch+1}/{epochs}")
                print(f"  Train Loss: {train_loss:.6f}")
                print(f"  Val Loss: {val_loss:.6f}")
                print(f"  Horizon MAE:")
                for h in sorted(horizon_mae.keys()):
                    print(f"    {h:2d}-day: {horizon_mae[h]:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        return self.history
    
    def predict(self, X: np.ndarray) -> Dict[int, np.ndarray]:
        """Make predictions for all horizons"""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        return {h: pred.cpu().numpy() for h, pred in predictions.items()}
    
    def save(self, path: str):
        """Save checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'horizons': self.model.horizons,
        }, path)
    
    def load(self, path: str):
        """Load checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint.get('history', self.history)


def prepare_multi_horizon_data(
    df: pd.DataFrame,
    lookback: int = 120,  # Longer lookback for better patterns
    horizons: List[int] = [1, 3, 5, 10, 15, 21],
    train_split: float = 0.8,
    use_returns: bool = True,  # Use returns instead of prices
    add_features: bool = True,  # Add trend features
    add_volume: bool = True,  # NEW: Add volume features
):
    """
    Prepare data for multi-horizon forecasting with returns, trend, and volume features
    
    Args:
        df: DataFrame with 'close' and 'volume' columns
        lookback: Number of historical days to use
        horizons: List of forecast horizons
        train_split: Train/test split ratio
        use_returns: If True, use log returns instead of raw prices (more stationary)
        add_features: If True, add trend indicators (SMA, momentum, volatility)
        add_volume: If True, add volume-based features
    
    Returns:
        X_train, y_train_dict, X_test, y_test_dict, scaler_info
    """
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    
    prices = df['close'].values.astype(np.float64)
    
    # Check if volume data exists
    has_volume = 'volume' in df.columns and df['volume'].sum() > 0
    if add_volume and has_volume:
        volumes = df['volume'].values.astype(np.float64)
        volumes = np.where(volumes == 0, 1, volumes)  # Avoid log(0)
    else:
        add_volume = False
    
    if use_returns:
        # Use log returns - more stationary and normalized signal
        returns = np.log(prices[1:] / prices[:-1])
        # Pad first value
        returns = np.concatenate([[0], returns])
        
        # Target: cumulative return over horizon (not single day return)
        # This represents total % change from current price
        primary_signal = returns
    else:
        primary_signal = prices
    
    # Build feature matrix
    if add_features:
        features = []
        
        # Feature 1: Primary signal (returns or prices)
        features.append(primary_signal)
        
        # Feature 2: 5-day momentum (short-term trend)
        momentum_5 = np.zeros(len(prices))
        momentum_5[5:] = (prices[5:] - prices[:-5]) / prices[:-5]
        features.append(momentum_5)
        
        # Feature 3: 20-day momentum (medium-term trend)
        momentum_20 = np.zeros(len(prices))
        momentum_20[20:] = (prices[20:] - prices[:-20]) / prices[:-20]
        features.append(momentum_20)
        
        # Feature 4: Price relative to 20-day SMA (mean reversion signal)
        sma_20 = np.convolve(prices, np.ones(20)/20, mode='same')
        sma_20[:20] = prices[:20]  # Handle edge
        price_to_sma = (prices - sma_20) / sma_20
        features.append(price_to_sma)
        
        # Feature 5: 10-day volatility (risk indicator)
        volatility = np.zeros(len(prices))
        for i in range(10, len(prices)):
            volatility[i] = np.std(returns[i-10:i])
        features.append(volatility)
        
        # Feature 6: RSI-like momentum (overbought/oversold)
        rsi_period = 14
        gains = np.zeros(len(returns))
        losses = np.zeros(len(returns))
        gains[returns > 0] = returns[returns > 0]
        losses[returns < 0] = -returns[returns < 0]
        
        avg_gain = np.convolve(gains, np.ones(rsi_period)/rsi_period, mode='same')
        avg_loss = np.convolve(losses, np.ones(rsi_period)/rsi_period, mode='same')
        avg_loss = np.where(avg_loss == 0, 1e-10, avg_loss)  # Avoid division by zero
        rs = avg_gain / avg_loss
        rsi = 1 - (1 / (1 + rs))  # Normalized 0-1
        features.append(rsi)
        
        # NEW: Volume-based features
        if add_volume:
            # Feature 7: Volume ratio to 20-day average (unusual volume detection)
            vol_sma_20 = np.convolve(volumes, np.ones(20)/20, mode='same')
            vol_sma_20[:20] = volumes[:20]
            vol_sma_20 = np.where(vol_sma_20 == 0, 1, vol_sma_20)
            volume_ratio = volumes / vol_sma_20
            features.append(np.clip(volume_ratio, 0, 5))  # Cap at 5x to avoid outliers
            
            # Feature 8: Volume momentum (5-day)
            vol_momentum = np.zeros(len(volumes))
            vol_momentum[5:] = (volumes[5:] - volumes[:-5]) / (volumes[:-5] + 1)
            features.append(np.clip(vol_momentum, -2, 2))
            
            # Feature 9: Price-Volume trend (OBV-like)
            # Positive when price up with high volume, negative when price down with high volume
            pv_trend = np.zeros(len(prices))
            for i in range(1, len(prices)):
                direction = 1 if returns[i] > 0 else -1
                pv_trend[i] = pv_trend[i-1] + direction * volume_ratio[i]
            # Normalize OBV to recent range
            pv_normalized = np.zeros(len(pv_trend))
            for i in range(20, len(pv_trend)):
                window = pv_trend[i-20:i+1]
                pv_range = window.max() - window.min()
                if pv_range > 0:
                    pv_normalized[i] = (pv_trend[i] - window.min()) / pv_range
                else:
                    pv_normalized[i] = 0.5
            features.append(pv_normalized)
            
            # Feature 10: Volume-weighted price change
            vwpc = np.zeros(len(prices))
            for i in range(5, len(prices)):
                weighted_returns = returns[i-5:i] * (volumes[i-5:i] / volumes[i-5:i].sum())
                vwpc[i] = weighted_returns.sum()
            features.append(vwpc)
        
        # Stack features: shape (n_samples, n_features)
        feature_matrix = np.column_stack(features)
        n_features = feature_matrix.shape[1]
    else:
        feature_matrix = primary_signal.reshape(-1, 1)
        n_features = 1
    
    # Normalize features
    feature_scaler = StandardScaler()
    scaled_features = feature_scaler.fit_transform(feature_matrix)
    
    # For targets: use cumulative returns over each horizon
    # y[h] = (price[t+h] - price[t]) / price[t] = cumulative return
    
    # Create sequences with multiple targets
    X = []
    y = {h: [] for h in horizons}
    
    max_horizon = max(horizons)
    for i in range(lookback, len(scaled_features) - max_horizon):
        X.append(scaled_features[i-lookback:i])  # Shape: (lookback, n_features)
        
        current_price = prices[i-1]  # Price at prediction time
        for horizon in horizons:
            future_price = prices[i + horizon - 1]
            # Target: percentage return (more stable than raw price)
            target_return = (future_price - current_price) / current_price
            y[horizon].append(target_return)
    
    X = np.array(X)  # Shape: (n_samples, lookback, n_features)
    
    # Normalize targets (returns typically in -0.5 to +0.5 range, but can spike)
    target_scaler = StandardScaler()
    all_targets = np.concatenate([np.array(y[h]) for h in horizons])
    target_scaler.fit(all_targets.reshape(-1, 1))
    
    y = {h: target_scaler.transform(np.array(targets).reshape(-1, 1)) 
         for h, targets in y.items()}
    
    # Split
    split_idx = int(len(X) * train_split)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train = {h: targets[:split_idx] for h, targets in y.items()}
    y_test = {h: targets[split_idx:] for h, targets in y.items()}
    
    # Return scaler info for inverse transform
    scaler_info = {
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'last_price': prices[-1],  # For converting returns to prices
        'n_features': n_features,
        'use_returns': use_returns,
    }
    
    return X_train, y_train, X_test, y_test, scaler_info


def prepare_multi_horizon_data_legacy(
    df: pd.DataFrame,
    lookback: int = 120,
    horizons: List[int] = [1, 3, 5, 10, 15, 21],
    train_split: float = 0.8,
):
    """Legacy version - uses raw prices (kept for backward compatibility)"""


def get_adaptive_lookback(horizon: int) -> int:
    """
    Adaptive lookback window based on forecast horizon.
    Shorter horizons need less history, longer horizons need more context.
    
    Based on research that:
    - Short-term: Noise dominates, use recent data (60 days ~ 3 months)
    - Medium-term: Trends matter, use 90 days
    - Long-term: Cycles matter, use 120+ days
    """
    if horizon <= 3:
        return 60  # ~3 months for 1-3 day predictions
    elif horizon <= 10:
        return 90  # ~4.5 months for 5-10 day predictions
    else:
        return 120  # ~6 months for 15-21 day predictions


def compute_technical_indicators(prices: np.ndarray, volumes: np.ndarray = None) -> dict:
    """
    Compute comprehensive technical indicators.
    Returns dict of indicator arrays.
    """
    n = len(prices)
    returns = np.zeros(n)
    returns[1:] = np.log(prices[1:] / prices[:-1])
    
    indicators = {}
    
    # === Trend Indicators ===
    
    # 1. Returns (already computed)
    indicators['returns'] = returns
    
    # 2-3. Momentum (5-day and 20-day)
    indicators['momentum_5'] = np.zeros(n)
    indicators['momentum_5'][5:] = (prices[5:] - prices[:-5]) / prices[:-5]
    
    indicators['momentum_20'] = np.zeros(n)
    indicators['momentum_20'][20:] = (prices[20:] - prices[:-20]) / prices[:-20]
    
    # 4. Price to SMA ratio
    sma_20 = np.convolve(prices, np.ones(20)/20, mode='same')
    sma_20[:20] = prices[:20]
    indicators['price_to_sma'] = (prices - sma_20) / (sma_20 + 1e-10)
    
    # 5. Volatility (10-day rolling std of returns)
    indicators['volatility'] = np.zeros(n)
    for i in range(10, n):
        indicators['volatility'][i] = np.std(returns[i-10:i])
    
    # 6. RSI (14-period)
    gains = np.zeros(n)
    losses = np.zeros(n)
    gains[returns > 0] = returns[returns > 0]
    losses[returns < 0] = -returns[returns < 0]
    avg_gain = np.convolve(gains, np.ones(14)/14, mode='same')
    avg_loss = np.convolve(losses, np.ones(14)/14, mode='same')
    avg_loss = np.where(avg_loss == 0, 1e-10, avg_loss)
    rs = avg_gain / avg_loss
    indicators['rsi'] = 1 - (1 / (1 + rs))  # Normalized 0-1
    
    # === NEW: Advanced Technical Indicators ===
    
    # 7. MACD (12, 26, 9) - Moving Average Convergence Divergence
    ema_12 = np.zeros(n)
    ema_26 = np.zeros(n)
    ema_12[0] = prices[0]
    ema_26[0] = prices[0]
    alpha_12, alpha_26 = 2/13, 2/27
    for i in range(1, n):
        ema_12[i] = alpha_12 * prices[i] + (1 - alpha_12) * ema_12[i-1]
        ema_26[i] = alpha_26 * prices[i] + (1 - alpha_26) * ema_26[i-1]
    macd_line = ema_12 - ema_26
    # Signal line (9-day EMA of MACD)
    signal_line = np.zeros(n)
    signal_line[0] = macd_line[0]
    alpha_9 = 2/10
    for i in range(1, n):
        signal_line[i] = alpha_9 * macd_line[i] + (1 - alpha_9) * signal_line[i-1]
    # MACD histogram normalized
    macd_hist = macd_line - signal_line
    macd_std = np.std(macd_hist[26:]) if len(macd_hist) > 26 else 1
    indicators['macd'] = macd_hist / (macd_std + 1e-10)
    
    # 8. Bollinger Bands position (where price is relative to bands)
    sma_20 = np.convolve(prices, np.ones(20)/20, mode='same')
    sma_20[:20] = prices[:20]
    bb_std = np.zeros(n)
    for i in range(20, n):
        bb_std[i] = np.std(prices[i-20:i])
    bb_std[:20] = bb_std[20] if n > 20 else 1
    upper_band = sma_20 + 2 * bb_std
    lower_band = sma_20 - 2 * bb_std
    band_width = upper_band - lower_band
    band_width = np.where(band_width == 0, 1, band_width)
    indicators['bb_position'] = (prices - lower_band) / band_width  # 0-1 range
    
    # 9. ADX (Average Directional Index) - trend strength
    # Simplified ADX computation
    tr = np.zeros(n)  # True Range
    plus_dm = np.zeros(n)  # +DM
    minus_dm = np.zeros(n)  # -DM
    for i in range(1, n):
        high_diff = prices[i] - prices[i-1] if prices[i] > prices[i-1] else 0
        low_diff = prices[i-1] - prices[i] if prices[i] < prices[i-1] else 0
        tr[i] = abs(prices[i] - prices[i-1])
        plus_dm[i] = high_diff if high_diff > low_diff else 0
        minus_dm[i] = low_diff if low_diff > high_diff else 0
    # 14-period smoothed
    atr = np.convolve(tr, np.ones(14)/14, mode='same')
    atr = np.where(atr == 0, 1e-10, atr)
    plus_di = 100 * np.convolve(plus_dm, np.ones(14)/14, mode='same') / atr
    minus_di = 100 * np.convolve(minus_dm, np.ones(14)/14, mode='same') / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    indicators['adx'] = np.convolve(dx, np.ones(14)/14, mode='same') / 100  # Normalized 0-1
    
    # 10. Rate of Change (ROC) - 10-period
    indicators['roc'] = np.zeros(n)
    indicators['roc'][10:] = (prices[10:] - prices[:-10]) / (prices[:-10] + 1e-10)
    
    # === Volume Indicators (if available) ===
    if volumes is not None and len(volumes) == n:
        volumes = np.where(volumes == 0, 1, volumes)
        
        # 11. Volume ratio to 20-day average
        vol_sma = np.convolve(volumes, np.ones(20)/20, mode='same')
        vol_sma[:20] = volumes[:20]
        vol_sma = np.where(vol_sma == 0, 1, vol_sma)
        indicators['volume_ratio'] = np.clip(volumes / vol_sma, 0, 5)
        
        # 12. Volume momentum
        indicators['vol_momentum'] = np.zeros(n)
        indicators['vol_momentum'][5:] = (volumes[5:] - volumes[:-5]) / (volumes[:-5] + 1)
        indicators['vol_momentum'] = np.clip(indicators['vol_momentum'], -2, 2)
        
        # 13. Money Flow Index (MFI) - volume-weighted RSI
        typical_price = prices  # Simplified (normally (H+L+C)/3)
        money_flow = typical_price * volumes
        pos_flow = np.zeros(n)
        neg_flow = np.zeros(n)
        for i in range(1, n):
            if typical_price[i] > typical_price[i-1]:
                pos_flow[i] = money_flow[i]
            else:
                neg_flow[i] = money_flow[i]
        pos_mf = np.convolve(pos_flow, np.ones(14)/14, mode='same')
        neg_mf = np.convolve(neg_flow, np.ones(14)/14, mode='same')
        neg_mf = np.where(neg_mf == 0, 1e-10, neg_mf)
        mf_ratio = pos_mf / neg_mf
        indicators['mfi'] = 1 - (1 / (1 + mf_ratio))  # 0-1 like RSI
        
        # 14. Volume-weighted price trend
        vwap = np.zeros(n)
        for i in range(20, n):
            window_vol = volumes[i-20:i]
            window_price = prices[i-20:i]
            if window_vol.sum() > 0:
                vwap[i] = (window_price * window_vol).sum() / window_vol.sum()
            else:
                vwap[i] = prices[i]
        vwap[:20] = prices[:20]
        indicators['vwap_diff'] = (prices - vwap) / (vwap + 1e-10)
    
    return indicators


class EnsembleForecaster:
    """
    Ensemble of multiple xLSTM models trained with different seeds.
    Provides predictions with uncertainty estimates (confidence intervals).
    """
    
    def __init__(
        self,
        n_models: int = 5,
        input_size: int = 14,  # Number of features
        hidden_size: int = 512,
        num_blocks: int = 7,
        horizons: List[int] = [1, 3, 5, 10, 15, 21],
        device: str = 'mps',
    ):
        self.n_models = n_models
        self.horizons = horizons
        self.device = device
        self.models = []
        self.trainers = []
        
        # Create ensemble of models with different random seeds
        for i in range(n_models):
            torch.manual_seed(42 + i * 100)  # Different seed per model
            np.random.seed(42 + i * 100)
            
            model = xLSTMStockForecaster(
                input_size=input_size,
                hidden_size=hidden_size,
                num_blocks=num_blocks,
                horizons=horizons,
            )
            trainer = xLSTMForecasterTrainer(model, device=device)
            
            self.models.append(model)
            self.trainers.append(trainer)
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        verbose: bool = True,
    ):
        """Train all models in ensemble"""
        histories = []
        
        for i, trainer in enumerate(self.trainers):
            if verbose:
                print(f"\n{'='*50}")
                print(f"Training ensemble model {i+1}/{self.n_models}")
                print(f"{'='*50}")
            
            history = trainer.fit(
                train_loader, val_loader, 
                epochs=epochs, 
                verbose=verbose and (i == 0),  # Only verbose for first model
                early_stopping_patience=15,
            )
            histories.append(history)
        
        return histories
    
    def predict_with_uncertainty(
        self, 
        X: np.ndarray,
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Predict with uncertainty estimates.
        
        Returns:
            Dict[horizon, {
                'mean': mean prediction,
                'std': standard deviation,
                'lower': 95% CI lower bound,
                'upper': 95% CI upper bound,
                'all_predictions': predictions from all models
            }]
        """
        all_predictions = {h: [] for h in self.horizons}
        
        for trainer in self.trainers:
            preds = trainer.predict(X)
            for h in self.horizons:
                all_predictions[h].append(preds[h])
        
        results = {}
        for h in self.horizons:
            stacked = np.stack(all_predictions[h], axis=0)  # (n_models, n_samples, 1)
            mean_pred = np.mean(stacked, axis=0)
            std_pred = np.std(stacked, axis=0)
            
            # 95% confidence interval (mean ± 1.96*std)
            results[h] = {
                'mean': mean_pred,
                'std': std_pred,
                'lower': mean_pred - 1.96 * std_pred,
                'upper': mean_pred + 1.96 * std_pred,
                'all_predictions': stacked,
            }
        
        return results
    
    def get_confidence_level(self, std: float, horizon: int) -> str:
        """
        Interpret uncertainty as confidence level.
        Lower std relative to horizon = higher confidence.
        """
        # Normalize std by horizon (longer horizons naturally have more uncertainty)
        normalized_std = std / np.sqrt(horizon)
        
        if normalized_std < 0.3:
            return "HIGH"
        elif normalized_std < 0.6:
            return "MEDIUM"
        else:
            return "LOW"


def prepare_enhanced_data(
    df: pd.DataFrame,
    lookback: int = 120,
    horizons: List[int] = [1, 3, 5, 10, 15, 21],
    train_split: float = 0.8,
    adaptive_lookback: bool = False,
):
    """
    Enhanced data preparation with all technical indicators.
    
    Features (14 total):
    - returns, momentum_5, momentum_20, price_to_sma, volatility, rsi
    - macd, bb_position, adx, roc (NEW)
    - volume_ratio, vol_momentum, mfi, vwap_diff (if volume available)
    """
    from sklearn.preprocessing import StandardScaler
    
    prices = df['close'].values.astype(np.float64)
    
    # Check for volume
    has_volume = 'volume' in df.columns and df['volume'].sum() > 0
    volumes = df['volume'].values.astype(np.float64) if has_volume else None
    
    # Compute all technical indicators
    indicators = compute_technical_indicators(prices, volumes)
    
    # Build feature matrix
    feature_names = [
        'returns', 'momentum_5', 'momentum_20', 'price_to_sma', 
        'volatility', 'rsi', 'macd', 'bb_position', 'adx', 'roc'
    ]
    
    if has_volume:
        feature_names.extend(['volume_ratio', 'vol_momentum', 'mfi', 'vwap_diff'])
    
    feature_matrix = np.column_stack([indicators[f] for f in feature_names])
    n_features = feature_matrix.shape[1]
    
    print(f"  Using {n_features} features: {feature_names}")
    
    # Normalize features
    feature_scaler = StandardScaler()
    scaled_features = feature_scaler.fit_transform(feature_matrix)
    
    # Use adaptive lookback if requested
    if adaptive_lookback:
        max_lookback = max(get_adaptive_lookback(h) for h in horizons)
    else:
        max_lookback = lookback
    
    # Create sequences
    X = []
    y = {h: [] for h in horizons}
    
    max_horizon = max(horizons)
    for i in range(max_lookback, len(scaled_features) - max_horizon):
        X.append(scaled_features[i-lookback:i])
        
        current_price = prices[i-1]
        for horizon in horizons:
            future_price = prices[i + horizon - 1]
            target_return = (future_price - current_price) / current_price
            y[horizon].append(target_return)
    
    X = np.array(X)
    
    # Normalize targets
    target_scaler = StandardScaler()
    all_targets = np.concatenate([np.array(y[h]) for h in horizons])
    target_scaler.fit(all_targets.reshape(-1, 1))
    
    y = {h: target_scaler.transform(np.array(targets).reshape(-1, 1)) 
         for h, targets in y.items()}
    
    # Split
    split_idx = int(len(X) * train_split)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train = {h: targets[:split_idx] for h, targets in y.items()}
    y_test = {h: targets[split_idx:] for h, targets in y.items()}
    
    scaler_info = {
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'last_price': prices[-1],
        'n_features': n_features,
        'feature_names': feature_names,
    }
    
    return X_train, y_train, X_test, y_test, scaler_info


def get_market_data(symbol: str, start_date: str = None, end_date: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Get NEPSE index and sector index data for a given stock symbol.
    
    Returns:
        Tuple of (nepse_df, sector_df, sector_code)
    """
    from sector_mapper import SectorMapper
    
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
    
    print(f"  NEPSE index: {len(nepse_df)} days")
    if not sector_df.empty:
        print(f"  Sector index ({sector_code}): {len(sector_df)} days")
    else:
        print(f"  Sector index: Not available for {symbol}")
    
    return nepse_df, sector_df, sector_code


def compute_market_features(
    prices: np.ndarray,
    nepse_closes: np.ndarray,
    sector_closes: np.ndarray = None,
) -> dict:
    """
    Compute market-relative features.
    
    Features:
    - NEPSE returns (market trend)
    - Stock beta (correlation with market)
    - Relative strength vs NEPSE
    - Sector returns (if available)
    - Relative strength vs sector
    """
    n = len(prices)
    features = {}
    
    # Stock returns
    stock_returns = np.zeros(n)
    stock_returns[1:] = np.log(prices[1:] / prices[:-1])
    
    # NEPSE returns
    nepse_returns = np.zeros(len(nepse_closes))
    nepse_returns[1:] = np.log(nepse_closes[1:] / nepse_closes[:-1])
    
    # Align lengths (use the shorter one)
    min_len = min(n, len(nepse_returns))
    stock_returns = stock_returns[-min_len:]
    nepse_returns = nepse_returns[-min_len:]
    
    # Feature 1: NEPSE returns (market trend)
    features['nepse_returns'] = nepse_returns
    
    # Feature 2: NEPSE momentum (5-day)
    features['nepse_momentum'] = np.zeros(min_len)
    features['nepse_momentum'][5:] = (nepse_closes[-min_len+5:] - nepse_closes[-min_len:-5]) / nepse_closes[-min_len:-5]
    features['nepse_momentum'][:5] = 0
    
    # Feature 3: Rolling beta (stock's sensitivity to market)
    # Beta = Cov(stock, market) / Var(market)
    features['rolling_beta'] = np.zeros(min_len)
    window = 20
    for i in range(window, min_len):
        stock_window = stock_returns[i-window:i]
        market_window = nepse_returns[i-window:i]
        cov = np.cov(stock_window, market_window)[0, 1]
        var = np.var(market_window)
        features['rolling_beta'][i] = cov / var if var > 0 else 1.0
    features['rolling_beta'][:window] = features['rolling_beta'][window] if min_len > window else 1.0
    
    # Feature 4: Relative strength vs NEPSE
    # How much stock outperformed/underperformed market
    features['relative_strength'] = np.zeros(min_len)
    for i in range(5, min_len):
        stock_cum = np.exp(stock_returns[i-5:i].sum()) - 1
        market_cum = np.exp(nepse_returns[i-5:i].sum()) - 1
        features['relative_strength'][i] = stock_cum - market_cum
    
    # Feature 5: Market regime (volatility state)
    features['market_volatility'] = np.zeros(min_len)
    for i in range(10, min_len):
        features['market_volatility'][i] = np.std(nepse_returns[i-10:i])
    
    # Sector features (if available)
    if sector_closes is not None and len(sector_closes) > 0:
        sector_returns = np.zeros(len(sector_closes))
        sector_returns[1:] = np.log(sector_closes[1:] / sector_closes[:-1])
        sector_min_len = min(min_len, len(sector_returns))
        sector_returns = sector_returns[-sector_min_len:]
        
        # Pad to match stock length
        if sector_min_len < min_len:
            padding = np.zeros(min_len - sector_min_len)
            sector_returns = np.concatenate([padding, sector_returns])
        
        # Feature 6: Sector returns
        features['sector_returns'] = sector_returns
        
        # Feature 7: Sector momentum
        features['sector_momentum'] = np.zeros(min_len)
        if len(sector_closes) >= 5:
            sector_closes_aligned = sector_closes[-min_len:] if len(sector_closes) >= min_len else np.concatenate([np.full(min_len - len(sector_closes), sector_closes[0]), sector_closes])
            features['sector_momentum'][5:] = (sector_closes_aligned[5:] - sector_closes_aligned[:-5]) / (sector_closes_aligned[:-5] + 1e-10)
        
        # Feature 8: Stock vs sector relative strength
        features['sector_relative_strength'] = np.zeros(min_len)
        for i in range(5, min_len):
            stock_cum = np.exp(stock_returns[i-5:i].sum()) - 1
            sector_cum = np.exp(sector_returns[i-5:i].sum()) - 1
            features['sector_relative_strength'][i] = stock_cum - sector_cum
    else:
        # No sector data - use zeros
        features['sector_returns'] = np.zeros(min_len)
        features['sector_momentum'] = np.zeros(min_len)
        features['sector_relative_strength'] = np.zeros(min_len)
    
    return features, min_len


def prepare_market_enhanced_data(
    df: pd.DataFrame,
    symbol: str,
    lookback: int = 120,
    horizons: List[int] = [1, 3, 5, 10, 15, 21],
    train_split: float = 0.8,
):
    """
    Enhanced data preparation with technical indicators + MARKET FEATURES.
    
    This version adds NEPSE index and sector index as features, which should
    improve predictions since stock movements are highly correlated with
    overall market and sector trends.
    
    Features (22 total if sector available, 19 otherwise):
    - Stock technical indicators (14): returns, momentum_5, momentum_20, price_to_sma, 
      volatility, rsi, macd, bb_position, adx, roc, volume_ratio, vol_momentum, mfi, vwap_diff
    - Market features (5): nepse_returns, nepse_momentum, rolling_beta, relative_strength, market_volatility
    - Sector features (3): sector_returns, sector_momentum, sector_relative_strength
    """
    from sklearn.preprocessing import StandardScaler
    
    print(f"\n📊 Preparing MARKET-ENHANCED data for {symbol}...")
    
    prices = df['close'].values.astype(np.float64)
    dates = pd.to_datetime(df['date'])
    
    # Check for volume
    has_volume = 'volume' in df.columns and df['volume'].sum() > 0
    volumes = df['volume'].values.astype(np.float64) if has_volume else None
    
    # Get market data
    print(f"  Loading market data...")
    nepse_df, sector_df, sector_code = get_market_data(symbol)
    
    # Compute stock technical indicators
    stock_indicators = compute_technical_indicators(prices, volumes)
    
    # Merge market data by date
    df_with_dates = df.copy()
    df_with_dates['date'] = pd.to_datetime(df_with_dates['date'])
    
    if not nepse_df.empty:
        df_with_dates = df_with_dates.merge(nepse_df, on='date', how='left')
        df_with_dates['nepse_close'] = df_with_dates['nepse_close'].ffill().bfill()
        df_with_dates['nepse_pct_change'] = df_with_dates['nepse_pct_change'].ffill().bfill()
    else:
        df_with_dates['nepse_close'] = prices  # Fallback to stock prices
        df_with_dates['nepse_pct_change'] = 0
    
    if not sector_df.empty:
        df_with_dates = df_with_dates.merge(sector_df, on='date', how='left')
        df_with_dates['sector_close'] = df_with_dates['sector_close'].ffill().bfill()
        df_with_dates['sector_pct_change'] = df_with_dates['sector_pct_change'].ffill().bfill()
    else:
        df_with_dates['sector_close'] = None
        df_with_dates['sector_pct_change'] = 0
    
    # Compute market features
    nepse_closes = df_with_dates['nepse_close'].values.astype(np.float64)
    sector_closes = df_with_dates['sector_close'].values.astype(np.float64) if sector_code and 'sector_close' in df_with_dates.columns and df_with_dates['sector_close'].notna().any() else None
    
    market_features, aligned_len = compute_market_features(prices, nepse_closes, sector_closes)
    
    # Align stock indicators to the same length
    offset = len(prices) - aligned_len
    
    # Build feature matrix
    stock_feature_names = [
        'returns', 'momentum_5', 'momentum_20', 'price_to_sma', 
        'volatility', 'rsi', 'macd', 'bb_position', 'adx', 'roc'
    ]
    if has_volume:
        stock_feature_names.extend(['volume_ratio', 'vol_momentum', 'mfi', 'vwap_diff'])
    
    market_feature_names = [
        'nepse_returns', 'nepse_momentum', 'rolling_beta', 
        'relative_strength', 'market_volatility',
        'sector_returns', 'sector_momentum', 'sector_relative_strength'
    ]
    
    # Align and stack features
    stock_features = [stock_indicators[f][offset:] for f in stock_feature_names]
    market_features_list = [market_features[f] for f in market_feature_names]
    
    all_features = stock_features + market_features_list
    feature_matrix = np.column_stack(all_features)
    
    all_feature_names = stock_feature_names + market_feature_names
    n_features = feature_matrix.shape[1]
    
    print(f"  ✓ Total features: {n_features}")
    print(f"    Stock indicators: {len(stock_feature_names)}")
    print(f"    Market features: {len(market_feature_names)}")
    if sector_code:
        print(f"    Sector: {sector_code}")
    
    # Handle NaN/inf values
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Normalize features
    feature_scaler = StandardScaler()
    scaled_features = feature_scaler.fit_transform(feature_matrix)
    
    # Create sequences
    prices_aligned = prices[offset:]
    X = []
    y = {h: [] for h in horizons}
    
    max_horizon = max(horizons)
    for i in range(lookback, len(scaled_features) - max_horizon):
        X.append(scaled_features[i-lookback:i])
        
        current_price = prices_aligned[i-1]
        for horizon in horizons:
            future_price = prices_aligned[i + horizon - 1]
            target_return = (future_price - current_price) / current_price
            y[horizon].append(target_return)
    
    X = np.array(X)
    
    # Normalize targets
    target_scaler = StandardScaler()
    all_targets = np.concatenate([np.array(y[h]) for h in horizons])
    target_scaler.fit(all_targets.reshape(-1, 1))
    
    y = {h: target_scaler.transform(np.array(targets).reshape(-1, 1)) 
         for h, targets in y.items()}
    
    # Split
    split_idx = int(len(X) * train_split)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train = {h: targets[:split_idx] for h, targets in y.items()}
    y_test = {h: targets[split_idx:] for h, targets in y.items()}
    
    scaler_info = {
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'last_price': prices[-1],
        'n_features': n_features,
        'feature_names': all_feature_names,
        'sector_code': sector_code,
        'has_market_features': True,
    }
    
    print(f"  ✓ Train: {len(X_train)}, Test: {len(X_test)}")
    
    return X_train, y_train, X_test, y_test, scaler_info


def forecast_with_market_context(
    symbol: str,
    lookback: int = 120,
    epochs: int = 50,
    n_models: int = 3,
    hidden_size: int = 256,
    num_blocks: int = 4,
    batch_size: int = 32,
    horizons: List[int] = [1, 3, 5, 10, 15, 21],
):
    """
    Forecast using ensemble with MARKET CONTEXT (NEPSE + Sector indices).
    
    This should produce better predictions because:
    1. Individual stocks are highly correlated with NEPSE index
    2. Stocks within a sector tend to move together
    3. Relative strength helps identify stock-specific vs market-wide movements
    """
    print(f"\n{'='*70}")
    print(f"🚀 xLSTM MARKET-AWARE FORECASTING: {symbol}")
    print(f"{'='*70}")
    print(f"Ensemble: {n_models} models")
    print(f"Features: Technical + NEPSE + Sector indices")
    print(f"Horizons: {horizons} days")
    print(f"{'='*70}\n")
    
    # Load stock data
    conn = sqlite3.connect(config.DB_PATH)
    query = f"""
        SELECT date, open, high, low, close, volume
        FROM price_history
        WHERE symbol = '{symbol}'
        ORDER BY date ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if len(df) < lookback + max(horizons) + 50:
        print(f"❌ Not enough data for {symbol}")
        return None
    
    print(f"✓ Loaded {len(df)} days of stock data")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Prepare market-enhanced data
    X_train, y_train, X_test, y_test, scaler_info = prepare_market_enhanced_data(
        df, symbol, lookback=lookback, horizons=horizons
    )
    
    if len(X_train) < 50:
        print(f"❌ Not enough training data after alignment")
        return None
    
    # Create datasets
    train_dataset = MultiHorizonDataset(X_train, y_train)
    test_dataset = MultiHorizonDataset(X_test, y_test)
    
    # Optimized for M1 Max - use multiprocessing for data loading
    num_workers = 4 if torch.backends.mps.is_available() else 0
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, persistent_workers=(num_workers > 0)
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, persistent_workers=(num_workers > 0)
    )
    
    # Create ensemble
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"\n🚀 Building market-aware ensemble ({device.upper()})...")
    
    ensemble = EnsembleForecaster(
        n_models=n_models,
        input_size=scaler_info['n_features'],
        hidden_size=hidden_size,
        num_blocks=num_blocks,
        horizons=horizons,
        device=device,
    )
    
    # Train ensemble
    import time
    start_time = time.time()
    ensemble.fit(train_loader, test_loader, epochs=epochs, verbose=True)
    training_time = time.time() - start_time
    
    print(f"\n✓ Training completed in {training_time:.1f}s")
    
    # Make predictions
    print(f"\n{'='*70}")
    print(f"📈 MARKET-AWARE FORECAST")
    print(f"{'='*70}")
    
    # Prepare latest data with market features
    prices = df['close'].values.astype(np.float64)
    volumes = df['volume'].values.astype(np.float64) if 'volume' in df.columns else None
    
    # Get market data
    nepse_df, sector_df, sector_code = get_market_data(symbol)
    
    # Merge with stock data
    df_with_dates = df.copy()
    df_with_dates['date'] = pd.to_datetime(df_with_dates['date'])
    
    if not nepse_df.empty:
        df_with_dates = df_with_dates.merge(nepse_df, on='date', how='left')
        df_with_dates['nepse_close'] = df_with_dates['nepse_close'].ffill().bfill()
    else:
        df_with_dates['nepse_close'] = prices
    
    if not sector_df.empty:
        df_with_dates = df_with_dates.merge(sector_df, on='date', how='left')
        df_with_dates['sector_close'] = df_with_dates['sector_close'].ffill().bfill()
    else:
        df_with_dates['sector_close'] = None
    
    nepse_closes = df_with_dates['nepse_close'].values.astype(np.float64)
    sector_closes = df_with_dates['sector_close'].values.astype(np.float64) if sector_code and df_with_dates['sector_close'].notna().any() else None
    
    # Compute features for latest data
    stock_indicators = compute_technical_indicators(prices, volumes)
    market_features, aligned_len = compute_market_features(prices, nepse_closes, sector_closes)
    
    offset = len(prices) - aligned_len
    
    # Build feature matrix for prediction
    stock_feature_names = ['returns', 'momentum_5', 'momentum_20', 'price_to_sma', 
                           'volatility', 'rsi', 'macd', 'bb_position', 'adx', 'roc']
    if volumes is not None:
        stock_feature_names.extend(['volume_ratio', 'vol_momentum', 'mfi', 'vwap_diff'])
    
    market_feature_names = ['nepse_returns', 'nepse_momentum', 'rolling_beta', 
                            'relative_strength', 'market_volatility',
                            'sector_returns', 'sector_momentum', 'sector_relative_strength']
    
    stock_features = [stock_indicators[f][offset:][-lookback:] for f in stock_feature_names]
    market_features_list = [market_features[f][-lookback:] for f in market_feature_names]
    
    latest_features = np.column_stack(stock_features + market_features_list)
    latest_features = np.nan_to_num(latest_features, nan=0.0, posinf=0.0, neginf=0.0)
    scaled_latest = scaler_info['feature_scaler'].transform(latest_features)
    latest_input = scaled_latest.reshape(1, lookback, -1)
    
    predictions = ensemble.predict_with_uncertainty(latest_input)
    
    current_price = df['close'].iloc[-1]
    current_date = pd.to_datetime(df['date'].iloc[-1])
    
    # Get latest market trend for context
    nepse_latest_change = nepse_df['nepse_pct_change'].iloc[-1] if not nepse_df.empty else 0
    
    print(f"\n Current Price: NPR {current_price:.2f} as of {current_date.date()}")
    print(f" NEPSE Change: {nepse_latest_change:+.2f}%")
    if sector_code:
        sector_latest = sector_df['sector_pct_change'].iloc[-1] if not sector_df.empty else 0
        print(f" {sector_code} Change: {sector_latest:+.2f}%")
    
    print(f"\n{'Horizon':<10} {'Predicted':<15} {'95% CI':<25} {'Confidence':<12}")
    print(f"{'-'*70}")
    
    results = {
        'symbol': symbol,
        'current_price': float(current_price),
        'current_date': str(current_date.date()),
        'sector': sector_code,
        'market_context': {
            'nepse_change': float(nepse_latest_change),
        },
        'predictions': {},
        'training_time': training_time,
    }
    
    for horizon in horizons:
        pred_data = predictions[horizon]
        
        # Convert normalized returns back to price
        mean_return = scaler_info['target_scaler'].inverse_transform(
            pred_data['mean'].reshape(-1, 1)
        )[0, 0]
        std_return = pred_data['std'][0, 0] * scaler_info['target_scaler'].scale_[0]
        
        pred_price = current_price * (1 + mean_return)
        lower_price = current_price * (1 + mean_return - 1.96 * std_return)
        upper_price = current_price * (1 + mean_return + 1.96 * std_return)
        
        confidence = ensemble.get_confidence_level(std_return, horizon)
        
        # Calculate target date
        target_date = current_date
        days_added = 0
        while days_added < horizon:
            target_date += timedelta(days=1)
            if target_date.weekday() < 5:
                days_added += 1
        
        pct_change = (pred_price - current_price) / current_price * 100
        
        results['predictions'][horizon] = {
            'target_date': str(target_date.date()),
            'predicted_price': float(pred_price),
            'lower_bound': float(lower_price),
            'upper_bound': float(upper_price),
            'pct_change': float(pct_change),
            'confidence': confidence,
        }
        
        ci_str = f"[{lower_price:.2f} - {upper_price:.2f}]"
        arrow = "📈" if pct_change > 0 else "📉" if pct_change < 0 else "➡️"
        print(f"{horizon:2d}-day{'':<5} NPR {pred_price:>8.2f} {arrow}{'':<3} {ci_str:<25} {confidence:<12}")
    
    print(f"\n{'='*70}")
    print(f"✓ Model uses {scaler_info['n_features']} features (Technical + Market)")
    if sector_code:
        print(f"✓ Sector-aware: {sector_code} index included")
    print(f"{'='*70}")
    
    return results


    from sklearn.preprocessing import MinMaxScaler
    
    prices = df['close'].values.reshape(-1, 1)
    
    # Normalize
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)
    
    # Create sequences with multiple targets
    X = []
    y = {h: [] for h in horizons}
    
    max_horizon = max(horizons)
    for i in range(lookback, len(scaled_prices) - max_horizon):
        X.append(scaled_prices[i-lookback:i, 0])
        
        for horizon in horizons:
            y[horizon].append(scaled_prices[i + horizon - 1, 0])
    
    X = np.array(X).reshape(-1, lookback, 1)
    y = {h: np.array(targets).reshape(-1, 1) for h, targets in y.items()}
    
    # Split
    split_idx = int(len(X) * train_split)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train = {h: targets[:split_idx] for h, targets in y.items()}
    y_test = {h: targets[split_idx:] for h, targets in y.items()}
    
    return X_train, y_train, X_test, y_test, scaler


def forecast_stock(
    symbol: str,
    lookback: int = 120,
    epochs: int = 100,
    hidden_size: int = 512,
    num_blocks: int = 7,
    batch_size: int = 32,
    horizons: List[int] = [1, 3, 5, 10, 15, 21],
):
    """Complete forecasting pipeline"""
    
    print(f"\n{'='*70}")
    print(f"xLSTM MULTI-HORIZON FORECASTING: {symbol}")
    print(f"{'='*70}")
    print(f"Paper: 'xLSTM: Extended Long Short-Term Memory' (Beck et al. 2024)")
    print(f"Horizons: {horizons} days")
    print(f"{'='*70}\n")
    
    # Load data
    conn = sqlite3.connect(config.DB_PATH)
    query = f"""
        SELECT date, open, high, low, close, volume
        FROM price_history
        WHERE symbol = '{symbol}'
        ORDER BY date ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if len(df) < lookback + max(horizons):
        print(f"❌ Not enough data")
        return None
    
    print(f"✓ Loaded {len(df)} days of data")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Current price: NPR {df['close'].iloc[-1]:.2f}")
    
    # Prepare data
    print(f"\n📊 Preparing multi-horizon data...")
    X_train, y_train, X_test, y_test, scaler = prepare_multi_horizon_data(
        df, lookback=lookback, horizons=horizons
    )
    
    print(f"✓ Train samples: {len(X_train)}")
    print(f"✓ Test samples: {len(X_test)}")
    
    # Create datasets
    train_dataset = MultiHorizonDataset(X_train, y_train)
    test_dataset = MultiHorizonDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"\n🚀 Building xLSTM model (device: {device.upper()})...")
    
    model = xLSTMStockForecaster(
        input_size=1,
        hidden_size=hidden_size,
        num_blocks=num_blocks,
        num_heads=8,
        dropout=0.1,
        horizons=horizons,
    )
    
    params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model parameters: {params:,}")
    
    # Train
    trainer = xLSTMForecasterTrainer(model, device=device, learning_rate=0.0001)
    
    print(f"\n{'='*70}")
    print(f"Training for {epochs} epochs...")
    print(f"{'='*70}")
    
    import time
    start_time = time.time()
    history = trainer.fit(train_loader, test_loader, epochs=epochs, verbose=True)
    training_time = time.time() - start_time
    
    print(f"\n✓ Training completed in {training_time:.2f}s")
    
    # Predict future prices
    print(f"\n{'='*70}")
    print(f"FORECASTING FUTURE PRICES")
    print(f"{'='*70}")
    
    latest_data = df['close'].values[-lookback:].reshape(-1, 1)
    latest_scaled = scaler.transform(latest_data)
    latest_input = latest_scaled.reshape(1, lookback, 1)
    
    predictions_scaled = trainer.predict(latest_input)
    
    current_price = df['close'].iloc[-1]
    current_date = pd.to_datetime(df['date'].iloc[-1])
    
    results = {
        'symbol': symbol,
        'current_price': float(current_price),
        'current_date': str(current_date.date()),
        'predictions': {},
        'training_time': training_time,
        'model_params': params,
    }
    
    print(f"\n Current Price (NPR {current_price:.2f}) as of {current_date.date()}")
    print(f"\n{'Horizon':<10} {'Target Date':<15} {'Predicted Price':<18} {'Change':<12} {'% Change':<10}")
    print(f"{'-'*70}")
    
    for horizon in horizons:
        pred_scaled = predictions_scaled[horizon][0][0]
        pred_price = scaler.inverse_transform([[pred_scaled]])[0][0]
        
        # Calculate target date (skip weekends)
        target_date = current_date
        days_added = 0
        while days_added < horizon:
            target_date += timedelta(days=1)
            if target_date.weekday() < 5:  # Monday = 0, Friday = 4
                days_added += 1
        
        change = pred_price - current_price
        pct_change = (change / current_price) * 100
        
        results['predictions'][horizon] = {
            'target_date': str(target_date.date()),
            'predicted_price': float(pred_price),
            'change': float(change),
            'pct_change': float(pct_change),
        }
        
        change_str = f"+NPR {change:.2f}" if change >= 0 else f"NPR {change:.2f}"
        pct_str = f"+{pct_change:.2f}%" if pct_change >= 0 else f"{pct_change:.2f}%"
        
        print(f"{horizon:2d}-day{'':<5} {str(target_date.date()):<15} NPR {pred_price:>10.2f}{'':<5} {change_str:<12} {pct_str:<10}")
    
    # Save model
    model_dir = Path('xlstm_models')
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f'{symbol}_xlstm_multihorizon.pt'
    trainer.save(str(model_path))
    
    results['model_path'] = str(model_path)
    
    # Save results
    results_dir = Path('xlstm_forecasts')
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f'{symbol}_forecast_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✓ Model saved: {model_path}")
    print(f"✓ Results saved: {results_file}")
    print(f"{'='*70}")
    
    return results


def forecast_with_ensemble(
    symbol: str,
    lookback: int = 120,
    epochs: int = 50,
    n_models: int = 5,
    hidden_size: int = 512,
    num_blocks: int = 7,
    batch_size: int = 32,
    horizons: List[int] = [1, 3, 5, 10, 15, 21],
):
    """
    Forecast using ensemble of models with uncertainty estimates.
    """
    print(f"\n{'='*70}")
    print(f"xLSTM ENSEMBLE FORECASTING: {symbol}")
    print(f"{'='*70}")
    print(f"Ensemble: {n_models} models")
    print(f"Features: 14 technical indicators")
    print(f"Horizons: {horizons} days")
    print(f"{'='*70}\n")
    
    # Load data
    conn = sqlite3.connect(config.DB_PATH)
    query = f"""
        SELECT date, open, high, low, close, volume
        FROM price_history
        WHERE symbol = '{symbol}'
        ORDER BY date ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if len(df) < lookback + max(horizons) + 50:
        print(f"❌ Not enough data")
        return None
    
    print(f"✓ Loaded {len(df)} days of data")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Prepare enhanced data
    print(f"\n📊 Preparing enhanced data with technical indicators...")
    X_train, y_train, X_test, y_test, scaler_info = prepare_enhanced_data(
        df, lookback=lookback, horizons=horizons
    )
    
    print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"✓ Features: {scaler_info['n_features']}")
    
    # Create datasets
    train_dataset = MultiHorizonDataset(X_train, y_train)
    test_dataset = MultiHorizonDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create ensemble
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"\n🚀 Building ensemble ({device.upper()})...")
    
    ensemble = EnsembleForecaster(
        n_models=n_models,
        input_size=scaler_info['n_features'],
        hidden_size=hidden_size,
        num_blocks=num_blocks,
        horizons=horizons,
        device=device,
    )
    
    # Train ensemble
    import time
    start_time = time.time()
    ensemble.fit(train_loader, test_loader, epochs=epochs, verbose=True)
    training_time = time.time() - start_time
    
    print(f"\n✓ Ensemble training completed in {training_time:.2f}s")
    
    # Make predictions with uncertainty
    print(f"\n{'='*70}")
    print(f"FORECASTING WITH UNCERTAINTY")
    print(f"{'='*70}")
    
    # Prepare latest data
    prices = df['close'].values.astype(np.float64)
    volumes = df['volume'].values.astype(np.float64) if 'volume' in df.columns else None
    indicators = compute_technical_indicators(prices, volumes)
    
    feature_names = scaler_info['feature_names']
    latest_features = np.column_stack([indicators[f][-lookback:] for f in feature_names])
    scaled_latest = scaler_info['feature_scaler'].transform(latest_features)
    latest_input = scaled_latest.reshape(1, lookback, -1)
    
    predictions = ensemble.predict_with_uncertainty(latest_input)
    
    current_price = df['close'].iloc[-1]
    current_date = pd.to_datetime(df['date'].iloc[-1])
    
    print(f"\n Current Price: NPR {current_price:.2f} as of {current_date.date()}")
    print(f"\n{'Horizon':<10} {'Predicted':<15} {'95% CI':<25} {'Confidence':<12}")
    print(f"{'-'*70}")
    
    results = {
        'symbol': symbol,
        'current_price': float(current_price),
        'predictions': {},
    }
    
    for horizon in horizons:
        pred_data = predictions[horizon]
        
        # Convert normalized returns back to price
        mean_return = scaler_info['target_scaler'].inverse_transform(
            pred_data['mean'].reshape(-1, 1)
        )[0, 0]
        std_return = pred_data['std'][0, 0] * scaler_info['target_scaler'].scale_[0]
        
        pred_price = current_price * (1 + mean_return)
        lower_price = current_price * (1 + mean_return - 1.96 * std_return)
        upper_price = current_price * (1 + mean_return + 1.96 * std_return)
        
        confidence = ensemble.get_confidence_level(std_return, horizon)
        
        results['predictions'][horizon] = {
            'predicted_price': float(pred_price),
            'lower_bound': float(lower_price),
            'upper_bound': float(upper_price),
            'std': float(std_return),
            'confidence': confidence,
        }
        
        ci_str = f"[{lower_price:.2f} - {upper_price:.2f}]"
        print(f"{horizon:2d}-day{'':<5} NPR {pred_price:>8.2f}{'':<5} {ci_str:<25} {confidence:<12}")
    
    print(f"\n{'='*70}")
    print(f"✓ Training time: {training_time:.2f}s")
    print(f"{'='*70}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='xLSTM Multi-Horizon Stock Forecaster')
    parser.add_argument('symbol', help='Stock symbol')
    parser.add_argument('--lookback', type=int, default=120, help='Lookback period (default: 120)')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs (default: 100)')
    parser.add_argument('--hidden-size', type=int, default=256, help='Hidden size (default: 256)')
    parser.add_argument('--num-blocks', type=int, default=4, help='Number of xLSTM blocks (default: 4)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--ensemble', action='store_true', help='Use ensemble with uncertainty')
    parser.add_argument('--market', action='store_true', help='Add NEPSE + sector index features (works with --ensemble)')
    parser.add_argument('--n-models', type=int, default=3, help='Number of models in ensemble (default: 3)')
    parser.add_argument('--fast', action='store_true', help='Use fast config (256h, 4b, 50 epochs)')
    parser.add_argument('--m1max', action='store_true', help='Optimized for Mac Studio M1 Max (512h, 6b, 64 batch, 100 epochs)')
    
    args = parser.parse_args()
    
    # M1 Max optimized mode - leverages 64GB unified memory & GPU
    if args.m1max:
        args.hidden_size = 512
        args.num_blocks = 6
        args.epochs = 100
        args.batch_size = 64
        args.n_models = 5
        print("\n🚀 M1 MAX OPTIMIZED MODE")
        print("   Hidden: 512, Blocks: 6, Batch: 64, Epochs: 100, Models: 5\n")
    # Fast mode overrides
    elif args.fast:
        args.hidden_size = 256
        args.num_blocks = 4
        args.epochs = 50
        args.batch_size = 64
        args.n_models = 3
    
    if args.market or args.ensemble:
        # Both --market and --ensemble use ensemble with uncertainty
        # --market adds NEPSE + sector index features
        if args.market:
            forecast_with_market_context(
                symbol=args.symbol,
                lookback=args.lookback,
                epochs=args.epochs,
                n_models=args.n_models,
                hidden_size=args.hidden_size,
                num_blocks=args.num_blocks,
                batch_size=args.batch_size,
            )
        else:
            forecast_with_ensemble(
                symbol=args.symbol,
                lookback=args.lookback,
                epochs=args.epochs,
                n_models=args.n_models,
                hidden_size=args.hidden_size,
                num_blocks=args.num_blocks,
                batch_size=args.batch_size,
            )
    else:
        forecast_stock(
            symbol=args.symbol,
            lookback=args.lookback,
            epochs=args.epochs,
            hidden_size=args.hidden_size,
            num_blocks=args.num_blocks,
            batch_size=args.batch_size,
        )
