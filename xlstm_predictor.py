"""
xLSTM Stock Price Predictor using PyTorch
Wrapper for the official xlstm package from NX-AI (Hochreiter et al. 2024)
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
    FeedForwardConfig,
)
from typing import Tuple, Dict, Optional
import json
from pathlib import Path


class StockDataset(Dataset):
    """PyTorch Dataset for stock price sequences"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class xLSTMStockPredictor(nn.Module):
    """
    Stock price predictor using xLSTM architecture
    
    Architecture:
    - Input layer
    - xLSTM block stack (sLSTM + mLSTM blocks)
    - Dense prediction head
    
    Reference: "xLSTM: Extended Long Short-Term Memory" (Hochreiter et al. 2024)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_blocks: int = 2,
        forecast_days: int = 7,
        dropout: float = 0.2,
        use_slstm: bool = True,
        use_mlstm: bool = True,
    ):
        """
        Args:
            input_size: Number of input features per timestep
            hidden_size: Hidden dimension size for xLSTM
            num_blocks: Number of xLSTM blocks (alternating sLSTM and mLSTM)
            forecast_days: Number of future days to predict
            dropout: Dropout rate
            use_slstm: Use scalar LSTM blocks (better for long-range dependencies)
            use_mlstm: Use matrix LSTM blocks (better for storage capacity)
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.forecast_days = forecast_days
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # Build xLSTM block configurations
        block_configs = []
        for i in range(num_blocks):
            if use_slstm and (i % 2 == 0 or not use_mlstm):
                # sLSTM: Better for long sequences
                block_configs.append(
                    sLSTMBlockConfig(
                        slstm=sLSTMLayerConfig(
                            backend="vanilla",  # Use vanilla backend (works on CPU/MPS)
                            num_heads=4,
                            conv1d_kernel_size=4,
                            bias_init="powerlaw_blockdependent",
                        ),
                    )
                )
            elif use_mlstm:
                # mLSTM: Better for storage
                block_configs.append(
                    mLSTMBlockConfig(
                        mlstm=mLSTMLayerConfig(
                            num_heads=4,
                            conv1d_kernel_size=4,
                        ),
                    )
                )
        
        # Create xLSTM stack
        xlstm_config = xLSTMBlockStackConfig(
            mlstm_block=block_configs[0] if isinstance(block_configs[0], mLSTMBlockConfig) else None,
            slstm_block=block_configs[0] if isinstance(block_configs[0], sLSTMBlockConfig) else None,
            context_length=200,  # Max sequence length
            num_blocks=num_blocks,
            embedding_dim=hidden_size,
            add_post_blocks_norm=True,
            bias=False,
            dropout=dropout,
        )
        
        self.xlstm_stack = xLSTMBlockStack(xlstm_config)
        
        # Output head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, forecast_days)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
        
        Returns:
            Predictions of shape (batch_size, forecast_days)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_proj(x)  # (batch, seq_len, hidden_size)
        
        # Pass through xLSTM stack
        x = self.xlstm_stack(x)  # (batch, seq_len, hidden_size)
        
        # Take last timestep output
        x = x[:, -1, :]  # (batch, hidden_size)
        
        # Prediction head
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # (batch, forecast_days)
        
        return x


class xLSTMTrainer:
    """Training wrapper for xLSTM stock predictor"""
    
    def __init__(
        self,
        model: xLSTMStockPredictor,
        device: str = 'mps',
        learning_rate: float = 0.001,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': []}
    
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_mae = 0
        
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(X_batch)
            loss = self.criterion(predictions, y_batch)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_mae += torch.mean(torch.abs(predictions - y_batch)).item()
        
        avg_loss = total_loss / len(dataloader)
        avg_mae = total_mae / len(dataloader)
        return avg_loss, avg_mae
    
    def validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        total_mae = 0
        
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                predictions = self.model(X_batch)
                loss = self.criterion(predictions, y_batch)
                
                total_loss += loss.item()
                total_mae += torch.mean(torch.abs(predictions - y_batch)).item()
        
        avg_loss = total_loss / len(dataloader)
        avg_mae = total_mae / len(dataloader)
        return avg_loss, avg_mae
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        early_stopping_patience: int = 15,
        verbose: bool = True,
    ) -> Dict:
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            early_stopping_patience: Stop if no improvement for this many epochs
            verbose: Print progress
        
        Returns:
            Training history dictionary
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss, train_mae = self.train_epoch(train_loader)
            val_loss, val_mae = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_mae'].append(train_mae)
            self.history['val_mae'].append(val_mae)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train Loss: {train_loss:.6f}, MAE: {train_mae:.2f}")
                print(f"  Val Loss: {val_loss:.6f}, MAE: {val_mae:.2f}")
            
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
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        return predictions.cpu().numpy()
    
    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)


def prepare_data_for_xlstm(
    df: pd.DataFrame,
    lookback: int = 60,
    forecast_days: int = 7,
    train_split: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, object, float, float]:
    """
    Prepare stock data for xLSTM training
    
    Args:
        df: DataFrame with stock data (must have 'close' column)
        lookback: Number of days to look back
        forecast_days: Number of days to forecast
        train_split: Fraction of data for training
    
    Returns:
        X_train, y_train, X_test, y_test, scaler, scale_factor, min_val
    """
    from sklearn.preprocessing import MinMaxScaler
    
    # Get close prices
    prices = df['close'].values.reshape(-1, 1)
    
    # Normalize
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)
    
    # Create sequences
    X, y = [], []
    for i in range(lookback, len(scaled_prices) - forecast_days + 1):
        X.append(scaled_prices[i-lookback:i, 0])
        y.append(scaled_prices[i:i+forecast_days, 0])
    
    X = np.array(X)
    y = np.array(y)
    
    # Reshape X for xLSTM: (samples, timesteps, features)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # Split train/test
    split_idx = int(len(X) * train_split)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    scale_factor = scaler.scale_[0]
    min_val = scaler.min_[0]
    
    return X_train, y_train, X_test, y_test, scaler, scale_factor, min_val


if __name__ == '__main__':
    print("xLSTM Stock Predictor Module")
    print("=" * 50)
    print("✓ PyTorch version:", torch.__version__)
    print("✓ MPS (Metal GPU) available:", torch.backends.mps.is_available())
    print("\nUsage:")
    print("  from xlstm_predictor import xLSTMStockPredictor, xLSTMTrainer, prepare_data_for_xlstm")
    print("\nExample:")
    print("  model = xLSTMStockPredictor(input_size=1, hidden_size=128)")
    print("  trainer = xLSTMTrainer(model, device='mps')")
    print("  history = trainer.fit(train_loader, val_loader, epochs=100)")
