"""
Enhanced xLSTM with Sliding Window Forecasting
Based on xLSTM paper recommendations for optimal time series forecasting
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from xlstm_stock_forecaster import xLSTMStockForecaster, xLSTMForecasterTrainer, prepare_multi_horizon_data
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd


class SlidingWindowForecaster:
    """
    Sliding window ensemble forecaster for xLSTM
    
    Key improvements from xLSTM paper:
    1. Multiple overlapping windows for robust predictions
    2. Exponential weighting favoring recent data
    3. Uncertainty estimation from window variance
    4. Longer lookback (xLSTM excels at long sequences)
    """
    
    def __init__(
        self,
        model: xLSTMStockForecaster,
        trainer: xLSTMForecasterTrainer,
        lookback: int = 120,  # Longer lookback - xLSTM handles it well
        stride: int = 10,      # Overlap windows
        min_windows: int = 5,  # Minimum windows for ensemble
    ):
        """
        Args:
            model: Trained xLSTM model
            trainer: Model trainer
            lookback: Window size (paper shows xLSTM works well with 100-200)
            stride: Step size between windows (smaller = more overlap)
            min_windows: Minimum windows needed for prediction
        """
        self.model = model
        self.trainer = trainer
        self.lookback = lookback
        self.stride = stride
        self.min_windows = min_windows
        self.device = trainer.device
    
    def _create_sliding_windows(self, data: np.ndarray) -> List[np.ndarray]:
        """
        Create overlapping windows from data
        
        Returns:
            List of windows, each of shape (lookback, features)
        """
        windows = []
        n = len(data)
        
        # Create windows with stride
        for i in range(0, n - self.lookback + 1, self.stride):
            window = data[i:i + self.lookback]
            windows.append(window)
        
        # Always include the most recent window
        if len(windows) == 0 or len(data[n - self.lookback:]) == self.lookback:
            latest_window = data[n - self.lookback:]
            if len(latest_window) == self.lookback:
                if len(windows) == 0 or not np.array_equal(windows[-1], latest_window):
                    windows.append(latest_window)
        
        return windows
    
    def _exponential_weights(self, n: int, decay: float = 0.9) -> np.ndarray:
        """
        Create exponential weights favoring recent windows
        
        Args:
            n: Number of windows
            decay: Decay factor (closer to 1 = stronger recency bias)
        
        Returns:
            Normalized weights array
        """
        weights = np.array([decay ** (n - i - 1) for i in range(n)])
        return weights / weights.sum()
    
    def predict_with_uncertainty(
        self,
        data: np.ndarray,
        scaler = None,
    ) -> Dict[int, Dict[str, float]]:
        """
        Make predictions with uncertainty estimates using sliding windows
        
        Args:
            data: Historical price data (n_days, 1)
            scaler: Scaler used for normalization
        
        Returns:
            Dict of {horizon: {
                'prediction': mean prediction,
                'std': standard deviation,
                'confidence_lower': lower bound (mean - 2*std),
                'confidence_upper': upper bound (mean + 2*std),
                'num_windows': number of windows used
            }}
        """
        self.model.eval()
        
        # Create sliding windows
        windows = self._create_sliding_windows(data)
        
        if len(windows) < self.min_windows:
            raise ValueError(
                f"Not enough data for sliding window prediction. "
                f"Need at least {self.lookback + (self.min_windows - 1) * self.stride} samples, "
                f"got {len(data)}"
            )
        
        # Get predictions from each window
        all_predictions = {h: [] for h in self.model.horizons}
        
        for window in windows:
            # Prepare input (1, lookback, 1)
            X = torch.FloatTensor(window).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                preds = self.model(X)
            
            # Store predictions for each horizon
            for h in self.model.horizons:
                pred = preds[h].cpu().numpy()[0, 0]
                all_predictions[h].append(pred)
        
        # Calculate weighted ensemble with uncertainty
        weights = self._exponential_weights(len(windows))
        results = {}
        
        for h in self.model.horizons:
            preds = np.array(all_predictions[h])
            
            # Weighted mean prediction
            mean_pred = np.average(preds, weights=weights)
            
            # Weighted standard deviation for uncertainty
            variance = np.average((preds - mean_pred) ** 2, weights=weights)
            std = np.sqrt(variance)
            
            # Inverse transform if scaler provided
            if scaler is not None:
                mean_pred_scaled = scaler.inverse_transform([[mean_pred]])[0, 0]
                std_scaled = std * (scaler.data_max_[0] - scaler.data_min_[0])
            else:
                mean_pred_scaled = mean_pred
                std_scaled = std
            
            results[h] = {
                'prediction': float(mean_pred_scaled),
                'std': float(std_scaled),
                'confidence_lower': float(mean_pred_scaled - 2 * std_scaled),  # 95% CI
                'confidence_upper': float(mean_pred_scaled + 2 * std_scaled),
                'num_windows': len(windows),
                'window_predictions': preds.tolist(),  # For debugging
            }
        
        return results
    
    def evaluate_with_sliding_window(
        self,
        X_test: np.ndarray,
        y_test: Dict[int, np.ndarray],
        scaler = None,
    ) -> Dict[int, Dict[str, float]]:
        """
        Evaluate model using sliding window on test set
        
        Args:
            X_test: Test sequences
            y_test: True values for each horizon
            scaler: Scaler for inverse transform
        
        Returns:
            Metrics for each horizon {horizon: {rmse, mae, r2, mape}}
        """
        # For each test sample, use sliding window on its history
        horizon_metrics = {h: {'predictions': [], 'actuals': []} for h in self.model.horizons}
        
        print(f"\nEvaluating with sliding window (stride={self.stride})...")
        
        for i in range(len(X_test)):
            try:
                # Get the sequence for this sample
                sequence = X_test[i]  # (lookback, 1)
                
                # Make prediction with sliding window
                results = self.predict_with_uncertainty(sequence, scaler)
                
                # Store predictions and actuals
                for h in self.model.horizons:
                    horizon_metrics[h]['predictions'].append(results[h]['prediction'])
                    
                    # Get actual value
                    actual = y_test[h][i, 0]
                    if scaler is not None:
                        actual = scaler.inverse_transform([[actual]])[0, 0]
                    horizon_metrics[h]['actuals'].append(actual)
            
            except Exception as e:
                print(f"Warning: Skipped sample {i}: {e}")
                continue
        
        # Calculate metrics for each horizon
        metrics = {}
        for h in self.model.horizons:
            preds = np.array(horizon_metrics[h]['predictions'])
            actuals = np.array(horizon_metrics[h]['actuals'])
            
            if len(preds) > 0 and len(actuals) > 0:
                rmse = np.sqrt(mean_squared_error(actuals, preds))
                mae = mean_absolute_error(actuals, preds)
                r2 = r2_score(actuals, preds)
                mape = np.mean(np.abs((actuals - preds) / (actuals + 1e-8))) * 100
                
                metrics[h] = {
                    'rmse': float(rmse),
                    'mae': float(mae),
                    'r2': float(r2),
                    'mape': float(mape),
                    'samples': len(preds),
                }
            else:
                metrics[h] = {
                    'error': 'No valid predictions',
                }
        
        return metrics


def create_sliding_window_forecaster(
    model_path: str,
    lookback: int = 120,
    stride: int = 10,
) -> SlidingWindowForecaster:
    """
    Load a trained model and wrap it with sliding window forecaster
    
    Args:
        model_path: Path to saved model checkpoint
        lookback: Window size for sliding window
        stride: Step size between windows
    
    Returns:
        SlidingWindowForecaster instance
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Recreate model
    horizons = checkpoint['horizons']
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # You'll need to pass the correct hyperparameters
    # For now, using defaults - ideally save these in checkpoint
    from xlstm_stock_forecaster import xLSTMStockForecaster, xLSTMForecasterTrainer
    
    model = xLSTMStockForecaster(
        input_size=1,
        hidden_size=512,  # Should match trained model
        num_blocks=7,     # Should match trained model
        horizons=horizons,
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create trainer (just for device management)
    trainer = xLSTMForecasterTrainer(model, device=device)
    
    # Create sliding window forecaster
    return SlidingWindowForecaster(
        model=model,
        trainer=trainer,
        lookback=lookback,
        stride=stride,
    )


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test sliding window forecasting')
    parser.add_argument('model_path', type=str, help='Path to trained model')
    parser.add_argument('--symbol', type=str, default='PFL', help='Stock symbol')
    parser.add_argument('--lookback', type=int, default=120, help='Window size')
    parser.add_argument('--stride', type=int, default=10, help='Stride between windows')
    
    args = parser.parse_args()
    
    # Load data
    from config import DB_PATH
    import sqlite3
    
    conn = sqlite3.connect(DB_PATH)
    query = f"""
        SELECT date, close 
        FROM price_history 
        WHERE symbol = '{args.symbol}'
        ORDER BY date
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Prepare data
    from sklearn.preprocessing import MinMaxScaler
    
    prices = df['close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(prices)
    
    # Create forecaster
    forecaster = create_sliding_window_forecaster(
        args.model_path,
        lookback=args.lookback,
        stride=args.stride,
    )
    
    # Make prediction with uncertainty
    recent_data = prices_scaled[-args.lookback:]
    results = forecaster.predict_with_uncertainty(recent_data, scaler)
    
    print(f"\n{'='*80}")
    print(f"Sliding Window Forecast for {args.symbol}")
    print(f"{'='*80}")
    print(f"Lookback: {args.lookback} days")
    print(f"Stride: {args.stride} days")
    print(f"Current price: ${prices[-1, 0]:.2f}")
    print()
    
    for h in sorted(results.keys()):
        r = results[h]
        print(f"{h:2d}-day forecast:")
        print(f"  Prediction: ${r['prediction']:.2f}")
        print(f"  Uncertainty: ±${r['std']:.2f}")
        print(f"  95% CI: [${r['confidence_lower']:.2f}, ${r['confidence_upper']:.2f}]")
        print(f"  Windows: {r['num_windows']}")
        print()
