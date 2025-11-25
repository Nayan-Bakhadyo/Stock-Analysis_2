"""
Improved Hyperparameter Tuning for LSTM Stock Price Predictor
Key improvements:
1. Rolling/expanding window validation with multiple test periods
2. Predicts log-returns instead of absolute prices for stationarity
3. Final hold-out period never touched during tuning
4. More robust cross-validation
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
import os
import gc
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# ML/DL libraries
try:
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
    import tensorflow as tf
    
    # Configure TensorFlow memory growth for M1
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (LSTM, Dense, Dropout, Bidirectional, GRU, 
                                         Conv1D, BatchNormalization, Activation, Add, 
                                         Input, GlobalAveragePooling1D, SpatialDropout1D,
                                         MultiHeadAttention, LayerNormalization, Flatten)
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers.legacy import Adam, RMSprop  # Legacy optimizers for M1/M2 performance
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available. Install with: pip install tensorflow scikit-learn")
    exit(1)

from data_fetcher import NepseDataFetcher


class ImprovedHyperparameterTuner:
    """Improved tuner with rolling validation and returns-based prediction"""
    
    def __init__(self, symbol: str, test_days: int = 7, final_holdout_days: int = 30):
        """
        Initialize tuner
        
        Args:
            symbol: Stock symbol to tune for
            test_days: Number of days per validation window (default: 7)
            final_holdout_days: Days reserved for final evaluation, never used in tuning (default: 30)
        """
        self.symbol = symbol
        self.test_days = test_days
        self.final_holdout_days = final_holdout_days
        self.data_fetcher = NepseDataFetcher()
        self.results = []
        
    def load_data_with_holdout(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load data with proper train/validation/holdout split
        
        Returns:
            train_df: Training data (never includes holdout)
            validation_windows: List of validation periods for rolling CV
            holdout_df: Final holdout (NEVER touched during tuning)
        """
        print(f"\n{'='*70}")
        print(f"LOADING DATA FOR {self.symbol} WITH IMPROVED VALIDATION")
        print('='*70)
        
        # Load price history
        df = self.data_fetcher.get_stock_price_history(self.symbol)
        
        if df is None or len(df) == 0:
            raise ValueError(f"No data found for {self.symbol}")
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"✓ Total data points: {len(df)}")
        print(f"✓ Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"✓ Latest price: {df['close'].iloc[-1]:.2f}")
        
        # Reserve final holdout (NEVER used in training or validation)
        holdout_split = len(df) - self.final_holdout_days
        available_df = df.iloc[:holdout_split].copy()
        holdout_df = df.iloc[holdout_split:].copy()
        
        print(f"\n📊 Data Split Strategy:")
        print(f"  Available for training/validation: {len(available_df)} days")
        print(f"  Final holdout (untouched): {len(holdout_df)} days ({holdout_df['date'].min()} to {holdout_df['date'].max()})")
        
        return available_df, holdout_df
    
    def create_rolling_validation_windows(self, df: pd.DataFrame, num_windows: int = 3) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create multiple disjoint validation windows for robust evaluation
        
        Args:
            df: Available data (excluding final holdout)
            num_windows: Number of validation windows to create
            
        Returns:
            List of (train, val) tuples
        """
        windows = []
        total_len = len(df)
        
        # Calculate window size (each validation is test_days long)
        # Leave enough room for minimum training data (e.g., 200 days)
        min_train_days = 200
        available_for_validation = total_len - min_train_days
        
        if available_for_validation < num_windows * self.test_days:
            num_windows = max(1, available_for_validation // self.test_days)
            print(f"⚠️ Adjusted validation windows to {num_windows} due to data constraints")
        
        # Create equally spaced validation windows
        step = (available_for_validation - self.test_days) // (num_windows - 1) if num_windows > 1 else 0
        
        for i in range(num_windows):
            val_start = min_train_days + i * step
            val_end = val_start + self.test_days
            
            if val_end > total_len:
                break
                
            train_df = df.iloc[:val_start].copy()
            val_df = df.iloc[val_start:val_end].copy()
            
            windows.append((train_df, val_df))
            
            print(f"  Window {i+1}: Train={len(train_df)} days, Val={len(val_df)} days ({val_df['date'].min()} to {val_df['date'].max()})")
        
        return windows
    
    def prepare_features_with_returns(self, df: pd.DataFrame, predict_returns: bool = True) -> pd.DataFrame:
        """
        Prepare features with log-returns as the primary target
        
        Args:
            df: Input dataframe
            predict_returns: If True, use log-returns as target; if False, use prices
        """
        data = df.copy()
        
        # Price-based features
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Moving averages (normalized by price)
        data['sma_5'] = data['close'].rolling(window=5).mean() / data['close']
        data['sma_10'] = data['close'].rolling(window=10).mean() / data['close']
        data['sma_20'] = data['close'].rolling(window=20).mean() / data['close']
        data['sma_50'] = data['close'].rolling(window=50).mean() / data['close']
        
        # EMA
        data['ema_12'] = data['close'].ewm(span=12, adjust=False).mean() / data['close']
        data['ema_26'] = data['close'].ewm(span=26, adjust=False).mean() / data['close']
        
        # MACD (normalized)
        ema12 = data['close'].ewm(span=12, adjust=False).mean()
        ema26 = data['close'].ewm(span=26, adjust=False).mean()
        data['macd'] = (ema12 - ema26) / data['close']
        data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
        data['macd_diff'] = data['macd'] - data['macd_signal']
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        data['rsi_norm'] = (data['rsi'] - 50) / 50  # Normalize RSI to [-1, 1]
        
        # Bollinger Bands (normalized)
        bb_middle = data['close'].rolling(window=20).mean()
        bb_std = data['close'].rolling(window=20).std()
        data['bb_position'] = (data['close'] - bb_middle) / bb_std  # Position within bands
        data['bb_width'] = (2 * bb_std) / bb_middle  # Width relative to price
        
        # Volume indicators (log-normalized)
        data['volume_log'] = np.log1p(data['volume'])
        data['volume_sma'] = data['volume'].rolling(window=20).mean()
        data['volume_ratio'] = data['volume'] / (data['volume_sma'] + 1e-8)
        
        # Volatility (ATR normalized by price)
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        data['atr'] = true_range.rolling(14).mean() / data['close']
        
        # Price momentum (as returns)
        data['momentum_5'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
        data['momentum_10'] = (data['close'] - data['close'].shift(10)) / data['close'].shift(10)
        
        # Rolling volatility
        data['volatility_20'] = data['returns'].rolling(20).std()
        
        # Drop NaN
        data = data.dropna()
        
        return data
    
    def create_sequences_returns(self, data: np.ndarray, prices: np.ndarray, 
                                 lookback: int, predict_returns: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sequences predicting log-returns instead of prices
        
        Args:
            data: Feature matrix (includes log_returns as first column if predict_returns=True)
            prices: Actual prices for reference
            lookback: Lookback window
            predict_returns: If True, predict log-returns; if False, predict prices
            
        Returns:
            X: Input sequences
            y: Target (log-returns or prices for next 7 days)
            base_prices: Reference prices for reconstructing predictions
        """
        X, y, base_prices = [], [], []
        
        for i in range(lookback, len(data) - self.test_days + 1):
            X.append(data[i-lookback:i])
            
            if predict_returns:
                # Predict log-returns for next 7 days
                future_returns = []
                for j in range(self.test_days):
                    if i + j < len(prices):
                        log_ret = np.log(prices[i + j] / prices[i - 1])
                        future_returns.append(log_ret)
                    else:
                        future_returns.append(0)
                y.append(future_returns)
                base_prices.append(prices[i - 1])  # Price at end of lookback window
            else:
                # Original: predict absolute prices
                y.append(prices[i:i+self.test_days])
                base_prices.append(prices[i - 1])
        
        return np.array(X), np.array(y), np.array(base_prices)
    
    def reconstruct_prices_from_returns(self, log_returns: np.ndarray, base_price: float) -> np.ndarray:
        """
        Reconstruct prices from predicted log-returns
        
        Args:
            log_returns: Array of log-returns for next N days
            base_price: Starting price
            
        Returns:
            Array of predicted prices
        """
        prices = []
        current_price = base_price
        
        for log_ret in log_returns:
            # P_t = P_{t-1} * exp(r_t)
            current_price = current_price * np.exp(log_ret)
            prices.append(current_price)
        
        return np.array(prices)
    
    def build_model(self, config: Dict, input_shape: Tuple[int, int]):
        """Build model with given configuration (same as before but works with returns)"""
        # Using bidirectional LSTM as default
        if config['architecture'] == 'bidirectional':
            model = Sequential()
            model.add(Bidirectional(LSTM(config['units_1'], return_sequences=True), input_shape=input_shape))
            model.add(Dropout(config['dropout_1']))
            
            if config['layers'] >= 2:
                model.add(Bidirectional(LSTM(config['units_2'], return_sequences=True)))
                model.add(Dropout(config['dropout_2']))
            
            if config['layers'] >= 3:
                model.add(LSTM(config['units_3'], return_sequences=False))
                model.add(Dropout(config['dropout_3']))
            else:
                model.add(LSTM(config['units_2'] // 2, return_sequences=False))
                model.add(Dropout(config['dropout_2']))
            
            model.add(Dense(config['dense_units'], activation='relu'))
            model.add(Dropout(0.1))
            model.add(Dense(config['dense_units'] // 2, activation='relu'))
            model.add(Dense(7))  # Output 7 predictions (returns or prices)
        
        else:  # GRU or standard LSTM
            model = Sequential()
            
            if config['architecture'] == 'gru':
                model.add(GRU(config['units_1'], return_sequences=True, input_shape=input_shape))
            else:
                model.add(LSTM(config['units_1'], return_sequences=True, input_shape=input_shape))
            model.add(Dropout(config['dropout_1']))
            
            if config['layers'] >= 2:
                if config['architecture'] == 'gru':
                    model.add(GRU(config['units_2'], return_sequences=True))
                else:
                    model.add(LSTM(config['units_2'], return_sequences=True))
                model.add(Dropout(config['dropout_2']))
            
            if config['layers'] >= 3:
                if config['architecture'] == 'gru':
                    model.add(GRU(config['units_3'], return_sequences=False))
                else:
                    model.add(LSTM(config['units_3'], return_sequences=False))
                model.add(Dropout(config['dropout_3']))
            else:
                if config['architecture'] == 'gru':
                    model.add(GRU(config['units_2'] // 2, return_sequences=False))
                else:
                    model.add(LSTM(config['units_2'] // 2, return_sequences=False))
                model.add(Dropout(config['dropout_2']))
            
            model.add(Dense(config['dense_units'], activation='relu'))
            model.add(Dropout(0.1))
            model.add(Dense(config['dense_units'] // 2, activation='relu'))
            model.add(Dense(7))
        
        optimizer = Adam(learning_rate=config['learning_rate'])
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
        
        return model
    
    def train_and_evaluate_rolling(self, config: Dict, available_df: pd.DataFrame, 
                                   holdout_df: pd.DataFrame, predict_returns: bool = True) -> Dict:
        """
        Train with rolling validation and evaluate on holdout
        
        Args:
            config: Model configuration
            available_df: Data available for training/validation
            holdout_df: Final holdout set (never used in training)
            predict_returns: If True, predict returns; if False, predict prices
        """
        print(f"\n{'='*70}")
        print(f"Testing Configuration (Returns-Based: {predict_returns}):")
        for key, val in config.items():
            print(f"  {key}: {val}")
        print('='*70)
        
        try:
            # Create rolling validation windows
            windows = self.create_rolling_validation_windows(available_df, num_windows=3)
            
            window_results = []
            
            # Train and validate on each window
            for window_idx, (train_df, val_df) in enumerate(windows):
                print(f"\n--- Validation Window {window_idx + 1}/{len(windows)} ---")
                
                # Prepare features
                train_data = self.prepare_features_with_returns(train_df, predict_returns)
                
                # Select features (log_returns is now first feature if predicting returns)
                if predict_returns:
                    feature_cols = ['log_returns', 'volume_ratio', 'rsi_norm', 'bb_position', 
                                   'macd', 'bb_width', 'atr', 'momentum_5', 'volatility_20']
                else:
                    feature_cols = ['close', 'volume', 'returns', 'sma_5', 'sma_20', 'rsi', 
                                   'macd', 'bb_width', 'volume_ratio', 'atr', 'momentum_5']
                
                feature_cols = [col for col in feature_cols if col in train_data.columns]
                
                features = train_data[feature_cols].values
                prices = train_data['close'].values
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Scale features
                scaler = StandardScaler() if predict_returns else MinMaxScaler(feature_range=(0, 1))
                scaled_features = scaler.fit_transform(features)
                
                # Create sequences
                X, y, base_prices = self.create_sequences_returns(
                    scaled_features, prices, config['lookback'], predict_returns
                )
                
                if len(X) < 10:
                    print("  ⚠️ Insufficient data for this window")
                    continue
                
                # Split into train/val within this window
                split_idx = int(len(X) * 0.85)  # Use 85% for training, 15% for validation
                X_train, X_val = X[:split_idx], X[split_idx:]
                y_train, y_val = y[:split_idx], y[split_idx:]
                base_prices_val = base_prices[split_idx:]
                
                print(f"  Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
                
                # Build model
                model = self.build_model(config, (config['lookback'], len(feature_cols)))
                
                # Callbacks
                early_stop = EarlyStopping(monitor='val_loss', patience=config['patience'], 
                                          restore_best_weights=True, verbose=0)
                reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, 
                                            min_lr=0.00001, verbose=0)
                
                # Train
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=config['epochs'],
                    batch_size=config['batch_size'],
                    callbacks=[early_stop, reduce_lr],
                    verbose=0
                )
                
                # Evaluate on validation set
                val_pred_returns = model.predict(X_val, verbose=0)
                
                # Convert returns to prices if predicting returns
                if predict_returns:
                    val_pred_prices = np.array([
                        self.reconstruct_prices_from_returns(pred, base) 
                        for pred, base in zip(val_pred_returns, base_prices_val)
                    ])
                    # Get actual prices for comparison
                    val_actual_prices = np.array([
                        self.reconstruct_prices_from_returns(actual, base)
                        for actual, base in zip(y_val, base_prices_val)
                    ])
                else:
                    val_pred_prices = val_pred_returns
                    val_actual_prices = y_val
                
                # Calculate metrics
                val_mae = mean_absolute_error(val_actual_prices.flatten(), val_pred_prices.flatten())
                val_mape = mean_absolute_percentage_error(val_actual_prices.flatten(), val_pred_prices.flatten()) * 100
                
                window_results.append({
                    'val_mae': val_mae,
                    'val_mape': val_mape
                })
                
                print(f"  ✓ Window {window_idx + 1} - Val MAE: {val_mae:.4f}, Val MAPE: {val_mape:.2f}%")
                
                # Cleanup
                del model
                gc.collect()
            
            # Average across windows
            if not window_results:
                return {'error': 'No valid windows'}
            
            avg_val_mae = np.mean([r['val_mae'] for r in window_results])
            avg_val_mape = np.mean([r['val_mape'] for r in window_results])
            
            print(f"\n  📊 Average across {len(window_results)} windows:")
            print(f"     Val MAE: {avg_val_mae:.4f}, Val MAPE: {avg_val_mape:.2f}%")
            
            # Now evaluate on final holdout (NEVER seen during training)
            print(f"\n  🎯 Evaluating on final holdout ({len(holdout_df)} days)...")
            
            # Prepare full training data (all available data except holdout)
            full_train_data = self.prepare_features_with_returns(available_df, predict_returns)
            full_features = full_train_data[feature_cols].values
            full_prices = full_train_data['close'].values
            full_features = np.nan_to_num(full_features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Scale
            full_scaler = StandardScaler() if predict_returns else MinMaxScaler(feature_range=(0, 1))
            full_scaled = full_scaler.fit_transform(full_features)
            
            # Create sequences for training
            X_full, y_full, base_prices_full = self.create_sequences_returns(
                full_scaled, full_prices, config['lookback'], predict_returns
            )
            
            # Train final model on ALL available data
            final_model = self.build_model(config, (config['lookback'], len(feature_cols)))
            
            early_stop = EarlyStopping(monitor='loss', patience=config['patience'], 
                                      restore_best_weights=True, verbose=0)
            
            final_model.fit(
                X_full, y_full,
                epochs=config['epochs'],
                batch_size=config['batch_size'],
                callbacks=[early_stop],
                verbose=0
            )
            
            # Predict on holdout
            # Use last lookback period from available data
            test_features = full_features[-config['lookback']:].reshape(1, config['lookback'], len(feature_cols))
            test_features_scaled = full_scaler.transform(test_features.reshape(-1, len(feature_cols))).reshape(1, config['lookback'], len(feature_cols))
            
            holdout_pred_returns = final_model.predict(test_features_scaled, verbose=0)[0]
            
            # Convert to prices
            base_price = full_prices[-1]
            if predict_returns:
                holdout_pred_prices = self.reconstruct_prices_from_returns(holdout_pred_returns, base_price)
            else:
                # Inverse transform if predicting prices directly
                dummy = np.zeros((7, len(feature_cols)))
                dummy[:, 0] = holdout_pred_returns
                unscaled = full_scaler.inverse_transform(dummy)
                holdout_pred_prices = unscaled[:, 0]
            
            # Compare with actual holdout prices
            holdout_actual_prices = holdout_df['close'].values[:7]
            
            # Calculate holdout metrics
            holdout_mae = mean_absolute_error(holdout_actual_prices, holdout_pred_prices)
            holdout_rmse = np.sqrt(mean_squared_error(holdout_actual_prices, holdout_pred_prices))
            holdout_mape = mean_absolute_percentage_error(holdout_actual_prices, holdout_pred_prices) * 100
            
            # Direction accuracy
            actual_directions = [1 if p > base_price else -1 for p in holdout_actual_prices]
            pred_directions = [1 if p > base_price else -1 for p in holdout_pred_prices]
            direction_accuracy = sum(a == p for a, p in zip(actual_directions, pred_directions)) / 7 * 100
            
            print(f"\n  🎯 HOLDOUT SET RESULTS (UNSEEN DATA):")
            print(f"     MAE: {holdout_mae:.4f}")
            print(f"     RMSE: {holdout_rmse:.4f}")
            print(f"     MAPE: {holdout_mape:.2f}%")
            print(f"     Direction Accuracy: {direction_accuracy:.1f}%")
            
            result = {
                'config': config,
                'avg_val_mae': float(avg_val_mae),
                'avg_val_mape': float(avg_val_mape),
                'num_windows': len(window_results),
                'holdout_mae': float(holdout_mae),
                'holdout_rmse': float(holdout_rmse),
                'holdout_mape': float(holdout_mape),
                'direction_accuracy': float(direction_accuracy),
                'predictions': holdout_pred_prices.tolist(),
                'actual': holdout_actual_prices.tolist(),
                'predict_returns': predict_returns
            }
            
            # Cleanup
            del final_model
            tf.keras.backend.clear_session()
            gc.collect()
            
            return result
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}


def main():
    """Test the improved tuner"""
    import sys
    
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'AHPC'
    
    print(f"\n{'='*70}")
    print(f"IMPROVED LSTM HYPERPARAMETER TUNING")
    print('='*70)
    print(f"Symbol: {symbol}")
    print(f"Improvements:")
    print(f"  ✓ Rolling/expanding window validation")
    print(f"  ✓ Predicting log-returns for stationarity")
    print(f"  ✓ Final holdout period never touched during tuning")
    print('='*70)
    
    # Create tuner
    tuner = ImprovedHyperparameterTuner(symbol, test_days=7, final_holdout_days=30)
    
    # Load data
    available_df, holdout_df = tuner.load_data_with_holdout()
    
    # Test configuration
    test_config = {
        'lookback': 60,
        'layers': 3,
        'units_1': 128,
        'units_2': 64,
        'units_3': 32,
        'dropout_1': 0.2,
        'dropout_2': 0.2,
        'dropout_3': 0.2,
        'dense_units': 32,
        'architecture': 'bidirectional',
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 100,
        'patience': 10
    }
    
    # Test with returns-based prediction
    print("\n\n" + "="*70)
    print("TESTING WITH RETURNS-BASED PREDICTION")
    print("="*70)
    result_returns = tuner.train_and_evaluate_rolling(test_config, available_df, holdout_df, predict_returns=True)
    
    # Test with price-based prediction (original method)
    print("\n\n" + "="*70)
    print("TESTING WITH PRICE-BASED PREDICTION (ORIGINAL)")
    print("="*70)
    result_prices = tuner.train_and_evaluate_rolling(test_config, available_df, holdout_df, predict_returns=False)
    
    # Compare
    if 'error' not in result_returns and 'error' not in result_prices:
        print("\n\n" + "="*70)
        print("COMPARISON: RETURNS vs PRICES")
        print("="*70)
        print(f"Returns-based:")
        print(f"  Holdout MAPE: {result_returns['holdout_mape']:.2f}%")
        print(f"  Direction Acc: {result_returns['direction_accuracy']:.1f}%")
        print(f"\nPrice-based:")
        print(f"  Holdout MAPE: {result_prices['holdout_mape']:.2f}%")
        print(f"  Direction Acc: {result_prices['direction_accuracy']:.1f}%")
        
        if result_returns['holdout_mape'] < result_prices['holdout_mape']:
            print(f"\n✅ Returns-based prediction performs BETTER by {result_prices['holdout_mape'] - result_returns['holdout_mape']:.2f} MAPE points!")
        else:
            print(f"\n⚠️ Price-based prediction performs better by {result_returns['holdout_mape'] - result_prices['holdout_mape']:.2f} MAPE points")


if __name__ == '__main__':
    main()
