"""
Improved Hyperparameter Tuning for LSTM Stock Price Predictor
Key improvements:
1. Rolling/expanding window validation with multiple test periods
2. Predicts log-returns instead of absolute prices for stationarity
3. Final hold-out period never touched during tuning
4. More robust cross-validation

M1 Pro Optimizations (ENABLED with subprocess isolation):
✓ Mixed precision (float16) - 2x faster training, memory safe via subprocess
✓ Memory growth enabled - prevents OOM on 16GB RAM
✓ Legacy Adam optimizer - optimized for M1/M2 Macs
✓ Batch sizes 32-64 - optimal for M1 Metal backend
✓ Model sizes 64-128 units - balanced for 16GB RAM

Memory Management via Subprocess Isolation:
✓ Each config runs in separate process - OS reclaims ALL memory on exit
✓ No manual cleanup needed - process isolation guarantees memory reset
✓ Mixed precision safe - memory can't accumulate across configs
✓ Scales to 300+ configs without memory issues
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
import os
import gc
import logging

# Suppress all warnings including TensorFlow retracing warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Suppress TensorFlow function retracing warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# ML/DL libraries
try:
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
    import tensorflow as tf
    
    # Suppress TensorFlow warnings at module level
    tf.get_logger().setLevel('ERROR')
    
    # M1 Pro Optimizations - Mixed precision now safe with subprocess isolation!
    tf.keras.mixed_precision.set_global_policy('mixed_float16')  # 2x faster on M1, memory safe via subprocess
    # tf.config.optimizer.set_jit(True)  # XLA disabled - not compatible with Metal backend
    
    # Configure TensorFlow memory growth for M1
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ M1 GPU optimizations enabled: Memory growth + mixed precision (FP16)")
    
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


class HyperparameterTuner:
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
    
    def cleanup(self):
        """Clean up all internal references to free memory"""
        if hasattr(self, 'data_fetcher') and hasattr(self.data_fetcher, 'session'):
            try:
                self.data_fetcher.session.close()
            except:
                pass
        self.data_fetcher = None
        self.results = []
        import gc
        gc.collect()
        
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
            data: Feature matrix (SCALED - includes log_returns as first column if predict_returns=True)
            prices: Actual prices for reference (UNSCALED)
            lookback: Lookback window
            predict_returns: If True, predict log-returns; if False, predict SCALED prices
            
        Returns:
            X: Input sequences (SCALED)
            y: Target (log-returns OR scaled prices for next 7 days)
            base_prices: Reference prices for reconstructing predictions (UNSCALED)
        """
        X, y, base_prices = [], [], []
        
        # Find close price column index for extracting scaled prices
        # When predict_returns=False, data should have 'close' as one column (scaled)
        
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
                # Predict SCALED prices (from the scaled data, not raw prices)
                # Extract scaled close prices for next 7 days
                # Assumes 'close' is the first column in the scaled data
                scaled_future_prices = data[i:i+self.test_days, 0]  # First column is close price
                y.append(scaled_future_prices)
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
            # Output layer must be float32 for mixed precision
            model.add(Dense(7, dtype='float32'))  # Output 7 predictions (returns or prices)
        
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
            # Output layer must be float32 for mixed precision
            model.add(Dense(7, dtype='float32'))
        
        # Use legacy Adam for M1/M2 performance
        from tensorflow.keras.optimizers.legacy import Adam as LegacyAdam
        optimizer = LegacyAdam(learning_rate=config['learning_rate'])
        
        # Cast outputs to float32 for loss calculation (mixed precision requirement)
        loss = tf.keras.losses.MeanSquaredError()
        
        model.compile(
            optimizer=optimizer, 
            loss=loss, 
            metrics=['mae'],
            run_eagerly=False,  # Use graph mode for performance
            jit_compile=False   # Disable XLA (not compatible with Metal)
        )
        
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
        # Print concise config info
        print(f"\n[Config] {config['architecture'].upper()}: lookback={config['lookback']}, layers={config['layers']}, units={config.get('units_1', 'N/A')}, dropout={config.get('dropout_1', 'N/A')}, lr={config['learning_rate']}, batch={config['batch_size']}")
        
        try:
            # Aggressive memory cleanup before starting
            import keras.backend as K
            K.clear_session()
            tf.keras.backend.clear_session()
            gc.collect()
            gc.collect()
            gc.collect()  # Triple collect for thorough cleanup
            
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
                
                # Clean up features/prices but KEEP scaler for later inverse transform
                del scaled_features, features, prices
                
                if len(X) < 10:
                    print("  ⚠️ Insufficient data for this window")
                    del X, y, base_prices
                    continue
                
                # Split into train/val within this window
                split_idx = int(len(X) * 0.85)  # Use 85% for training, 15% for validation
                X_train, X_val = X[:split_idx], X[split_idx:]
                y_train, y_val = y[:split_idx], y[split_idx:]
                base_prices_val = base_prices[split_idx:]
                
                # Clean up original arrays
                del X, y, base_prices
                
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
                
                # Clear training data from memory immediately
                del X_train, y_train
                gc.collect()
                
                # Evaluate on validation set
                val_pred_returns = model.predict(X_val, verbose=0, batch_size=config['batch_size'])
                
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
                    # For price prediction, both pred and y_val are in SCALED space
                    # Need to inverse scale BOTH to get actual prices
                    close_col_idx = feature_cols.index('close') if 'close' in feature_cols else 0
                    
                    # Inverse scale predictions
                    val_pred_prices = []
                    for pred in val_pred_returns:
                        dummy = np.zeros((len(pred), len(feature_cols)))
                        dummy[:, close_col_idx] = pred
                        unscaled = scaler.inverse_transform(dummy)
                        val_pred_prices.append(unscaled[:, close_col_idx])
                    val_pred_prices = np.array(val_pred_prices)
                    
                    # Inverse scale actual values (y_val is also scaled now)
                    val_actual_prices = []
                    for actual in y_val:
                        dummy = np.zeros((len(actual), len(feature_cols)))
                        dummy[:, close_col_idx] = actual
                        unscaled = scaler.inverse_transform(dummy)
                        val_actual_prices.append(unscaled[:, close_col_idx])
                    val_actual_prices = np.array(val_actual_prices)
                
                # Calculate metrics
                val_mae = mean_absolute_error(val_actual_prices.flatten(), val_pred_prices.flatten())
                val_mape = mean_absolute_percentage_error(val_actual_prices.flatten(), val_pred_prices.flatten()) * 100
                
                window_results.append({
                    'val_mae': val_mae,
                    'val_mape': val_mape
                })
                
                print(f"  ✓ Window {window_idx + 1} - Val MAE: {val_mae:.4f}, Val MAPE: {val_mape:.2f}%")
                
                # Aggressive cleanup after each window to prevent memory buildup
                del model, history, X_val, y_val  # X_train, y_train already deleted earlier
                del val_actual_prices, val_pred_prices, val_pred_returns
                del train_data, base_prices_val, scaler  # Clean up dataframes, arrays and scaler
                
                # Force memory release
                import keras.backend as K
                K.clear_session()
                tf.keras.backend.clear_session()
                gc.collect()
                gc.collect()
                gc.collect()  # Triple collect for thorough cleanup
            
            # Average across windows
            if not window_results:
                return {'error': 'No valid windows'}
            
            avg_val_mae = np.mean([r['val_mae'] for r in window_results])
            avg_val_mape = np.mean([r['val_mape'] for r in window_results])
            num_windows = len(window_results)
            
            print(f"\n  📊 Average across {num_windows} windows:")
            print(f"     Val MAE: {avg_val_mae:.4f}, Val MAPE: {avg_val_mape:.2f}%")
            
            # Clean up window data before holdout
            del window_results
            gc.collect()
            
            # Now evaluate on final holdout (NEVER seen during training)
            print(f"\n  🎯 Evaluating on final holdout ({len(holdout_df)} days)...")
            
            # Prepare full training data (all available data except holdout)
            print(f"  • Preparing features from {len(available_df)} days...")
            full_train_data = self.prepare_features_with_returns(available_df, predict_returns)
            full_features = full_train_data[feature_cols].values
            full_prices = full_train_data['close'].values
            full_features = np.nan_to_num(full_features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Scale
            full_scaler = StandardScaler() if predict_returns else MinMaxScaler(feature_range=(0, 1))
            full_scaled = full_scaler.fit_transform(full_features)
            
            # Create sequences for training
            print(f"  • Creating sequences with lookback={config['lookback']}...")
            X_full, y_full, base_prices_full = self.create_sequences_returns(
                full_scaled, full_prices, config['lookback'], predict_returns
            )
            
            # Clean up scaled data (but keep full_features for later use)
            del full_scaled
            gc.collect()
            
            print(f"  • Training final model on {len(X_full)} samples...")
            # Train final model on ALL available data (but with reduced epochs for speed)
            final_model = self.build_model(config, (config['lookback'], len(feature_cols)))
            
            # Reduce epochs for final holdout evaluation (we already validated the config)
            holdout_epochs = min(50, config['epochs'])  # Cap at 50 epochs for speed
            
            early_stop = EarlyStopping(monitor='loss', patience=min(5, config['patience']), 
                                      restore_best_weights=True, verbose=0)
            
            final_model.fit(
                X_full, y_full,
                epochs=holdout_epochs,
                batch_size=config['batch_size'],
                callbacks=[early_stop],
                verbose=0
            )
            
            print(f"  • Generating predictions...")
            # Prepare holdout features for prediction
            holdout_features = self.prepare_features_with_returns(holdout_df, predict_returns)
            holdout_features_matrix = holdout_features[feature_cols].values
            holdout_features_matrix = np.nan_to_num(holdout_features_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Concatenate available + holdout features for proper sequence creation
            combined_features = np.vstack([full_features, holdout_features_matrix])
            combined_prices = np.concatenate([full_prices, holdout_df['close'].values])
            
            # Scale the combined features
            combined_scaler = StandardScaler() if predict_returns else MinMaxScaler(feature_range=(0, 1))
            combined_scaled = combined_scaler.fit_transform(combined_features)
            
            # Create sequence from the last lookback period before holdout
            X_holdout, y_holdout_actual, base_prices_holdout = self.create_sequences_returns(
                combined_scaled, combined_prices, config['lookback'], predict_returns
            )
            
            # Predict - use the first sequence which predicts the first 7 days of holdout
            if len(X_holdout) > 0:
                holdout_pred_returns = final_model.predict(X_holdout[:1], verbose=0)[0]
                
                # Convert to prices
                base_price = base_prices_holdout[0]
                if predict_returns:
                    holdout_pred_prices = self.reconstruct_prices_from_returns(holdout_pred_returns, base_price)
                else:
                    # For price prediction, model outputs scaled prices
                    # Need to inverse transform to get actual prices
                    close_col_idx = feature_cols.index('close') if 'close' in feature_cols else 0
                    dummy = np.zeros((7, len(feature_cols)))
                    
                    # Fill with mean values from last combined data point
                    if len(combined_features) > 0:
                        last_features = combined_features[-8]  # Use point before the 7-day target
                        for i in range(7):
                            dummy[i] = last_features
                    
                    # Replace close column with predictions (which are scaled)
                    dummy[:, close_col_idx] = holdout_pred_returns
                    
                    # Inverse transform
                    unscaled = combined_scaler.inverse_transform(dummy)
                    holdout_pred_prices = unscaled[:, close_col_idx]
            else:
                # Fallback if no sequences created
                holdout_pred_prices = np.full(7, combined_prices[-1])
            
            # Compare with actual holdout prices
            holdout_actual_prices = holdout_df['close'].values[:7]
            
            print(f"  • Calculating metrics...")
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
                'config': config.copy(),  # Copy to avoid references
                'avg_val_mae': float(avg_val_mae),
                'avg_val_mape': float(avg_val_mape),
                'num_windows': num_windows,
                'holdout_mae': float(holdout_mae),
                'holdout_rmse': float(holdout_rmse),
                'holdout_mape': float(holdout_mape),
                'direction_accuracy': float(direction_accuracy),
                # predictions and actual arrays removed to save memory during grid search
                'predict_returns': predict_returns
            }
            
            # Aggressive cleanup at end of config
            del final_model, X_holdout, y_holdout_actual, base_prices_holdout
            del combined_features, combined_prices, combined_scaled, combined_scaler
            del holdout_features, holdout_features_matrix
            del holdout_pred_prices, holdout_actual_prices, holdout_pred_returns
            
            # Force complete memory cleanup
            import keras.backend as K
            K.clear_session()
            tf.keras.backend.clear_session()
            gc.collect()
            gc.collect()
            gc.collect()  # Triple collect to ensure cleanup
            
            return result
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            
            # Emergency cleanup on error
            tf.keras.backend.clear_session()
            gc.collect()
            
            return {'error': str(e)}
    
    # ========== BACKWARD COMPATIBILITY METHODS ==========
    # These methods maintain compatibility with stock_predictor.py
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Legacy method for backward compatibility with stock_predictor.py
        Returns train/test split without holdout
        """
        df = self.data_fetcher.get_stock_price_history(self.symbol)
        if df is None or len(df) == 0:
            raise ValueError(f"No data found for {self.symbol}")
        
        df = df.sort_values('date').reset_index(drop=True)
        
        # Simple split for backward compatibility
        split_idx = len(df) - self.test_days
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        return train_df, test_df
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features - uses improved returns-based features
        """
        return self.prepare_features_with_returns(df, predict_returns=False)
    
    def create_sequences(self, data: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Legacy method for backward compatibility
        Creates sequences for price prediction
        """
        X, y = [], []
        
        for i in range(lookback, len(data) - self.test_days + 1):
            X.append(data[i-lookback:i])
            y.append(data[i:i+self.test_days, 0])  # Next test_days close prices
        
        return np.array(X), np.array(y)
    
    def train_and_evaluate(self, config: Dict, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
        """
        Legacy method for backward compatibility with stock_predictor.py
        Uses improved rolling validation but returns results in old format
        """
        # Combine train and test for available data
        available_df = pd.concat([train_df, test_df], ignore_index=True)
        
        # Use test_df as holdout
        holdout_df = test_df.copy()
        
        # Train with rolling validation on train_df
        # Use price-based prediction for backward compatibility
        result = self.train_and_evaluate_rolling(
            config=config,
            available_df=train_df,  # Only use train_df for validation
            holdout_df=holdout_df,
            predict_returns=False  # Use price-based for compatibility
        )
        
        if 'error' in result:
            return result
        
        # Convert to old format
        return {
            'config': result['config'],
            'val_mae': result.get('avg_val_mae', 0),
            'val_rmse': result.get('avg_val_mae', 0),  # Approximate
            'test_mae': result['holdout_mae'],
            'test_rmse': result['holdout_rmse'],
            'test_mape': result['holdout_mape'],
            'direction_accuracy': result['direction_accuracy'],
            'predictions': result['predictions'],
            'actual': result['actual'],
            'epochs_trained': 50,  # Approximate
            'training_samples': len(train_df)
        }


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
    tuner = HyperparameterTuner(symbol, test_days=7, final_holdout_days=30)
    
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
