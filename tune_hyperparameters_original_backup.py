"""
Hyperparameter Tuning for LSTM Stock Price Predictor
Tests various configurations to find optimal settings for 7-day predictions
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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Prevent TensorFlow from allocating all GPU memory at once

# ML/DL libraries
try:
    from sklearn.preprocessing import MinMaxScaler
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
    from tensorflow.keras.optimizers import Adam, RMSprop
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available. Install with: pip install tensorflow scikit-learn")
    exit(1)

from data_fetcher import NepseDataFetcher
from ml_predictor import MLStockPredictor


class HyperparameterTuner:
    """Tune LSTM hyperparameters for optimal 7-day predictions"""
    
    def __init__(self, symbol: str, test_days: int = 7):
        """
        Initialize tuner
        
        Args:
            symbol: Stock symbol to tune for
            test_days: Number of days to hold out for testing (default: 7)
        """
        self.symbol = symbol
        self.test_days = test_days
        self.data_fetcher = NepseDataFetcher()
        self.results = []
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and split data into train and test sets
        
        Returns:
            train_df, test_df
        """
        print(f"\n{'='*70}")
        print(f"LOADING DATA FOR {self.symbol}")
        print('='*70)
        
        # Load price history
        df = self.data_fetcher.get_stock_price_history(self.symbol)
        
        if df is None or len(df) == 0:
            raise ValueError(f"No data found for {self.symbol}")
        
        # Sort by date
        df = df.sort_values('date')
        
        print(f"✓ Total data points: {len(df)}")
        print(f"✓ Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"✓ Latest price: {df['close'].iloc[-1]:.2f}")
        
        # Split: hold out last 7 days for testing
        split_idx = len(df) - self.test_days
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        print(f"\n📊 Data Split:")
        print(f"  Training: {len(train_df)} days ({train_df['date'].min()} to {train_df['date'].max()})")
        print(f"  Testing: {len(test_df)} days ({test_df['date'].min()} to {test_df['date'].max()})")
        
        # Show actual test prices
        print(f"\n🎯 Actual prices to predict (next 7 days):")
        for i, (idx, row) in enumerate(test_df.iterrows(), 1):
            change_pct = ((row['close'] - train_df['close'].iloc[-1]) / train_df['close'].iloc[-1]) * 100
            print(f"  Day {i} ({row['date']}): {row['close']:.2f} ({change_pct:+.2f}%)")
        
        return train_df, test_df
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features with technical indicators"""
        data = df.copy()
        
        # Price-based features
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Moving averages
        data['sma_5'] = data['close'].rolling(window=5).mean()
        data['sma_10'] = data['close'].rolling(window=10).mean()
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['sma_50'] = data['close'].rolling(window=50).mean()
        
        # EMA
        data['ema_12'] = data['close'].ewm(span=12, adjust=False).mean()
        data['ema_26'] = data['close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        data['macd'] = data['ema_12'] - data['ema_26']
        data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
        data['macd_diff'] = data['macd'] - data['macd_signal']
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        data['bb_middle'] = data['close'].rolling(window=20).mean()
        bb_std = data['close'].rolling(window=20).std()
        data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
        data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        
        # Volume indicators
        data['volume_sma'] = data['volume'].rolling(window=20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_sma']
        
        # Volatility (ATR)
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        data['atr'] = true_range.rolling(14).mean()
        
        # Price position relative to high/low
        data['high_low_ratio'] = (data['close'] - data['low']) / (data['high'] - data['low'])
        
        # Momentum
        data['momentum_5'] = data['close'] - data['close'].shift(5)
        data['momentum_10'] = data['close'] - data['close'].shift(10)
        
        # Drop NaN
        data = data.dropna()
        
        return data
    
    def create_sequences(self, data: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM"""
        X, y = [], []
        
        for i in range(lookback, len(data) - self.test_days + 1):
            X.append(data[i-lookback:i])
            y.append(data[i:i+self.test_days, 0])  # Next 7 days close prices
        
        return np.array(X), np.array(y)
    
    def build_tcn_block(self, x, filters: int, kernel_size: int, dilation_rate: int, dropout: float):
        """
        Build a single TCN residual block with dilated causal convolutions
        
        Args:
            x: Input tensor
            filters: Number of filters
            kernel_size: Kernel size for convolution
            dilation_rate: Dilation rate for dilated convolution
            dropout: Dropout rate
        """
        # First dilated causal convolution
        conv1 = Conv1D(filters=filters,
                      kernel_size=kernel_size,
                      dilation_rate=dilation_rate,
                      padding='causal',
                      activation='relu')(x)
        conv1 = SpatialDropout1D(dropout)(conv1)
        
        # Second dilated causal convolution
        conv2 = Conv1D(filters=filters,
                      kernel_size=kernel_size,
                      dilation_rate=dilation_rate,
                      padding='causal',
                      activation='relu')(conv1)
        conv2 = SpatialDropout1D(dropout)(conv2)
        
        # Residual connection
        if x.shape[-1] != filters:
            # Match dimensions with 1x1 convolution
            residual = Conv1D(filters=filters, kernel_size=1, padding='same')(x)
        else:
            residual = x
        
        # Add residual connection
        output = Add()([conv2, residual])
        output = Activation('relu')(output)
        
        return output
    
    def build_transformer_block(self, x, num_heads: int, key_dim: int, ff_dim: int, dropout: float):
        """
        Build a single Transformer encoder block with multi-head attention
        
        Args:
            x: Input tensor
            num_heads: Number of attention heads
            key_dim: Dimension of attention keys
            ff_dim: Dimension of feed-forward network
            dropout: Dropout rate
        """
        # Multi-head self-attention
        attn_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout
        )(x, x)
        attn_output = Dropout(dropout)(attn_output)
        
        # Add & Norm (residual connection)
        x1 = LayerNormalization(epsilon=1e-6)(x + attn_output)
        
        # Feed-forward network
        ff_output = Dense(ff_dim, activation='relu')(x1)
        ff_output = Dropout(dropout)(ff_output)
        ff_output = Dense(x.shape[-1])(ff_output)
        ff_output = Dropout(dropout)(ff_output)
        
        # Add & Norm (residual connection)
        x2 = LayerNormalization(epsilon=1e-6)(x1 + ff_output)
        
        return x2
    
    def build_model(self, config: Dict, input_shape: Tuple[int, int]):
        """
        Build model with given configuration (LSTM/GRU/TCN/Transformer)
        
        Args:
            config: Dictionary with hyperparameters
            input_shape: (timesteps, features)
        """
        # Transformer Architecture with Multi-Head Attention
        if config['architecture'] == 'transformer':
            inputs = Input(shape=input_shape)
            x = inputs
            
            # Number of attention heads and key dimension
            num_heads = 8
            key_dim = config['units_1'] // num_heads  # Ensure key_dim * num_heads = units_1
            ff_dim = config['units_1'] * 2  # Feed-forward dimension (typically 2-4x model dim)
            dropout = config['dropout_1']
            
            # Stack multiple transformer blocks
            num_blocks = config['layers']
            for _ in range(num_blocks):
                x = self.build_transformer_block(x, num_heads, key_dim, ff_dim, dropout)
            
            # Global average pooling to get fixed-size representation
            x = GlobalAveragePooling1D()(x)
            
            # Dense layers for final prediction
            x = Dense(config['dense_units'], activation='relu')(x)
            x = Dropout(0.1)(x)
            x = Dense(config['dense_units'] // 2, activation='relu')(x)
            outputs = Dense(7)(x)
            
            model = Model(inputs=inputs, outputs=outputs)
        
        # TCN Architecture (Temporal Convolutional Network)
        elif config['architecture'] == 'tcn':
            inputs = Input(shape=input_shape)
            x = inputs
            
            # Stack of TCN blocks with increasing dilation rates
            filters = config['units_1']
            kernel_size = 3
            dropout = config['dropout_1']
            
            # Multiple TCN blocks with exponentially increasing dilation
            dilations = [1, 2, 4, 8, 16]  # Receptive field of ~60 steps
            for dilation in dilations:
                x = self.build_tcn_block(x, filters, kernel_size, dilation, dropout)
            
            # Additional blocks for deeper networks
            if config['layers'] >= 2:
                filters = config['units_2']
                for dilation in [1, 2, 4]:
                    x = self.build_tcn_block(x, filters, kernel_size, dilation, config['dropout_2'])
            
            if config['layers'] >= 3:
                filters = config['units_3']
                for dilation in [1, 2]:
                    x = self.build_tcn_block(x, filters, kernel_size, dilation, config['dropout_3'])
            
            # Global pooling and dense layers
            x = GlobalAveragePooling1D()(x)
            x = Dense(config['dense_units'], activation='relu')(x)
            x = Dropout(0.1)(x)
            x = Dense(config['dense_units'] // 2, activation='relu')(x)
            outputs = Dense(7)(x)
            
            model = Model(inputs=inputs, outputs=outputs)
            
        # LSTM/GRU Architectures
        else:
            model = Sequential()
            
            # Architecture type
            if config['architecture'] == 'bidirectional':
                model.add(Bidirectional(LSTM(config['units_1'], return_sequences=True), input_shape=input_shape))
            elif config['architecture'] == 'gru':
                model.add(GRU(config['units_1'], return_sequences=True, input_shape=input_shape))
            else:  # standard
                model.add(LSTM(config['units_1'], return_sequences=True, input_shape=input_shape))
            
            model.add(Dropout(config['dropout_1']))
            
            # Second layer
            if config['layers'] >= 2:
                if config['architecture'] == 'bidirectional':
                    model.add(Bidirectional(LSTM(config['units_2'], return_sequences=True)))
                elif config['architecture'] == 'gru':
                    model.add(GRU(config['units_2'], return_sequences=True))
                else:
                    model.add(LSTM(config['units_2'], return_sequences=True))
                model.add(Dropout(config['dropout_2']))
            
            # Third layer
            if config['layers'] >= 3:
                if config['architecture'] == 'gru':
                    model.add(GRU(config['units_3'], return_sequences=False))
                else:
                    model.add(LSTM(config['units_3'], return_sequences=False))
                model.add(Dropout(config['dropout_3']))
            else:
                # Final LSTM/GRU layer
                if config['architecture'] == 'gru':
                    model.add(GRU(config['units_2'] // 2, return_sequences=False))
                else:
                    model.add(LSTM(config['units_2'] // 2, return_sequences=False))
                model.add(Dropout(config['dropout_2']))
            
            # Dense layers
            model.add(Dense(config['dense_units'], activation='relu'))
            model.add(Dropout(0.1))
            model.add(Dense(config['dense_units'] // 2, activation='relu'))
            model.add(Dense(7))  # Output 7 predictions
        
        # Compile (works for both Sequential and Functional API models)
        optimizer = Adam(learning_rate=config['learning_rate']) if config['optimizer'] == 'adam' else RMSprop(learning_rate=config['learning_rate'])
        
        model.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        return model
    
    def train_and_evaluate(self, config: Dict, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
        """
        Train model with given config and evaluate on test set
        
        Returns:
            Dictionary with metrics and predictions
        """
        print(f"\n{'='*70}")
        print(f"Testing Configuration:")
        for key, val in config.items():
            print(f"  {key}: {val}")
        print('='*70)
        
        try:
            # Prepare features
            data = self.prepare_features(train_df)
            
            # Select features
            feature_cols = ['close', 'volume', 'returns', 'sma_5', 'sma_20', 'rsi', 
                           'macd', 'bb_width', 'volume_ratio', 'atr', 'momentum_5']
            feature_cols = [col for col in feature_cols if col in data.columns]
            
            features = data[feature_cols].values
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Scale
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_features = scaler.fit_transform(features)
            
            # Create sequences
            X, y = self.create_sequences(scaled_features, config['lookback'])
            
            if len(X) < 10:
                return {'error': 'Insufficient training data'}
            
            # Split train/val
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
            
            # Build model
            model = self.build_model(config, (config['lookback'], len(feature_cols)))
            
            # Callbacks
            early_stop = EarlyStopping(monitor='val_loss', patience=config['patience'], restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
            
            # Train
            print("Training...")
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=config['epochs'],
                batch_size=config['batch_size'],
                callbacks=[early_stop, reduce_lr],
                verbose=0
            )
            
            # Evaluate on validation
            val_pred = model.predict(X_val, verbose=0)
            val_mae = mean_absolute_error(y_val, val_pred)
            val_mse = mean_squared_error(y_val, val_pred)
            val_rmse = np.sqrt(val_mse)
            
            print(f"✓ Validation MAE: {val_mae:.6f}")
            print(f"✓ Validation RMSE: {val_rmse:.6f}")
            
            # Now test on actual held-out 7 days
            # Prepare test features using ALL training data
            test_input = self.prepare_features(train_df)
            test_features = test_input[feature_cols].iloc[-config['lookback']:].values
            test_features = np.nan_to_num(test_features, nan=0.0, posinf=0.0, neginf=0.0)
            test_scaled = scaler.transform(test_features)
            
            X_test = test_scaled.reshape(1, config['lookback'], len(feature_cols))
            
            # Predict
            predictions_scaled = model.predict(X_test, verbose=0)[0]
            
            # Inverse transform
            dummy = np.zeros((7, len(feature_cols)))
            dummy[:, 0] = predictions_scaled
            unscaled = scaler.inverse_transform(dummy)
            predicted_prices = unscaled[:, 0]
            
            # Compare with actual test prices
            actual_prices = test_df['close'].values
            
            # Calculate test metrics
            test_mae = mean_absolute_error(actual_prices, predicted_prices)
            test_mse = mean_squared_error(actual_prices, predicted_prices)
            test_rmse = np.sqrt(test_mse)
            test_mape = mean_absolute_percentage_error(actual_prices, predicted_prices) * 100
            
            # Direction accuracy (did we predict the right trend?)
            base_price = train_df['close'].iloc[-1]
            actual_directions = [1 if p > base_price else -1 for p in actual_prices]
            pred_directions = [1 if p > base_price else -1 for p in predicted_prices]
            direction_accuracy = sum(a == p for a, p in zip(actual_directions, pred_directions)) / 7 * 100
            
            print(f"\n🎯 TEST SET RESULTS:")
            print(f"  MAE: {test_mae:.4f}")
            print(f"  RMSE: {test_rmse:.4f}")
            print(f"  MAPE: {test_mape:.2f}%")
            print(f"  Direction Accuracy: {direction_accuracy:.1f}%")
            
            print(f"\n📊 Predictions vs Actual:")
            for i in range(7):
                actual = actual_prices[i]
                pred = predicted_prices[i]
                error = pred - actual
                error_pct = (error / actual) * 100
                print(f"  Day {i+1}: Pred={pred:.2f}, Actual={actual:.2f}, Error={error:+.2f} ({error_pct:+.1f}%)")
            
            result = {
                'config': config,
                'val_mae': float(val_mae),
                'val_rmse': float(val_rmse),
                'test_mae': float(test_mae),
                'test_rmse': float(test_rmse),
                'test_mape': float(test_mape),
                'direction_accuracy': float(direction_accuracy),
                'predictions': predicted_prices.tolist(),
                'actual': actual_prices.tolist(),
                'epochs_trained': len(history.history['loss']),
                'training_samples': len(X_train)
            }
            
            # Cleanup model and free memory (CRITICAL for grid search)
            del model
            del history
            tf.keras.backend.clear_session()
            import gc
            gc.collect()
            
            return result
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return {'error': str(e)}
    
    def generate_grid_search_configs(self, mode: str = 'quick') -> List[Dict]:
        """
        Generate configurations using grid search approach
        
        Args:
            mode: 'quick', 'medium', or 'full'
        """
        if mode == 'quick':
            # Quick: Test key variations including TCN and Transformer
            lookbacks = [60, 90]
            architectures = ['bidirectional', 'gru', 'tcn', 'transformer']
            layer_configs = [(128, 64, 32)]
            learning_rates = [0.001]
            batch_sizes = [32]
            dropouts = [(0.2, 0.2, 0.2)]
        elif mode == 'medium':
            # Medium: More combinations including TCN and Transformer
            lookbacks = [30, 60, 90]
            architectures = ['bidirectional', 'gru', 'tcn', 'transformer']
            layer_configs = [(128, 64, 32), (256, 128, 64)]
            learning_rates = [0.001, 0.0001]
            batch_sizes = [32]
            dropouts = [(0.2, 0.2, 0.2), (0.3, 0.3, 0.2)]
        else:  # full
            # Full: Comprehensive grid with all architectures
            lookbacks = [30, 60, 90]
            architectures = ['bidirectional', 'gru', 'standard', 'tcn', 'transformer']
            layer_configs = [(64, 32, 0), (128, 64, 32), (256, 128, 64), (256, 256, 128)]
            learning_rates = [0.0001, 0.001, 0.005]
            batch_sizes = [16, 32, 64]
            dropouts = [(0.2, 0.2, 0.2), (0.3, 0.3, 0.2), (0.4, 0.4, 0.3)]
        
        configs = []
        config_id = 1
        
        for lookback in lookbacks:
            for arch in architectures:
                for units in layer_configs:
                    for lr in learning_rates:
                        for batch_size in batch_sizes:
                            for dropout in dropouts:
                                layers = 3 if units[2] > 0 else 2
                                dense_units = units[0] // 4
                                
                                config = {
                                    'name': f'config_{config_id}',
                                    'lookback': lookback,
                                    'layers': layers,
                                    'units_1': units[0],
                                    'units_2': units[1],
                                    'units_3': units[2],
                                    'dropout_1': dropout[0],
                                    'dropout_2': dropout[1],
                                    'dropout_3': dropout[2],
                                    'dense_units': dense_units,
                                    'architecture': arch,
                                    'optimizer': 'adam',
                                    'learning_rate': lr,
                                    'batch_size': batch_size,
                                    'epochs': 100 if lr < 0.001 else 50,
                                    'patience': 15 if lr < 0.001 else 10
                                }
                                configs.append(config)
                                config_id += 1
        
        return configs
    
    def run_tuning(self, quick_mode: bool = False, grid_search: bool = True):
        """
        Run hyperparameter tuning with various configurations
        
        Args:
            quick_mode: If True, test fewer configurations for speed
            grid_search: If True, use grid search combinations (recommended)
        """
        print(f"\n{'='*70}")
        print(f"HYPERPARAMETER TUNING FOR {self.symbol}")
        print('='*70)
        
        # Load data
        train_df, test_df = self.load_data()
        
        # Define configurations to test
        if grid_search:
            mode = 'quick' if quick_mode else 'medium'
            configs = self.generate_grid_search_configs(mode)
            print(f"🔍 Using grid search mode: {mode}")
        elif quick_mode:
            configs = [
                # Baseline
                {'name': 'baseline', 'lookback': 60, 'layers': 3, 'units_1': 128, 'units_2': 64, 'units_3': 32, 
                 'dropout_1': 0.2, 'dropout_2': 0.2, 'dropout_3': 0.2, 'dense_units': 32, 
                 'architecture': 'bidirectional', 'optimizer': 'adam', 'learning_rate': 0.001, 
                 'batch_size': 32, 'epochs': 50, 'patience': 10},
                
                # More lookback
                {'name': 'long_lookback', 'lookback': 90, 'layers': 3, 'units_1': 128, 'units_2': 64, 'units_3': 32,
                 'dropout_1': 0.2, 'dropout_2': 0.2, 'dropout_3': 0.2, 'dense_units': 32,
                 'architecture': 'bidirectional', 'optimizer': 'adam', 'learning_rate': 0.001,
                 'batch_size': 32, 'epochs': 50, 'patience': 10},
                
                # Deeper network
                {'name': 'deeper', 'lookback': 60, 'layers': 3, 'units_1': 256, 'units_2': 128, 'units_3': 64,
                 'dropout_1': 0.3, 'dropout_2': 0.3, 'dropout_3': 0.2, 'dense_units': 64,
                 'architecture': 'bidirectional', 'optimizer': 'adam', 'learning_rate': 0.001,
                 'batch_size': 32, 'epochs': 50, 'patience': 10},
                
                # GRU architecture
                {'name': 'gru', 'lookback': 60, 'layers': 3, 'units_1': 128, 'units_2': 64, 'units_3': 32,
                 'dropout_1': 0.2, 'dropout_2': 0.2, 'dropout_3': 0.2, 'dense_units': 32,
                 'architecture': 'gru', 'optimizer': 'adam', 'learning_rate': 0.001,
                 'batch_size': 32, 'epochs': 50, 'patience': 10},
            ]
        else:
            configs = [
                # Baseline - current config
                {'name': 'baseline', 'lookback': 60, 'layers': 3, 'units_1': 128, 'units_2': 64, 'units_3': 32, 
                 'dropout_1': 0.2, 'dropout_2': 0.2, 'dropout_3': 0.2, 'dense_units': 32, 
                 'architecture': 'bidirectional', 'optimizer': 'adam', 'learning_rate': 0.001, 
                 'batch_size': 32, 'epochs': 50, 'patience': 10},
                
                # Lookback variations
                {'name': 'short_lookback', 'lookback': 30, 'layers': 3, 'units_1': 128, 'units_2': 64, 'units_3': 32,
                 'dropout_1': 0.2, 'dropout_2': 0.2, 'dropout_3': 0.2, 'dense_units': 32,
                 'architecture': 'bidirectional', 'optimizer': 'adam', 'learning_rate': 0.001,
                 'batch_size': 32, 'epochs': 50, 'patience': 10},
                
                {'name': 'long_lookback', 'lookback': 90, 'layers': 3, 'units_1': 128, 'units_2': 64, 'units_3': 32,
                 'dropout_1': 0.2, 'dropout_2': 0.2, 'dropout_3': 0.2, 'dense_units': 32,
                 'architecture': 'bidirectional', 'optimizer': 'adam', 'learning_rate': 0.001,
                 'batch_size': 32, 'epochs': 50, 'patience': 10},
                
                # Architecture variations
                {'name': 'deeper', 'lookback': 60, 'layers': 3, 'units_1': 256, 'units_2': 128, 'units_3': 64,
                 'dropout_1': 0.3, 'dropout_2': 0.3, 'dropout_3': 0.2, 'dense_units': 64,
                 'architecture': 'bidirectional', 'optimizer': 'adam', 'learning_rate': 0.001,
                 'batch_size': 32, 'epochs': 50, 'patience': 10},
                
                {'name': 'wider', 'lookback': 60, 'layers': 3, 'units_1': 256, 'units_2': 256, 'units_3': 128,
                 'dropout_1': 0.3, 'dropout_2': 0.3, 'dropout_3': 0.3, 'dense_units': 128,
                 'architecture': 'bidirectional', 'optimizer': 'adam', 'learning_rate': 0.001,
                 'batch_size': 32, 'epochs': 50, 'patience': 10},
                
                {'name': 'simpler', 'lookback': 60, 'layers': 2, 'units_1': 64, 'units_2': 32, 'units_3': 0,
                 'dropout_1': 0.2, 'dropout_2': 0.2, 'dropout_3': 0.0, 'dense_units': 16,
                 'architecture': 'bidirectional', 'optimizer': 'adam', 'learning_rate': 0.001,
                 'batch_size': 32, 'epochs': 50, 'patience': 10},
                
                # GRU
                {'name': 'gru', 'lookback': 60, 'layers': 3, 'units_1': 128, 'units_2': 64, 'units_3': 32,
                 'dropout_1': 0.2, 'dropout_2': 0.2, 'dropout_3': 0.2, 'dense_units': 32,
                 'architecture': 'gru', 'optimizer': 'adam', 'learning_rate': 0.001,
                 'batch_size': 32, 'epochs': 50, 'patience': 10},
                
                # Learning rate variations
                {'name': 'lower_lr', 'lookback': 60, 'layers': 3, 'units_1': 128, 'units_2': 64, 'units_3': 32,
                 'dropout_1': 0.2, 'dropout_2': 0.2, 'dropout_3': 0.2, 'dense_units': 32,
                 'architecture': 'bidirectional', 'optimizer': 'adam', 'learning_rate': 0.0001,
                 'batch_size': 32, 'epochs': 100, 'patience': 15},
                
                {'name': 'higher_lr', 'lookback': 60, 'layers': 3, 'units_1': 128, 'units_2': 64, 'units_3': 32,
                 'dropout_1': 0.2, 'dropout_2': 0.2, 'dropout_3': 0.2, 'dense_units': 32,
                 'architecture': 'bidirectional', 'optimizer': 'adam', 'learning_rate': 0.005,
                 'batch_size': 32, 'epochs': 50, 'patience': 10},
                
                # Batch size variations
                {'name': 'small_batch', 'lookback': 60, 'layers': 3, 'units_1': 128, 'units_2': 64, 'units_3': 32,
                 'dropout_1': 0.2, 'dropout_2': 0.2, 'dropout_3': 0.2, 'dense_units': 32,
                 'architecture': 'bidirectional', 'optimizer': 'adam', 'learning_rate': 0.001,
                 'batch_size': 16, 'epochs': 50, 'patience': 10},
                
                {'name': 'large_batch', 'lookback': 60, 'layers': 3, 'units_1': 128, 'units_2': 64, 'units_3': 32,
                 'dropout_1': 0.2, 'dropout_2': 0.2, 'dropout_3': 0.2, 'dense_units': 32,
                 'architecture': 'bidirectional', 'optimizer': 'adam', 'learning_rate': 0.001,
                 'batch_size': 64, 'epochs': 50, 'patience': 10},
                
                # High dropout (regularization)
                {'name': 'high_dropout', 'lookback': 60, 'layers': 3, 'units_1': 128, 'units_2': 64, 'units_3': 32,
                 'dropout_1': 0.4, 'dropout_2': 0.4, 'dropout_3': 0.3, 'dense_units': 32,
                 'architecture': 'bidirectional', 'optimizer': 'adam', 'learning_rate': 0.001,
                 'batch_size': 32, 'epochs': 50, 'patience': 10},
            ]
        
        # Test each configuration
        print(f"\n🔍 Testing {len(configs)} configurations...\n")
        
        for i, config in enumerate(configs, 1):
            print(f"\n{'#'*70}")
            print(f"CONFIG {i}/{len(configs)}: {config['name']}")
            print('#'*70)
            
            result = self.train_and_evaluate(config, train_df, test_df)
            
            if 'error' not in result:
                self.results.append(result)
        
        # Sort by test performance
        self.results.sort(key=lambda x: x['test_mae'])
        
        # Display final results
        self.display_results()
        
        # Save results
        self.save_results()
    
    def display_results(self):
        """Display tuning results"""
        print(f"\n{'='*70}")
        print(f"TUNING RESULTS - RANKED BY TEST MAE")
        print('='*70)
        
        print(f"\nTested {len(self.results)} configurations\n")
        
        for i, result in enumerate(self.results[:10], 1):  # Top 10
            config = result['config']
            print(f"\n{i}. {config['name'].upper()}")
            print(f"   Test MAE: {result['test_mae']:.4f}")
            print(f"   Test RMSE: {result['test_rmse']:.4f}")
            print(f"   Test MAPE: {result['test_mape']:.2f}%")
            print(f"   Direction Accuracy: {result['direction_accuracy']:.1f}%")
            print(f"   Val MAE: {result['val_mae']:.6f}")
            print(f"   Key params: lookback={config['lookback']}, layers={config['layers']}, " +
                  f"arch={config['architecture']}, lr={config['learning_rate']}")
        
        # Best configuration
        best = self.results[0]
        print(f"\n{'='*70}")
        print(f"🏆 BEST CONFIGURATION: {best['config']['name']}")
        print('='*70)
        print(json.dumps(best['config'], indent=2))
        
        print(f"\n📊 BEST PREDICTIONS vs ACTUAL:")
        for i in range(7):
            pred = best['predictions'][i]
            actual = best['actual'][i]
            error = pred - actual
            error_pct = (error / actual) * 100
            print(f"  Day {i+1}: Pred={pred:.2f}, Actual={actual:.2f}, Error={error:+.2f} ({error_pct:+.1f}%)")
    
    def save_results(self):
        """Save results to individual file and append to master log"""
        # Individual results file
        output = {
            'symbol': self.symbol,
            'tuning_date': datetime.now().isoformat(),
            'test_days': self.test_days,
            'total_configs': len(self.results),
            'best_config': self.results[0]['config'] if self.results else None,
            'best_metrics': {
                'test_mae': self.results[0]['test_mae'],
                'test_rmse': self.results[0]['test_rmse'],
                'test_mape': self.results[0]['test_mape'],
                'direction_accuracy': self.results[0]['direction_accuracy']
            } if self.results else None,
            'all_results': self.results
        }
        
        # Save individual file
        filename = f'tuning_results_{self.symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")
        
        # Append to master log
        self.append_to_master_log(output)
        
        # Save best config to easy reference file
        if self.results:
            self.save_best_config()
    
    def append_to_master_log(self, output: Dict):
        """Append results to master log file for cross-stock analysis"""
        master_log = 'hyperparameter_tuning_master_log.json'
        
        # Load existing log
        if os.path.exists(master_log):
            with open(master_log, 'r') as f:
                master_data = json.load(f)
        else:
            master_data = {
                'created': datetime.now().isoformat(),
                'total_stocks_tested': 0,
                'stocks': {}
            }
        
        # Add this stock's results
        master_data['stocks'][self.symbol] = {
            'last_tested': output['tuning_date'],
            'best_config': output['best_config'],
            'best_metrics': output['best_metrics'],
            'total_configs_tested': output['total_configs']
        }
        master_data['total_stocks_tested'] = len(master_data['stocks'])
        master_data['last_updated'] = datetime.now().isoformat()
        
        # Save master log
        with open(master_log, 'w') as f:
            json.dump(master_data, f, indent=2)
        
        print(f"📝 Updated master log: {master_log}")
    
    def save_best_config(self):
        """Save best configuration to easy-to-read file"""
        best = self.results[0]
        
        best_config_file = f'best_config_{self.symbol}.txt'
        with open(best_config_file, 'w') as f:
            f.write(f"BEST HYPERPARAMETERS FOR {self.symbol}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Tested on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Test MAE: {best['test_mae']:.4f}\n")
            f.write(f"Test RMSE: {best['test_rmse']:.4f}\n")
            f.write(f"Test MAPE: {best['test_mape']:.2f}%\n")
            f.write(f"Direction Accuracy: {best['direction_accuracy']:.1f}%\n\n")
            
            f.write(f"CONFIGURATION:\n")
            f.write(f"{'-'*60}\n")
            config = best['config']
            for key, val in config.items():
                f.write(f"{key:20s}: {val}\n")
            
            f.write(f"\n{'='*60}\n")
            f.write(f"PREDICTIONS vs ACTUAL:\n")
            f.write(f"{'-'*60}\n")
            for i in range(7):
                pred = best['predictions'][i]
                actual = best['actual'][i]
                error = pred - actual
                error_pct = (error / actual) * 100
                f.write(f"Day {i+1}: Pred={pred:7.2f}, Actual={actual:7.2f}, Error={error:+7.2f} ({error_pct:+6.1f}%)\n")
        
        print(f"📄 Best config saved to: {best_config_file}")


def main():
    """Main entry point"""
    import sys
    
    # Get symbol from command line or use default
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'AHPC'
    quick_mode = '--quick' in sys.argv
    grid_search = '--grid' in sys.argv or '--quick' not in sys.argv  # Grid by default unless --quick
    
    print(f"\n{'='*70}")
    print(f"LSTM HYPERPARAMETER TUNING")
    print('='*70)
    print(f"Symbol: {symbol}")
    print(f"Mode: {'QUICK GRID' if quick_mode and grid_search else 'MEDIUM GRID' if grid_search else 'QUICK PRESET' if quick_mode else 'FULL PRESET'}")
    if grid_search:
        mode = 'quick' if quick_mode else 'medium'
        tuner = HyperparameterTuner(symbol, test_days=7)
        configs = tuner.generate_grid_search_configs(mode)
        print(f"Testing: {len(configs)} hyperparameter combinations")
    print('='*70)
    
    # Run tuning
    tuner = HyperparameterTuner(symbol, test_days=7)
    tuner.run_tuning(quick_mode=quick_mode, grid_search=grid_search)
    
    print(f"\n{'='*70}")
    print(f"✅ TUNING COMPLETE")
    print('='*70)
    print(f"\nFiles created:")
    print(f"  1. tuning_results_{symbol}_*.json - Full results")
    print(f"  2. best_config_{symbol}.txt - Best configuration")
    print(f"  3. hyperparameter_tuning_master_log.json - Cross-stock log")


if __name__ == '__main__':
    main()
