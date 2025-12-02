"""
ML-based Stock Price Predictor
Uses LSTM neural networks for multi-week price predictions (1-6 weeks)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML/DL libraries
try:
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available. Install with: pip install tensorflow scikit-learn")

# Technical indicators
from technical_analyzer import TechnicalAnalyzer


class MLStockPredictor:
    """ML-based stock price predictor with LSTM"""
    
    def __init__(self, lookback_days: int = 60):
        """
        Initialize ML predictor
        
        Args:
            lookback_days: Number of historical days to use for predictions
        """
        self.lookback_days = lookback_days
        self.scaler = MinMaxScaler(feature_range=(0, 1)) if TENSORFLOW_AVAILABLE else None
        self.model = None
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1)) if TENSORFLOW_AVAILABLE else None
        self.technical_analyzer = TechnicalAnalyzer()
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare feature set with technical indicators
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with engineered features
        """
        # Make a copy to avoid modifying original
        data = df.copy()
        
        # Calculate technical indicators
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
        
        # Drop NaN values
        data = data.dropna()
        
        return data
    
    def create_sequences(self, data: np.ndarray, lookback: int, forecast_days: int = 7) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training with multi-output (7-day predictions)
        
        Args:
            data: Scaled feature array
            lookback: Number of historical timesteps
            forecast_days: Number of days to forecast (default: 7)
            
        Returns:
            X, y arrays for training
            X shape: (samples, lookback, features)
            y shape: (samples, forecast_days) - predicts next 7 days
        """
        X, y = [], []
        
        # Need enough data for lookback + forecast_days
        for i in range(lookback, len(data) - forecast_days + 1):
            X.append(data[i-lookback:i])
            # y contains next 7 days of close prices
            y.append(data[i:i+forecast_days, 0])  # Next 7 days close prices
        
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, input_shape: Tuple[int, int], forecast_days: int = 7) -> Sequential:
        """
        Build LSTM model architecture with multi-output (7-day predictions)
        
        Args:
            input_shape: Shape of input (timesteps, features)
            forecast_days: Number of days to forecast (default: 7)
            
        Returns:
            Compiled Keras model that outputs 7 price predictions
        """
        model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
            Dropout(0.2),
            
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.2),
            
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(16, activation='relu'),
            Dense(forecast_days)  # Output 7 predictions (Day 1-7)
        ])
        
        model.compile(
            optimizer=keras.optimizers.legacy.Adam(learning_rate=0.001),  # Legacy for M1/M2 performance
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        return model
    
    def train_model(self, df: pd.DataFrame, epochs: int = 50, batch_size: int = 32, 
                   validation_split: float = 0.2) -> Dict:
        """
        Train LSTM model on historical data
        
        Args:
            df: DataFrame with OHLCV data
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data for validation
            
        Returns:
            Training metrics dictionary
        """
        if not TENSORFLOW_AVAILABLE:
            return {'error': 'TensorFlow not available'}
        
        # Prepare features
        print("📊 Preparing features...")
        data = self.prepare_features(df)
        
        # Select feature columns
        feature_cols = ['close', 'volume', 'returns', 'sma_5', 'sma_20', 'rsi', 
                       'macd', 'bb_width', 'volume_ratio', 'atr', 'momentum_5']
        
        # Ensure all columns exist
        feature_cols = [col for col in feature_cols if col in data.columns]
        features = data[feature_cols].values
        
        # Check for infinity/NaN and replace
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        scaled_features = self.feature_scaler.fit_transform(features)
        
        # CRITICAL FIX: Scale target prices separately (first column is 'close')
        # This ensures model learns patterns in 0-1 range, preventing it from outputting mean
        self.price_scaler = MinMaxScaler()
        scaled_prices = self.price_scaler.fit_transform(features[:, 0].reshape(-1, 1)).flatten()
        
        # Replace close prices in scaled_features with properly scaled prices
        scaled_features[:, 0] = scaled_prices
        
        # Create sequences (multi-output: 7 days)
        print(f"🔄 Creating sequences (lookback={self.lookback_days}, forecast=7 days)...")
        X, y = self.create_sequences(scaled_features, self.lookback_days, forecast_days=7)
        
        if len(X) < 10:
            return {'error': f'Insufficient data. Need at least {self.lookback_days + 17} days'}
        
        # Split train/validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"✓ Training set: {len(X_train)} samples (each predicts 7 days)")
        print(f"✓ Validation set: {len(X_val)} samples")
        
        # Build model
        print("🏗️ Building multi-output LSTM model (7-day predictions)...")
        self.model = self.build_lstm_model((self.lookback_days, len(feature_cols)), forecast_days=7)
        
        # Callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
        
        # Train
        print("🚀 Training model...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        
        # Evaluate
        train_pred = self.model.predict(X_train, verbose=0)
        val_pred = self.model.predict(X_val, verbose=0)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        
        print(f"✓ Training MAE: {train_mae:.6f}")
        print(f"✓ Validation MAE: {val_mae:.6f}")
        
        return {
            'train_mae': float(train_mae),
            'val_mae': float(val_mae),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'epochs_trained': len(history.history['loss'])
        }
    
    def save_model(self, symbol: str, save_dir: str = 'models'):
        """Save trained model for a specific stock symbol"""
        import os
        if self.model is None:
            print("⚠️ No model to save")
            return
        
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f'lstm_{symbol}.keras')
        scaler_path = os.path.join(save_dir, f'scaler_{symbol}.pkl')
        
        self.model.save(model_path)
        
        # Save scalers
        import pickle
        scalers_to_save = {
            'scaler': self.scaler,
            'feature_scaler': self.feature_scaler
        }
        # Include price_scaler if it exists (new models)
        if hasattr(self, 'price_scaler'):
            scalers_to_save['price_scaler'] = self.price_scaler
            
        with open(scaler_path, 'wb') as f:
            pickle.dump(scalers_to_save, f)
        
        print(f"  ✓ Model saved: {model_path}")
    
    def load_model(self, symbol: str, load_dir: str = 'models'):
        """Load trained model for a specific stock symbol"""
        import os
        model_path = os.path.join(load_dir, f'lstm_{symbol}.keras')
        scaler_path = os.path.join(load_dir, f'scaler_{symbol}.pkl')
        
        # Try .keras first, fall back to .h5 for old models
        if not os.path.exists(model_path):
            model_path_h5 = os.path.join(load_dir, f'lstm_{symbol}.h5')
            if os.path.exists(model_path_h5):
                model_path = model_path_h5
            else:
                print(f"  ℹ️ No saved model found for {symbol}")
                return False
        
        self.model = keras.models.load_model(model_path)
        
        import pickle
        with open(scaler_path, 'rb') as f:
            scalers = pickle.load(f)
            self.scaler = scalers['scaler']
            self.feature_scaler = scalers['feature_scaler']
            # Load price_scaler if it exists (new models)
            if 'price_scaler' in scalers:
                self.price_scaler = scalers['price_scaler']
        
        print(f"  ✓ Model loaded: {model_path}")
        return True
    
    def predict_future_prices(self, df: pd.DataFrame, days: List[int] = None) -> Dict:
        """
        Predict future prices using multi-output LSTM (predicts 7 days at once)
        
        Args:
            df: DataFrame with historical OHLCV data
            days: List of days to return (1-7). If None, returns all 7 days
            
        Returns:
            Dictionary with predictions for each day (1-7)
        """
        if not TENSORFLOW_AVAILABLE:
            return {'error': 'TensorFlow not available'}
        
        if self.model is None:
            print("⚠️ Model not trained. Training now...")
            metrics = self.train_model(df)
            if 'error' in metrics:
                return metrics
        
        # Prepare features
        data = self.prepare_features(df)
        
        # Select feature columns (same as training)
        feature_cols = ['close', 'volume', 'returns', 'sma_5', 'sma_20', 'rsi', 
                       'macd', 'bb_width', 'volume_ratio', 'atr', 'momentum_5']
        feature_cols = [col for col in feature_cols if col in data.columns]
        
        # Get current price
        current_price = float(df['close'].iloc[-1])
        
        # Use recent data for prediction
        recent_data = data[feature_cols].iloc[-self.lookback_days:].values
        
        # Check for infinity/NaN and replace
        recent_data = np.nan_to_num(recent_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # CRITICAL FIX: Scale features the same way as training
        scaled_recent = self.feature_scaler.transform(recent_data)
        
        # Also scale prices separately if price_scaler exists (from training)
        if hasattr(self, 'price_scaler'):
            scaled_prices = self.price_scaler.transform(recent_data[:, 0].reshape(-1, 1)).flatten()
            scaled_recent[:, 0] = scaled_prices
        
        # Prepare input (1, 60, 11)
        X_pred = scaled_recent.reshape(1, self.lookback_days, len(feature_cols))
        
        # Get all 7 predictions in ONE forward pass (no error accumulation!)
        predictions_scaled = self.model.predict(X_pred, verbose=0)[0]  # Shape: (7,)
        
        # CRITICAL FIX: Inverse transform using price_scaler (not feature_scaler)
        if hasattr(self, 'price_scaler'):
            predicted_prices = self.price_scaler.inverse_transform(
                predictions_scaled.reshape(-1, 1)
            ).flatten()
        else:
            # Fallback to old method if price_scaler not available (backward compatibility)
            dummy = np.zeros((7, len(feature_cols)))
            dummy[:, 0] = predictions_scaled
            unscaled = self.feature_scaler.inverse_transform(dummy)
            predicted_prices = unscaled[:, 0]

        
        # Build results
        if days is None:
            days = list(range(1, 8))  # All 7 days
        
        predictions = {
            'current_price': current_price,
            'prediction_date': datetime.now().strftime('%Y-%m-%d'),
            'model_type': 'multi_output_lstm',
            'days': {}
        }
        
        for day in days:
            if 1 <= day <= 7:
                predicted_price = float(predicted_prices[day-1])
                price_change = predicted_price - current_price
                price_change_pct = (price_change / current_price) * 100
                
                predictions['days'][f'day_{day}'] = {
                    'day': day,
                    'target_date': (datetime.now() + timedelta(days=day)).strftime('%Y-%m-%d'),
                    'predicted_price': round(predicted_price, 2),
                    'price_change': round(price_change, 2),
                    'price_change_pct': round(price_change_pct, 2),
                    'trend': 'UP' if price_change_pct > 0 else 'DOWN' if price_change_pct < 0 else 'FLAT'
                }
        
        # Add summary statistics
        predictions['summary'] = {
            'avg_predicted_price': round(float(np.mean(predicted_prices[:len(days)])), 2),
            'min_predicted_price': round(float(np.min(predicted_prices[:len(days)])), 2),
            'max_predicted_price': round(float(np.max(predicted_prices[:len(days)])), 2),
            'total_change_pct': round(((predicted_prices[len(days)-1] - current_price) / current_price) * 100, 2),
            'volatility': round(float(np.std(predicted_prices[:len(days)])), 2)
        }
        
        return predictions
    
    def get_trend_analysis(self, predictions: Dict) -> Dict:
        """
        Analyze predicted trends across daily predictions
        
        Args:
            predictions: Predictions dictionary from predict_future_prices
            
        Returns:
            Trend analysis summary
        """
        if 'error' in predictions or not predictions.get('days'):
            return {}
        
        days = predictions['days']
        
        # Analyze trends
        trends = [d['trend'] for d in days.values()]
        avg_change = np.mean([d['price_change_pct'] for d in days.values()])
        
        # Determine overall trend
        up_count = trends.count('UP')
        down_count = trends.count('DOWN')
        
        if up_count > down_count:
            overall_trend = 'BULLISH'
        elif down_count > up_count:
            overall_trend = 'BEARISH'
        else:
            overall_trend = 'NEUTRAL'
        
        # Get best and worst predictions
        best_day = max(days.items(), key=lambda x: x[1]['price_change_pct'])
        worst_day = min(days.items(), key=lambda x: x[1]['price_change_pct'])
        
        return {
            'overall_trend': overall_trend,
            'avg_predicted_change': round(avg_change, 2),
            'bullish_days': up_count,
            'bearish_days': down_count,
            'best_day': {
                'day': best_day[0],
                'change_pct': round(best_day[1]['price_change_pct'], 2),
                'predicted_price': best_day[1]['predicted_price']
            },
            'worst_day': {
                'day': worst_day[0],
                'change_pct': round(worst_day[1]['price_change_pct'], 2),
                'predicted_price': worst_day[1]['predicted_price']
            },
            'volatility': predictions['summary']['volatility']
        }


if __name__ == "__main__":
    # Test the predictor
    print("Testing ML Stock Predictor")
    print("=" * 60)
    
    if not TENSORFLOW_AVAILABLE:
        print("❌ TensorFlow not installed. Install with:")
        print("   pip install tensorflow scikit-learn")
        exit(1)
    
    # Load sample data
    from data_fetcher import NepseDataFetcher
    
    fetcher = NepseDataFetcher()
    symbol = "IGI"
    
    print(f"\n📊 Testing predictions for {symbol}...")
    df = fetcher.get_stock_price_history(symbol, days=None)
    
    if df.empty:
        print(f"❌ No data available for {symbol}")
        exit(1)
    
    print(f"✓ Loaded {len(df)} days of historical data")
    
    # Initialize predictor
    predictor = MLStockPredictor(lookback_days=60)
    
    # Train model
    print("\n🚀 Training model...")
    metrics = predictor.train_model(df, epochs=50, batch_size=32)
    
    if 'error' in metrics:
        print(f"❌ Error: {metrics['error']}")
        exit(1)
    
    # Make predictions
    print("\n🔮 Making 7-day predictions (multi-output LSTM)...")
    predictions = predictor.predict_future_prices(df, days=[1, 2, 3, 4, 5, 6, 7])
    
    if 'error' in predictions:
        print(f"❌ Error: {predictions['error']}")
        exit(1)
    
    # Display results
    print(f"\n📈 PREDICTIONS FOR {symbol}")
    print("=" * 60)
    print(f"Current Price: NPR {predictions['current_price']:.2f}")
    print(f"Prediction Date: {predictions['prediction_date']}")
    print("\n")
    
    for horizon_key, horizon_data in predictions['horizons'].items():
        print(f"{'='*60}")
        print(f"{horizon_data['weeks_ahead']}-Week Forecast ({horizon_data['target_date']})")
        print(f"{'='*60}")
        print(f"  Predicted Price: NPR {horizon_data['predicted_price']:.2f}")
        print(f"  Price Change: NPR {horizon_data['price_change']:+.2f} ({horizon_data['price_change_pct']:+.2f}%)")
        print(f"  Price Range: NPR {horizon_data['price_range']['min']:.2f} - {horizon_data['price_range']['max']:.2f}")
        print(f"  Trend: {horizon_data['trend']}")
        print(f"  Confidence: {horizon_data['confidence_score']:.1f}%")
        print()
    
    # Trend analysis
    trend_analysis = predictor.get_trend_analysis(predictions)
    print(f"\n🎯 TREND ANALYSIS")
    print("=" * 60)
    print(f"Overall Trend: {trend_analysis['overall_trend']}")
    print(f"Avg Predicted Change: {trend_analysis['avg_predicted_change']:+.2f}%")
    print(f"Avg Confidence: {trend_analysis['avg_confidence']:.1f}%")
    print(f"Bullish Horizons: {trend_analysis['bullish_horizons']}/{len(predictions['horizons'])}")
    print(f"\nBest Outlook: {trend_analysis['best_horizon']['period']} ({trend_analysis['best_horizon']['change_pct']:+.2f}%)")
    print(f"Worst Outlook: {trend_analysis['worst_horizon']['period']} ({trend_analysis['worst_horizon']['change_pct']:+.2f}%)")
