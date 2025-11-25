"""
Quick predictor using a specific configuration (no grid search)
Useful for predicting with already-known best config
"""
import sys
sys.path.insert(0, '/Users/Nayan/Documents/Business/Stock_Analysis')

# Suppress TensorFlow warnings and optimize for M1
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

# Configure TensorFlow for M1 optimization
import tensorflow as tf
# Disable mixed precision and XLA for prediction script (causes XLA errors on Metal)
# Mixed precision is only used during grid search in subprocesses
# tf.keras.mixed_precision.set_global_policy('mixed_float16')
# tf.config.optimizer.set_jit(True)

from tune_hyperparameters import HyperparameterTuner
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def predict_with_config(symbol, config, performance_metrics, output_file='stock_predictions.json'):
    """
    Predict next 7 days using a specific configuration
    
    Args:
        symbol: Stock symbol
        config: Dictionary with model configuration
        output_file: JSON file to save results
    """
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Predicting {symbol} with Best Config")
    print(f"{'='*60}")
    print(f"Config: {config}")
    
    # Create tuner to load ALL data and train final model
    print(f"\nTraining on ALL available data for future predictions...")
    
    # Use test_days=0 to load ALL available data (no test split)
    tuner = HyperparameterTuner(symbol=symbol, test_days=0)
    full_train_df, _ = tuner.load_data()
    
    # Add required fields to config
    final_config = config.copy()
    final_config['epochs'] = 100
    final_config['optimizer'] = 'adam'
    
    # Prepare features on ALL available data
    data_with_features = tuner.prepare_features(full_train_df)
    feature_cols = ['close', 'volume', 'returns', 'sma_5', 'sma_20', 'rsi', 
                   'macd', 'bb_width', 'volume_ratio', 'atr', 'momentum_5']
    feature_cols = [col for col in feature_cols if col in data_with_features.columns]
    
    features = data_with_features[feature_cols].values
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Scale
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features)
    
    # Create sequences for 7-step-ahead prediction (same as grid search)
    lookback = final_config['lookback']
    X, y = [], []
    for i in range(lookback, len(scaled_features) - 6):  # -6 because we need 7 future values
        X.append(scaled_features[i-lookback:i])
        y.append(scaled_features[i:i+7, 0])  # Next 7 close prices
    X, y = np.array(X), np.array(y)
    
    # Build model (will output 7 predictions as designed)
    model = tuner.build_model(final_config, (final_config['lookback'], len(feature_cols)))
    
    # Callbacks with validation monitoring for better early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=final_config['patience'], restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
    
    print(f"Training on {len(X)} samples with 15% validation split...")
    history = model.fit(
        X, y,
        epochs=final_config['epochs'],
        batch_size=final_config['batch_size'],
        validation_split=0.15,  # Use 15% for validation to enable early stopping
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # Save the trained model
    model_path = models_dir / f"{symbol}_model.keras"
    model.save(model_path)
    print(f"\n✓ Model saved to: {model_path}")
    
    # Predict 7 days directly (model outputs 7 predictions at once)
    last_sequence = scaled_features[-final_config['lookback']:]
    X_pred = last_sequence.reshape(1, final_config['lookback'], len(feature_cols))
    predictions_scaled = model.predict(X_pred, verbose=0)[0]  # Get 7 predictions
    
    # Inverse transform
    dummy = np.zeros((7, len(feature_cols)))
    dummy[:, 0] = predictions_scaled
    predictions = scaler.inverse_transform(dummy)[:, 0].tolist()
    
    # Calculate directional accuracy for future predictions
    base_price = full_train_df['close'].iloc[-1]
    pred_directions = [1 if p > base_price else -1 for p in predictions]
    up_days = sum(1 for d in pred_directions if d == 1)
    down_days = 7 - up_days
    direction_confidence = (max(up_days, down_days) / 7) * 100
    overall_direction = "BULLISH" if up_days > down_days else "BEARISH" if down_days > up_days else "NEUTRAL"
    
    # Generate dates
    last_date = pd.to_datetime(full_train_df['date'].iloc[-1])
    prediction_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') 
                       for i in range(7)]
    
    # Recent prices
    recent_prices = full_train_df['close'].tail(30).tolist()
    recent_dates = pd.to_datetime(full_train_df['date']).tail(30).dt.strftime('%Y-%m-%d').tolist()
    
    # Display
    print(f"\n{'='*60}")
    print(f"TRUE FUTURE Predictions")
    print(f"{'='*60}")
    print(f"Last known date: {last_date.strftime('%Y-%m-%d')}")
    print(f"Last known price: Rs. {full_train_df['close'].iloc[-1]:.2f}")
    
    print(f"\n🎯 Trading Signal:")
    print(f"  Direction: {overall_direction}")
    print(f"  Confidence: {direction_confidence:.1f}% ({up_days} up days, {down_days} down days)")
    if overall_direction == "BULLISH":
        print(f"  Recommendation: CONSIDER BUY (model predicts upward trend)")
    elif overall_direction == "BEARISH":
        print(f"  Recommendation: CONSIDER SELL/HOLD (model predicts downward trend)")
    else:
        print(f"  Recommendation: HOLD (model predicts sideways movement)")
    
    print(f"\n📈 7-Day Price Forecast:")
    for date, price in zip(prediction_dates, predictions):
        change = price - full_train_df['close'].iloc[-1]
        change_pct = (change / full_train_df['close'].iloc[-1]) * 100
        direction = "↑" if price > full_train_df['close'].iloc[-1] else "↓" if price < full_train_df['close'].iloc[-1] else "→"
        print(f"  {date}: Rs. {price:.2f} ({change_pct:+.2f}%) {direction}")
    
    # Save to JSON
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            data = json.load(f)
    else:
        data = {'stocks': {}, 'last_updated': None}
    
    # Calculate ratings if not present in performance_metrics
    test_mape = performance_metrics.get('test_mape', 0)
    direction_acc = performance_metrics.get('direction_accuracy', 0)
    
    # Calculate MAPE rating
    if test_mape < 2:
        mape_rating = 'Excellent'
    elif test_mape < 5:
        mape_rating = 'Good'
    elif test_mape < 10:
        mape_rating = 'Fair'
    else:
        mape_rating = 'Poor'
    
    # Calculate signal strength
    if direction_acc > 70:
        signal_strength = 'Strong'
    elif direction_acc >= 50:
        signal_strength = 'Moderate'
    else:
        signal_strength = 'Weak'
    
    stock_data = {
        'symbol': symbol,
        'predictions': {
            'dates': prediction_dates,
            'prices': [round(p, 2) for p in predictions]
        },
        'recent_actual': {
            'dates': recent_dates,
            'prices': [round(p, 2) for p in recent_prices]
        },
        'trading_signal': {
            'direction': overall_direction,
            'confidence': round(direction_confidence, 1),
            'up_days': up_days,
            'down_days': down_days,
            'recommendation': 'BUY' if overall_direction == 'BULLISH' else 'SELL' if overall_direction == 'BEARISH' else 'HOLD'
        },
        'model': {
            'model_path': str(model_path),
            'architecture': config['architecture'],
            'lookback_days': config['lookback'],
            'layers': config['layers'],
            'training_samples': len(full_train_df),
            'last_actual_price': round(full_train_df['close'].iloc[-1], 2),
            'performance': {
                'test_mape': performance_metrics.get('test_mape', 0),
                'test_mae': performance_metrics.get('test_mae', 0),
                'test_rmse': performance_metrics.get('test_rmse', 0),
                'direction_accuracy': performance_metrics.get('direction_accuracy', 0),
                'mape_rating': mape_rating,
                'signal_strength': signal_strength
            }
        },
        'config': config,
        'analysis_date': datetime.now().isoformat(),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    data['stocks'][symbol] = stock_data
    data['last_updated'] = datetime.now().isoformat()
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✓ Predictions saved to {output_file}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python3 predict_single_config.py <SYMBOL>")
        print("Example: python3 predict_single_config.py SPC")
        sys.exit(1)
    
    symbol = sys.argv[1].upper()
    config_file = f"best_configs/{symbol}_best_config.json"
    
    # Load best config
    if not os.path.exists(config_file):
        print(f"❌ Config file not found: {config_file}")
        print(f"Please run grid search first:")
        print(f"  python3 stock_predictor.py {symbol}")
        sys.exit(1)
    
    with open(config_file, 'r') as f:
        config_data = json.load(f)
    
    best_config = config_data['best_config']
    performance_metrics = config_data['performance']
    print(f"✓ Loaded best config for {symbol}")
    print(f"  Performance: MAPE={config_data['performance']['test_mape']:.2f}%, "
          f"Direction Acc={config_data['performance']['direction_accuracy']:.1f}%")
    
    predict_with_config(symbol, best_config, performance_metrics)
