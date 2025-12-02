"""
Comprehensive Comparison: LSTM vs xLSTM for Multi-Horizon Stock Forecasting

Compares:
- LSTM (TensorFlow) - Traditional approach  
- xLSTM (PyTorch) - Extended LSTM with improved memory (Beck et al. 2024)

Forecast horizons: 1, 3, 5, 10, 15, 21 days
Metrics: MAPE, RMSE, R²
"""

import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import json
from pathlib import Path
import time
from typing import Dict, List, Tuple
import argparse

# Import both predictors
from ml_predictor import MLStockPredictor
from xlstm_stock_forecaster import (
    xLSTMStockForecaster,
    xLSTMForecasterTrainer,
    prepare_multi_horizon_data,
    prepare_enhanced_data,
    EnsembleForecaster,
    compute_technical_indicators,
)
import torch
from torch.utils.data import TensorDataset, DataLoader

import config


def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error"""
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))


def calculate_r2(y_true, y_pred):
    """Calculate R-squared (coefficient of determination)"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0


def load_stock_data(symbol: str, lookback_days: int = 252) -> pd.DataFrame:
    """Load stock data from database"""
    db_path = Path(config.DATA_DIR) / "nepse_stocks.db"
    
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    conn = sqlite3.connect(db_path)
    
    query = """
    SELECT date, open, high, low, close, volume
    FROM price_history
    WHERE symbol = ?
    ORDER BY date DESC
    LIMIT ?
    """
    
    df = pd.read_sql_query(query, conn, params=(symbol, lookback_days * 2))
    conn.close()
    
    if df.empty:
        raise ValueError(f"No data found for symbol: {symbol}")
    
    # Reverse to chronological order
    df = df.iloc[::-1].reset_index(drop=True)
    
    # Rename columns - lowercase for xLSTM, capitalized for LSTM
    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    df['date'] = pd.to_datetime(df['date'])
    
    # Add capitalized columns for LSTM compatibility
    df['Date'] = df['date']
    df['Open'] = df['open']
    df['High'] = df['high']
    df['Low'] = df['low']
    df['Close'] = df['close']
    df['Volume'] = df['volume']
    
    print(f"✓ Loaded {len(df)} days of data")
    print(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    
    return df


def train_lstm_multi_horizon(
    symbol: str,
    horizons: List[int] = [1, 3, 5, 10, 15, 21],
    lookback: int = 60,
    epochs: int = 30,
) -> Dict:
    """
    Train separate LSTM models for each horizon
    
    Returns:
        Dict with predictions and metrics for each horizon
    """
    print("\n" + "="*80)
    print("TRAINING LSTM (TensorFlow) - Multi-Horizon")
    print("="*80)
    
    df = load_stock_data(symbol, lookback_days=lookback * 5)
    current_price = df['Close'].iloc[-1]
    
    results = {
        'symbol': symbol,
        'horizons': {},
    }
    
    start_time = time.time()
    
    # Train a model for each horizon
    for horizon in horizons:
        print(f"\n--- Training LSTM for {horizon}-day forecast ---")
        
        predictor = MLStockPredictor(lookback_days=lookback)
        
        try:
            # Train model (LSTM predicts 7 days)
            metrics = predictor.train_model(
                df,
                epochs=epochs,
                batch_size=32,
                validation_split=0.2,
            )
            
            # Get predictions - ml_predictor returns dict with 'days' key containing day_1, day_3, etc.
            predictions_dict = predictor.predict_future_prices(df, days=list(range(1, 8)))
            
            if not predictions_dict or 'error' in predictions_dict:
                print(f"  ✗ Failed to generate prediction")
                results['horizons'][horizon] = {'error': 'Prediction failed'}
                continue
            
            # Extract predictions from nested dict structure
            # predictions_dict = {'days': {'day_1': {...}, 'day_2': {...}, ...}}
            prediction = []
            if 'days' in predictions_dict:
                for i in range(1, 8):
                    day_key = f'day_{i}'
                    if day_key in predictions_dict['days']:
                        prediction.append(predictions_dict['days'][day_key]['predicted_price'])
                    else:
                        prediction.append(current_price)
            else:
                # Fallback if structure is different
                prediction = [predictions_dict.get(i, current_price) for i in range(1, 8)]

            
            if horizon <= 7:
                pred_price = prediction[horizon - 1]
            else:
                # Extrapolate for horizons > 7
                trend = (prediction[6] - prediction[0]) / 6
                pred_price = prediction[6] + trend * (horizon - 7)
            
            pred_return = ((pred_price - current_price) / current_price) * 100
            
            # Calculate actual MAPE from prediction (not from normalized validation MAE)
            # This is a single-point MAPE, not test set MAPE like xLSTM
            mape = abs(pred_price - current_price) / current_price * 100
            
            # Get MAE from training metrics (normalized space, not comparable)
            mae = metrics.get('val_mae', metrics.get('mae', 0))
            
            direction_acc = metrics.get('direction_accuracy', 0)
            
            results['horizons'][horizon] = {
                'predicted_price': float(pred_price),
                'predicted_return': float(pred_return),
                'mape': float(mape),
                'mae': float(mae),
                'direction_accuracy': float(direction_acc),
                'rmse': 0.0,  # LSTM doesn't provide test set predictions
                'r2': 0.0,
            }
            
            print(f"  ✓ Predicted price: {pred_price:.2f} ({pred_return:+.2f}%)")
            print(f"    MAPE: {mape:.2f}%, Direction Acc: {direction_acc:.1f}%")
        
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results['horizons'][horizon] = {'error': str(e)}
    
    results['training_time'] = time.time() - start_time
    print(f"\n✓ LSTM training completed in {results['training_time']:.2f}s")
    
    return results


def train_xlstm_multi_horizon(
    symbol: str,
    horizons: List[int] = [1, 3, 5, 10, 15, 21],
    lookback: int = 90,
    epochs: int = 100,
    hidden_size: int = 512,
    num_blocks: int = 7,
) -> Dict:
    """
    Train single xLSTM model with multi-horizon heads
    
    Returns:
        Dict with predictions, metrics (MAPE, RMSE, R²), and training info
    """
    print("\n" + "="*80)
    print("TRAINING xLSTM (PyTorch) - Multi-Horizon")
    print("="*80)
    
    df = load_stock_data(symbol, lookback_days=lookback * 5)
    current_price = df['Close'].iloc[-1]
    
    start_time = time.time()
    
    # Use ENHANCED data with 14 technical indicators
    print(f"\n📊 Preparing enhanced data with 14 technical indicators...")
    X_train, y_train_dict, X_test, y_test_dict, scaler_info = prepare_enhanced_data(
        df, horizons=horizons, lookback=lookback, train_split=0.8,
    )
    
    # Get number of features from data shape
    n_features = X_train.shape[2]
    feature_names = scaler_info.get('feature_names', [])
    
    print(f"\n✓ Data prepared (ENHANCED TECHNICAL INDICATORS)")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features ({n_features}): {', '.join(feature_names[:7])}...")
    print(f"  Horizons: {horizons}")
    
    # Determine device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n✓ Using device: {device}")
    
    # Create model with correct input size for multiple features
    model = xLSTMStockForecaster(
        input_size=n_features,  # Now 10 features with volume
        hidden_size=hidden_size,
        num_blocks=num_blocks,
        num_heads=8,
        horizons=horizons,
        dropout=0.2,
    ).to(device)
    
    print(f"Building xLSTM forecaster:")
    print(f"  Input features: {n_features}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Num blocks: {num_blocks}")
    print(f"  Num heads: 8")
    print(f"  Horizons: {horizons}")
    print(f"  Loss: DirectionalLoss (α=0.3)")
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model created - {num_params:,} parameters")
    
    # Create trainer
    trainer = xLSTMForecasterTrainer(
        model=model,
        device=device,
        learning_rate=0.0005,
        weight_decay=0.01,
    )
    
    print(f"\n{'='*80}")
    print(f"Training for {epochs} epochs...")
    print(f"{'='*80}\n")
    
    # Create DataLoaders - convert numpy to torch tensors first
    train_targets = torch.stack([torch.FloatTensor(y_train_dict[h]) for h in horizons], dim=1)
    val_targets = torch.stack([torch.FloatTensor(y_test_dict[h]) for h in horizons], dim=1)
    
    # Convert X to tensors as well
    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)
    
    train_dataset = TensorDataset(X_train_tensor, train_targets)
    val_dataset = TensorDataset(X_test_tensor, val_targets)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    trainer.fit(
        train_loader,
        val_loader,
        epochs=epochs,
        early_stopping_patience=15,
        verbose=True,
    )
    
    # Evaluate on test set
    print(f"\n{'='*80}")
    print(f"Evaluating model...")
    print(f"{'='*80}\n")
    
    predictions = trainer.predict(X_test_tensor)  # Returns dict {horizon: predictions}
    
    results = {
        'symbol': symbol,
        'training_time': time.time() - start_time,
        'parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'horizons': {},
    }
    
    # Get scaler for inverse transform
    target_scaler = scaler_info['target_scaler']
    
    # Calculate metrics for each horizon
    for horizon in horizons:
        # Get normalized predictions and actuals (these are RETURNS, not prices)
        y_true_norm = y_test_dict[horizon].flatten()  # Normalized returns
        y_pred_norm = predictions[horizon].flatten()   # Normalized returns
        
        # Denormalize returns using target_scaler
        y_true_returns = target_scaler.inverse_transform(y_true_norm.reshape(-1, 1)).flatten()
        y_pred_returns = target_scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
        
        # Calculate metrics on RETURNS (more meaningful for forecasting)
        mae_returns = np.abs(y_true_returns - y_pred_returns).mean()
        
        # For MAPE, use absolute returns to avoid division by zero
        mape = (np.abs(y_true_returns - y_pred_returns) / (np.abs(y_true_returns) + 1e-8)).mean() * 100
        
        # RMSE on returns
        rmse = np.sqrt(np.mean((y_true_returns - y_pred_returns) ** 2))
        
        # R² on returns - this should be much better now
        r2 = calculate_r2(y_true_returns, y_pred_returns)
        
        # NEW: Direction Accuracy (most important for trading!)
        true_direction = np.sign(y_true_returns)
        pred_direction = np.sign(y_pred_returns)
        direction_acc = (true_direction == pred_direction).mean() * 100
        
        # Convert last prediction return to price
        # return = (future - current) / current, so future = current * (1 + return)
        last_pred_return = y_pred_returns[-1]
        pred_price = current_price * (1 + last_pred_return)
        pred_return_pct = last_pred_return * 100
        
        # Also calculate price-based MAE for comparison
        # Convert all returns to prices for price-based metrics
        # We need the base prices for each test sample
        # For simplicity, use current_price as base (approximate)
        y_true_prices = current_price * (1 + y_true_returns)
        y_pred_prices = current_price * (1 + y_pred_returns)
        mae_price = np.abs(y_true_prices - y_pred_prices).mean()
        mape_price = (np.abs((y_true_prices - y_pred_prices) / y_true_prices)).mean() * 100
        
        results['horizons'][horizon] = {
            'mae': float(mae_price),  # Price-based for comparison
            'mape': float(mape_price),  # Price-based MAPE
            'rmse': float(rmse * current_price),  # Convert to price scale
            'r2': float(r2),
            'direction_accuracy': float(direction_acc),  # NEW
            'mae_returns': float(mae_returns),  # Return-based MAE
            'predicted_price': float(pred_price),
            'predicted_return': float(pred_return_pct),
        }
        
        print(f"Horizon {horizon:2d} days: MAE={mae_price:.2f}, MAPE={mape_price:.2f}%, R²={r2:.4f}, DirAcc={direction_acc:.1f}%")
    
    print(f"\n✓ xLSTM training completed in {results['training_time']:.2f}s")
    
    return results


def train_xlstm_ensemble(
    symbol: str,
    horizons: List[int] = [1, 3, 5, 10, 15, 21],
    lookback: int = 90,
    epochs: int = 50,
    n_models: int = 3,
    hidden_size: int = 512,
    num_blocks: int = 7,
) -> Dict:
    """
    Train xLSTM ensemble with uncertainty estimates.
    """
    print("\n" + "="*80)
    print("TRAINING xLSTM ENSEMBLE - Multi-Horizon with Uncertainty")
    print("="*80)
    
    df = load_stock_data(symbol, lookback_days=lookback * 5)
    current_price = df['Close'].iloc[-1]
    
    start_time = time.time()
    
    # Prepare enhanced data
    print(f"\n📊 Preparing enhanced data with 14 technical indicators...")
    X_train, y_train_dict, X_test, y_test_dict, scaler_info = prepare_enhanced_data(
        df, horizons=horizons, lookback=lookback, train_split=0.8,
    )
    
    n_features = X_train.shape[2]
    print(f"\n✓ Features: {n_features}")
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n✓ Device: {device}")
    
    # Create ensemble
    print(f"\n🚀 Building ensemble of {n_models} models...")
    ensemble = EnsembleForecaster(
        n_models=n_models,
        input_size=n_features,
        hidden_size=hidden_size,
        num_blocks=num_blocks,
        horizons=horizons,
        device=str(device),
    )
    
    # Create data loaders
    train_targets = torch.stack([torch.FloatTensor(y_train_dict[h]) for h in horizons], dim=1)
    val_targets = torch.stack([torch.FloatTensor(y_test_dict[h]) for h in horizons], dim=1)
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), train_targets)
    val_dataset = TensorDataset(torch.FloatTensor(X_test), val_targets)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Train ensemble
    ensemble.fit(train_loader, val_loader, epochs=epochs, verbose=True)
    
    # Get predictions with uncertainty
    print(f"\n{'='*80}")
    print("Evaluating ensemble with uncertainty...")
    print(f"{'='*80}")
    
    predictions = ensemble.predict_with_uncertainty(X_test)
    
    target_scaler = scaler_info['target_scaler']
    
    results = {
        'symbol': symbol,
        'training_time': time.time() - start_time,
        'n_models': n_models,
        'horizons': {},
    }
    
    for horizon in horizons:
        y_true_norm = y_test_dict[horizon].flatten()
        y_pred_norm = predictions[horizon]['mean'].flatten()
        y_std_norm = predictions[horizon]['std'].flatten()
        
        # Denormalize
        y_true_returns = target_scaler.inverse_transform(y_true_norm.reshape(-1, 1)).flatten()
        y_pred_returns = target_scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
        y_std_returns = y_std_norm * target_scaler.scale_[0]
        
        # Metrics
        y_true_prices = current_price * (1 + y_true_returns)
        y_pred_prices = current_price * (1 + y_pred_returns)
        
        mae_price = np.abs(y_true_prices - y_pred_prices).mean()
        mape_price = (np.abs((y_true_prices - y_pred_prices) / y_true_prices)).mean() * 100
        r2 = calculate_r2(y_true_returns, y_pred_returns)
        
        # Direction accuracy
        direction_acc = (np.sign(y_true_returns) == np.sign(y_pred_returns)).mean() * 100
        
        # Mean uncertainty
        mean_std = y_std_returns.mean()
        confidence = ensemble.get_confidence_level(mean_std, horizon)
        
        results['horizons'][horizon] = {
            'mae': float(mae_price),
            'mape': float(mape_price),
            'r2': float(r2),
            'direction_accuracy': float(direction_acc),
            'uncertainty': float(mean_std),
            'confidence': confidence,
            'predicted_price': float(y_pred_prices[-1]),
        }
        
        print(f"Horizon {horizon:2d}: MAE={mae_price:.2f}, MAPE={mape_price:.2f}%, R²={r2:.4f}, DirAcc={direction_acc:.1f}%, Conf={confidence}")
    
    print(f"\n✓ Ensemble training completed in {results['training_time']:.2f}s")
    
    return results


def compare_models(
    symbol: str,
    horizons: List[int] = [1, 3, 5, 10, 15, 21],
    lstm_epochs: int = 30,
    xlstm_epochs: int = 100,
    xlstm_hidden: int = 512,
    xlstm_blocks: int = 7,
    use_ensemble: bool = False,
    n_ensemble_models: int = 3,
) -> Dict:
    """
    Compare LSTM and xLSTM on multi-horizon forecasting
    
    Returns:
        Comprehensive comparison results with MAPE, RMSE, R²
    """
    print("\n" + "="*80)
    print(f"COMPARING LSTM vs xLSTM - {symbol}")
    print("="*80)
    print(f"Horizons: {horizons}")
    print(f"LSTM epochs: {lstm_epochs}")
    print(f"xLSTM epochs: {xlstm_epochs}, hidden: {xlstm_hidden}, blocks: {xlstm_blocks}")
    print("="*80)
    
    # Train LSTM
    lstm_results = train_lstm_multi_horizon(
        symbol=symbol,
        horizons=horizons,
        epochs=lstm_epochs,
    )
    
    # Train xLSTM (single or ensemble)
    if use_ensemble:
        print(f"\n🔥 Using ENSEMBLE mode with {n_ensemble_models} models")
        xlstm_results = train_xlstm_ensemble(
            symbol=symbol,
            horizons=horizons,
            epochs=xlstm_epochs,
            n_models=n_ensemble_models,
            hidden_size=xlstm_hidden,
            num_blocks=xlstm_blocks,
        )
    else:
        xlstm_results = train_xlstm_multi_horizon(
            symbol=symbol,
            horizons=horizons,
            epochs=xlstm_epochs,
            hidden_size=xlstm_hidden,
            num_blocks=xlstm_blocks,
        )
    
    # Print comparison
    print(f"\n{'='*80}")
    print(f"COMPARISON RESULTS - {symbol}")
    print(f"{'='*80}")
    print(f"\nModel Parameters:")
    print(f"  LSTM:  N/A (TensorFlow)")
    if 'parameters' in xlstm_results:
        print(f"  xLSTM: {xlstm_results['parameters']:,}")
    elif 'n_models' in xlstm_results:
        print(f"  xLSTM: Ensemble of {xlstm_results['n_models']} models")
    
    print(f"\nTraining Time:")
    print(f"  LSTM:  {lstm_results['training_time']:.2f}s")
    print(f"  xLSTM: {xlstm_results['training_time']:.2f}s")
    
    # MAPE Comparison
    print(f"\n{'='*80}")
    print(f"PERFORMANCE METRICS: MAPE (Lower is Better)")
    print(f"{'='*80}")
    print(f"{'Horizon':<10} {'LSTM MAPE':<12} {'xLSTM MAPE':<12} {'Winner':<10}")
    print("-" * 50)
    
    for h in horizons:
        lstm_mape = lstm_results['horizons'].get(h, {}).get('mape', float('inf'))
        xlstm_mape = xlstm_results['horizons'].get(h, {}).get('mape', float('inf'))
        
        winner = "LSTM" if lstm_mape < xlstm_mape else "xLSTM"
        if lstm_mape == float('inf') or xlstm_mape == float('inf'):
            winner = "N/A"
        
        print(f"{h:<10} {lstm_mape:>10.2f}%  {xlstm_mape:>10.2f}%  {winner:<10}")
    
    # RMSE Comparison (xLSTM only)
    print(f"\n{'='*80}")
    print(f"PERFORMANCE METRICS: RMSE (Lower is Better)")
    print(f"{'='*80}")
    print(f"{'Horizon':<10} {'xLSTM RMSE':<15}")
    print("-" * 30)
    
    for h in horizons:
        xlstm_rmse = xlstm_results['horizons'].get(h, {}).get('rmse', 0)
        print(f"{h:<10} {xlstm_rmse:>12.4f}")
    
    # R² Comparison (xLSTM only)
    print(f"\n{'='*80}")
    print(f"PERFORMANCE METRICS: R² (Higher is Better, max=1.0)")
    print(f"{'='*80}")
    print(f"{'Horizon':<10} {'xLSTM R²':<15}")
    print("-" * 30)
    
    for h in horizons:
        xlstm_r2 = xlstm_results['horizons'].get(h, {}).get('r2', 0)
        print(f"{h:<10} {xlstm_r2:>12.4f}")
    
    # Save results
    comparison = {
        'symbol': symbol,
        'comparison_date': datetime.now().isoformat(),
        'horizons': horizons,
        'lstm': lstm_results,
        'xlstm': xlstm_results,
    }
    
    # Save to file
    output_dir = Path("model_comparisons")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{symbol}_comparison_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n✓ Comparison saved to: {output_file}")
    
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare LSTM vs xLSTM")
    parser.add_argument("symbol", type=str, help="Stock symbol (e.g., PFL)")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5, 10, 15, 21],
                       help="Forecast horizons in days")
    parser.add_argument("--lstm-epochs", type=int, default=30,
                       help="LSTM training epochs")
    parser.add_argument("--xlstm-epochs", type=int, default=100,
                       help="xLSTM training epochs")
    parser.add_argument("--xlstm-hidden", type=int, default=512,
                       help="xLSTM hidden size")
    parser.add_argument("--xlstm-blocks", type=int, default=7,
                       help="xLSTM number of blocks")
    parser.add_argument("--ensemble", action="store_true",
                       help="Use ensemble of xLSTM models with uncertainty")
    parser.add_argument("--n-models", type=int, default=3,
                       help="Number of models in ensemble (default: 3)")
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    if args.ensemble:
        print(f"Using ENSEMBLE mode with {args.n_models} xLSTM models")
        print(f"  - Multiple models with different random seeds")
        print(f"  - Predictions with uncertainty (95% CI)")
        print(f"  - Confidence levels (HIGH/MEDIUM/LOW)")
    else:
        print("Using STANDARD single xLSTM model")
    print(f"\n14 Technical Indicators: returns, momentum, MACD, BB, ADX, RSI, volume...")
    print(f"{'='*80}\n")
    
    comparison = compare_models(
        symbol=args.symbol,
        horizons=args.horizons,
        lstm_epochs=args.lstm_epochs,
        xlstm_epochs=args.xlstm_epochs,
        xlstm_hidden=args.xlstm_hidden,
        xlstm_blocks=args.xlstm_blocks,
        use_ensemble=args.ensemble,
        n_ensemble_models=args.n_models,
    )
