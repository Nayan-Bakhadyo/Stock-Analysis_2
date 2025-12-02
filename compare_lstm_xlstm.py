"""
Comprehensive Comparison: LSTM vs xLSTM for Multi-Horizon Stock Forecasting

Compares:
- LSTM (TensorFlow) - Traditional approach
- xLSTM (PyTorch) - Extended LSTM with improved memory (Beck et al. 2024)

Forecast horizons: 1, 3, 5, 10, 15, 21 days
"""

import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import json
from pathlib import Path
import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Import both predictors
from ml_predictor import MLStockPredictor
from xlstm_stock_forecaster import (
    xLSTMStockForecaster,
    xLSTMForecasterTrainer,
    prepare_multi_horizon_data,
)

import config


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
        raise ValueError(f"No data found for {symbol}")
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    return df


def train_lstm_multi_horizon(
    symbol: str,
    horizons: List[int] = [1, 3, 5, 10, 15, 21],
    lookback: int = 60,
    epochs: int = 100,
) -> Dict:
    """
    Train separate LSTM models for each horizon
    
    Returns:
        Dict with predictions, metrics, and training info
    """
    print("\n" + "="*80)
    print("TRAINING LSTM (TensorFlow) - Multi-Horizon")
    print("="*80)
    
    results = {
        'symbol': symbol,
        'model': 'LSTM',
        'horizons': {},
        'training_time': 0,
        'total_params': 0,
    }
    
    start_time = time.time()
    
    # Load data
    df = load_stock_data(symbol, lookback_days=1000)
    print(f"✓ Loaded {len(df)} days of data")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    current_price = df['close'].iloc[-1]
    
    # Train a model for each horizon
    for horizon in horizons:
        print(f"\n--- Training LSTM for {horizon}-day forecast ---")
        
        predictor = MLStockPredictor(lookback_days=lookback)
        
        try:
            # Train model (LSTM always predicts 7 days)
            metrics = predictor.train_model(
                df,
                epochs=epochs,
                batch_size=32,
                validation_split=0.2,
            )
            
            # Get prediction (LSTM returns dict with day keys)
            predictions_dict = predictor.predict_future_prices(df, days=list(range(1, 8)))
            prediction = [predictions_dict.get(i, current_price) for i in range(1, 8)]
            
            if prediction and len(prediction) >= 1:
                # Get prediction for this horizon (max 7 days)
                if horizon <= 7:
                    pred_price = prediction[horizon - 1]
                else:
                    # For horizons > 7, extrapolate linearly
                    trend = (prediction[6] - prediction[0]) / 6
                    pred_price = prediction[6] + trend * (horizon - 7)
                
                pred_return = ((pred_price - current_price) / current_price) * 100
                
                # Calculate metrics
                mape = metrics.get('val_mape', metrics.get('mape', 0))
                if mape == 0:
                    # Estimate MAPE from MAE
                    mae = metrics.get('val_mae', metrics.get('mae', 0))
                    mape = (mae / current_price) * 100 if current_price > 0 else 0
                else:
                    mae = metrics.get('val_mae', metrics.get('mae', 0))
                direction_acc = metrics.get('direction_accuracy', 0)
                
                results['horizons'][horizon] = {
                    'predicted_price': float(pred_price),
                    'predicted_return': float(pred_return),
                    'mape': float(mape),
                    'mae': float(mae),
                    'direction_accuracy': float(direction_acc),
                }
                
                print(f"  ✓ Predicted price: {pred_price:.2f} ({pred_return:+.2f}%)")
                print(f"    MAPE: {mape:.2f}%, Direction Acc: {direction_acc:.1f}%")
            else:
                print(f"  ✗ Failed to generate prediction")
                results['horizons'][horizon] = {
                    'error': 'Prediction failed',
                }
        
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results['horizons'][horizon] = {
                'error': str(e),
            }
    
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
        Dict with predictions, metrics, and training info
    """
    print("\n" + "="*80)
    print("TRAINING xLSTM (PyTorch) - Multi-Horizon")
    print("="*80)
    
    results = {
        'symbol': symbol,
        'model': 'xLSTM',
        'horizons': {},
        'training_time': 0,
        'total_params': 0,
    }
    
    start_time = time.time()
    
    # Load data
    df = load_stock_data(symbol, lookback_days=1000)
    print(f"✓ Loaded {len(df)} days of data")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    current_price = df['close'].iloc[-1]
    
    # Prepare multi-horizon data
    X_train, y_train_dict, X_test, y_test_dict, scaler = prepare_multi_horizon_data(
        df,
        horizons=horizons,
        lookback=lookback,
        train_split=0.8,
    )
    
    print(f"\n✓ Data prepared")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Horizons: {horizons}")
    
    # Create model
    import torch
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"\n✓ Using device: {device.upper()}")
    
    model = xLSTMStockForecaster(
        input_size=1,
        hidden_size=hidden_size,
        num_blocks=num_blocks,
        num_heads=8,
        dropout=0.1,
        horizons=horizons,
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    results['total_params'] = int(total_params)
    print(f"✓ Model created - {total_params:,} parameters")
    
    # Train model
    trainer = xLSTMForecasterTrainer(
        model,
        device=device,
        learning_rate=0.0001,
        weight_decay=0.01,
    )
    
    print(f"\n{'='*80}")
    print(f"Training for {epochs} epochs...")
    print(f"{'='*80}\n")
    
    # Create DataLoaders
    from torch.utils.data import TensorDataset, DataLoader
    import torch
    
    train_targets = torch.stack([y_train_dict[h] for h in horizons], dim=1)
    val_targets = torch.stack([y_test_dict[h] for h in horizons], dim=1)
    
    train_dataset = TensorDataset(X_train, train_targets)
    val_dataset = TensorDataset(X_test, val_targets)
    
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
    print("Evaluating model...")
    print(f"{'='*80}\n")
    
    predictions_dict, metrics_dict = trainer.predict(X_test, y_test_dict)
    
    # Get predictions for latest data point (last sequence)
    latest_sequence = X_test[-1:] if len(X_test) > 0 else X_train[-1:]
    latest_preds, _ = trainer.predict(latest_sequence, None)
    
    # Store results for each horizon
    for horizon in horizons:
        if horizon in latest_preds:
            # Inverse transform prediction
            pred_scaled = latest_preds[horizon][0].item()
            pred_price = scaler.inverse_transform([[pred_scaled]])[0, 0]
            pred_return = ((pred_price - current_price) / current_price) * 100
            
            # Get metrics
            mape = metrics_dict.get(horizon, {}).get('mape', 0)
            mae = metrics_dict.get(horizon, {}).get('mae', 0)
            direction_acc = metrics_dict.get(horizon, {}).get('direction_accuracy', 0)
            
            results['horizons'][horizon] = {
                'predicted_price': float(pred_price),
                'predicted_return': float(pred_return),
                'mape': float(mape),
                'mae': float(mae),
                'direction_accuracy': float(direction_acc),
            }
            
            print(f"✓ {horizon}-day prediction: {pred_price:.2f} ({pred_return:+.2f}%)")
            print(f"  MAPE: {mape:.2f}%, MAE: {mae:.4f}, Direction: {direction_acc:.1f}%")
    
    results['training_time'] = time.time() - start_time
    
    print(f"\n✓ xLSTM training completed in {results['training_time']:.2f}s")
    
    # Save model
    model_dir = Path("xlstm_models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"{symbol}_xlstm_multihorizon.pt"
    
    import torch
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'hidden_size': hidden_size,
            'num_blocks': num_blocks,
            'horizons': horizons,
            'lookback': lookback,
        },
        'scaler': scaler,
    }, model_path)
    
    print(f"✓ Model saved to: {model_path}")
    
    return results


def compare_models(
    symbol: str,
    horizons: List[int] = [1, 3, 5, 10, 15, 21],
    lstm_epochs: int = 100,
    xlstm_epochs: int = 100,
    xlstm_hidden: int = 512,
    xlstm_blocks: int = 7,
) -> Dict:
    """
    Run comprehensive comparison between LSTM and xLSTM
    
    Returns:
        Complete comparison results
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
        symbol,
        horizons=horizons,
        epochs=lstm_epochs,
    )
    
    # Train xLSTM
    xlstm_results = train_xlstm_multi_horizon(
        symbol,
        horizons=horizons,
        epochs=xlstm_epochs,
        hidden_size=xlstm_hidden,
        num_blocks=xlstm_blocks,
    )
    
    # Create comparison
    comparison = {
        'symbol': symbol,
        'timestamp': datetime.now().isoformat(),
        'horizons': horizons,
        'lstm': lstm_results,
        'xlstm': xlstm_results,
        'comparison': {},
    }
    
    # Compare each horizon
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"\n{'Horizon':<10} {'Metric':<20} {'LSTM':<15} {'xLSTM':<15} {'Winner':<10}")
    print("-" * 80)
    
    for horizon in horizons:
        lstm_h = lstm_results['horizons'].get(horizon, {})
        xlstm_h = xlstm_results['horizons'].get(horizon, {})
        
        if 'error' in lstm_h or 'error' in xlstm_h:
            continue
        
        comparison['comparison'][horizon] = {}
        
        # Compare MAPE (lower is better)
        lstm_mape = lstm_h.get('mape', float('inf'))
        xlstm_mape = xlstm_h.get('mape', float('inf'))
        mape_winner = 'LSTM' if lstm_mape < xlstm_mape else 'xLSTM'
        
        print(f"{horizon}d{'':<8} {'MAPE':<20} {lstm_mape:<14.2f}% {xlstm_mape:<14.2f}% {mape_winner:<10}")
        
        # Compare Direction Accuracy (higher is better)
        lstm_dir = lstm_h.get('direction_accuracy', 0)
        xlstm_dir = xlstm_h.get('direction_accuracy', 0)
        dir_winner = 'LSTM' if lstm_dir > xlstm_dir else 'xLSTM'
        
        print(f"{'':<10} {'Direction Acc':<20} {lstm_dir:<14.1f}% {xlstm_dir:<14.1f}% {dir_winner:<10}")
        
        # Compare predictions
        lstm_pred = lstm_h.get('predicted_return', 0)
        xlstm_pred = xlstm_h.get('predicted_return', 0)
        
        print(f"{'':<10} {'Predicted Return':<20} {lstm_pred:<+14.2f}% {xlstm_pred:<+14.2f}%")
        print("-" * 80)
        
        comparison['comparison'][horizon] = {
            'mape': {'lstm': lstm_mape, 'xlstm': xlstm_mape, 'winner': mape_winner},
            'direction_accuracy': {'lstm': lstm_dir, 'xlstm': xlstm_dir, 'winner': dir_winner},
            'predicted_return': {'lstm': lstm_pred, 'xlstm': xlstm_pred},
        }
    
    # Overall comparison
    print("\nOVERALL PERFORMANCE:")
    print(f"  LSTM training time:  {lstm_results['training_time']:.2f}s")
    print(f"  xLSTM training time: {xlstm_results['training_time']:.2f}s")
    print(f"  xLSTM parameters:    {xlstm_results['total_params']:,}")
    
    # Save results
    output_dir = Path("model_comparisons")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{symbol}_lstm_vs_xlstm_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n✓ Comparison saved to: {output_file}")
    
    return comparison


def visualize_comparison(comparison: Dict, symbol: str):
    """Create visualization comparing LSTM vs xLSTM"""
    horizons = comparison['horizons']
    
    # Extract data
    lstm_mapes = []
    xlstm_mapes = []
    lstm_dirs = []
    xlstm_dirs = []
    lstm_rets = []
    xlstm_rets = []
    
    for h in horizons:
        comp = comparison['comparison'].get(h, {})
        if comp:
            lstm_mapes.append(comp['mape']['lstm'])
            xlstm_mapes.append(comp['mape']['xlstm'])
            lstm_dirs.append(comp['direction_accuracy']['lstm'])
            xlstm_dirs.append(comp['direction_accuracy']['xlstm'])
            lstm_rets.append(comp['predicted_return']['lstm'])
            xlstm_rets.append(comp['predicted_return']['xlstm'])
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{symbol}: LSTM vs xLSTM Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: MAPE comparison
    x = np.arange(len(horizons))
    width = 0.35
    
    axes[0].bar(x - width/2, lstm_mapes, width, label='LSTM', alpha=0.8, color='#3498db')
    axes[0].bar(x + width/2, xlstm_mapes, width, label='xLSTM', alpha=0.8, color='#e74c3c')
    axes[0].set_xlabel('Forecast Horizon (days)', fontweight='bold')
    axes[0].set_ylabel('MAPE (%)', fontweight='bold')
    axes[0].set_title('Prediction Error (Lower is Better)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(horizons)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Direction Accuracy
    axes[1].bar(x - width/2, lstm_dirs, width, label='LSTM', alpha=0.8, color='#3498db')
    axes[1].bar(x + width/2, xlstm_dirs, width, label='xLSTM', alpha=0.8, color='#e74c3c')
    axes[1].set_xlabel('Forecast Horizon (days)', fontweight='bold')
    axes[1].set_ylabel('Direction Accuracy (%)', fontweight='bold')
    axes[1].set_title('Trend Prediction (Higher is Better)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(horizons)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_ylim([0, 105])
    
    # Plot 3: Predicted Returns
    axes[2].plot(horizons, lstm_rets, marker='o', linewidth=2, markersize=8, label='LSTM', color='#3498db')
    axes[2].plot(horizons, xlstm_rets, marker='s', linewidth=2, markersize=8, label='xLSTM', color='#e74c3c')
    axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('Forecast Horizon (days)', fontweight='bold')
    axes[2].set_ylabel('Predicted Return (%)', fontweight='bold')
    axes[2].set_title('Multi-Horizon Predictions')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path("model_comparisons")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = output_dir / f"{symbol}_comparison_{timestamp}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to: {fig_path}")
    
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare LSTM vs xLSTM for multi-horizon forecasting")
    parser.add_argument("symbol", help="Stock symbol to analyze")
    parser.add_argument("--horizons", nargs='+', type=int, default=[1, 3, 5, 10, 15, 21],
                        help="Forecast horizons in days")
    parser.add_argument("--lstm-epochs", type=int, default=100,
                        help="Training epochs for LSTM")
    parser.add_argument("--xlstm-epochs", type=int, default=100,
                        help="Training epochs for xLSTM")
    parser.add_argument("--xlstm-hidden", type=int, default=512,
                        help="xLSTM hidden size")
    parser.add_argument("--xlstm-blocks", type=int, default=7,
                        help="Number of xLSTM blocks")
    parser.add_argument("--model", choices=['lstm', 'xlstm', 'compare'], default='compare',
                        help="Which model(s) to run")
    parser.add_argument("--visualize", action='store_true',
                        help="Create comparison visualization")
    
    args = parser.parse_args()
    
    if args.model == 'lstm':
        results = train_lstm_multi_horizon(
            args.symbol,
            horizons=args.horizons,
            epochs=args.lstm_epochs,
        )
        print("\n" + json.dumps(results, indent=2))
    
    elif args.model == 'xlstm':
        results = train_xlstm_multi_horizon(
            args.symbol,
            horizons=args.horizons,
            epochs=args.xlstm_epochs,
            hidden_size=args.xlstm_hidden,
            num_blocks=args.xlstm_blocks,
        )
        print("\n" + json.dumps(results, indent=2))
    
    else:  # compare
        comparison = compare_models(
            args.symbol,
            horizons=args.horizons,
            lstm_epochs=args.lstm_epochs,
            xlstm_epochs=args.xlstm_epochs,
            xlstm_hidden=args.xlstm_hidden,
            xlstm_blocks=args.xlstm_blocks,
        )
        
        if args.visualize:
            visualize_comparison(comparison, args.symbol)
