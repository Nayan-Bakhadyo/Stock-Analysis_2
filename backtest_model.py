#!/usr/bin/env python3
"""
Backtest Script for xLSTM Stock Forecaster
Evaluates model accuracy on historical data with proper train/test split.

Usage:
    python3 backtest_model.py NABIL --days 30
    python3 backtest_model.py NABIL --market --days 60
"""

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
import config
from xlstm_stock_forecaster import (
    xLSTMStockForecaster,
    xLSTMForecasterTrainer,
    EnsembleForecaster,
    MultiHorizonDataset,
    compute_technical_indicators,
    compute_market_features,
    get_market_data,
    prepare_enhanced_data,
    prepare_market_enhanced_data,
)


def backtest_model(
    symbol: str,
    test_days: int = 30,
    use_market: bool = True,
    lookback: int = 120,
    epochs: int = 50,
    n_models: int = 3,
    hidden_size: int = 256,
    num_blocks: int = 4,
    horizons: list = [1, 3, 5, 10],
):
    """
    Backtest the model by:
    1. Training on data up to (today - test_days)
    2. Making predictions for each day in the test period
    3. Comparing predictions vs actual prices
    """
    print(f"\n{'='*70}")
    print(f"📊 BACKTESTING: {symbol}")
    print(f"{'='*70}")
    print(f"Test period: Last {test_days} trading days")
    print(f"Model: {'Market-aware' if use_market else 'Standard'} xLSTM Ensemble")
    print(f"Horizons: {horizons}")
    print(f"{'='*70}\n")
    
    # Load all data
    conn = sqlite3.connect(config.DB_PATH)
    query = f"""
        SELECT date, open, high, low, close, volume
        FROM price_history
        WHERE symbol = '{symbol}'
        ORDER BY date ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if len(df) < lookback + max(horizons) + test_days + 50:
        print(f"❌ Not enough data for backtesting")
        return None
    
    print(f"✓ Loaded {len(df)} days of data")
    print(f"  Full range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
    
    # Split into train and test
    train_df = df.iloc[:-test_days].copy()
    test_df = df.iloc[-test_days:].copy()
    
    print(f"  Training: {len(train_df)} days ({train_df['date'].iloc[0]} to {train_df['date'].iloc[-1]})")
    print(f"  Testing: {len(test_df)} days ({test_df['date'].iloc[0]} to {test_df['date'].iloc[-1]})")
    
    # Prepare training data
    print(f"\n📊 Preparing training data...")
    if use_market:
        X_train, y_train, X_val, y_val, scaler_info = prepare_market_enhanced_data(
            train_df, symbol, lookback=lookback, horizons=horizons, train_split=0.9
        )
    else:
        X_train, y_train, X_val, y_val, scaler_info = prepare_enhanced_data(
            train_df, lookback=lookback, horizons=horizons, train_split=0.9
        )
    
    # Create data loaders
    train_dataset = MultiHorizonDataset(X_train, y_train)
    val_dataset = MultiHorizonDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Build and train ensemble
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"\n🚀 Training ensemble ({device.upper()})...")
    
    ensemble = EnsembleForecaster(
        n_models=n_models,
        input_size=scaler_info['n_features'],
        hidden_size=hidden_size,
        num_blocks=num_blocks,
        horizons=horizons,
        device=device,
    )
    
    start_time = time.time()
    ensemble.fit(train_loader, val_loader, epochs=epochs, verbose=False)
    training_time = time.time() - start_time
    print(f"✓ Training completed in {training_time:.1f}s")
    
    # Now make predictions for each test day
    print(f"\n📈 Running backtest predictions...")
    
    results = {
        'symbol': symbol,
        'test_days': test_days,
        'use_market': use_market,
        'predictions': [],
        'metrics': {},
    }
    
    prices = train_df['close'].values.astype(np.float64)
    volumes = train_df['volume'].values.astype(np.float64) if 'volume' in train_df.columns else None
    
    # Get market data for feature computation
    if use_market:
        nepse_df, sector_df, sector_code = get_market_data(symbol)
    
    for i, (_, test_row) in enumerate(test_df.iterrows()):
        test_date = test_row['date']
        actual_price = test_row['close']
        
        # Use data up to this point for prediction
        current_df = pd.concat([train_df, test_df.iloc[:i]]) if i > 0 else train_df
        current_prices = current_df['close'].values.astype(np.float64)
        current_volumes = current_df['volume'].values.astype(np.float64) if 'volume' in current_df.columns else None
        
        # Compute features
        stock_indicators = compute_technical_indicators(current_prices, current_volumes)
        
        if use_market:
            # Merge market data
            df_with_dates = current_df.copy()
            df_with_dates['date'] = pd.to_datetime(df_with_dates['date'])
            
            if not nepse_df.empty:
                df_with_dates = df_with_dates.merge(nepse_df, on='date', how='left')
                df_with_dates['nepse_close'] = df_with_dates['nepse_close'].ffill().bfill()
            else:
                df_with_dates['nepse_close'] = current_prices
            
            if not sector_df.empty:
                df_with_dates = df_with_dates.merge(sector_df, on='date', how='left')
                df_with_dates['sector_close'] = df_with_dates['sector_close'].ffill().bfill()
            else:
                df_with_dates['sector_close'] = None
            
            nepse_closes = df_with_dates['nepse_close'].values.astype(np.float64)
            sector_closes = df_with_dates['sector_close'].values.astype(np.float64) if sector_code and df_with_dates['sector_close'].notna().any() else None
            
            market_features, aligned_len = compute_market_features(current_prices, nepse_closes, sector_closes)
            offset = len(current_prices) - aligned_len
            
            # Build feature matrix
            stock_feature_names = ['returns', 'momentum_5', 'momentum_20', 'price_to_sma', 
                                   'volatility', 'rsi', 'macd', 'bb_position', 'adx', 'roc']
            if current_volumes is not None:
                stock_feature_names.extend(['volume_ratio', 'vol_momentum', 'mfi', 'vwap_diff'])
            
            market_feature_names = ['nepse_returns', 'nepse_momentum', 'rolling_beta', 
                                    'relative_strength', 'market_volatility',
                                    'sector_returns', 'sector_momentum', 'sector_relative_strength']
            
            stock_features = [stock_indicators[f][offset:][-lookback:] for f in stock_feature_names]
            market_features_list = [market_features[f][-lookback:] for f in market_feature_names]
            
            latest_features = np.column_stack(stock_features + market_features_list)
        else:
            # Standard features only
            feature_names = ['returns', 'momentum_5', 'momentum_20', 'price_to_sma', 
                            'volatility', 'rsi', 'macd', 'bb_position', 'adx', 'roc']
            if current_volumes is not None:
                feature_names.extend(['volume_ratio', 'vol_momentum', 'mfi', 'vwap_diff'])
            
            latest_features = np.column_stack([stock_indicators[f][-lookback:] for f in feature_names])
        
        latest_features = np.nan_to_num(latest_features, nan=0.0, posinf=0.0, neginf=0.0)
        scaled_latest = scaler_info['feature_scaler'].transform(latest_features)
        latest_input = scaled_latest.reshape(1, lookback, -1)
        
        # Get predictions
        predictions = ensemble.predict_with_uncertainty(latest_input)
        
        base_price = current_prices[-1]
        
        pred_result = {
            'date': str(test_date),
            'actual_price': float(actual_price),
            'base_price': float(base_price),
            'horizons': {}
        }
        
        for horizon in horizons:
            pred_data = predictions[horizon]
            mean_return = scaler_info['target_scaler'].inverse_transform(
                pred_data['mean'].reshape(-1, 1)
            )[0, 0]
            
            pred_price = base_price * (1 + mean_return)
            pred_result['horizons'][horizon] = {
                'predicted': float(pred_price),
                'actual': float(actual_price) if horizon == 1 else None,
            }
        
        results['predictions'].append(pred_result)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{test_days} days...")
    
    # Calculate metrics
    print(f"\n📊 Calculating metrics...")
    
    for horizon in horizons:
        errors = []
        direction_correct = 0
        total_direction = 0
        
        for i, pred in enumerate(results['predictions']):
            if i + horizon <= len(results['predictions']) - 1:
                predicted = pred['horizons'][horizon]['predicted']
                actual_future = results['predictions'][i + horizon]['actual_price'] if i + horizon < len(results['predictions']) else None
                
                if actual_future:
                    error = abs(predicted - actual_future) / actual_future * 100
                    errors.append(error)
                    
                    # Direction accuracy
                    pred_direction = predicted > pred['base_price']
                    actual_direction = actual_future > pred['base_price']
                    if pred_direction == actual_direction:
                        direction_correct += 1
                    total_direction += 1
        
        if errors:
            results['metrics'][f'{horizon}-day'] = {
                'mape': float(np.mean(errors)),
                'max_error': float(np.max(errors)),
                'min_error': float(np.min(errors)),
                'direction_accuracy': float(direction_correct / total_direction * 100) if total_direction > 0 else 0,
            }
    
    # Print results
    print(f"\n{'='*70}")
    print(f"📊 BACKTEST RESULTS: {symbol}")
    print(f"{'='*70}")
    print(f"Model: {'Market-aware' if use_market else 'Standard'} xLSTM")
    print(f"Test period: {test_days} days")
    print(f"Training time: {training_time:.1f}s")
    print(f"\n{'Horizon':<12} {'MAPE':<10} {'Direction Acc':<15} {'Max Error':<12}")
    print(f"{'-'*50}")
    
    for horizon in horizons:
        if f'{horizon}-day' in results['metrics']:
            m = results['metrics'][f'{horizon}-day']
            print(f"{horizon}-day{'':<7} {m['mape']:.2f}%{'':<5} {m['direction_accuracy']:.1f}%{'':<10} {m['max_error']:.2f}%")
    
    print(f"{'='*70}\n")
    
    # Save results
    output_file = Path(config.DATA_DIR) / f"backtest_{symbol}_{'market' if use_market else 'standard'}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"✓ Results saved to {output_file}")
    
    return results


def compare_models(symbol: str, test_days: int = 30):
    """Compare market-aware vs standard model."""
    print(f"\n{'='*70}")
    print(f"🔬 MODEL COMPARISON: {symbol}")
    print(f"{'='*70}\n")
    
    # Run standard model
    print("Running STANDARD model (14 features)...")
    standard_results = backtest_model(symbol, test_days=test_days, use_market=False)
    
    print("\n" + "="*70 + "\n")
    
    # Run market-aware model
    print("Running MARKET-AWARE model (22 features)...")
    market_results = backtest_model(symbol, test_days=test_days, use_market=True)
    
    # Compare
    print(f"\n{'='*70}")
    print(f"📊 COMPARISON SUMMARY: {symbol}")
    print(f"{'='*70}")
    print(f"\n{'Horizon':<10} {'Standard MAPE':<15} {'Market MAPE':<15} {'Improvement':<12}")
    print(f"{'-'*55}")
    
    for horizon in [1, 3, 5, 10]:
        std_key = f'{horizon}-day'
        if std_key in standard_results['metrics'] and std_key in market_results['metrics']:
            std_mape = standard_results['metrics'][std_key]['mape']
            mkt_mape = market_results['metrics'][std_key]['mape']
            improvement = (std_mape - mkt_mape) / std_mape * 100
            
            better = "✓" if mkt_mape < std_mape else "✗"
            print(f"{horizon}-day{'':<5} {std_mape:.2f}%{'':<10} {mkt_mape:.2f}%{'':<10} {improvement:+.1f}% {better}")
    
    print(f"\n{'Horizon':<10} {'Std Direction':<15} {'Mkt Direction':<15}")
    print(f"{'-'*40}")
    
    for horizon in [1, 3, 5, 10]:
        std_key = f'{horizon}-day'
        if std_key in standard_results['metrics'] and std_key in market_results['metrics']:
            std_dir = standard_results['metrics'][std_key]['direction_accuracy']
            mkt_dir = market_results['metrics'][std_key]['direction_accuracy']
            print(f"{horizon}-day{'':<5} {std_dir:.1f}%{'':<10} {mkt_dir:.1f}%")
    
    print(f"{'='*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backtest xLSTM Stock Forecaster')
    parser.add_argument('symbol', help='Stock symbol to backtest')
    parser.add_argument('--days', type=int, default=30, help='Number of test days (default: 30)')
    parser.add_argument('--market', action='store_true', help='Use market-aware model')
    parser.add_argument('--compare', action='store_true', help='Compare market vs standard model')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_models(args.symbol, test_days=args.days)
    else:
        backtest_model(
            args.symbol,
            test_days=args.days,
            use_market=args.market,
            epochs=args.epochs,
        )
