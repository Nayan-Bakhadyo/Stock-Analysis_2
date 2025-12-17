"""
Train and Predict: xLSTM Multi-Horizon Model
=============================================
This script:
1. Reads priority stocks from priority_stocks/priority_list.txt
2. Trains xLSTM multi-horizon model for each stock
3. Runs inference and saves predictions to xlstm_predictions.json

Run this after update_price_data.py, update_fundamental.py, update_sentiment.py
"""

import json
import os
import sys
import gc
from pathlib import Path
from datetime import datetime

import torch

# Import from xlstm module
from xlstm_multihorizon_bayesian import (
    quick_train, inference, bayesian_optimization,
    HORIZONS, MODEL_DIR, validate_stock
)


PRIORITY_FILE = Path("priority_stocks/priority_list.txt")
PREDICTIONS_FILE = Path("xlstm_predictions.json")


def load_priority_stocks():
    """Load list of priority stocks from file"""
    if not PRIORITY_FILE.exists():
        print(f"⚠️  Priority file not found: {PRIORITY_FILE}")
        return []
    
    stocks = []
    with open(PRIORITY_FILE, 'r') as f:
        for line in f:
            symbol = line.strip()
            if symbol and not symbol.startswith('#'):
                stocks.append(symbol)
    return stocks


def convert_inference_to_json(result: dict) -> dict:
    """Convert inference result to JSON-serializable format for website"""
    if not result:
        return None
    
    predictions = result['predictions']
    accuracy = result['accuracy_metrics']
    model_info = result['model_info']
    
    # Build arrays for each horizon
    pred_dates = []
    pred_prices = []
    pred_horizons = []
    pred_directions = []
    pred_confidences = []
    pred_returns = []
    
    up_days = 0
    down_days = 0
    flat_days = 0
    
    for h in HORIZONS:
        pred = predictions[h]
        pred_dates.append(pred['target_date'])
        pred_prices.append(pred['predicted_price'])
        pred_horizons.append(h)
        pred_directions.append(pred['direction'])
        pred_confidences.append(pred['confidence'])
        pred_returns.append(pred['predicted_return'])
        
        if pred['direction'] == 'UP':
            up_days += 1
        elif pred['direction'] == 'DOWN':
            down_days += 1
        else:
            flat_days += 1
    
    # Overall signal
    if up_days > down_days:
        overall_direction = 'BULLISH'
        recommendation = 'BUY' if up_days >= 4 else 'HOLD'
    elif down_days > up_days:
        overall_direction = 'BEARISH'
        recommendation = 'SELL' if down_days >= 4 else 'HOLD'
    else:
        overall_direction = 'NEUTRAL'
        recommendation = 'HOLD'
    
    avg_confidence = sum(pred_confidences) / len(pred_confidences)
    avg_mape = accuracy['avg_mape']
    avg_dir_acc = accuracy['avg_direction_accuracy']
    
    # Ratings
    if avg_mape < 2:
        mape_rating = 'Excellent'
    elif avg_mape < 5:
        mape_rating = 'Good'
    elif avg_mape < 10:
        mape_rating = 'Fair'
    else:
        mape_rating = 'Poor'
    
    if avg_dir_acc >= 70:
        signal_strength = 'Strong'
    elif avg_dir_acc >= 55:
        signal_strength = 'Moderate'
    else:
        signal_strength = 'Weak'
    
    return {
        'symbol': result['symbol'],
        'current_price': result['current_price'],
        'current_date': result['current_date'],
        'sector': result.get('sector', 'Unknown'),
        'predictions': {
            'dates': pred_dates,
            'prices': pred_prices,
            'horizons': pred_horizons,
            'directions': pred_directions,
            'confidences': pred_confidences,
            'returns': pred_returns,
        },
        'multi_horizon': True,
        'horizon_details': {
            str(h): {
                'date': predictions[h]['target_date'],
                'price': predictions[h]['predicted_price'],
                'return': predictions[h]['predicted_return'],
                'direction': predictions[h]['direction'],
                'confidence': predictions[h]['confidence'],
                'mape': predictions[h]['mape'],
                'direction_accuracy': predictions[h]['direction_accuracy'],
            } for h in HORIZONS
        },
        'trading_signal': {
            'direction': overall_direction,
            'confidence': round(avg_confidence, 1),
            'up_days': up_days,
            'down_days': down_days,
            'flat_days': flat_days,
            'recommendation': recommendation,
        },
        'model': {
            'architecture': 'xlstm_multihorizon',
            'type': model_info['type'],
            'lookback_days': model_info['lookback'],
            'n_models': model_info['n_models'],
            'n_features': model_info['features'],
            'last_actual_price': result['current_price'],
            'sector': result.get('sector', 'Unknown'),
            'performance': {
                'test_mape': avg_mape,
                'avg_direction_accuracy': avg_dir_acc,
                'mape_rating': mape_rating,
                'signal_strength': signal_strength,
                'strength': accuracy['strength'],
                'by_horizon': accuracy['by_horizon'],
            }
        },
        'last_updated': result['generated_at'],
    }


def train_and_predict(symbols: list, mode: str = 'train', 
                      n_models: int = 5, epochs: int = 100,
                      hidden_size: int = 128, lookback: int = 60,
                      max_trials: int = 216):
    """
    Train xLSTM models and run inference for all symbols
    
    Args:
        symbols: List of stock symbols
        mode: 'train' for quick training, 'optimize' for Bayesian optimization
        n_models: Number of ensemble models
        epochs: Training epochs
        hidden_size: LSTM hidden size
        lookback: Lookback window
        max_trials: Max Bayesian optimization trials
    """
    
    print(f"\n{'='*70}")
    print(f"🚀 xLSTM MULTI-HORIZON TRAINING & PREDICTION")
    print(f"{'='*70}")
    print(f"Stocks: {', '.join(symbols)}")
    print(f"Mode: {mode}")
    print(f"Horizons: {HORIZONS}")
    print(f"{'='*70}\n")
    
    # Load existing predictions
    all_predictions = {}
    if PREDICTIONS_FILE.exists():
        try:
            with open(PREDICTIONS_FILE, 'r') as f:
                existing = json.load(f)
                all_predictions = existing.get('stocks', {})
        except:
            pass
    
    success_count = 0
    error_count = 0
    
    for i, symbol in enumerate(symbols, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(symbols)}] Processing {symbol}")
        print(f"{'='*70}")
        
        try:
            # Validate stock has sector info
            print(f"📋 Validating {symbol}...")
            sector_info = validate_stock(symbol)
            print(f"   ✅ Sector: {sector_info['sector']} ({sector_info['sector_index_code']})")
            
            # Train model
            if mode == 'optimize':
                print(f"\n🔬 Running Bayesian Optimization...")
                all_results, best = bayesian_optimization(
                    symbol=symbol,
                    max_trials=max_trials,
                    n_models=n_models
                )
            else:
                print(f"\n🏋️ Training model...")
                quick_train(
                    symbol=symbol,
                    n_models=n_models,
                    epochs=epochs,
                    hidden_size=hidden_size,
                    lookback=lookback
                )
            
            # Run inference
            print(f"\n🔮 Running inference...")
            result = inference(symbol)
            
            if result:
                # Convert to JSON format
                pred_json = convert_inference_to_json(result)
                all_predictions[symbol] = pred_json
                
                # Print summary
                print(f"\n✅ {symbol} Complete:")
                print(f"   MAPE: {pred_json['model']['performance']['test_mape']:.2f}%")
                print(f"   Direction Accuracy: {pred_json['model']['performance']['avg_direction_accuracy']:.1f}%")
                print(f"   Signal: {pred_json['trading_signal']['direction']} ({pred_json['trading_signal']['recommendation']})")
                
                # Print predictions
                print(f"\n   Predictions:")
                for h in HORIZONS:
                    p = result['predictions'][h]
                    arrow = "📈" if p['direction'] == 'UP' else "📉" if p['direction'] == 'DOWN' else "➡️"
                    print(f"     t+{h:2d}: NPR {p['predicted_price']:.2f} {arrow} {p['direction']} ({p['confidence']:.1f}%)")
                
                success_count += 1
                
                # Save after each stock (in case of crash)
                _save_predictions(all_predictions)
                
                # Generate website after each stock
                _generate_website()
                
            else:
                print(f"⚠️  Inference failed for {symbol}")
                error_count += 1
                
        except Exception as e:
            print(f"❌ Error processing {symbol}: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1
        
        finally:
            # Force memory cleanup after each stock
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            print(f"🧹 Memory cleared after {symbol}")
    
    # Final save
    _save_predictions(all_predictions)
    
    print(f"\n{'='*70}")
    print(f"📊 TRAINING & PREDICTION COMPLETE")
    print(f"{'='*70}")
    print(f"  ✅ Success: {success_count} stocks")
    print(f"  ❌ Errors: {error_count} stocks")
    print(f"  📁 Predictions saved to: {PREDICTIONS_FILE}")
    print(f"{'='*70}\n")
    
    return all_predictions


def _save_predictions(predictions: dict):
    """Save predictions to JSON file"""
    output = {
        'stocks': predictions,
        'last_updated': datetime.now().isoformat(),
        'model_type': 'xlstm_multihorizon',
        'horizons': HORIZONS,
    }
    with open(PREDICTIONS_FILE, 'w') as f:
        json.dump(output, f, indent=2)


def _generate_website():
    """Generate website after each stock analysis"""
    import subprocess
    print(f"\n🌐 Generating website...")
    try:
        result = subprocess.run(
            [sys.executable, 'generate_website.py'],
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )
        if result.returncode == 0:
            print(f"   ✅ Website updated: stock_website/index.html")
        else:
            print(f"   ⚠️ Website generation failed: {result.stderr[:100]}")
    except subprocess.TimeoutExpired:
        print(f"   ⚠️ Website generation timed out")
    except Exception as e:
        print(f"   ⚠️ Website generation error: {e}")


def predict_only(symbols: list = None, all_models: bool = True):
    """
    Run inference only (no training) for symbols that have trained models
    
    Args:
        symbols: List of symbols to predict. If None, predict all trained models.
        all_models: If True, predict all stocks with saved models (default)
    """
    import glob
    
    # Get trained models
    model_files = glob.glob(str(MODEL_DIR / "*_best_model.pt"))
    trained_symbols = [os.path.basename(f).replace("_best_model.pt", "") for f in model_files]
    
    print(f"\n📁 Found {len(trained_symbols)} trained models: {', '.join(trained_symbols)}")
    
    if symbols and not all_models:
        # Filter to only requested symbols that have models
        target_symbols = [s for s in symbols if s in trained_symbols]
        missing = [s for s in symbols if s not in trained_symbols]
        if missing:
            print(f"⚠️  No trained models for: {', '.join(missing)}")
    else:
        # Use all trained models
        target_symbols = trained_symbols
    
    if not target_symbols:
        print("⚠️  No trained models found! Run optimization first.")
        print("   Use: ./run_optimization.sh")
        return {}
    
    print(f"\n{'='*70}")
    print(f"🔮 xLSTM INFERENCE ONLY")
    print(f"{'='*70}")
    print(f"Stocks: {', '.join(target_symbols)}")
    print(f"{'='*70}\n")
    
    # Load existing predictions
    all_predictions = {}
    if PREDICTIONS_FILE.exists():
        try:
            with open(PREDICTIONS_FILE, 'r') as f:
                existing = json.load(f)
                all_predictions = existing.get('stocks', {})
        except:
            pass
    
    for symbol in target_symbols:
        print(f"🔮 {symbol}...", end=" ")
        try:
            result = inference(symbol)
            if result:
                pred_json = convert_inference_to_json(result)
                all_predictions[symbol] = pred_json
                print(f"✅ {pred_json['trading_signal']['direction']} | "
                      f"MAPE={pred_json['model']['performance']['test_mape']:.2f}%")
            else:
                print("⚠️ Failed")
        except Exception as e:
            print(f"❌ {e}")
    
    _save_predictions(all_predictions)
    print(f"\n✅ Predictions saved to {PREDICTIONS_FILE}")
    
    return all_predictions


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train and Predict xLSTM Multi-Horizon Models')
    parser.add_argument('--mode', choices=['train', 'optimize', 'predict', 'inference'], default='train',
                       help='Mode: train (quick), optimize (bayesian), predict/inference (inference only)')
    parser.add_argument('--symbols', nargs='*', help='Specific symbols (default: priority list)')
    parser.add_argument('--models', type=int, default=5, help='Number of ensemble models')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--hidden', type=int, default=128, help='Hidden size')
    parser.add_argument('--lookback', type=int, default=60, help='Lookback period')
    parser.add_argument('--max-trials', type=int, default=216, help='Max optimization trials (216 = all combinations)')
    
    args = parser.parse_args()
    
    # Get symbols
    if args.symbols:
        symbols = args.symbols
    else:
        symbols = load_priority_stocks()
        if not symbols:
            print("❌ No symbols provided and priority list is empty!")
            print(f"   Add symbols to {PRIORITY_FILE} or use --symbols")
            sys.exit(1)
    
    print(f"📋 Symbols to process: {symbols}")
    
    if args.mode in ['predict', 'inference']:
        predict_only(symbols)
    else:
        train_and_predict(
            symbols=symbols,
            mode=args.mode,
            n_models=args.models,
            epochs=args.epochs,
            hidden_size=args.hidden,
            lookback=args.lookback,
            max_trials=args.max_trials
        )
