"""
Export xLSTM Multi-Horizon Predictions to Website Format
=========================================================
This script runs inference on all trained multi-horizon xLSTM models
and exports the results to stock_predictions.json for website display.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
import glob

# Import the inference function from the multi-horizon module
from xlstm_multihorizon_bayesian import inference, HORIZONS, MODEL_DIR


def get_trained_symbols():
    """Get list of symbols that have trained multi-horizon models"""
    model_files = glob.glob(str(MODEL_DIR / "*_best_model.pt"))
    symbols = []
    for f in model_files:
        filename = os.path.basename(f)
        # Extract symbol from filename like "NABIL_best_model.pt"
        symbol = filename.replace("_best_model.pt", "")
        symbols.append(symbol)
    return symbols


def convert_to_website_format(inference_result: dict) -> dict:
    """
    Convert multi-horizon inference result to website-compatible format
    
    New format includes:
    - Multi-horizon predictions (t+1, t+3, t+5, t+10, t+15, t+20)
    - Accuracy metrics per horizon
    - Model info (xLSTM Multi-Horizon Ensemble)
    """
    if not inference_result:
        return None
    
    symbol = inference_result['symbol']
    predictions = inference_result['predictions']
    accuracy = inference_result['accuracy_metrics']
    model_info = inference_result['model_info']
    
    # Create predictions in array format (for website compatibility)
    pred_dates = []
    pred_prices = []
    pred_horizons = []
    pred_directions = []
    pred_confidences = []
    pred_returns = []
    
    # Count up/down days
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
    
    # Determine overall direction based on majority
    if up_days > down_days:
        overall_direction = 'BULLISH'
        recommendation = 'BUY' if up_days >= 4 else 'HOLD'
    elif down_days > up_days:
        overall_direction = 'BEARISH'
        recommendation = 'SELL' if down_days >= 4 else 'HOLD'
    else:
        overall_direction = 'NEUTRAL'
        recommendation = 'HOLD'
    
    # Calculate average confidence
    avg_confidence = sum(pred_confidences) / len(pred_confidences)
    
    # Get model performance rating
    avg_mape = accuracy['avg_mape']
    avg_dir_acc = accuracy['avg_direction_accuracy']
    
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
    
    # Create website-compatible format
    website_data = {
        'symbol': symbol,
        'predictions': {
            'dates': pred_dates,
            'prices': pred_prices,
            'horizons': pred_horizons,
            'directions': pred_directions,
            'confidences': pred_confidences,
            'returns': pred_returns,
        },
        'multi_horizon': True,  # Flag for website to use new display
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
            'model_path': f'models/{symbol}_best_model.pt',
            'architecture': 'xlstm_multihorizon',
            'type': model_info['type'],
            'lookback_days': model_info['lookback'],
            'layers': 2,  # Fixed in multi-horizon model
            'n_models': model_info['n_models'],
            'n_features': model_info['features'],
            'last_actual_price': inference_result['current_price'],
            'sector': inference_result.get('sector', 'Unknown'),
            'performance': {
                'test_mape': avg_mape,
                'avg_direction_accuracy': avg_dir_acc,
                'mape_rating': mape_rating,
                'signal_strength': signal_strength,
                'strength': accuracy['strength'],
                'by_horizon': accuracy['by_horizon'],
            }
        },
        'analysis_date': inference_result['generated_at'],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'last_updated': datetime.now().isoformat(),
    }
    
    return website_data


def export_all_to_website(output_file='stock_predictions.json', symbols=None):
    """
    Export all trained multi-horizon models to website format
    
    Args:
        output_file: Path to output JSON file
        symbols: Optional list of symbols to export. If None, exports all trained models.
    """
    # Load existing predictions (to preserve old LSTM predictions for stocks without multi-horizon)
    existing_data = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                existing_data = json.load(f)
        except:
            existing_data = {}
    
    stocks_data = existing_data.get('stocks', {})
    
    # Get list of symbols to process
    if symbols:
        target_symbols = symbols
    else:
        target_symbols = get_trained_symbols()
    
    if not target_symbols:
        print("⚠️  No trained multi-horizon models found!")
        print(f"   Run: python xlstm_multihorizon_bayesian.py SYMBOL --mode train")
        return
    
    print(f"\n{'='*70}")
    print(f"📤 EXPORTING xLSTM MULTI-HORIZON PREDICTIONS TO WEBSITE")
    print(f"{'='*70}")
    print(f"Models found: {len(target_symbols)}")
    print(f"Symbols: {', '.join(target_symbols)}")
    print(f"Output: {output_file}")
    print(f"{'='*70}\n")
    
    success_count = 0
    error_count = 0
    
    for symbol in target_symbols:
        print(f"\n🔄 Processing {symbol}...")
        
        try:
            # Run inference
            result = inference(symbol)
            
            if result:
                # Convert to website format
                website_data = convert_to_website_format(result)
                stocks_data[symbol] = website_data
                
                # Print summary
                accuracy = result['accuracy_metrics']
                print(f"   ✅ {symbol}: MAPE={accuracy['avg_mape']:.2f}%, "
                      f"Direction={accuracy['avg_direction_accuracy']:.1f}%, "
                      f"Strength={accuracy['strength']}")
                success_count += 1
            else:
                print(f"   ⚠️  {symbol}: Inference returned None")
                error_count += 1
                
        except Exception as e:
            print(f"   ❌ {symbol}: Error - {str(e)}")
            error_count += 1
    
    # Save to file
    output_data = {
        'stocks': stocks_data,
        'last_updated': datetime.now().isoformat(),
        'model_type': 'xlstm_multihorizon_bayesian',
        'horizons': HORIZONS,
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"📊 EXPORT SUMMARY")
    print(f"{'='*70}")
    print(f"  ✅ Success: {success_count} stocks")
    print(f"  ❌ Errors: {error_count} stocks")
    print(f"  📁 Output: {output_file}")
    print(f"{'='*70}\n")
    
    return output_data


def export_single(symbol: str, output_file='stock_predictions.json'):
    """Export a single symbol to website format"""
    return export_all_to_website(output_file=output_file, symbols=[symbol])


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Export xLSTM predictions to website')
    parser.add_argument('symbols', nargs='*', help='Symbols to export (default: all trained)')
    parser.add_argument('--output', '-o', default='stock_predictions.json', 
                       help='Output JSON file')
    
    args = parser.parse_args()
    
    if args.symbols:
        export_all_to_website(output_file=args.output, symbols=args.symbols)
    else:
        export_all_to_website(output_file=args.output)
