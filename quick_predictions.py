"""
Quick ML Price Predictions for All Stocks
Uses saved MULTI-OUTPUT LSTM models to predict 1-7 day ahead prices
Predicts all 7 days simultaneously (no error accumulation!)
Ranks stocks by profitability from highest to lowest
"""
import os
import json
import glob
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from ml_predictor import MLStockPredictor
from data_fetcher import NepseDataFetcher


def predict_all_stocks():
    """
    Load all saved ML models and generate 1-7 day predictions using multi-output LSTM
    Predicts all 7 days simultaneously in ONE forward pass (no error accumulation)
    Rank by profitability
    """
    print("\n" + "="*80)
    print("ML PRICE PREDICTIONS FOR ALL STOCKS (1-7 DAYS AHEAD)")
    print("Multi-Output LSTM - Predicts All Days Simultaneously")
    print("="*80)
    
    # Find all saved models
    model_files = glob.glob('models/lstm_*.h5')
    
    if not model_files:
        print("❌ No saved ML models found in models/ directory")
        return
    
    print(f"Found {len(model_files)} saved ML models\n")
    
    # Initialize
    ml_predictor = MLStockPredictor()
    data_fetcher = NepseDataFetcher()
    
    predictions_data = []
    
    # Process each model
    for i, model_file in enumerate(model_files, 1):
        # Extract symbol from filename
        symbol = os.path.basename(model_file).replace('lstm_', '').replace('.h5', '')
        
        print(f"[{i}/{len(model_files)}] {symbol}...", end=' ')
        
        try:
            # Load the model
            if not ml_predictor.load_model(symbol):
                print("❌ Failed to load model")
                continue
            
            # Get price data
            price_data = data_fetcher.get_stock_price_history(symbol, days=None)
            
            if price_data.empty or len(price_data) < 60:
                print("❌ Insufficient data")
                continue
            
            # Get current price
            current_price = float(price_data['close'].iloc[-1])
            
            # Initialize predictions dictionary
            predictions = {}
            
            # Predict all 7 days at once using multi-output LSTM
            try:
                pred = ml_predictor.predict_future_prices(price_data, days=[1, 2, 3, 4, 5, 6, 7])
                
                if pred and 'days' in pred:
                    # Extract predictions for each day
                    for day in range(1, 8):
                        day_key = f'day_{day}'
                        if day_key in pred['days']:
                            predictions[day_key] = pred['days'][day_key]['predicted_price']
                        else:
                            predictions[day_key] = current_price
                else:
                    # Fallback: use current price
                    for day in range(1, 8):
                        predictions[f'day_{day}'] = current_price
                    
            except Exception as e:
                # Fallback: use current price for all days
                for day in range(1, 8):
                    predictions[f'day_{day}'] = current_price
            
            # Calculate returns for each day
            returns = {}
            for day, price in predictions.items():
                return_pct = ((price - current_price) / current_price) * 100
                returns[day] = return_pct
            
            # Calculate average return across all days
            avg_return = np.mean(list(returns.values()))
            
            # Store results (now with 7 days)
            predictions_data.append({
                'symbol': symbol,
                'current_price': current_price,
                'day_1_price': predictions.get('day_1', current_price),
                'day_1_return': returns.get('day_1', 0),
                'day_2_price': predictions.get('day_2', current_price),
                'day_2_return': returns.get('day_2', 0),
                'day_3_price': predictions.get('day_3', current_price),
                'day_3_return': returns.get('day_3', 0),
                'day_4_price': predictions.get('day_4', current_price),
                'day_4_return': returns.get('day_4', 0),
                'day_5_price': predictions.get('day_5', current_price),
                'day_5_return': returns.get('day_5', 0),
                'day_6_price': predictions.get('day_6', current_price),
                'day_6_return': returns.get('day_6', 0),
                'day_7_price': predictions.get('day_7', current_price),
                'day_7_return': returns.get('day_7', 0),
                'avg_return': avg_return,
                'total_7day_return': returns.get('day_7', 0)
            })
            
            print(f"✓ Avg return: {avg_return:+.2f}%")
            
        except Exception as e:
            print(f"❌ Error: {str(e)[:40]}")
            continue
    
    # Sort by average return (most profitable first)
    predictions_data.sort(key=lambda x: x['avg_return'], reverse=True)
    
    # Display results
    print("\n" + "="*100)
    print("PREDICTIONS RANKED BY PROFITABILITY (HIGHEST TO LOWEST) - 7 DAY FORECAST")
    print("="*100)
    print(f"{'Rank':<5} {'Symbol':<10} {'Current':<10} {'1D %':<8} {'2D %':<8} {'3D %':<8} {'4D %':<8} {'5D %':<8} {'6D %':<8} {'7D %':<8} {'Avg %':<8}")
    print("-"*100)
    
    for rank, stock in enumerate(predictions_data, 1):
        print(f"{rank:<5} {stock['symbol']:<10} NPR {stock['current_price']:<7.2f} "
              f"{stock['day_1_return']:+6.2f}% "
              f"{stock['day_2_return']:+6.2f}% "
              f"{stock['day_3_return']:+6.2f}% "
              f"{stock['day_4_return']:+6.2f}% "
              f"{stock['day_5_return']:+6.2f}% "
              f"{stock['day_6_return']:+6.2f}% "
              f"{stock['day_7_return']:+6.2f}% "
              f"{stock['avg_return']:+6.2f}%")
    
    # Top 10 Most Profitable
    print("\n" + "="*80)
    print("TOP 10 MOST PROFITABLE STOCKS (Next 7 Days)")
    print("="*80)
    
    for rank, stock in enumerate(predictions_data[:10], 1):
        print(f"\n{rank}. {stock['symbol']} - Average Return: {stock['avg_return']:+.2f}%")
        print(f"   Current Price: NPR {stock['current_price']:.2f}")
        print(f"   Day 1: NPR {stock['day_1_price']:.2f} ({stock['day_1_return']:+.2f}%)")
        print(f"   Day 2: NPR {stock['day_2_price']:.2f} ({stock['day_2_return']:+.2f}%)")
        print(f"   Day 3: NPR {stock['day_3_price']:.2f} ({stock['day_3_return']:+.2f}%)")
        print(f"   Day 4: NPR {stock['day_4_price']:.2f} ({stock['day_4_return']:+.2f}%)")
        print(f"   Day 5: NPR {stock['day_5_price']:.2f} ({stock['day_5_return']:+.2f}%)")
        print(f"   Day 6: NPR {stock['day_6_price']:.2f} ({stock['day_6_return']:+.2f}%)")
        print(f"   Day 7: NPR {stock['day_7_price']:.2f} ({stock['day_7_return']:+.2f}%)")
    
    # Bottom 10 Least Profitable
    print("\n" + "="*80)
    print("BOTTOM 10 LEAST PROFITABLE STOCKS (Next 7 Days)")
    print("="*80)
    
    for rank, stock in enumerate(predictions_data[-10:], 1):
        print(f"\n{rank}. {stock['symbol']} - Average Return: {stock['avg_return']:+.2f}%")
        print(f"   Current Price: NPR {stock['current_price']:.2f}")
        print(f"   Day 7 Predicted: NPR {stock['day_7_price']:.2f} ({stock['day_7_return']:+.2f}%)")
    
    # Save to JSON
    output_file = 'ml_predictions_ranked.json'
    with open(output_file, 'w') as f:
        json.dump(predictions_data, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✅ Predictions saved to: {output_file}")
    print(f"Total stocks analyzed: {len(predictions_data)}")
    print(f"{'='*80}\n")
    
    return predictions_data


def predict_quick(symbol):
    """
    Quick 7-day prediction for a single stock using multi-output LSTM
    
    Args:
        symbol: Stock symbol
    """
    ml_predictor = MLStockPredictor()
    data_fetcher = NepseDataFetcher()
    
    print(f"\n{'='*60}")
    print(f"QUICK PREDICTION FOR {symbol}")
    print(f"{'='*60}")
    
    # Load model
    if not ml_predictor.load_model(symbol):
        print(f"❌ No saved model found for {symbol}")
        return
    
    # Get data
    price_data = data_fetcher.get_stock_price_history(symbol, days=None)
    
    if price_data.empty:
        print(f"❌ No price data found for {symbol}")
        return
    
    current_price = float(price_data['close'].iloc[-1])
    
    print(f"\nCurrent Price: NPR {current_price:.2f}")
    print(f"Last Updated: {price_data['date'].iloc[-1]}")
    print(f"\nPredictions (7-Day Multi-Output LSTM):")
    print("-"*60)
    
    # Predict all 7 days at once
    try:
        pred = ml_predictor.predict_future_prices(price_data, days=[1, 2, 3, 4, 5, 6, 7])
        
        if pred and 'days' in pred:
            for day_num in range(1, 8):
                day_key = f'day_{day_num}'
                if day_key in pred['days']:
                    predicted_price = pred['days'][day_key]['predicted_price']
                    return_pct = pred['days'][day_key]['price_change_pct']
                    print(f"Day {day_num}: NPR {predicted_price:7.2f} ({return_pct:+6.2f}%)")
                else:
                    print(f"Day {day_num}: NPR {current_price:7.2f} (+0.00%)")
        else:
            print("❌ Could not generate predictions")
            
    except Exception as e:
        print(f"❌ Error generating predictions: {str(e)}")
    
    print(f"{'='*60}\n")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        # Single stock prediction
        symbol = sys.argv[1].upper()
        predict_quick(symbol)
    else:
        # All stocks prediction
        predict_all_stocks()
