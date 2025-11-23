"""
Dynamic Stock Prediction System
- Performs grid search for any stock
- Finds best model based on MAPE and other metrics
- Trains final model and predicts next 7 days
- Saves results to JSON file for web display
"""
import sys
sys.path.insert(0, '/Users/Nayan/Documents/Business/Stock_Analysis')

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error
# Enable Metal GPU acceleration for M1 Mac
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

# Configure TensorFlow for M1 optimization
import tensorflow as tf
# Enable mixed precision for faster training on M1
tf.keras.mixed_precision.set_global_policy('mixed_float16')
# Enable XLA compilation for faster execution
tf.config.optimizer.set_jit(True)

from tune_hyperparameters import HyperparameterTuner
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
import config

class StockPredictor:
    def __init__(self, symbol, output_file='stock_predictions.json'):
        """
        Initialize predictor for a specific stock
        
        Args:
            symbol: Stock symbol to analyze
            output_file: JSON file to save predictions
        """
        self.symbol = symbol.upper()
        self.output_file = output_file
        self.db_path = config.DB_PATH
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
    def grid_search(self, architectures=['bidirectional'], 
                   lookback_periods=[60, 90, 120, 180, 240, 365],
                   verbose=True):
        """
        Perform comprehensive grid search across architectures and hyperparameters
        
        Returns:
            best_config: Dictionary with best configuration
            best_results: Dictionary with best results
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Starting Grid Search for {self.symbol}")
            print(f"{'='*60}")
        
        # Comprehensive grid search: 6 lookbacks × 2 layers × 2 units × 2 dropout × 2 LR × 2 batch = 384 configs
        TCN_CONFIGS = {
            'layers': [2, 3],              
            'units': [64, 128],
            'dropout': [0.2, 0.3],
            'learning_rate': [0.0005, 0.001],   
            'batch_size': [32, 64],         
            'patience': [10]
        }
        
        LSTM_CONFIGS = {
            'layers': [2, 3],              
            'units': [64, 128],
            'dropout': [0.2, 0.3],
            'learning_rate': [0.0005, 0.001],   
            'batch_size': [32, 64],         
            'patience': [10]
        }
        
        best_mape = float('inf')
        best_config = None
        best_results = None
        all_results = []
        total_config_count = 0  # Track total configs tested across all lookbacks
        
        for architecture in architectures:
            config_grid = TCN_CONFIGS if architecture == 'tcn' else LSTM_CONFIGS
            
            for lookback in lookback_periods:
                if verbose:
                    print(f"\n--- Testing {architecture.upper()} with lookback={lookback} ---")
                
                # Generate configurations
                configs = self._generate_configs(config_grid, architecture, lookback)
                
                # Create tuner once per lookback/architecture
                try:
                    tuner = HyperparameterTuner(
                        symbol=self.symbol,
                        test_days=7
                    )
                    
                    # Load data once
                    train_df, test_df = tuner.load_data()
                    
                    if verbose:
                        print(f"  Loaded {len(train_df)} training samples, {len(test_df)} test samples")
                    
                except Exception as e:
                    if verbose:
                        print(f"  ✗ Failed to load data: {str(e)}")
                    continue
                
                for i, config in enumerate(configs):
                    total_config_count += 1  # Increment total counter
                    try:
                        # Add architecture to config
                        test_config = config.copy()
                        test_config['epochs'] = 100  # Max epochs, early stopping will handle
                        
                        # Train and evaluate
                        results = tuner.train_and_evaluate(
                            config=test_config,
                            train_df=train_df,
                            test_df=test_df
                        )
                        
                        if results and 'test_mape' in results and 'error' not in results:
                            test_mape = results['test_mape']
                            
                            # Store result
                            result_entry = {
                                'symbol': self.symbol,
                                **config,
                                'test_mae': results.get('test_mae'),
                                'test_rmse': results.get('test_rmse'),
                                'test_mape': test_mape,
                                'val_mae': results.get('val_mae'),
                                'timestamp': datetime.now().isoformat()
                            }
                            all_results.append(result_entry)
                            
                            # Update best
                            if test_mape < best_mape:
                                best_mape = test_mape
                                best_config = config.copy()
                                best_results = results.copy()
                                
                                if verbose:
                                    print(f"  ✓ Config {total_config_count} (lookback {lookback}, {i+1}/{len(configs)}): New best! MAPE={test_mape:.2f}%, MAE={results.get('test_mae', 0):.2f}")
                                
                                # Early exit if MAPE < 1.0% (excellent performance)
                                if test_mape < 1.0:
                                    if verbose:
                                        print(f"\n  🎯 EXCELLENT PERFORMANCE ACHIEVED! MAPE={test_mape:.2f}% < 1.5%")
                                        print(f"  Stopping grid search early - further exploration unnecessary.")
                                    return best_config, best_results, all_results
                            else:
                                if verbose and (i+1) % 10 == 0:
                                    print(f"  · Config {total_config_count} (lookback {lookback}, {i+1}/{len(configs)}): MAPE={test_mape:.2f}%")
                        
                    except Exception as e:
                        if verbose:
                            print(f"  ✗ Config {i+1}/{len(configs)}: Error - {str(e)[:100]}")
                        continue
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Grid Search Complete!")
            print(f"Total configurations tested: {len(all_results)}")
            if best_config:
                print(f"Best Configuration: {best_config['architecture'].upper()} with lookback={best_config['lookback']}")
                print(f"Best MAPE: {best_mape:.2f}%")
                
                # Show summary by lookback period
                print(f"\n--- Performance by Lookback Period ---")
                lookback_summary = {}
                for result in all_results:
                    lb = result['lookback']
                    if lb not in lookback_summary:
                        lookback_summary[lb] = []
                    lookback_summary[lb].append(result['test_mape'])
                
                for lb in sorted(lookback_summary.keys()):
                    mapes = lookback_summary[lb]
                    print(f"  Lookback {lb}: Best MAPE = {min(mapes):.2f}%, Avg MAPE = {sum(mapes)/len(mapes):.2f}% ({len(mapes)} configs)")
            else:
                print(f"No valid configurations found!")
            print(f"{'='*60}\n")
        
        return best_config, best_results, all_results
    
    def _generate_configs(self, config_grid, architecture, lookback):
        """Generate all configuration combinations"""
        configs = []
        
        for layers in config_grid['layers']:
            for units in config_grid['units']:
                for dropout in config_grid['dropout']:
                    for lr in config_grid['learning_rate']:
                        for batch_size in config_grid['batch_size']:
                            for patience in config_grid['patience']:
                                config = {
                                    'architecture': architecture,
                                    'lookback': lookback,
                                    'layers': layers,
                                    'learning_rate': lr,
                                    'batch_size': batch_size,
                                    'patience': patience,
                                    'optimizer': 'adam'
                                }
                                
                                # Build layer-wise units and dropout
                                if layers == 2:
                                    config['units_1'] = units
                                    config['units_2'] = units // 2
                                    config['dropout_1'] = dropout
                                    config['dropout_2'] = dropout
                                elif layers == 3:
                                    config['units_1'] = units
                                    config['units_2'] = units // 2
                                    config['units_3'] = units // 4
                                    config['dropout_1'] = dropout
                                    config['dropout_2'] = dropout
                                    config['dropout_3'] = dropout / 2
                                
                                config['dense_units'] = 32
                                configs.append(config)
        
        return configs
    
    def predict_next_7_days(self, config, verbose=True):
        """
        Train final model with best config and predict next 7 days into the FUTURE
        
        Returns:
            predictions: List of predicted prices for next 7 days
            dates: List of dates for predictions
            model_info: Dictionary with model details
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training Final Model for True Future Prediction")
            print(f"{'='*60}")
        
        # Create tuner - first with test_days to get validation metrics
        tuner = HyperparameterTuner(
            symbol=self.symbol,
            test_days=7
        )
        
        # Load data for validation
        train_df, test_df = tuner.load_data()
        
        # Add epochs and optimizer to config
        final_config = config.copy()
        final_config['epochs'] = 100
        final_config['optimizer'] = 'adam'
        
        # First: Get validation metrics (this is what grid search found)
        if verbose:
            print("\n[1/2] Getting validation metrics on test set...")
        
        validation_results = tuner.train_and_evaluate(
            config=final_config,
            train_df=train_df,
            test_df=test_df
        )
        
        # Now: Train on ALL data for true future predictions
        if verbose:
            print("\n[2/2] Training on ALL data for future predictions...")
        
        # For final training, we need to load ALL data without any test split
        # But we still use test_days=7 for the tuner to work properly with create_sequences
        all_data_tuner = HyperparameterTuner(
            symbol=self.symbol,
            test_days=7  # Keep at 7 for create_sequences to work
        )
        
        # Load ALL data by combining train and test
        train_df, test_df = all_data_tuner.load_data()
        # Combine both to get ALL available data
        full_train_df = pd.concat([train_df, test_df], ignore_index=True)
        
        # Prepare features on full data
        data_with_features = all_data_tuner.prepare_features(full_train_df)
        
        # Select features
        feature_cols = ['close', 'volume', 'returns', 'sma_5', 'sma_20', 'rsi', 
                       'macd', 'bb_width', 'volume_ratio', 'atr', 'momentum_5']
        feature_cols = [col for col in feature_cols if col in data_with_features.columns]
        
        features = data_with_features[feature_cols].values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_features = scaler.fit_transform(features)
        
        # Create sequences for training (predicting 7 days ahead for sequence compatibility)
        # We create sequences manually to use ALL data
        lookback = final_config['lookback']
        X, y = [], []
        for i in range(lookback, len(scaled_features) - 6):  # -6 because we predict 7 days
            X.append(scaled_features[i-lookback:i])
            y.append(scaled_features[i:i+7, 0])  # Next 7 days of close prices
        
        X = np.array(X)
        y = np.array(y)
        
        # Build and train model on ALL data
        model = all_data_tuner.build_model(final_config, (final_config['lookback'], len(feature_cols)))
        
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        early_stop = EarlyStopping(monitor='loss', patience=final_config['patience'], restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.00001)
        
        if verbose:
            print(f"Training on {len(X)} samples with all available data...")
        
        history = model.fit(
            X, y,
            epochs=final_config['epochs'],
            batch_size=final_config['batch_size'],
            callbacks=[early_stop, reduce_lr],
            verbose=1 if verbose else 0
        )
        
        # Save the trained model
        model_path = self.models_dir / f"{self.symbol}_model.keras"
        model.save(model_path)
        if verbose:
            print(f"\n✓ Model saved to: {model_path}")
        
        # Now predict TRUE FUTURE (7 days beyond last available data)
        # Get last lookback sequence
        last_sequence = scaled_features[-final_config['lookback']:]
        
        # Predict next 7 days iteratively
        predictions_scaled = []
        current_sequence = last_sequence.copy()
        
        for day in range(7):
            # Reshape for prediction
            X_pred = current_sequence.reshape(1, final_config['lookback'], len(feature_cols))
            
            # Predict next day (this returns 7 values, we take the first)
            next_pred_scaled = model.predict(X_pred, verbose=0)[0, 0]
            predictions_scaled.append(next_pred_scaled)
            
            # Update sequence - simple approach: assume other features follow close price
            next_row = current_sequence[-1].copy()
            next_row[0] = next_pred_scaled  # Update close price
            
            # Slide window
            current_sequence = np.vstack([current_sequence[1:], next_row])
        
        # Inverse transform
        dummy = np.zeros((7, len(feature_cols)))
        dummy[:, 0] = predictions_scaled
        predictions = scaler.inverse_transform(dummy)[:, 0].tolist()
        
        # Calculate directional signal
        base_price = full_train_df['close'].iloc[-1]
        pred_directions = [1 if p > base_price else -1 for p in predictions]
        up_days = sum(1 for d in pred_directions if d == 1)
        down_days = 7 - up_days
        direction_confidence = (max(up_days, down_days) / 7) * 100
        overall_direction = "BULLISH" if up_days > down_days else "BEARISH" if down_days > up_days else "NEUTRAL"
        
        # Generate TRUE FUTURE dates (beyond last available data)
        last_date = pd.to_datetime(full_train_df['date'].iloc[-1])
        prediction_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') 
                           for i in range(7)]
        
        # Get recent actual prices for context (last 30 days)
        recent_prices = full_train_df['close'].tail(30).tolist()
        recent_dates = pd.to_datetime(full_train_df['date']).tail(30).dt.strftime('%Y-%m-%d').tolist()
        
        if verbose:
            print(f"\n✓ TRUE FUTURE Predictions (beyond all available data):")
            print(f"  Last known date: {last_date.strftime('%Y-%m-%d')}")
            print(f"  Last known price: Rs. {full_train_df['close'].iloc[-1]:.2f}")
            
            print(f"\n  📊 Model Performance:")
            mape = validation_results.get('test_mape', 0)
            dir_acc = validation_results.get('direction_accuracy', 0)
            print(f"    MAPE: {mape:.2f}% ({'Excellent' if mape < 2 else 'Good' if mape < 5 else 'Fair' if mape < 10 else 'Poor'})")
            print(f"    Direction Accuracy: {dir_acc:.1f}% ({'Strong' if dir_acc > 70 else 'Moderate' if dir_acc > 50 else 'Weak'})")
            
            print(f"\n  🎯 Trading Signal: {overall_direction}")
            print(f"    Confidence: {direction_confidence:.1f}% ({up_days} up, {down_days} down)")
            print(f"    Recommendation: {'BUY' if overall_direction == 'BULLISH' else 'SELL' if overall_direction == 'BEARISH' else 'HOLD'}")
            
            print(f"\n  📈 7-Day Forecast:")
            for date, price in zip(prediction_dates, predictions):
                change = price - full_train_df['close'].iloc[-1]
                change_pct = (change / full_train_df['close'].iloc[-1]) * 100
                direction = "↑" if price > full_train_df['close'].iloc[-1] else "↓"
                print(f"    {date}: Rs. {price:.2f} ({change_pct:+.2f}%) {direction}")
        
        model_info = {
            'model_path': str(model_path),
            'architecture': config['architecture'],
            'lookback': config['lookback'],
            'layers': config['layers'],
            'test_mape': validation_results.get('test_mape', 0),
            'test_mae': validation_results.get('test_mae', 0),
            'test_rmse': validation_results.get('test_rmse', 0),
            'direction_accuracy': validation_results.get('direction_accuracy', 0),
            'training_samples': len(full_train_df),
            'last_actual_price': float(full_train_df['close'].iloc[-1]),
            'overall_direction': overall_direction,
            'direction_confidence': direction_confidence,
            'up_days': up_days,
            'down_days': down_days
        }
        
        return predictions, prediction_dates, recent_prices, recent_dates, model_info
    
    def save_to_json(self, predictions, dates, recent_prices, recent_dates, 
                     model_info, best_config, all_results=None):
        """
        Save predictions to JSON file
        Overwrites if same symbol exists, otherwise appends
        """
        # Load existing data
        if os.path.exists(self.output_file):
            with open(self.output_file, 'r') as f:
                data = json.load(f)
        else:
            data = {'stocks': {}, 'last_updated': None}
        
        # Prepare stock data
        stock_data = {
            'symbol': self.symbol,
            'predictions': {
                'dates': dates,
                'prices': [round(p, 2) for p in predictions]
            },
            'recent_actual': {
                'dates': recent_dates,
                'prices': [round(p, 2) for p in recent_prices]
            },
            'trading_signal': {
                'direction': model_info.get('overall_direction', 'NEUTRAL'),
                'confidence': round(model_info.get('direction_confidence', 50), 1),
                'up_days': model_info.get('up_days', 0),
                'down_days': model_info.get('down_days', 0),
                'recommendation': 'BUY' if model_info.get('overall_direction') == 'BULLISH' else 'SELL' if model_info.get('overall_direction') == 'BEARISH' else 'HOLD'
            },
            'model': {
                'architecture': model_info['architecture'],
                'lookback_days': model_info['lookback'],
                'layers': model_info['layers'],
                'performance': {
                    'mape': round(model_info['test_mape'], 2),
                    'mape_rating': 'Excellent' if model_info['test_mape'] < 2 else 'Good' if model_info['test_mape'] < 5 else 'Fair' if model_info['test_mape'] < 10 else 'Poor',
                    'mae': round(model_info['test_mae'], 2),
                    'rmse': round(model_info['test_rmse'], 2),
                    'direction_accuracy': round(model_info['direction_accuracy'], 1),
                    'signal_strength': 'Strong' if model_info['direction_accuracy'] > 70 else 'Moderate' if model_info['direction_accuracy'] > 50 else 'Weak'
                },
                'training_samples': model_info['training_samples'],
                'last_actual_price': round(model_info['last_actual_price'], 2)
            },
            'config': best_config,
            'analysis_date': datetime.now().isoformat(),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Add grid search summary if available
        if all_results:
            maes = [r['test_mae'] for r in all_results if 'test_mae' in r]
            mapes = [r['test_mape'] for r in all_results if 'test_mape' in r]
            stock_data['grid_search'] = {
                'configs_tested': len(all_results),
                'best_mape': round(min(mapes), 2) if mapes else None,
                'avg_mape': round(sum(mapes) / len(mapes), 2) if mapes else None,
                'best_mae': round(min(maes), 2) if maes else None,
                'avg_mae': round(sum(maes) / len(maes), 2) if maes else None
            }
        
        # Update or add stock data
        data['stocks'][self.symbol] = stock_data
        data['last_updated'] = datetime.now().isoformat()
        
        # Save to file
        with open(self.output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n✓ Predictions saved to {self.output_file}")
        print(f"  Symbol: {self.symbol}")
        print(f"  Total stocks in file: {len(data['stocks'])}")
    
    def run_full_analysis(self, architectures=['bidirectional'],
                         lookback_periods=[60,90, 120, 180, 240, 365],
                         verbose=True):
        """
        Run grid search to find best configuration and save it
        
        This does NOT train final model or make predictions.
        Use predict_single_config.py with saved config for that.
        """
        try:
            # Grid search to find best config
            best_config, best_results, all_results = self.grid_search(
                architectures=architectures,
                lookback_periods=lookback_periods,
                verbose=verbose
            )
            
            if not best_config:
                print(f"✗ No valid configuration found for {self.symbol}")
                return False
            
            # Save best config to file
            config_file = f"best_configs/{self.symbol}_best_config.json"
            os.makedirs("best_configs", exist_ok=True)
            
            config_data = {
                'symbol': self.symbol,
                'best_config': best_config,
                'performance': {
                    'test_mape': best_results.get('test_mape', 0),
                    'test_mae': best_results.get('test_mae', 0),
                    'test_rmse': best_results.get('test_rmse', 0),
                    'direction_accuracy': best_results.get('direction_accuracy', 0)
                },
                'timestamp': datetime.now().isoformat()
            }
            
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            if verbose:
                print(f"\n{'='*60}")
                print(f"✓ Best config saved to: {config_file}")
                print(f"{'='*60}")
                print(f"  Architecture: {best_config['architecture']}")
                print(f"  Lookback: {best_config['lookback']}")
                print(f"  Test MAPE: {best_results.get('test_mape', 0):.2f}%")
                print(f"  Direction Accuracy: {best_results.get('direction_accuracy', 0):.1f}%")
                print(f"\nTo make predictions with this config, run:")
                print(f"  python3 predict_single_config.py {self.symbol}")
            
            return True
            
        except Exception as e:
            print(f"\n✗ Error analyzing {self.symbol}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


def analyze_stock(symbol, output_file='stock_predictions.json', 
                 architectures=['bidirectional'],
                 lookback_periods=[60,90, 120, 180, 240, 365],
                 verbose=True):
    """
    Convenience function to analyze a single stock
    
    Args:
        symbol: Stock symbol to analyze
        output_file: JSON file to save results
        architectures: List of architectures to test
        lookback_periods: List of lookback periods to test
        verbose: Print progress
    """
    predictor = StockPredictor(symbol, output_file)
    return predictor.run_full_analysis(
        architectures=architectures,
        lookback_periods=lookback_periods,
        verbose=verbose
    )


def analyze_multiple_stocks(symbols, output_file='stock_predictions.json',
                           architectures=['bidirectional'],
                           lookback_periods=[60, 90, 120, 180, 240, 365],
                           verbose=True):
    """
    Analyze multiple stocks in sequence
    
    Args:
        symbols: List of stock symbols
        output_file: JSON file to save results
        architectures: List of architectures to test
        lookback_periods: List of lookback periods to test
        verbose: Print progress
    """
    results = {}
    
    for symbol in symbols:
        print(f"\n{'#'*60}")
        print(f"# Analyzing {symbol}")
        print(f"{'#'*60}")
        
        success = analyze_stock(
            symbol, output_file, architectures, 
            lookback_periods, verbose
        )
        results[symbol] = success
    
    # Summary
    print(f"\n{'='*60}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*60}")
    for symbol, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"{symbol:10} {status}")
    print(f"{'='*60}\n")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze stock(s) and predict next 7 days')
    parser.add_argument('symbols', nargs='+', help='Stock symbol(s) to analyze')
    parser.add_argument('--output', default='stock_predictions.json', 
                       help='Output JSON file (default: stock_predictions.json)')
    parser.add_argument('--architectures', nargs='+', 
                       default=['bidirectional'],
                       help='Architectures to test (default: bidirectional)')
    parser.add_argument('--lookback', nargs='+', type=int,
                       default=[60, 90, 120, 180, 240, 365],
                       help='Lookback periods to test (default: 60 90 120 180 240 365)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress output')
    
    args = parser.parse_args()
    
    if len(args.symbols) == 1:
        analyze_stock(
            args.symbols[0], 
            args.output, 
            args.architectures,
            args.lookback,
            not args.quiet
        )
    else:
        analyze_multiple_stocks(
            args.symbols,
            args.output,
            args.architectures,
            args.lookback,
            not args.quiet
        )
