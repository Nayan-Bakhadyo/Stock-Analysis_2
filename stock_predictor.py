"""
Dynamic Stock Prediction System
- Performs grid search for any stock
- Finds best model based on MAPE and other metrics
- Trains final model and predicts next 7 days
- Saves results to JSON file for web display
- Uses subprocess isolation to guarantee memory cleanup
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
# Mixed precision is set in subprocess workers (config_worker.py) to avoid warnings
# tf.keras.mixed_precision.set_global_policy('mixed_float16')
# XLA compilation disabled due to Metal compatibility issues
# tf.config.optimizer.set_jit(True)

from tune_hyperparameters import HyperparameterTuner
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
import config
import subprocess
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

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
        
    def _run_config_in_subprocess(self, config, lookback_period):
        """
        Run a single config in an isolated subprocess.
        When subprocess exits, OS automatically reclaims ALL memory.
        
        Returns:
            dict: Results from the config, or None if failed
        """
        try:
            # Get Python executable from conda environment
            python_path = shutil.which('python3') or shutil.which('python')
            
            # Prepare arguments
            config_json = json.dumps(config)
            args = [
                python_path,
                'config_worker.py',
                self.symbol,
                config_json,
                str(lookback_period)
            ]
            
            # Run subprocess
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout per config
                cwd=os.getcwd(),
                env={**os.environ, 'TF_CPP_MIN_LOG_LEVEL': '3'}  # Suppress TF warnings in subprocess
            )
            
            if result.returncode == 0:
                # Parse JSON output
                results = json.loads(result.stdout)
                if results.get('success'):
                    return results
                else:
                    print(f"    ✗ Config failed: {results.get('error', 'Unknown error')}")
                    return None
            else:
                print(f"    ✗ Subprocess failed with code {result.returncode}")
                if result.stderr:
                    print(f"    Error: {result.stderr[:200]}")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"    ✗ Config timed out after 10 minutes")
            return None
        except Exception as e:
            print(f"    ✗ Subprocess error: {str(e)[:100]}")
            return None
    
    def _run_config_batch(self, config_batch, lookback_period, verbose=True):
        """
        Run a batch of configs in parallel using ProcessPoolExecutor.
        
        Args:
            config_batch: List of configs to run
            lookback_period: Lookback period for all configs
            verbose: Print progress
            
        Returns:
            List of (config, results) tuples
        """
        # Determine number of parallel workers (leave 1 core free for system)
        n_workers = max(1, multiprocessing.cpu_count() - 1)
        # Limit to 2-3 for memory safety (each needs ~500MB)
        n_workers = min(n_workers, 3)
        
        results_list = []
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all configs
            future_to_config = {
                executor.submit(self._run_config_in_subprocess, config, lookback_period): config
                for config in config_batch
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    results = future.result()
                    results_list.append((config, results))
                except Exception as e:
                    if verbose:
                        print(f"    ✗ Config exception: {str(e)[:100]}")
                    results_list.append((config, None))
        
        return results_list
    
    def grid_search(self, architectures=['bidirectional'], 
                   lookback_periods=[60, 90, 120, 180, 365],
                   verbose=True,
                   predict_returns=True):  # NEW: Enable returns-based prediction
        """
        Perform comprehensive grid search across architectures and hyperparameters
        Now uses improved rolling validation and returns-based prediction
        
        Args:
            predict_returns: If True, predict log-returns; if False, predict prices (default: True)
        
        Returns:
            best_config: Dictionary with best configuration
            best_results: Dictionary with best results
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Starting Grid Search for {self.symbol}")
            print(f"{'='*60}")
        
        # Expanded grid search for better MAPE: more learning rates and dropout values
        TCN_CONFIGS = {
            'layers': [2, 3],              
            'units': [64, 128],
            'dropout': [0.1, 0.2, 0.3],  # Added 0.1
            'learning_rate': [0.0003, 0.0005, 0.001, 0.002],  # Added 0.0003 and 0.002  
            'batch_size': [32, 64],         
            'patience': [10]
        }
        
        LSTM_CONFIGS = {
            'layers': [2, 3],              
            'units': [64, 128],
            'dropout': [0.1, 0.2, 0.3],  # Added 0.1
            'learning_rate': [0.0003, 0.0005, 0.001, 0.002],  # Added 0.0003 and 0.002
            'batch_size': [32, 64],         
            'patience': [10]
        }
        
        best_mape = float('inf')
        best_config = None
        best_results = None
        all_results = []
        total_config_count = 0  # Track total configs tested across all lookbacks
        early_stop = False  # Flag to break out of all loops
        
        for architecture in architectures:
            if early_stop:
                break
                
            config_grid = TCN_CONFIGS if architecture == 'tcn' else LSTM_CONFIGS
            
            for lookback in lookback_periods:
                if early_stop:
                    break
                    
                if verbose:
                    print(f"\n--- Testing {architecture.upper()} with lookback={lookback} ---")
                
                # Generate configurations
                configs = self._generate_configs(config_grid, architecture, lookback)
                
                # Process configs in parallel batches
                batch_size = 3  # Run 3 configs at once (M1 Pro can handle this)
                for batch_start in range(0, len(configs), batch_size):
                    if early_stop:
                        break
                    
                    batch_end = min(batch_start + batch_size, len(configs))
                    config_batch = configs[batch_start:batch_end]
                    
                    if verbose:
                        print(f"\n  Processing batch {batch_start//batch_size + 1} (configs {batch_start+1}-{batch_end})...")
                    
                    # Run batch in parallel
                    batch_results = self._run_config_batch(config_batch, lookback, verbose=verbose)
                    
                    # Process results
                    for config, results in batch_results:
                        if early_stop:
                            break
                        
                        total_config_count += 1
                        
                        if verbose:
                            print(f"\n  Config {total_config_count}: {architecture.upper()}, lookback={lookback}, layers={config.get('layers', 2)}, units={config.get('units_1', config.get('units', 64))}, lr={config['learning_rate']}, batch={config['batch_size']}")
                        
                        if results and results.get('success'):
                            # Extract metrics
                            test_mape = results.get('test_mape', 999)
                            
                            # Store result (only metrics, no predictions)
                            result_entry = {
                                'symbol': self.symbol,
                                **config,
                                'lookback': lookback,
                                'test_mae': results.get('test_mae'),
                                'test_rmse': results.get('test_rmse'),
                                'test_mape': test_mape,
                                'val_mae': results.get('val_mae'),
                                'val_mape': results.get('val_mape'),
                                'direction_accuracy': results.get('direction_accuracy'),
                                'timestamp': datetime.now().isoformat()
                            }
                            all_results.append(result_entry)
                            
                            # Update best based on test MAPE
                            if test_mape < best_mape:
                                best_mape = test_mape
                                best_config = config.copy()
                                best_config['lookback'] = lookback
                                best_results = {
                                    'test_mae': results.get('test_mae'),
                                    'test_rmse': results.get('test_rmse'),
                                    'test_mape': results.get('test_mape'),
                                    'val_mape': results.get('val_mape'),
                                    'direction_accuracy': results.get('direction_accuracy')
                                }
                                
                                # Save best config immediately (incremental save)
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
                                    print(f"    ✓ NEW BEST! Test MAPE={test_mape:.2f}%, Dir Acc={results.get('direction_accuracy', 0):.1f}%")
                                    print(f"    [Best MAPE so far: {best_mape:.2f}%]")
                                    print(f"    💾 Saved to: {config_file}")
                                
                                # Early exit if test MAPE < 1.0% (excellent performance)
                                if test_mape < 1.0:
                                    if verbose:
                                        print(f"\n  🎯 EXCELLENT PERFORMANCE! Test MAPE={test_mape:.2f}% < 1.0%")
                                        print(f"  Direction Accuracy: {results.get('direction_accuracy', 0):.1f}%")
                                        print(f"  Stopping grid search early.")
                                    early_stop = True
                                    break
                            else:
                                if verbose:
                                    print(f"    Test MAPE={test_mape:.2f}% [Best: {best_mape:.2f}%]")
                        else:
                            if verbose:
                                print(f"    Config failed")
                    
                    # Subprocess already exited and freed memory - no cleanup needed!
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Grid Search Complete!")
            print(f"Total configurations tested: {len(all_results)}")
            if best_config:
                print(f"Best Configuration: {best_config.get('architecture', best_config.get('model_type', 'Unknown')).upper()} with lookback={best_config.get('lookback', 'Unknown')}")
                print(f"Best Test MAPE: {best_mape:.2f}%")
                
                # Show summary by lookback period
                print(f"\n--- Performance by Lookback Period ---")
                lookback_summary = {}
                for result in all_results:
                    lb = result.get('lookback', 0)
                    if lb not in lookback_summary:
                        lookback_summary[lb] = []
                    lookback_summary[lb].append(result.get('test_mape', 999))
                
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
        
        # Initialize tuner (it will load data internally)
        tuner = HyperparameterTuner(
            symbol=self.symbol,
            test_days=7,
            final_holdout_days=30
        )
        
        # Add epochs to config
        final_config = config.copy()
        final_config['epochs'] = 100
        
        # First: Get validation metrics using rolling validation
        if verbose:
            print("\n[1/2] Getting validation metrics with rolling CV...")
        
        validation_results = tuner.train_and_evaluate_rolling(
            config=final_config
        )
        
        # Now: Train on ALL data for true future predictions
        if verbose:
            print("\n[2/2] Training on ALL data for future predictions...")
        
        # Create a fresh tuner with no holdout for final training
        final_tuner = HyperparameterTuner(
            symbol=self.symbol,
            test_days=7,
            final_holdout_days=0  # No holdout for final training - use everything
        )
        
        # Get all available data
        train_df, _, _ = final_tuner.load_data_with_holdout()
        
        # Use the improved training method with returns prediction
        # Set predict_returns flag based on config (default True)
        predict_returns = final_config.get('predict_returns', True)
        
        if predict_returns:
            # Use returns-based prediction
            data_with_features = final_tuner.prepare_features_with_returns(train_df)
            
            # Select features
            feature_cols = ['log_returns', 'volume_ratio', 'sma_5_ratio', 'sma_20_ratio', 
                           'rsi_norm', 'macd_norm', 'bb_position', 'atr_norm', 'momentum_5']
            feature_cols = [col for col in feature_cols if col in data_with_features.columns]
            
            features = data_with_features[feature_cols].values
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Scale with StandardScaler for returns
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            # Create sequences for returns prediction
            lookback = final_config['lookback']
            X, y = final_tuner.create_sequences_returns(scaled_features, lookback, 7)
        else:
            # Use price-based prediction (backward compatibility)
            data_with_features = final_tuner.prepare_features(train_df)
            
            feature_cols = ['close', 'volume', 'returns', 'sma_5', 'sma_20', 'rsi', 
                           'macd', 'bb_width', 'volume_ratio', 'atr', 'momentum_5']
            feature_cols = [col for col in feature_cols if col in data_with_features.columns]
            
            features = data_with_features[feature_cols].values
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_features = scaler.fit_transform(features)
            
            lookback = final_config['lookback']
            X, y = final_tuner.create_sequences(scaled_features, lookback, 7)
        
        # Build and train model on ALL data
        model = final_tuner.build_model(final_config, (final_config['lookback'], len(feature_cols)))
        
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        early_stop = EarlyStopping(monitor='loss', patience=final_config.get('patience', 10), restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.00001)
        
        if verbose:
            print(f"Training on {len(X)} samples with all available data...")
            print(f"Prediction mode: {'Returns-based' if predict_returns else 'Price-based'}")
        
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
        
        # Store data_with_features for predictions
        full_data = data_with_features
        
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
        
        # Inverse transform predictions
        if predict_returns:
            # For returns-based prediction, reconstruct prices
            base_price = full_data['close'].iloc[-1]
            predictions = final_tuner.reconstruct_prices_from_returns(predictions_scaled, base_price)
        else:
            # For price-based prediction, inverse transform
            dummy = np.zeros((7, len(feature_cols)))
            dummy[:, 0] = predictions_scaled
            predictions = scaler.inverse_transform(dummy)[:, 0].tolist()
        
        # Calculate directional signal
        base_price = full_data['close'].iloc[-1]
        pred_directions = [1 if p > base_price else -1 for p in predictions]
        up_days = sum(1 for d in pred_directions if d == 1)
        down_days = 7 - up_days
        direction_confidence = (max(up_days, down_days) / 7) * 100
        overall_direction = "BULLISH" if up_days > down_days else "BEARISH" if down_days > up_days else "NEUTRAL"
        
        # Generate TRUE FUTURE dates (beyond last available data)
        last_date = pd.to_datetime(full_data['date'].iloc[-1])
        prediction_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') 
                           for i in range(7)]
        
        # Get recent actual prices for context (last 30 days)
        recent_prices = full_data['close'].tail(30).tolist()
        recent_dates = pd.to_datetime(full_data['date']).tail(30).dt.strftime('%Y-%m-%d').tolist()
        
        if verbose:
            print(f"\n✓ TRUE FUTURE Predictions (beyond all available data):")
            print(f"  Last known date: {last_date.strftime('%Y-%m-%d')}")
            print(f"  Last known price: Rs. {full_data['close'].iloc[-1]:.2f}")
            
            print(f"\n  📊 Model Performance:")
            mape = validation_results.get('avg_mape', validation_results.get('test_mape', 0))
            dir_acc = validation_results.get('avg_direction_accuracy', validation_results.get('direction_accuracy', 0))
            print(f"    MAPE: {mape:.2f}% ({'Excellent' if mape < 2 else 'Good' if mape < 5 else 'Fair' if mape < 10 else 'Poor'})")
            print(f"    Direction Accuracy: {dir_acc:.1f}% ({'Strong' if dir_acc > 70 else 'Moderate' if dir_acc > 50 else 'Weak'})")
            
            print(f"\n  🎯 Trading Signal: {overall_direction}")
            print(f"    Confidence: {direction_confidence:.1f}% ({up_days} up, {down_days} down)")
            print(f"    Recommendation: {'BUY' if overall_direction == 'BULLISH' else 'SELL' if overall_direction == 'BEARISH' else 'HOLD'}")
            
            print(f"\n  📈 7-Day Forecast:")
            for date, price in zip(prediction_dates, predictions):
                change = price - full_data['close'].iloc[-1]
                change_pct = (change / full_data['close'].iloc[-1]) * 100
                direction = "↑" if price > full_data['close'].iloc[-1] else "↓"
                print(f"    {date}: Rs. {price:.2f} ({change_pct:+.2f}%) {direction}")
        
        model_info = {
            'model_path': str(model_path),
            'architecture': config['architecture'],
            'lookback': config['lookback'],
            'layers': config['layers'],
            'predict_returns': predict_returns,
            'test_mape': validation_results.get('avg_mape', validation_results.get('test_mape', 0)),
            'test_mae': validation_results.get('avg_mae', validation_results.get('test_mae', 0)),
            'test_rmse': validation_results.get('avg_rmse', validation_results.get('test_rmse', 0)),
            'direction_accuracy': validation_results.get('avg_direction_accuracy', validation_results.get('direction_accuracy', 0)),
            'training_samples': len(full_data),
            'last_actual_price': float(full_data['close'].iloc[-1]),
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
