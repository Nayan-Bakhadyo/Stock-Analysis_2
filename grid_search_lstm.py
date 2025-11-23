"""
Grid search for LSTM/Bi-LSTM architecture across multiple stocks and lookback periods
Tests lookback periods: 60, 120, 180, 365, 500 days
"""
import sys
sys.path.insert(0, '/Users/Nayan/Documents/Business/Stock_Analysis')

from tune_hyperparameters import HyperparameterTuner
import pandas as pd
from datetime import datetime
import os

# Stocks to test
STOCKS = ['HRL', 'SPC', 'IGI', 'AHPC', 'ADBL']

# Lookback periods to test
LOOKBACK_PERIODS = [60, 120, 180, 365, 500]

# LSTM hyperparameter grid
LSTM_CONFIGS = {
    'layers': [2, 3],
    'units_1': [64, 128],
    'units_2': [32, 64],
    'units_3': [16, 32],
    'dropout_1': [0.1, 0.2, 0.3],
    'dropout_2': [0.1, 0.2, 0.3],
    'dropout_3': [0.1, 0.2],
    'dense_units': [16, 32],
    'learning_rate': [0.001, 0.0005],
    'batch_size': [32, 64],
    'patience': [10, 15]
}

def generate_lstm_configs(lookback):
    """Generate grid search configurations for Bi-LSTM"""
    configs = []
    
    # Test a subset of combinations to keep runtime reasonable
    for layers in LSTM_CONFIGS['layers']:
        for units_1 in LSTM_CONFIGS['units_1']:
            for dropout_1 in LSTM_CONFIGS['dropout_1']:
                for lr in LSTM_CONFIGS['learning_rate']:
                    for batch_size in LSTM_CONFIGS['batch_size']:
                        for patience in LSTM_CONFIGS['patience']:
                            # Scale units down for deeper networks
                            if units_1 == 128:
                                units_2 = 64
                                units_3 = 32
                            else:
                                units_2 = 32
                                units_3 = 16
                            
                            config = {
                                'name': f'bilstm_lb{lookback}_l{layers}_u{units_1}_d{dropout_1}_lr{lr}_bs{batch_size}',
                                'lookback': lookback,
                                'layers': layers,
                                'units_1': units_1,
                                'units_2': units_2,
                                'units_3': units_3,
                                'dropout_1': dropout_1,
                                'dropout_2': dropout_1,  # Use same dropout for consistency
                                'dropout_3': dropout_1 * 0.5,  # Lower dropout in final layer
                                'dense_units': 32 if units_1 == 128 else 16,
                                'architecture': 'bidirectional',
                                'optimizer': 'adam',
                                'learning_rate': lr,
                                'batch_size': batch_size,
                                'epochs': 100,
                                'patience': patience
                            }
                            configs.append(config)
    
    return configs

def main():
    print("\n" + "="*80)
    print("BIDIRECTIONAL LSTM GRID SEARCH - MULTIPLE STOCKS AND LOOKBACK PERIODS")
    print("="*80)
    print(f"Stocks to test: {len(STOCKS)}")
    print(f"Lookback periods: {LOOKBACK_PERIODS}")
    print(f"Configurations per lookback: ~{len(generate_lstm_configs(60))}")
    
    # Estimate total configurations
    total_configs = len(STOCKS) * len(LOOKBACK_PERIODS) * len(generate_lstm_configs(60))
    print(f"Total configurations to test: ~{total_configs}")
    print("="*80)
    
    # Load or create master log
    master_log_path = 'lstm_grid_search_master_log.csv'
    if os.path.exists(master_log_path):
        master_log = pd.read_csv(master_log_path)
        print(f"\n✓ Loaded existing master log with {len(master_log)} entries")
    else:
        master_log = pd.DataFrame()
        print(f"\n✓ Creating new master log")
    
    # Track progress
    stock_count = 0
    total_stocks = len(STOCKS)
    
    for symbol in STOCKS:
        stock_count += 1
        print("\n" + "="*80)
        print(f"TESTING STOCK {stock_count}/{total_stocks}: {symbol}")
        print("="*80)
        
        try:
            # Create tuner for this stock
            tuner = HyperparameterTuner(symbol, test_days=7)
            train_df, test_df = tuner.load_data()
            
            # Check if we have enough data
            min_required = max(LOOKBACK_PERIODS) + 100  # Extra buffer for sequences
            if len(train_df) < min_required:
                print(f"⚠️ Insufficient data for {symbol}: {len(train_df)} rows (need {min_required})")
                print(f"   Skipping {symbol}")
                continue
            
            print(f"✓ Loaded data: {len(train_df)} training rows, {len(test_df)} test rows")
            
            # Test each lookback period
            lookback_count = 0
            for lookback in LOOKBACK_PERIODS:
                lookback_count += 1
                
                # Check if this lookback has enough data
                if len(train_df) < lookback + 50:
                    print(f"\n⚠️ Skipping lookback {lookback} for {symbol} (insufficient data)")
                    continue
                
                print(f"\n{'─'*80}")
                print(f"Lookback Period {lookback_count}/{len(LOOKBACK_PERIODS)}: {lookback} days")
                print(f"{'─'*80}")
                
                # Generate configs for this lookback
                configs = generate_lstm_configs(lookback)
                print(f"Testing {len(configs)} Bi-LSTM configurations...")
                
                config_count = 0
                for config in configs:
                    config_count += 1
                    
                    # Check if already tested
                    if not master_log.empty:
                        existing = master_log[
                            (master_log['symbol'] == symbol) & 
                            (master_log['config_name'] == config['name'])
                        ]
                        if not existing.empty:
                            print(f"  [{config_count}/{len(configs)}] ⏭️  Skipping {config['name']} (already tested)")
                            continue
                    
                    print(f"  [{config_count}/{len(configs)}] 🔄 Testing {config['name']}...", end='', flush=True)
                    
                    # Train and evaluate
                    result = tuner.train_and_evaluate(config, train_df, test_df)
                    
                    if 'error' not in result:
                        print(f" ✅ MAE: {result['test_mae']:.2f}, MAPE: {result['test_mape']:.2f}%")
                        
                        # Log result
                        log_entry = {
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'symbol': symbol,
                            'config_name': config['name'],
                            'architecture': 'bidirectional',
                            'lookback': lookback,
                            'layers': config['layers'],
                            'units_1': config['units_1'],
                            'units_2': config['units_2'],
                            'units_3': config['units_3'],
                            'dropout_1': config['dropout_1'],
                            'dropout_2': config['dropout_2'],
                            'dropout_3': config['dropout_3'],
                            'dense_units': config['dense_units'],
                            'learning_rate': config['learning_rate'],
                            'batch_size': config['batch_size'],
                            'patience': config['patience'],
                            'test_mae': result['test_mae'],
                            'test_rmse': result['test_rmse'],
                            'test_mape': result['test_mape'],
                            'direction_accuracy': result['direction_accuracy']
                        }
                        
                        # Append to master log
                        master_log = pd.concat([master_log, pd.DataFrame([log_entry])], ignore_index=True)
                        
                        # Save after each successful test
                        master_log.to_csv(master_log_path, index=False)
                    else:
                        print(f" ❌ Error: {result['error']}")
                
                print(f"\n✓ Completed lookback {lookback} for {symbol}")
            
            print(f"\n✅ Completed all lookback periods for {symbol}")
            
        except Exception as e:
            print(f"\n❌ Error processing {symbol}: {str(e)}")
            continue
    
    # Final summary
    print("\n" + "="*80)
    print("GRID SEARCH COMPLETE")
    print("="*80)
    print(f"Total configurations tested: {len(master_log)}")
    print(f"Master log saved to: {master_log_path}")
    
    if not master_log.empty:
        print("\n📊 SUMMARY BY STOCK:")
        print("-" * 80)
        summary = master_log.groupby('symbol').agg({
            'test_mae': ['mean', 'min'],
            'test_mape': ['mean', 'min'],
            'config_name': 'count'
        }).round(2)
        print(summary)
        
        print("\n📊 SUMMARY BY LOOKBACK PERIOD:")
        print("-" * 80)
        lookback_summary = master_log.groupby('lookback').agg({
            'test_mae': ['mean', 'min'],
            'test_mape': ['mean', 'min'],
            'direction_accuracy': 'mean',
            'config_name': 'count'
        }).round(2)
        print(lookback_summary)
        
        print("\n🏆 TOP 10 BEST CONFIGURATIONS (by MAE):")
        print("-" * 80)
        top_configs = master_log.nsmallest(10, 'test_mae')[
            ['symbol', 'lookback', 'layers', 'units_1', 'dropout_1', 
             'learning_rate', 'test_mae', 'test_mape', 'direction_accuracy']
        ]
        print(top_configs.to_string(index=False))
        
        print("\n🏆 TOP 10 BEST CONFIGURATIONS (by MAPE):")
        print("-" * 80)
        top_configs_mape = master_log.nsmallest(10, 'test_mape')[
            ['symbol', 'lookback', 'layers', 'units_1', 'dropout_1', 
             'learning_rate', 'test_mae', 'test_mape', 'direction_accuracy']
        ]
        print(top_configs_mape.to_string(index=False))
        
        print("\n🏆 BEST CONFIGURATION PER STOCK:")
        print("-" * 80)
        best_per_stock = master_log.loc[master_log.groupby('symbol')['test_mae'].idxmin()][
            ['symbol', 'lookback', 'layers', 'units_1', 'dropout_1', 
             'learning_rate', 'test_mae', 'test_mape', 'direction_accuracy']
        ]
        print(best_per_stock.to_string(index=False))

if __name__ == "__main__":
    main()
