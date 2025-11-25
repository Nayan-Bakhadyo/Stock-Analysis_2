"""
Debug script to test memory usage during grid search
"""
import psutil
import os
import gc
from stock_predictor import StockPredictor

def get_memory_mb():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def main():
    print("="*70)
    print("MEMORY DEBUG TEST - Running 3 configs only")
    print("="*70)
    
    initial_memory = get_memory_mb()
    print(f"\n📊 Initial memory: {initial_memory:.2f} MB\n")
    
    # Create predictor
    predictor = StockPredictor(symbol='IGI', output_file='test_predictions.json')
    
    # Override to test only 3 configs with one lookback
    print("Testing with minimal grid: 1 lookback, 3 configs\n")
    
    # Run grid search with just 3 configs
    import sys
    from tune_hyperparameters import HyperparameterTuner
    import tensorflow as tf
    
    tuner = HyperparameterTuner(symbol='IGI', test_days=7, final_holdout_days=30)
    available_df, holdout_df = tuner.load_data_with_holdout()
    
    print(f"\n📊 After loading data: {get_memory_mb():.2f} MB (Δ {get_memory_mb() - initial_memory:.2f} MB)\n")
    
    # Test 3 configs manually
    configs = [
        {
            'architecture': 'bidirectional',
            'lookback': 60,
            'layers': 2,
            'units_1': 64,
            'units_2': 32,
            'dropout_1': 0.2,
            'dropout_2': 0.2,
            'dense_units': 32,
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'batch_size': 32,
            'patience': 10,
            'epochs': 50  # Reduced for testing
        },
        {
            'architecture': 'bidirectional',
            'lookback': 60,
            'layers': 2,
            'units_1': 64,
            'units_2': 32,
            'dropout_1': 0.2,
            'dropout_2': 0.2,
            'dense_units': 32,
            'optimizer': 'adam',
            'learning_rate': 0.0005,
            'batch_size': 32,
            'patience': 10,
            'epochs': 50
        },
        {
            'architecture': 'bidirectional',
            'lookback': 60,
            'layers': 2,
            'units_1': 64,
            'units_2': 32,
            'dropout_1': 0.3,
            'dropout_2': 0.3,
            'dense_units': 32,
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'batch_size': 64,
            'patience': 10,
            'epochs': 50
        }
    ]
    
    for i, config in enumerate(configs):
        config['predict_returns'] = True
        
        mem_before = get_memory_mb()
        print(f"\n{'='*70}")
        print(f"Config {i+1}/3 - Memory before: {mem_before:.2f} MB")
        print(f"{'='*70}")
        
        # Run training
        result = tuner.train_and_evaluate_rolling(
            config=config,
            available_df=available_df,
            holdout_df=holdout_df,
            predict_returns=True
        )
        
        mem_after = get_memory_mb()
        print(f"\n📊 Config {i+1} complete:")
        print(f"   Memory after: {mem_after:.2f} MB")
        print(f"   Increase: {mem_after - mem_before:.2f} MB")
        print(f"   Total increase: {mem_after - initial_memory:.2f} MB")
        
        # Force cleanup
        del result
        import keras.backend as K
        K.clear_session()
        tf.keras.backend.clear_session()
        gc.collect()
        gc.collect()
        gc.collect()
        
        mem_after_cleanup = get_memory_mb()
        print(f"   After cleanup: {mem_after_cleanup:.2f} MB")
        print(f"   Released: {mem_after - mem_after_cleanup:.2f} MB")
        
        if i == 2:  # After 3rd config, recreate tuner
            print(f"\n🔄 Recreating tuner...")
            tuner.cleanup()
            del tuner, available_df, holdout_df
            K.clear_session()
            tf.keras.backend.clear_session()
            gc.collect()
            gc.collect()
            gc.collect()
            
            mem_after_tuner_cleanup = get_memory_mb()
            print(f"   After tuner cleanup: {mem_after_tuner_cleanup:.2f} MB")
            print(f"   Released: {mem_after_cleanup - mem_after_tuner_cleanup:.2f} MB")
    
    final_memory = get_memory_mb()
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS:")
    print(f"  Initial: {initial_memory:.2f} MB")
    print(f"  Final: {final_memory:.2f} MB")
    print(f"  Total leaked: {final_memory - initial_memory:.2f} MB")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
