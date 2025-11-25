#!/usr/bin/env python3
"""
Subprocess worker to run a single config in isolation.
When this process exits, all TensorFlow memory is guaranteed to be freed by the OS.
"""

import sys
import json
import os

# Suppress ALL output before importing TensorFlow
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHONWARNINGS'] = 'ignore'
# Suppress TensorFlow mixed precision warnings
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'

# Redirect stdout/stderr temporarily to suppress TensorFlow warnings
import io
original_stdout = sys.stdout
original_stderr = sys.stderr
sys.stdout = io.StringIO()
sys.stderr = io.StringIO()

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

# Set deterministic seeds for reproducibility
import numpy as np
import random
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# Enable mixed precision while output is suppressed (no warning printed)
tf.keras.mixed_precision.set_global_policy('mixed_float16')

from tune_hyperparameters import HyperparameterTuner

# Restore stdout/stderr for final JSON output only
sys.stdout = original_stdout
sys.stderr = original_stderr

def run_single_config(symbol, config, lookback_period):
    """
    Run a single config and return results.
    This runs in a subprocess that will be terminated after completion.
    """
    try:
        # Suppress all print output during execution
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
        # Initialize tuner with faster settings
        tuner = HyperparameterTuner(
            symbol=symbol,
            test_days=7,
            final_holdout_days=30
        )
        
        # Disable verbose TensorFlow logs for faster startup
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        
        # Load data with holdout split
        available_df, holdout_df = tuner.load_data_with_holdout()
        
        # Prepare config in the format expected by train_and_evaluate_rolling
        # It needs 'architecture' not 'model_type', and 'lookback' not 'lookback_period'
        train_config = {
            'architecture': config.get('architecture', config.get('model_type', 'bidirectional')),
            'lookback': lookback_period,
            'layers': config.get('layers', 2),
            'learning_rate': config['learning_rate'],
            'batch_size': config['batch_size'],
            'epochs': 100,
            'predict_returns': False  # Use price prediction
        }
        
        # Add layer-specific config if available
        layers = train_config['layers']
        if 'units_1' in config:
            train_config['units_1'] = config['units_1']
            # For 2-layer networks, add units_2
            if layers >= 2:
                train_config['units_2'] = config.get('units_2', config['units_1'] // 2)
            # For 3-layer networks, add units_3
            if layers >= 3:
                train_config['units_3'] = config.get('units_3', config['units_1'] // 4)
        else:
            units = config.get('units', 64)
            train_config['units_1'] = units
            if layers >= 2:
                train_config['units_2'] = units // 2
            if layers >= 3:
                train_config['units_3'] = units // 4
        
        if 'dropout_1' in config:
            train_config['dropout_1'] = config['dropout_1']
            # For 2-layer networks, add dropout_2
            if layers >= 2:
                train_config['dropout_2'] = config.get('dropout_2', config['dropout_1'])
            # For 3-layer networks, add dropout_3
            if layers >= 3:
                train_config['dropout_3'] = config.get('dropout_3', config['dropout_1'] / 2)
        else:
            dropout = config.get('dropout', 0.2)
            train_config['dropout_1'] = dropout
            if layers >= 2:
                train_config['dropout_2'] = dropout
            if layers >= 3:
                train_config['dropout_3'] = dropout / 2
        
        # Add other expected config keys
        train_config['dense_units'] = config.get('dense_units', 32)
        train_config['optimizer'] = config.get('optimizer', 'adam')
        train_config['patience'] = config.get('patience', 10)
        
        # Run training and evaluation
        results = tuner.train_and_evaluate_rolling(
            config=train_config,
            available_df=available_df,
            holdout_df=holdout_df,
            predict_returns=False
        )
        
        # Clean up (though process will exit anyway)
        try:
            tuner.cleanup()
        except:
            pass
        
        # Restore stdout for JSON output
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        
        # Return only the metrics we need (not predictions/arrays)
        if results and 'error' not in results:
            clean_results = {
                'model_type': train_config.get('model_type'),
                'lookback_period': lookback_period,
                'num_layers': train_config.get('num_layers'),
                'learning_rate': train_config.get('learning_rate'),
                'batch_size': train_config.get('batch_size'),
                'val_mae': results.get('avg_val_mae'),
                'val_mape': results.get('avg_val_mape'),
                'test_mae': results.get('holdout_mae'),
                'test_rmse': results.get('holdout_rmse'),
                'test_mape': results.get('holdout_mape'),
                'direction_accuracy': results.get('direction_accuracy'),
                'success': True
            }
        else:
            clean_results = {
                'success': False,
                'error': results.get('error', 'Unknown error') if results else 'No results returned'
            }
        
        return clean_results
        
    except KeyError as e:
        # Restore output for error reporting
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        return {
            'success': False,
            'error': f'Missing config key: {str(e)}'
        }
    except Exception as e:
        # Restore output for error reporting
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        import traceback
        return {
            'success': False,
            'error': f'{type(e).__name__}: {str(e)}\n{traceback.format_exc()}'
        }

if __name__ == '__main__':
    # Read config from command line arguments
    if len(sys.argv) != 4:
        print(json.dumps({'success': False, 'error': 'Usage: config_worker.py <symbol> <config_json> <lookback_period>'}))
        sys.exit(1)
    
    symbol = sys.argv[1]
    config = json.loads(sys.argv[2])
    lookback_period = int(sys.argv[3])
    
    # Run config
    results = run_single_config(symbol, config, lookback_period)
    
    # Output results as JSON
    print(json.dumps(results))
    
    # Exit cleanly - OS will reclaim all memory
    sys.exit(0)
