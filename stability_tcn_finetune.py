"""
Stability evaluation + fine-tuning for TCN using recommended configuration.

Usage:
  python3 stability_tcn_finetune.py --dry-run
  python3 stability_tcn_finetune.py --run --stocks 20 --seeds 3 --lrs 0.0005,0.001,0.002 --batches 32,64

By default script performs a dry-run (no training) to validate dataset and configs.
Results are saved to `tcn_stability_finetune_log.csv` and `tcn_stability_summary.json`.
"""

import argparse
import random
import json
import os
from datetime import datetime
import pandas as pd
import numpy as np

from tune_hyperparameters import HyperparameterTuner

# Try to load recommended config if present
RECOMMENDED_PATH = 'tcn_recommended_config.json'
if os.path.exists(RECOMMENDED_PATH):
    with open(RECOMMENDED_PATH) as f:
        recommended = json.load(f)
else:
    # default fallback
    recommended = {
        'lookback': 60,
        'layers': 2,
        'units_1': 128,
        'units_2': 64,
        'units_3': 32,
        'dropout_1': 0.2,
        'dropout_2': 0.2,
        'dropout_3': 0.1,
        'dense_units': 32,
        'learning_rate': 0.0005,
        'batch_size': 32,
        'patience': 10
    }

DEFAULT_STOCK_FILE = 'all_symbols.txt'


def sample_stocks(n):
    # prefer using all_symbols.txt if available
    symbols = []
    if os.path.exists(DEFAULT_STOCK_FILE):
        with open(DEFAULT_STOCK_FILE) as f:
            symbols = [s.strip() for s in f.readlines() if s.strip()]
    else:
        # fallback: check grid search master log for symbols
        if os.path.exists('tcn_grid_search_master_log.csv'):
            df = pd.read_csv('tcn_grid_search_master_log.csv')
            symbols = sorted(df['symbol'].unique().tolist())

    if not symbols:
        raise RuntimeError('No symbols list found (all_symbols.txt or log).')

    random.shuffle(symbols)
    return symbols[:n]


def set_seeds(seed):
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    import random as pyrandom
    pyrandom.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass


def build_config(base, lr, batch):
    cfg = base.copy()
    cfg['learning_rate'] = lr
    cfg['batch_size'] = batch
    # name for bookkeeping
    cfg['name'] = f"tcn_warm_l{cfg['lookback']}_u{cfg['units_1']}_lr{lr}_bs{batch}"
    cfg['architecture'] = 'tcn'
    cfg['epochs'] = cfg.get('epochs', 100)
    cfg['patience'] = cfg.get('patience', 10)
    return cfg


def check_data_has_enough_rows(tuner, lookback, min_extra=100):
    train_df, test_df = tuner.load_data()
    required = lookback + min_extra
    return len(train_df) >= required, len(train_df), len(test_df)


def run_experiment(symbols, seeds, lrs, batches, dry_run=True, out_csv='tcn_stability_finetune_log.csv'):
    results = []
    tried = 0

    for symbol in symbols:
        print(f"\n=== Symbol: {symbol} ===")
        tuner = HyperparameterTuner(symbol, test_days=7)

        # Load data (may raise ValueError if no data available)
        try:
            train_df, test_df = tuner.load_data()
        except Exception as e:
            print(f"  ⚠️ Skipping {symbol}: {e}")
            continue

        # check data sufficiency
        required = recommended['lookback'] + 100
        if len(train_df) < required:
            print(f"  ⚠️ Skipping {symbol}: insufficient training rows ({len(train_df)}), need >= {required}")
            continue
        print(f"  ✓ Data rows: train={len(train_df)}, test={len(test_df)}")

        for lr in lrs:
            for batch in batches:
                for seed in seeds:
                    tried += 1
                    print(f"  -> lr={lr}, batch={batch}, seed={seed}")
                    cfg = build_config(recommended, lr, batch)
                    cfg['name'] = f"stability_{symbol}_lr{lr}_bs{batch}_s{seed}"

                    if dry_run:
                        # perform only a light validation: ensure sequences can be created
                        features = tuner.prepare_features(train_df)
                        feature_cols = [c for c in ['close','volume','returns','sma_5','sma_20','rsi','macd','bb_width','volume_ratio','atr','momentum_5'] if c in features.columns]
                        arr = features[feature_cols].values
                        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                        X, y = tuner.create_sequences(arr, cfg['lookback'])
                        print(f"    ✓ sequences: {len(X)} (lookback={cfg['lookback']})")

                        # record metadata only
                        entry = {
                            'timestamp': datetime.now().isoformat(),
                            'symbol': symbol,
                            'seed': seed,
                            'lookback': cfg['lookback'],
                            'layers': cfg['layers'],
                            'units_1': cfg['units_1'],
                            'dropout_1': cfg['dropout_1'],
                            'learning_rate': lr,
                            'batch_size': batch,
                            'status': 'dry-run'
                        }
                        results.append(entry)

                    else:
                        # full run: set seed and train
                        set_seeds(seed)
                        train_df, test_df = tuner.load_data()
                        result = tuner.train_and_evaluate(cfg, train_df, test_df)
                        # attach metadata
                        result.update({
                            'timestamp': datetime.now().isoformat(),
                            'symbol': symbol,
                            'seed': seed,
                            'lookback': cfg['lookback'],
                            'learning_rate': lr,
                            'batch_size': batch
                        })
                        results.append(result)

    # save results
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {len(df)} rows to {out_csv}")

    # summary
    if not dry_run and len(df) > 0:
        summary = df.groupby('symbol').agg({'test_mae':['mean','std','min'],'test_mape':['mean','std','min']})
        summary.to_json('tcn_stability_summary.json', orient='split')
        print('Saved summary to tcn_stability_summary.json')

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true', help='Run full experiments (training).')
    parser.add_argument('--stocks', type=int, default=20, help='Number of random stocks to sample.')
    parser.add_argument('--seeds', type=int, default=3, help='Number of random seeds per config.')
    parser.add_argument('--lrs', type=str, default='0.0005,0.001,0.002', help='Comma-separated LR values to test.')
    parser.add_argument('--batches', type=str, default='32,64', help='Comma-separated batch sizes to test.')
    parser.add_argument('--dry-run', action='store_true', default=True, help='Dry-run: validate configs and sequences but do not train.')
    args = parser.parse_args()

    # parse values
    lrs = [float(x) for x in args.lrs.split(',')]
    batches = [int(x) for x in args.batches.split(',')]
    seeds = list(range(args.seeds))

    symbols = sample_stocks(args.stocks)
    print(f"Sampling {len(symbols)} symbols: {symbols}")

    df = run_experiment(symbols, seeds, lrs, batches, dry_run=not args.run)
    print('\nDone.')
