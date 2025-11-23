"""
Analyze TCN grid search results and find optimal hyperparameters
"""
import pandas as pd
import numpy as np

# Load the log
print("\n" + "="*80)
print("TCN GRID SEARCH ANALYSIS")
print("="*80)

df = pd.read_csv('tcn_grid_search_master_log.csv')
print(f"\n✓ Loaded {len(df)} completed tests")
print(f"✓ Stocks tested: {df['symbol'].nunique()} ({', '.join(sorted(df['symbol'].unique()))})")
print(f"✓ Lookback periods tested: {sorted(df['lookback'].unique())}")

# Overall statistics
print("\n" + "="*80)
print("OVERALL STATISTICS")
print("="*80)
print(f"Mean MAE: {df['test_mae'].mean():.2f}")
print(f"Median MAE: {df['test_mae'].median():.2f}")
print(f"Best MAE: {df['test_mae'].min():.2f}")
print(f"Worst MAE: {df['test_mae'].max():.2f}")
print(f"\nMean MAPE: {df['test_mape'].mean():.2f}%")
print(f"Median MAPE: {df['test_mape'].median():.2f}%")
print(f"Best MAPE: {df['test_mape'].min():.2f}%")
print(f"Mean Direction Accuracy: {df['direction_accuracy'].mean():.2f}%")

# Top 10 configurations by MAE
print("\n" + "="*80)
print("🏆 TOP 10 BEST CONFIGURATIONS (by MAE)")
print("="*80)
top_10 = df.nsmallest(10, 'test_mae')
for idx, row in top_10.iterrows():
    print(f"\n#{top_10.index.get_loc(idx) + 1}: {row['symbol']} - MAE: {row['test_mae']:.2f}, MAPE: {row['test_mape']:.2f}%, Dir: {row['direction_accuracy']:.1f}%")
    print(f"   Lookback: {int(row['lookback'])}, Layers: {int(row['layers'])}, Units: {int(row['units_1'])}/{int(row['units_2'])}/{int(row['units_3'])}")
    print(f"   Dropout: {row['dropout_1']:.2f}, LR: {row['learning_rate']}, Batch: {int(row['batch_size'])}, Patience: {int(row['patience'])}")

# Top 10 by MAPE
print("\n" + "="*80)
print("🏆 TOP 10 BEST CONFIGURATIONS (by MAPE)")
print("="*80)
top_10_mape = df.nsmallest(10, 'test_mape')
for idx, row in top_10_mape.iterrows():
    print(f"\n#{top_10_mape.index.get_loc(idx) + 1}: {row['symbol']} - MAPE: {row['test_mape']:.2f}%, MAE: {row['test_mae']:.2f}, Dir: {row['direction_accuracy']:.1f}%")
    print(f"   Lookback: {int(row['lookback'])}, Layers: {int(row['layers'])}, Units: {int(row['units_1'])}/{int(row['units_2'])}/{int(row['units_3'])}")
    print(f"   Dropout: {row['dropout_1']:.2f}, LR: {row['learning_rate']}, Batch: {int(row['batch_size'])}, Patience: {int(row['patience'])}")

# Performance by lookback period
print("\n" + "="*80)
print("📊 PERFORMANCE BY LOOKBACK PERIOD")
print("="*80)
lookback_summary = df.groupby('lookback').agg({
    'test_mae': ['mean', 'median', 'min', 'std'],
    'test_mape': ['mean', 'median', 'min'],
    'direction_accuracy': ['mean', 'max'],
    'symbol': 'count'
}).round(2)
lookback_summary.columns = ['_'.join(col).strip() for col in lookback_summary.columns.values]
print(lookback_summary)

# Best lookback period
best_lookback = df.groupby('lookback')['test_mae'].mean().idxmin()
print(f"\n🎯 Best Overall Lookback Period: {int(best_lookback)} days")

# Performance by stock
print("\n" + "="*80)
print("📊 PERFORMANCE BY STOCK")
print("="*80)
stock_summary = df.groupby('symbol').agg({
    'test_mae': ['mean', 'min'],
    'test_mape': ['mean', 'min'],
    'direction_accuracy': 'mean',
    'config_name': 'count'
}).round(2)
stock_summary.columns = ['_'.join(col).strip() for col in stock_summary.columns.values]
stock_summary = stock_summary.sort_values('test_mae_mean')
print(stock_summary)

# Best configuration per stock
print("\n" + "="*80)
print("🏆 BEST CONFIGURATION PER STOCK")
print("="*80)
best_per_stock = df.loc[df.groupby('symbol')['test_mae'].idxmin()].sort_values('test_mae')
for idx, row in best_per_stock.iterrows():
    print(f"\n{row['symbol']}: MAE={row['test_mae']:.2f}, MAPE={row['test_mape']:.2f}%, Dir={row['direction_accuracy']:.1f}%")
    print(f"  Lookback: {int(row['lookback'])}, Layers: {int(row['layers'])}, Units: {int(row['units_1'])}/{int(row['units_2'])}/{int(row['units_3'])}")
    print(f"  Dropout: {row['dropout_1']:.2f}, LR: {row['learning_rate']}, Batch: {int(row['batch_size'])}, Patience: {int(row['patience'])}")

# Hyperparameter frequency analysis
print("\n" + "="*80)
print("📊 HYPERPARAMETER FREQUENCY IN TOP 50 CONFIGS")
print("="*80)
top_50 = df.nsmallest(50, 'test_mae')

print("\nLookback Period Distribution:")
print(top_50['lookback'].value_counts().sort_index())

print("\nLayers Distribution:")
print(top_50['layers'].value_counts().sort_index())

print("\nUnits (Layer 1) Distribution:")
print(top_50['units_1'].value_counts().sort_index())

print("\nDropout Distribution:")
print(top_50['dropout_1'].value_counts().sort_index())

print("\nLearning Rate Distribution:")
print(top_50['learning_rate'].value_counts().sort_index())

print("\nBatch Size Distribution:")
print(top_50['batch_size'].value_counts().sort_index())

print("\nPatience Distribution:")
print(top_50['patience'].value_counts().sort_index())

# Recommended configuration based on frequency
print("\n" + "="*80)
print("💡 RECOMMENDED TCN CONFIGURATION (Based on Top 50)")
print("="*80)
recommended = {
    'lookback': int(top_50['lookback'].mode()[0]),
    'layers': int(top_50['layers'].mode()[0]),
    'units_1': int(top_50['units_1'].mode()[0]),
    'units_2': int(top_50['units_2'].mode()[0]),
    'units_3': int(top_50['units_3'].mode()[0]),
    'dropout_1': float(top_50['dropout_1'].mode()[0]),
    'dropout_2': float(top_50['dropout_2'].mode()[0]),
    'dropout_3': float(top_50['dropout_3'].mode()[0]),
    'dense_units': int(top_50['dense_units'].mode()[0]),
    'learning_rate': float(top_50['learning_rate'].mode()[0]),
    'batch_size': int(top_50['batch_size'].mode()[0]),
    'patience': int(top_50['patience'].mode()[0])
}

print("\nRecommended TCN Configuration:")
print(f"  Lookback: {recommended['lookback']} days")
print(f"  Layers: {recommended['layers']}")
print(f"  Units: {recommended['units_1']}/{recommended['units_2']}/{recommended['units_3']}")
print(f"  Dropout: {recommended['dropout_1']:.2f}/{recommended['dropout_2']:.2f}/{recommended['dropout_3']:.2f}")
print(f"  Dense Units: {recommended['dense_units']}")
print(f"  Learning Rate: {recommended['learning_rate']}")
print(f"  Batch Size: {recommended['batch_size']}")
print(f"  Patience: {recommended['patience']}")

# Expected performance with recommended config
configs_with_recommended = df[
    (df['lookback'] == recommended['lookback']) &
    (df['layers'] == recommended['layers']) &
    (df['units_1'] == recommended['units_1']) &
    (df['dropout_1'] == recommended['dropout_1']) &
    (df['learning_rate'] == recommended['learning_rate']) &
    (df['batch_size'] == recommended['batch_size'])
]

if len(configs_with_recommended) > 0:
    print(f"\n📈 Expected Performance with Recommended Config:")
    print(f"  Mean MAE: {configs_with_recommended['test_mae'].mean():.2f}")
    print(f"  Mean MAPE: {configs_with_recommended['test_mape'].mean():.2f}%")
    print(f"  Mean Direction Accuracy: {configs_with_recommended['direction_accuracy'].mean():.2f}%")
    print(f"  Tested on {len(configs_with_recommended)} stock(s): {', '.join(configs_with_recommended['symbol'].unique())}")

# Save recommended config
import json
with open('tcn_recommended_config.json', 'w') as f:
    json.dump(recommended, f, indent=2)
print(f"\n✓ Recommended configuration saved to: tcn_recommended_config.json")

# Compare with other architectures
print("\n" + "="*80)
print("📊 TCN vs OTHER ARCHITECTURES (Comparison)")
print("="*80)
print(f"TCN Best MAE: {df['test_mae'].min():.2f}")
print(f"TCN Average MAE: {df['test_mae'].mean():.2f}")
print(f"TCN Best MAPE: {df['test_mape'].min():.2f}%")
print(f"TCN Average MAPE: {df['test_mape'].mean():.2f}%")
print(f"\nPrevious Results:")
print(f"  Transformer (365d): MAE=46.54, MAPE=11.40%")
print(f"  Bi-LSTM (60d): MAE=52.68, MAPE=12.92%")
print(f"  TCN (60d): MAE=123.88, MAPE=30.37%")

# Correlation analysis
print("\n" + "="*80)
print("📊 CORRELATION WITH PERFORMANCE")
print("="*80)
numeric_cols = ['lookback', 'layers', 'units_1', 'dropout_1', 'learning_rate', 
                'batch_size', 'patience', 'test_mae', 'test_mape']
corr_with_mae = df[numeric_cols].corr()['test_mae'].sort_values()
print("\nCorrelation with MAE:")
for param, corr_val in corr_with_mae.items():
    if param != 'test_mae':
        print(f"  {param:20s}: {corr_val:+.3f}")

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE")
print("="*80)
