"""
Analyze hyperparameter tuning results across multiple stocks
Find common patterns in best configurations
"""
import json
import os
from collections import Counter
from typing import Dict, List


def analyze_master_log():
    """Analyze master log to find common patterns"""
    master_log = 'hyperparameter_tuning_master_log.json'
    
    if not os.path.exists(master_log):
        print("❌ No master log found. Run hyperparameter tuning first.")
        return
    
    with open(master_log, 'r') as f:
        data = json.load(f)
    
    stocks = data.get('stocks', {})
    
    if not stocks:
        print("❌ No stock data in master log.")
        return
    
    print(f"\n{'='*70}")
    print(f"HYPERPARAMETER ANALYSIS ACROSS {len(stocks)} STOCKS")
    print('='*70)
    
    # Collect all best configs
    lookbacks = []
    architectures = []
    learning_rates = []
    batch_sizes = []
    layers = []
    dropouts = []
    
    stock_results = []
    
    for symbol, info in stocks.items():
        config = info.get('best_config', {})
        metrics = info.get('best_metrics', {})
        
        if config:
            lookbacks.append(config['lookback'])
            architectures.append(config['architecture'])
            learning_rates.append(config['learning_rate'])
            batch_sizes.append(config['batch_size'])
            layers.append(config['layers'])
            dropouts.append(config['dropout_1'])
            
            stock_results.append({
                'symbol': symbol,
                'mae': metrics.get('test_mae', 0),
                'mape': metrics.get('test_mape', 0),
                'direction_acc': metrics.get('direction_accuracy', 0),
                'config': config
            })
    
    # Print summary statistics
    print(f"\n📊 PARAMETER FREQUENCY ANALYSIS")
    print(f"{'-'*70}")
    
    print(f"\n🔢 Lookback Days:")
    for lookback, count in Counter(lookbacks).most_common():
        pct = (count / len(lookbacks)) * 100
        print(f"  {lookback:3d} days: {count:2d} stocks ({pct:5.1f}%)")
    
    print(f"\n🏗️ Architecture:")
    for arch, count in Counter(architectures).most_common():
        pct = (count / len(architectures)) * 100
        print(f"  {arch:15s}: {count:2d} stocks ({pct:5.1f}%)")
    
    print(f"\n📈 Learning Rate:")
    for lr, count in Counter(learning_rates).most_common():
        pct = (count / len(learning_rates)) * 100
        print(f"  {lr:.5f}: {count:2d} stocks ({pct:5.1f}%)")
    
    print(f"\n📦 Batch Size:")
    for bs, count in Counter(batch_sizes).most_common():
        pct = (count / len(batch_sizes)) * 100
        print(f"  {bs:3d}: {count:2d} stocks ({pct:5.1f}%)")
    
    print(f"\n🔗 Network Layers:")
    for layer, count in Counter(layers).most_common():
        pct = (count / len(layers)) * 100
        print(f"  {layer:1d} layers: {count:2d} stocks ({pct:5.1f}%)")
    
    print(f"\n💧 Dropout Rate:")
    for dropout, count in Counter(dropouts).most_common():
        pct = (count / len(dropouts)) * 100
        print(f"  {dropout:.1f}: {count:2d} stocks ({pct:5.1f}%)")
    
    # Recommended configuration (most common)
    print(f"\n{'='*70}")
    print(f"🏆 RECOMMENDED CONFIGURATION (Most Common)")
    print(f"{'='*70}")
    print(f"  Lookback: {Counter(lookbacks).most_common(1)[0][0]} days")
    print(f"  Architecture: {Counter(architectures).most_common(1)[0][0]}")
    print(f"  Learning Rate: {Counter(learning_rates).most_common(1)[0][0]}")
    print(f"  Batch Size: {Counter(batch_sizes).most_common(1)[0][0]}")
    print(f"  Layers: {Counter(layers).most_common(1)[0][0]}")
    print(f"  Dropout: {Counter(dropouts).most_common(1)[0][0]}")
    
    # Best performing stocks
    print(f"\n{'='*70}")
    print(f"⭐ TOP 5 BEST PREDICTIONS (by MAE)")
    print(f"{'='*70}")
    
    stock_results.sort(key=lambda x: x['mae'])
    for i, result in enumerate(stock_results[:5], 1):
        config = result['config']
        print(f"\n{i}. {result['symbol']}")
        print(f"   MAE: {result['mae']:.4f}, MAPE: {result['mape']:.2f}%, Direction: {result['direction_acc']:.1f}%")
        print(f"   Config: lookback={config['lookback']}, arch={config['architecture']}, " +
              f"lr={config['learning_rate']}, batch={config['batch_size']}")
    
    # Save recommended config
    save_recommended_config(Counter(lookbacks).most_common(1)[0][0],
                           Counter(architectures).most_common(1)[0][0],
                           Counter(learning_rates).most_common(1)[0][0],
                           Counter(batch_sizes).most_common(1)[0][0],
                           Counter(layers).most_common(1)[0][0],
                           Counter(dropouts).most_common(1)[0][0])


def save_recommended_config(lookback, arch, lr, batch_size, layers, dropout):
    """Save recommended configuration based on analysis"""
    config = {
        'name': 'recommended_from_analysis',
        'lookback': lookback,
        'layers': layers,
        'units_1': 128,
        'units_2': 64,
        'units_3': 32 if layers == 3 else 0,
        'dropout_1': dropout,
        'dropout_2': dropout,
        'dropout_3': dropout,
        'dense_units': 32,
        'architecture': arch,
        'optimizer': 'adam',
        'learning_rate': lr,
        'batch_size': batch_size,
        'epochs': 100 if lr < 0.001 else 50,
        'patience': 15 if lr < 0.001 else 10
    }
    
    with open('recommended_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n💾 Recommended config saved to: recommended_config.json")


def compare_stocks():
    """Compare performance across tested stocks"""
    master_log = 'hyperparameter_tuning_master_log.json'
    
    if not os.path.exists(master_log):
        print("❌ No master log found.")
        return
    
    with open(master_log, 'r') as f:
        data = json.load(f)
    
    stocks = data.get('stocks', {})
    
    print(f"\n{'='*70}")
    print(f"STOCK COMPARISON")
    print(f"{'='*70}")
    
    results = []
    for symbol, info in stocks.items():
        metrics = info.get('best_metrics', {})
        results.append({
            'symbol': symbol,
            'mae': metrics.get('test_mae', 999),
            'rmse': metrics.get('test_rmse', 999),
            'mape': metrics.get('test_mape', 999),
            'direction': metrics.get('direction_accuracy', 0),
            'tested': info.get('last_tested', 'Unknown')
        })
    
    # Sort by MAE
    results.sort(key=lambda x: x['mae'])
    
    print(f"\n{'Symbol':<10} {'MAE':<8} {'RMSE':<8} {'MAPE':<8} {'Dir%':<8} {'Tested'}")
    print(f"{'-'*70}")
    
    for r in results:
        print(f"{r['symbol']:<10} {r['mae']:<8.4f} {r['rmse']:<8.4f} {r['mape']:<7.2f}% {r['direction']:<7.1f}% {r['tested'][:10]}")


def main():
    """Main entry point"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--compare':
        compare_stocks()
    else:
        analyze_master_log()
    
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()
