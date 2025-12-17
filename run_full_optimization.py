#!/usr/bin/env python3
"""
Full Stock Optimization Script
==============================
Run Bayesian optimization on ALL stocks with sector information.

Features:
- Tracks progress in optimization_tracker.json
- Can resume from where it left off if interrupted
- Runs caffeinate to prevent Mac sleep
- Memory cleanup after each stock

Usage:
    python run_full_optimization.py                    # Run all unoptimized stocks
    python run_full_optimization.py --resume           # Resume from last position
    python run_full_optimization.py --stock NABIL      # Optimize single stock
    python run_full_optimization.py --list             # List all stocks with sectors
    python run_full_optimization.py --status           # Show optimization status
    python run_full_optimization.py --reset            # Reset progress tracker
    
Run with caffeinate (recommended):
    caffeinate -ims python run_full_optimization.py
"""

import json
import sqlite3
import sys
import gc
import argparse
from pathlib import Path
from datetime import datetime

import torch

# Import from xlstm module
from xlstm_multihorizon_bayesian import (
    bayesian_optimization, inference, validate_stock,
    HORIZONS, MODEL_DIR
)

TRACKER_FILE = Path("optimization_tracker.json")
DB_PATH = Path("data/nepse_stocks.db")


def get_all_stocks_with_sectors():
    """Get all stocks that have sector information"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute('''
        SELECT ss.symbol, ss.sector, ss.sector_index_code, 
               COUNT(ph.date) as price_days
        FROM stock_sectors ss
        LEFT JOIN price_history ph ON ss.symbol = ph.symbol
        GROUP BY ss.symbol
        HAVING price_days >= 100
        ORDER BY ss.sector, ss.symbol
    ''')
    stocks = []
    for row in cursor.fetchall():
        stocks.append({
            'symbol': row[0],
            'sector': row[1],
            'sector_index_code': row[2],
            'price_days': row[3]
        })
    conn.close()
    return stocks


def load_tracker():
    """Load optimization tracker"""
    if TRACKER_FILE.exists():
        with open(TRACKER_FILE, 'r') as f:
            return json.load(f)
    return {
        'started_at': None,
        'last_updated': None,
        'completed': [],
        'failed': [],
        'in_progress': None,
        'pending': [],
        'results': {}
    }


def save_tracker(tracker):
    """Save optimization tracker"""
    tracker['last_updated'] = datetime.now().isoformat()
    with open(TRACKER_FILE, 'w') as f:
        json.dump(tracker, f, indent=2, default=str)


PRIORITY_FILE = Path("priority_stocks/priority_list.txt")


def load_priority_stocks():
    """Load priority stocks from file"""
    if not PRIORITY_FILE.exists():
        return []
    stocks = []
    with open(PRIORITY_FILE, 'r') as f:
        for line in f:
            symbol = line.strip()
            if symbol and not symbol.startswith('#'):
                stocks.append(symbol)
    return stocks


def initialize_tracker(stocks):
    """Initialize tracker with all stocks - priority stocks first"""
    tracker = load_tracker()
    
    if tracker['started_at'] is None:
        tracker['started_at'] = datetime.now().isoformat()
    
    # Get priority stocks
    priority_symbols = load_priority_stocks()
    
    # Add any new stocks to pending
    existing = set(tracker['completed'] + tracker['failed'] + tracker['pending'])
    if tracker['in_progress']:
        existing.add(tracker['in_progress'])
    
    # Add priority stocks first (in order)
    for symbol in priority_symbols:
        if symbol not in existing:
            # Check if it's in the stocks list (has sector info)
            if any(s['symbol'] == symbol for s in stocks):
                tracker['pending'].append(symbol)
                print(f"   ⭐ Priority: {symbol}")
    
    # Then add remaining stocks
    for stock in stocks:
        if stock['symbol'] not in existing and stock['symbol'] not in priority_symbols:
            tracker['pending'].append(stock['symbol'])
    
    save_tracker(tracker)
    return tracker


def mark_in_progress(tracker, symbol):
    """Mark a stock as in progress"""
    if symbol in tracker['pending']:
        tracker['pending'].remove(symbol)
    tracker['in_progress'] = symbol
    save_tracker(tracker)


def mark_completed(tracker, symbol, results, sector=None):
    """Mark a stock as completed and store best hyperparams"""
    tracker['in_progress'] = None
    if symbol not in tracker['completed']:
        tracker['completed'].append(symbol)
    
    # Store results including hyperparams for warm starting
    tracker['results'][symbol] = {
        'completed_at': datetime.now().isoformat(),
        'best_mape': results.get('avg_mape'),
        'best_direction_acc': results.get('avg_direction_acc'),
        'trials_run': results.get('trial'),
        'criteria_met': results.get('criteria_met', False),
        'hyperparams': results.get('hyperparams', {}),
        'sector': sector,
    }
    
    # Update sector best configs for adaptive warm starting
    if 'sector_best' not in tracker:
        tracker['sector_best'] = {}
    
    if sector and results.get('hyperparams'):
        hp = results['hyperparams']
        acc = results.get('avg_direction_acc', 0)
        
        # Keep top 3 configs per sector
        if sector not in tracker['sector_best']:
            tracker['sector_best'][sector] = []
        
        tracker['sector_best'][sector].append({
            'symbol': symbol,
            'direction_acc': acc,
            **hp
        })
        
        # Sort by accuracy and keep top 3
        tracker['sector_best'][sector] = sorted(
            tracker['sector_best'][sector],
            key=lambda x: x.get('direction_acc', 0),
            reverse=True
        )[:3]
    
    save_tracker(tracker)


def mark_failed(tracker, symbol, error):
    """Mark a stock as failed"""
    tracker['in_progress'] = None
    if symbol not in tracker['failed']:
        tracker['failed'].append(symbol)
    tracker['results'][symbol] = {
        'failed_at': datetime.now().isoformat(),
        'error': str(error)[:200]
    }
    save_tracker(tracker)


def get_warm_start_configs(tracker, sector):
    """
    Get warm start configs for adaptive optimization.
    Priority: same sector > other high-performing sectors > defaults
    """
    configs = []
    sector_best = tracker.get('sector_best', {})
    
    # 1. First, add configs from same sector (most relevant)
    if sector in sector_best:
        for cfg in sector_best[sector]:
            configs.append(cfg)
            print(f"   🔥 Same sector ({sector}): {cfg.get('symbol')} - {cfg.get('direction_acc', 0):.1f}%")
    
    # 2. Add best configs from other sectors (cross-sector learning)
    other_sectors = [s for s in sector_best.keys() if s != sector]
    for other_sector in other_sectors:
        if len(configs) >= 5:
            break
        # Get best config from each other sector
        if sector_best[other_sector]:
            best_other = sector_best[other_sector][0]
            configs.append(best_other)
            print(f"   🌐 Cross-sector ({other_sector}): {best_other.get('symbol')} - {best_other.get('direction_acc', 0):.1f}%")
    
    return configs[:5]  # Max 5 configs


def print_status(tracker, stocks):
    """Print current optimization status"""
    print(f"\n{'='*70}")
    print(f"📊 OPTIMIZATION STATUS")
    print(f"{'='*70}")
    print(f"Started: {tracker.get('started_at', 'Not started')}")
    print(f"Last updated: {tracker.get('last_updated', 'Never')}")
    print(f"\n📈 Progress:")
    print(f"   ✅ Completed: {len(tracker['completed'])}")
    print(f"   ❌ Failed: {len(tracker['failed'])}")
    print(f"   ⏳ In Progress: {tracker['in_progress'] or 'None'}")
    print(f"   📋 Pending: {len(tracker['pending'])}")
    print(f"   📊 Total stocks: {len(stocks)}")
    
    if tracker['completed']:
        print(f"\n✅ Completed stocks:")
        for symbol in tracker['completed']:
            result = tracker['results'].get(symbol, {})
            mape = result.get('best_mape', '?')
            acc = result.get('best_direction_acc', '?')
            met = "🎉" if result.get('criteria_met') else ""
            print(f"   {symbol}: MAPE={mape:.2f}%, Dir={acc:.1f}% {met}" if isinstance(mape, float) else f"   {symbol}")
    
    if tracker['failed']:
        print(f"\n❌ Failed stocks:")
        for symbol in tracker['failed']:
            result = tracker['results'].get(symbol, {})
            error = result.get('error', 'Unknown error')
            print(f"   {symbol}: {error[:50]}")
    
    if tracker['pending']:
        print(f"\n📋 Pending stocks ({len(tracker['pending'])}):")
        print(f"   {', '.join(tracker['pending'][:10])}")
        if len(tracker['pending']) > 10:
            print(f"   ... and {len(tracker['pending']) - 10} more")
    
    # Show sector best configs (for adaptive warm starting)
    sector_best = tracker.get('sector_best', {})
    if sector_best:
        print(f"\n🔥 Sector Best Configs (for warm starting):")
        for sector, configs in sector_best.items():
            if configs:
                best = configs[0]
                print(f"   {sector}: {best.get('symbol')} - {best.get('direction_acc', 0):.1f}%")
    
    print(f"{'='*70}\n")


def optimize_stock(symbol, tracker, stocks):
    """Run optimization for a single stock with adaptive warm starting"""
    print(f"\n{'='*70}")
    print(f"🚀 OPTIMIZING: {symbol}")
    print(f"{'='*70}")
    
    mark_in_progress(tracker, symbol)
    
    try:
        # Validate stock and get sector
        print(f"📋 Validating {symbol}...")
        sector_info = validate_stock(symbol)
        sector = sector_info.get('sector')
        print(f"   ✅ Sector: {sector} ({sector_info['sector_index_code']})")
        
        # Get warm start configs from previously optimized stocks
        print(f"\n🔥 Looking for warm start configs...")
        warm_configs = get_warm_start_configs(tracker, sector)
        
        if warm_configs:
            print(f"   Found {len(warm_configs)} configs to warm start from")
        else:
            print(f"   No previous configs found, using defaults")
        
        # Run Bayesian optimization with warm start
        print(f"\n🔬 Running Bayesian Optimization (max 900 trials)...")
        all_results, best = bayesian_optimization(
            symbol=symbol,
            max_trials=900,
            n_models=5,
            warm_start_configs=warm_configs
        )
        
        if best:
            print(f"\n✅ {symbol} OPTIMIZATION COMPLETE!")
            print(f"   Best MAPE: {best['avg_mape']:.2f}%")
            print(f"   Best Direction Accuracy: {best['avg_direction_acc']:.1f}%")
            print(f"   Trials run: {best['trial']}")
            print(f"   Criteria met: {'Yes 🎉' if best.get('criteria_met') else 'No'}")
            
            # Run inference
            print(f"\n🔮 Running inference...")
            result = inference(symbol)
            if result:
                print(f"   ✅ Inference complete")
            
            mark_completed(tracker, symbol, best, sector=sector)
            return True
        else:
            mark_failed(tracker, symbol, "No results returned")
            return False
            
    except Exception as e:
        print(f"❌ Error optimizing {symbol}: {e}")
        import traceback
        traceback.print_exc()
        mark_failed(tracker, symbol, str(e))
        return False
    
    finally:
        # Memory cleanup
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print(f"🧹 Memory cleared after {symbol}")


def run_optimization(args):
    """Main optimization runner"""
    
    # Get all stocks with sectors
    stocks = get_all_stocks_with_sectors()
    
    if not stocks:
        print("❌ No stocks found with sector information!")
        return
    
    # List mode
    if args.list:
        print(f"\n📋 Stocks with sector information ({len(stocks)}):")
        print(f"{'='*70}")
        current_sector = None
        for stock in stocks:
            if stock['sector'] != current_sector:
                current_sector = stock['sector']
                print(f"\n📁 {current_sector}:")
            print(f"   {stock['symbol']:<10} ({stock['price_days']} days)")
        print(f"{'='*70}\n")
        return
    
    # Initialize tracker
    tracker = initialize_tracker(stocks)
    
    # Status mode
    if args.status:
        print_status(tracker, stocks)
        return
    
    # Reset mode
    if args.reset:
        TRACKER_FILE.unlink(missing_ok=True)
        print("✅ Tracker reset. Run again to start fresh.")
        return
    
    # Single stock mode
    if args.stock:
        symbol = args.stock.upper()
        if symbol not in [s['symbol'] for s in stocks]:
            print(f"❌ Stock {symbol} not found or doesn't have sector info!")
            sys.exit(1)
        success = optimize_stock(symbol, tracker, stocks)
        sys.exit(0 if success else 1)
    
    # Resume mode - continue from in_progress or pending
    print(f"\n{'#'*70}")
    print(f"#  FULL STOCK OPTIMIZATION")
    print(f"#  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"#  Total stocks: {len(stocks)}")
    print(f"#  Already completed: {len(tracker['completed'])}")
    print(f"#  To process: {len(tracker['pending']) + (1 if tracker['in_progress'] else 0)}")
    print(f"{'#'*70}")
    
    # Resume in_progress first
    if tracker['in_progress']:
        print(f"\n⏳ Resuming interrupted optimization: {tracker['in_progress']}")
        optimize_stock(tracker['in_progress'], tracker, stocks)
    
    # Process pending stocks
    while tracker['pending']:
        symbol = tracker['pending'][0]
        
        remaining = len(tracker['pending'])
        completed = len(tracker['completed'])
        total = completed + remaining + len(tracker['failed'])
        
        print(f"\n📊 Progress: {completed}/{total} completed, {remaining} remaining")
        
        success = optimize_stock(symbol, tracker, stocks)
        
        # Reload tracker in case it was modified
        tracker = load_tracker()
    
    # Final summary
    print(f"\n{'#'*70}")
    print(f"#  OPTIMIZATION COMPLETE!")
    print(f"{'#'*70}")
    print_status(tracker, stocks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Full Stock Optimization')
    parser.add_argument('--resume', action='store_true', help='Resume from last position')
    parser.add_argument('--stock', type=str, help='Optimize single stock')
    parser.add_argument('--list', action='store_true', help='List all stocks with sectors')
    parser.add_argument('--status', action='store_true', help='Show optimization status')
    parser.add_argument('--reset', action='store_true', help='Reset progress tracker')
    
    args = parser.parse_args()
    run_optimization(args)
