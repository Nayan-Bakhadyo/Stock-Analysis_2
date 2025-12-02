#!/usr/bin/env python3
"""
Batch Stock Prediction Script
Runs market-aware xLSTM predictions for all stocks and saves results.

Usage:
    python3 run_all_predictions.py                    # Run all stocks
    python3 run_all_predictions.py --limit 10         # Run first 10 stocks
    python3 run_all_predictions.py --sector "Commercial Bank"  # Run only banks
    python3 run_all_predictions.py --symbol NABIL,NICA,GBIME   # Run specific symbols
    python3 run_all_predictions.py --resume           # Resume from last run
"""

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
import config


def get_all_symbols(sector: str = None) -> list:
    """Get all stock symbols from database, optionally filtered by sector."""
    conn = sqlite3.connect(config.DB_PATH)
    
    if sector:
        query = """
            SELECT DISTINCT s.symbol 
            FROM stock_sectors s
            JOIN price_history p ON s.symbol = p.symbol
            WHERE s.sector = ?
            GROUP BY s.symbol
            HAVING COUNT(p.date) >= 200
            ORDER BY s.symbol
        """
        cursor = conn.execute(query, (sector,))
    else:
        query = """
            SELECT symbol 
            FROM price_history
            GROUP BY symbol
            HAVING COUNT(date) >= 200
            ORDER BY symbol
        """
        cursor = conn.execute(query)
    
    symbols = [row[0] for row in cursor.fetchall()]
    conn.close()
    return symbols


def get_completed_symbols(output_file: Path) -> set:
    """Get symbols that have already been processed."""
    if not output_file.exists():
        return set()
    
    try:
        with open(output_file, 'r') as f:
            data = json.load(f)
            return set(data.get('predictions', {}).keys())
    except:
        return set()


def run_prediction(symbol: str, n_models: int = 5, epochs: int = 100, 
                   hidden_size: int = 512, num_blocks: int = 7) -> dict:
    """Run market-aware prediction for a single symbol."""
    from xlstm_stock_forecaster import forecast_with_market_context
    
    try:
        result = forecast_with_market_context(
            symbol=symbol,
            lookback=120,
            epochs=epochs,
            n_models=n_models,
            hidden_size=hidden_size,
            num_blocks=num_blocks,
            batch_size=32,
        )
        return result
    except Exception as e:
        print(f"❌ Error predicting {symbol}: {e}")
        return {'error': str(e), 'symbol': symbol}


def save_results(output_file: Path, results: dict):
    """Save results to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(description='Batch Stock Prediction')
    parser.add_argument('--limit', type=int, help='Limit number of stocks to process')
    parser.add_argument('--sector', type=str, help='Filter by sector (e.g., "Commercial Bank")')
    parser.add_argument('--symbol', type=str, help='Comma-separated list of symbols')
    parser.add_argument('--resume', action='store_true', help='Resume from last run')
    parser.add_argument('--output', type=str, default='batch_predictions.json', help='Output file')
    parser.add_argument('--n-models', type=int, default=5, help='Number of ensemble models')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--hidden-size', type=int, default=512, help='Hidden size')
    parser.add_argument('--num-blocks', type=int, default=7, help='Number of xLSTM blocks')
    parser.add_argument('--fast', action='store_true', help='Use fast config (256h, 4b, 50e, 3m)')
    
    args = parser.parse_args()
    
    # Fast mode overrides
    if args.fast:
        args.n_models = 3
        args.epochs = 50
        args.hidden_size = 256
        args.num_blocks = 4
    
    output_file = Path(config.DATA_DIR) / args.output
    
    # Get symbols
    if args.symbol:
        symbols = [s.strip().upper() for s in args.symbol.split(',')]
    else:
        symbols = get_all_symbols(sector=args.sector)
    
    if args.limit:
        symbols = symbols[:args.limit]
    
    # Resume support
    completed = set()
    existing_results = {'predictions': {}, 'errors': [], 'metadata': {}}
    
    if args.resume and output_file.exists():
        completed = get_completed_symbols(output_file)
        try:
            with open(output_file, 'r') as f:
                existing_results = json.load(f)
        except:
            pass
        symbols = [s for s in symbols if s not in completed]
        print(f"📂 Resuming: {len(completed)} already done, {len(symbols)} remaining")
    
    if not symbols:
        print("✅ All symbols already processed!")
        return
    
    print(f"\n{'='*70}")
    print(f"🚀 BATCH STOCK PREDICTION")
    print(f"{'='*70}")
    print(f"Symbols to process: {len(symbols)}")
    print(f"Config: {args.n_models} models, {args.epochs} epochs, {args.hidden_size}h, {args.num_blocks}b")
    print(f"Output: {output_file}")
    print(f"{'='*70}\n")
    
    # Estimate time
    est_time_per_stock = 90 if not args.fast else 30  # seconds
    est_total = len(symbols) * est_time_per_stock / 60
    print(f"⏱️  Estimated time: {est_total:.0f} minutes ({est_total/60:.1f} hours)\n")
    
    results = existing_results
    results['metadata'] = {
        'run_date': datetime.now().isoformat(),
        'config': {
            'n_models': args.n_models,
            'epochs': args.epochs,
            'hidden_size': args.hidden_size,
            'num_blocks': args.num_blocks,
            'lookback': 120,
        },
        'total_symbols': len(symbols) + len(completed),
    }
    
    start_time = time.time()
    
    for i, symbol in enumerate(symbols, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(symbols)}] Processing {symbol}...")
        print(f"{'='*70}")
        
        symbol_start = time.time()
        result = run_prediction(
            symbol=symbol,
            n_models=args.n_models,
            epochs=args.epochs,
            hidden_size=args.hidden_size,
            num_blocks=args.num_blocks,
        )
        symbol_time = time.time() - symbol_start
        
        if result and 'error' not in result:
            result['processing_time'] = symbol_time
            results['predictions'][symbol] = result
            print(f"✅ {symbol} completed in {symbol_time:.1f}s")
        else:
            results['errors'].append({
                'symbol': symbol,
                'error': result.get('error', 'Unknown error') if result else 'No result',
                'time': datetime.now().isoformat(),
            })
            print(f"❌ {symbol} failed")
        
        # Save after each symbol (for resume support)
        save_results(output_file, results)
        
        # Progress update
        elapsed = time.time() - start_time
        avg_time = elapsed / i
        remaining = (len(symbols) - i) * avg_time / 60
        print(f"📊 Progress: {i}/{len(symbols)} | Avg: {avg_time:.1f}s/stock | ETA: {remaining:.0f} min")
    
    # Final summary
    total_time = time.time() - start_time
    successful = len(results['predictions'])
    failed = len(results['errors'])
    
    results['metadata']['total_time_seconds'] = total_time
    results['metadata']['successful'] = successful
    results['metadata']['failed'] = failed
    
    save_results(output_file, results)
    
    print(f"\n{'='*70}")
    print(f"✅ BATCH PREDICTION COMPLETE")
    print(f"{'='*70}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
