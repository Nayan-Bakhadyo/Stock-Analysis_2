"""
Daily Price Data Update
- Incremental sync of price history
- Only fetches new data (days since last update)
- Fast and lightweight
"""
import json
import os
from datetime import datetime
from sync_manager import SyncManager
from stock_tracker import StockTracker


def update_price_data(symbols=None):
    """
    Update price history for all analyzed stocks
    - Incremental sync only (fetches new data)
    - Fast and efficient
    
    Args:
        symbols: List of symbols to update (None = update all from analysis_results.json)
    """
    print("\n" + "="*70)
    print("PRICE DATA UPDATE (Incremental)")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Get symbols to update
    if symbols is None:
        try:
            with open('analysis_results.json', 'r') as f:
                existing_results = json.load(f)
            
            # Filter out failed stocks
            symbols = [r['symbol'] for r in existing_results if not r.get('error')]
            
            print(f"Found {len(symbols)} stocks to update")
            print("="*70 + "\n")
            
        except FileNotFoundError:
            print("❌ No analysis_results.json found. Provide symbols manually.")
            return
    
    # Initialize
    sync_manager = SyncManager()
    tracker = StockTracker()
    
    # Track statistics
    success_count = 0
    error_count = 0
    total_records_added = 0
    
    # Update each stock
    for i, symbol in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] Updating price data: {symbol}")
        
        try:
            # Sync price history (incremental)
            result = sync_manager.sync_price_history(symbol, force_full=False)
            
            print(f"  ✅ Status: {result['status']}")
            print(f"  ✅ Records added: {result['records_added']}")
            print(f"  ✅ Total records: {result['total_records']}")
            
            total_records_added += result['records_added']
            success_count += 1
            
            # Mark as processed
            tracker.mark_processed(symbol, status='success')
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)[:80]}...")
            tracker.mark_processed(symbol, status='failed', error=str(e))
            error_count += 1
    
    # Summary
    print(f"\n{'='*70}")
    print(f"PRICE DATA UPDATE COMPLETE")
    print(f"{'='*70}")
    print(f"Total stocks processed: {len(symbols)}")
    print(f"Successful: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Total new records added: {total_records_added}")
    print(f"{'='*70}\n")
    
    # Print tracker summary
    tracker.print_summary()


if __name__ == '__main__':
    import sys
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        symbols = [s.strip().upper() for s in sys.argv[1:]]
        update_price_data(symbols)
    else:
        # Update all stocks from analysis_results.json
        update_price_data()
