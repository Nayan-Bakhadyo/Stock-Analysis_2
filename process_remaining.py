"""
Process remaining unanalyzed stocks one at a time
Allows stopping and resuming without losing progress
"""
import json
import sys
import time
from simple_analysis import analyze_stock
from stock_tracker import StockTracker


def get_remaining_stocks(skip_failed_insufficient_data=True):
    """
    Get list of stocks not yet analyzed
    
    Args:
        skip_failed_insufficient_data: Skip stocks that failed due to insufficient data
    """
    # Load all available stocks
    with open('all_stocks_tracker.json', 'r') as f:
        tracker_data = json.load(f)
    all_symbols = set(tracker_data['all_symbols'])
    
    # Load already analyzed stocks
    try:
        with open('analysis_results.json', 'r') as f:
            analyzed = json.load(f)
        
        # Separate successful and failed
        analyzed_symbols = set()
        failed_insufficient = set()
        
        for stock in analyzed:
            analyzed_symbols.add(stock['symbol'])
            # Skip stocks with insufficient data error
            if skip_failed_insufficient_data and stock.get('error'):
                if 'Insufficient historical data' in stock.get('error', ''):
                    failed_insufficient.add(stock['symbol'])
        
        # Don't include failed insufficient data stocks in analyzed count
        if skip_failed_insufficient_data:
            effective_analyzed = analyzed_symbols - failed_insufficient
        else:
            effective_analyzed = analyzed_symbols
            
    except:
        analyzed_symbols = set()
        effective_analyzed = set()
    
    # Get remaining
    remaining = sorted(all_symbols - analyzed_symbols)
    return remaining, len(effective_analyzed), len(all_symbols)


def process_one_stock():
    """Process next unanalyzed stock"""
    # Get remaining stocks
    remaining, analyzed_count, total_count = get_remaining_stocks()
    
    if not remaining:
        print("\n✅ All stocks have been analyzed!")
        print(f"Total: {total_count}/{total_count} stocks completed")
        return False
    
    # Show progress
    print(f"\n{'='*70}")
    print(f"PROCESSING REMAINING STOCKS")
    print(f"{'='*70}")
    print(f"Progress: {analyzed_count}/{total_count} stocks analyzed")
    print(f"Remaining: {len(remaining)} stocks")
    print(f"Next stock: {remaining[0]}")
    print(f"{'='*70}\n")
    
    # Load existing results
    try:
        with open('analysis_results.json', 'r') as f:
            existing_results = json.load(f)
        existing_dict = {r['symbol']: r for r in existing_results}
    except:
        existing_dict = {}
    
    # Initialize tracker
    tracker = StockTracker()
    
    # Process next stock
    symbol = remaining[0]
    
    try:
        result = analyze_stock(symbol, time_horizon='short', tracker=tracker)
        
        # Add to results
        existing_dict[symbol] = result
        
        # Save immediately
        all_results = list(existing_dict.values())
        with open('analysis_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n✅ {symbol} completed and saved!")
        print(f"💾 Progress: {analyzed_count + 1}/{total_count} stocks")
        
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user. Progress saved.")
        return False
    except Exception as e:
        print(f"\n❌ Error processing {symbol}: {e}")
        # Still save the error result
        existing_dict[symbol] = {
            'symbol': symbol,
            'error': str(e),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        all_results = list(existing_dict.values())
        with open('analysis_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        tracker.mark_processed(symbol, status='failed', error=str(e))
        return True


def process_batch(count=None):
    """
    Process multiple stocks in sequence
    
    Args:
        count: Number of stocks to process (None = all remaining)
    """
    processed = 0
    
    while True:
        # Check if we should stop
        if count is not None and processed >= count:
            print(f"\n✅ Batch complete: Processed {processed} stocks")
            break
        
        # Process one stock
        should_continue = process_one_stock()
        
        if not should_continue:
            break
        
        processed += 1
        
        # Small delay between stocks
        print("\n⏳ Waiting 2 seconds before next stock...")
        time.sleep(2)
    
    # Final summary
    remaining, analyzed_count, total_count = get_remaining_stocks()
    print(f"\n{'='*70}")
    print(f"SESSION COMPLETE")
    print(f"{'='*70}")
    print(f"Total analyzed: {analyzed_count}/{total_count}")
    print(f"Remaining: {len(remaining)}")
    if remaining[:10]:
        print(f"Next 10 to process: {', '.join(remaining[:10])}")
    print(f"{'='*70}\n")


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        try:
            count = int(sys.argv[1])
            print(f"📊 Processing {count} stocks...")
            process_batch(count)
        except ValueError:
            print("Usage: python3 process_remaining.py [count]")
            print("  count: Number of stocks to process (optional, default=1)")
            sys.exit(1)
    else:
        # Default: process one stock
        process_one_stock()


if __name__ == '__main__':
    main()
