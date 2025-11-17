"""
Retry failed stock analyses
Some stocks fail due to insufficient data or temporary errors
This script identifies and retries them
"""
import json
import sys
from simple_analysis import analyze_stock
from stock_tracker import StockTracker
import time


def get_failed_stocks():
    """Get list of stocks that failed analysis"""
    try:
        with open('analysis_results.json', 'r') as f:
            results = json.load(f)
        
        failed = [r['symbol'] for r in results if r.get('error')]
        return failed
    except:
        return []


def get_failed_stock_details():
    """Get detailed info about failed stocks"""
    try:
        with open('analysis_results.json', 'r') as f:
            results = json.load(f)
        
        failed = [(r['symbol'], r.get('error', 'Unknown error')) for r in results if r.get('error')]
        return failed
    except:
        return []


def retry_failed_stocks(skip_insufficient_data=True):
    """
    Retry all failed stock analyses
    
    Args:
        skip_insufficient_data: If True, skip stocks that failed due to insufficient data
    """
    failed_details = get_failed_stock_details()
    
    if not failed_details:
        print("✅ No failed stocks to retry!")
        return
    
    print(f"\n{'='*70}")
    print(f"RETRYING FAILED STOCK ANALYSES")
    print(f"{'='*70}")
    print(f"Total failed stocks: {len(failed_details)}\n")
    
    # Filter out insufficient data errors if requested
    to_retry = []
    skipped = []
    
    for symbol, error in failed_details:
        if skip_insufficient_data and 'Insufficient historical data' in error:
            skipped.append(symbol)
        else:
            to_retry.append(symbol)
    
    if skipped:
        print(f"⏭️  Skipping {len(skipped)} stocks with insufficient data:")
        print(f"   {', '.join(skipped[:20])}")
        if len(skipped) > 20:
            print(f"   ... and {len(skipped) - 20} more")
        print()
    
    if not to_retry:
        print("ℹ️  No stocks to retry (all have insufficient data)")
        print("💡 Tip: Run with --retry-all to attempt them anyway")
        return
    
    print(f"🔄 Retrying {len(to_retry)} stocks...\n")
    
    # Load existing results
    with open('analysis_results.json', 'r') as f:
        results = json.load(f)
    results_dict = {r['symbol']: r for r in results}
    
    # Initialize tracker
    tracker = StockTracker()
    
    # Retry each stock
    success_count = 0
    still_failed_count = 0
    
    for i, symbol in enumerate(to_retry, 1):
        print(f"\n[{i}/{len(to_retry)}] Retrying {symbol}...")
        
        try:
            result = analyze_stock(symbol, time_horizon='short', tracker=tracker)
            
            if result.get('error'):
                print(f"  ❌ Still failed: {result['error'][:60]}...")
                still_failed_count += 1
            else:
                print(f"  ✅ Success!")
                success_count += 1
            
            # Update results
            results_dict[symbol] = result
            
            # Save after each retry
            with open('analysis_results.json', 'w') as f:
                json.dump(list(results_dict.values()), f, indent=2)
            
            print(f"  💾 Saved")
            
            # Small delay
            time.sleep(2)
            
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
            break
        except Exception as e:
            print(f"  ❌ Error: {e}")
            still_failed_count += 1
    
    # Summary
    print(f"\n{'='*70}")
    print(f"RETRY SUMMARY")
    print(f"{'='*70}")
    print(f"Attempted: {success_count + still_failed_count}")
    print(f"Now successful: {success_count}")
    print(f"Still failed: {still_failed_count}")
    print(f"Skipped (insufficient data): {len(skipped)}")
    print(f"{'='*70}\n")


def show_failed_stats():
    """Show statistics about failed stocks"""
    failed_details = get_failed_stock_details()
    
    if not failed_details:
        print("✅ No failed stocks!")
        return
    
    # Categorize errors
    error_types = {}
    for symbol, error in failed_details:
        error_key = error.split(':')[0] if ':' in error else error[:50]
        if error_key not in error_types:
            error_types[error_key] = []
        error_types[error_key].append(symbol)
    
    print(f"\n{'='*70}")
    print(f"FAILED STOCKS SUMMARY")
    print(f"{'='*70}")
    print(f"Total failed: {len(failed_details)}\n")
    
    for error_type, symbols in sorted(error_types.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"📊 {error_type}")
        print(f"   Count: {len(symbols)}")
        print(f"   Stocks: {', '.join(symbols[:10])}")
        if len(symbols) > 10:
            print(f"   ... and {len(symbols) - 10} more")
        print()
    
    print(f"{'='*70}\n")


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        if sys.argv[1] == '--show':
            show_failed_stats()
        elif sys.argv[1] == '--retry-all':
            retry_failed_stocks(skip_insufficient_data=False)
        elif sys.argv[1] == '--retry':
            retry_failed_stocks(skip_insufficient_data=True)
        else:
            print("Usage:")
            print("  python3 retry_failed.py --show         # Show failed stocks")
            print("  python3 retry_failed.py --retry        # Retry (skip insufficient data)")
            print("  python3 retry_failed.py --retry-all    # Retry all failed stocks")
    else:
        show_failed_stats()


if __name__ == '__main__':
    main()
