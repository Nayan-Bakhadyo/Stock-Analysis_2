"""
Daily Stock Analysis Update
- Syncs latest price history (incremental)
- Syncs latest news
- Reuses existing ML models (no retraining)
- Fast updates for all analyzed stocks
"""
import json
import os
from datetime import datetime
from simple_analysis import analyze_stock
from stock_tracker import StockTracker


def daily_update():
    """
    Perform daily update on all analyzed stocks
    - Updates price data (incremental sync)
    - Updates news sentiment
    - Reuses ML models (fast)
    """
    print("\n" + "="*70)
    print("DAILY STOCK ANALYSIS UPDATE")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Load existing analysis results
    try:
        with open('analysis_results.json', 'r') as f:
            existing_results = json.load(f)
        
        # Filter out failed stocks
        stocks_to_update = [r for r in existing_results if not r.get('error')]
        failed_stocks = [r for r in existing_results if r.get('error')]
        
        print(f"Stocks to update: {len(stocks_to_update)}")
        print(f"Skipping failed stocks: {len(failed_stocks)}")
        print("="*70 + "\n")
        
    except FileNotFoundError:
        print("❌ No analysis_results.json found. Run initial analysis first.")
        return
    
    # Initialize
    tracker = StockTracker()
    updated_results = {}
    
    # Track statistics
    success_count = 0
    error_count = 0
    
    # Update each stock
    for i, stock_data in enumerate(stocks_to_update, 1):
        symbol = stock_data['symbol']
        
        print(f"\n[{i}/{len(stocks_to_update)}] Updating {symbol}...")
        
        try:
            # Perform update using analyze_stock for consistent format
            # ML disabled for daily updates - only updates technical/fundamental/sentiment
            result = analyze_stock(symbol, time_horizon='short', tracker=tracker, reuse_ml_model=True, enable_ml=False)
            
            if result.get('error'):
                print(f"  ❌ Error: {result['error'][:60]}...")
                updated_results[symbol] = result
                error_count += 1
            else:
                print(f"  ✅ Updated successfully")
                updated_results[symbol] = result
                success_count += 1
            
            # Mark as processed
            tracker.mark_processed(symbol, 
                                 status='success' if not result.get('error') else 'failed',
                                 error=result.get('error'))
            
        except Exception as e:
            print(f"  ❌ Exception: {str(e)[:60]}...")
            updated_results[symbol] = {
                'symbol': symbol,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            tracker.mark_processed(symbol, status='failed', error=str(e))
            error_count += 1
        
        # Save progress every 10 stocks
        if i % 10 == 0:
            print(f"\n💾 Saving progress... ({i}/{len(stocks_to_update)})")
            # Merge with failed stocks (keep them as-is)
            all_results = list(updated_results.values()) + failed_stocks
            with open('analysis_results.json', 'w') as f:
                json.dump(all_results, f, indent=2)
    
    # Final save
    print(f"\n💾 Saving final results...")
    all_results = list(updated_results.values()) + failed_stocks
    with open('analysis_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"DAILY UPDATE COMPLETE")
    print(f"{'='*70}")
    print(f"Total stocks updated: {len(stocks_to_update)}")
    print(f"Successful: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Results saved to: analysis_results.json")
    print(f"{'='*70}\n")
    
    # Print tracker summary
    tracker.print_summary()


if __name__ == '__main__':
    daily_update()
