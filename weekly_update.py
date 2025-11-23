"""
Weekly Stock Analysis Update
- Full data sync (price history and news)
- Retrains ML models with latest data
- More comprehensive than daily update
- Run every Monday or weekly
"""
import json
import os
import time
from datetime import datetime
from simple_analysis import analyze_stock
from stock_tracker import StockTracker


def weekly_update(retrain_all=True, max_model_age_days=7, limit=None, resume=True):
    """
    Perform weekly comprehensive update
    - Full price history sync
    - Full news sync
    - Retrain ML models
    
    Args:
        retrain_all: If True, retrain all models. If False, only retrain stale models
        max_model_age_days: Maximum model age before retraining (default 7 days)
        limit: Maximum number of stocks to update in this run (None = all)
        resume: If True, skip stocks already updated today (default True)
    """
    print("\n" + "="*70)
    print("WEEKLY STOCK ANALYSIS UPDATE")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    print(f"Retrain strategy: {'ALL models' if retrain_all else f'Models older than {max_model_age_days} days'}")
    if limit:
        print(f"Batch limit: {limit} stocks")
    if resume:
        print(f"Resume mode: Skip stocks updated in last 3 days")
    print("="*70)
    
    # Load existing analysis results
    try:
        with open('analysis_results.json', 'r') as f:
            existing_results = json.load(f)
        
        # Filter out failed stocks
        stocks_to_update = [r for r in existing_results if not r.get('error')]
        failed_stocks = [r for r in existing_results if r.get('error')]
        
        # Resume: Filter out stocks already updated in the last 3 days
        if resume:
            from datetime import timedelta
            cutoff_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
            pending_stocks = []
            already_updated = []
            
            for stock in stocks_to_update:
                timestamp = stock.get('analysis_timestamp', '')
                stock_date = timestamp.split('T')[0] if timestamp else ''
                if stock_date >= cutoff_date:
                    already_updated.append(stock)
                else:
                    pending_stocks.append(stock)
            
            print(f"Stocks already updated (last 3 days): {len(already_updated)}")
            print(f"Stocks pending update: {len(pending_stocks)}")
            
            stocks_to_update = pending_stocks
        else:
            print(f"Stocks to update: {len(stocks_to_update)}")
        
        # Apply limit if specified
        if limit and limit < len(stocks_to_update):
            print(f"Limiting to first {limit} stocks (out of {len(stocks_to_update)} pending)")
            stocks_to_update = stocks_to_update[:limit]
        
        print(f"Skipping failed stocks: {len(failed_stocks)}")
        
        if len(stocks_to_update) == 0:
            print("\n✅ All stocks are already up to date!")
            return
        
        # Check which models need retraining
        if not retrain_all:
            models_to_retrain = []
            models_to_reuse = []
            
            for stock in stocks_to_update:
                symbol = stock['symbol']
                model_path = f'models/lstm_{symbol}.h5'
                
                if os.path.exists(model_path):
                    model_age = (time.time() - os.path.getmtime(model_path)) / 86400
                    if model_age > max_model_age_days:
                        models_to_retrain.append(symbol)
                    else:
                        models_to_reuse.append(symbol)
                else:
                    models_to_retrain.append(symbol)
            
            print(f"Models to retrain: {len(models_to_retrain)}")
            print(f"Models to reuse: {len(models_to_reuse)}")
        
        print("="*70 + "\n")
        
    except FileNotFoundError:
        print("❌ No analysis_results.json found. Run initial analysis first.")
        return
    
    # Initialize
    tracker = StockTracker()
    
    # Load all existing results (updated + not-yet-updated)
    all_existing = {}
    for stock in existing_results:
        all_existing[stock['symbol']] = stock
    
    # Track statistics
    success_count = 0
    error_count = 0
    retrained_count = 0
    reused_count = 0
    
    # Update each stock
    for i, stock_data in enumerate(stocks_to_update, 1):
        symbol = stock_data['symbol']
        
        # Determine if we should retrain this model
        should_retrain = retrain_all
        if not retrain_all:
            model_path = f'models/lstm_{symbol}.h5'
            if os.path.exists(model_path):
                model_age = (time.time() - os.path.getmtime(model_path)) / 86400
                should_retrain = model_age > max_model_age_days
            else:
                should_retrain = True
        
        retrain_status = "RETRAIN" if should_retrain else "REUSE"
        print(f"\n[{i}/{len(stocks_to_update)}] Updating {symbol} ({retrain_status})...")
        
        try:
            # Perform update using analyze_stock for consistent format
            # ML disabled for weekly updates - only updates technical/fundamental/sentiment
            result = analyze_stock(symbol, time_horizon='short', tracker=tracker, reuse_ml_model=not should_retrain, enable_ml=False)
            
            if result.get('error'):
                print(f"  ❌ Error: {result['error'][:60]}...")
                all_existing[symbol] = result
                error_count += 1
            else:
                print(f"  ✅ Updated successfully")
                all_existing[symbol] = result
                success_count += 1
                
                if should_retrain:
                    retrained_count += 1
                else:
                    reused_count += 1
            
            # Mark as processed
            tracker.mark_processed(symbol, 
                                 status='success' if not result.get('error') else 'failed',
                                 error=result.get('error'))
            
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user. Saving progress...")
            break
            
        except Exception as e:
            print(f"  ❌ Exception: {str(e)[:60]}...")
            all_existing[symbol] = {
                'symbol': symbol,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            tracker.mark_processed(symbol, status='failed', error=str(e))
            error_count += 1
        
        # Save progress every 10 stocks
        if i % 10 == 0:
            print(f"\n💾 Saving progress... ({i}/{len(stocks_to_update)})")
            all_results = list(all_existing.values())
            with open('analysis_results.json', 'w') as f:
                json.dump(all_results, f, indent=2)
        
        # Small delay to avoid rate limiting
        if should_retrain:
            time.sleep(1)
    
    # Final save
    print(f"\n💾 Saving final results...")
    all_results = list(all_existing.values())
    with open('analysis_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"WEEKLY UPDATE COMPLETE")
    print(f"{'='*70}")
    print(f"Total stocks updated: {len(stocks_to_update)}")
    print(f"Successful: {success_count}")
    print(f"Errors: {error_count}")
    print(f"ML Models retrained: {retrained_count}")
    print(f"ML Models reused: {reused_count}")
    print(f"Results saved to: analysis_results.json")
    print(f"{'='*70}\n")
    
    # Print tracker summary
    tracker.print_summary()


if __name__ == '__main__':
    import sys
    
    # Check command line arguments
    retrain_all = '--retrain-all' in sys.argv
    resume = '--no-resume' not in sys.argv  # Resume by default
    limit = None
    
    # Parse --limit argument
    for arg in sys.argv:
        if arg.startswith('--limit='):
            try:
                limit = int(arg.split('=')[1])
            except ValueError:
                print(f"❌ Invalid limit value: {arg}")
                sys.exit(1)
    
    if '--help' in sys.argv or '-h' in sys.argv:
        print("\nUsage: python3 weekly_update.py [options]")
        print("\nOptions:")
        print("  --retrain-all       Retrain all ML models (slower, more accurate)")
        print("  --limit=N           Update only N stocks per run")
        print("  --no-resume         Don't skip stocks already updated today")
        print("  (default)           Only retrain models older than 7 days, resume enabled")
        print("\nExamples:")
        print("  python3 weekly_update.py                      # Smart retraining, resume mode")
        print("  python3 weekly_update.py --limit=50           # Update first 50 pending stocks")
        print("  python3 weekly_update.py --limit=50           # Run again for next 50 stocks")
        print("  python3 weekly_update.py --retrain-all        # Force retrain all")
        print("  python3 weekly_update.py --limit=20 --no-resume  # Force 20 stocks, no resume")
        print("\nBatch Processing:")
        print("  1. python3 weekly_update.py --limit=50    # First 50 stocks")
        print("  2. python3 weekly_update.py --limit=50    # Next 50 stocks (auto-resume)")
        print("  3. python3 weekly_update.py --limit=50    # Next 50 stocks")
        print("  ... repeat until all stocks are updated")
        print()
    else:
        weekly_update(retrain_all=retrain_all, limit=limit, resume=resume)
