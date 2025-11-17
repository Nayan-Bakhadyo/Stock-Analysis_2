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
from trading_insights import TradingInsightsEngine
from stock_tracker import StockTracker


def weekly_update(retrain_all=True, max_model_age_days=7):
    """
    Perform weekly comprehensive update
    - Full price history sync
    - Full news sync
    - Retrain ML models
    
    Args:
        retrain_all: If True, retrain all models. If False, only retrain stale models
        max_model_age_days: Maximum model age before retraining (default 7 days)
    """
    print("\n" + "="*70)
    print("WEEKLY STOCK ANALYSIS UPDATE")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    print(f"Retrain strategy: {'ALL models' if retrain_all else f'Models older than {max_model_age_days} days'}")
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
    insights_engine = TradingInsightsEngine()
    updated_results = {}
    
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
            # Perform update
            result = insights_engine.calculate_profitability_probability(
                symbol=symbol,
                time_horizon='short',
                include_broker_analysis=False,
                use_cache=True,
                reuse_ml_model=not should_retrain  # Retrain if should_retrain=True
            )
            
            if result.get('error'):
                print(f"  ❌ Error: {result['error'][:60]}...")
                updated_results[symbol] = result
                error_count += 1
            else:
                print(f"  ✅ Updated successfully")
                updated_results[symbol] = result
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
            all_results = list(updated_results.values()) + failed_stocks
            with open('analysis_results.json', 'w') as f:
                json.dump(all_results, f, indent=2)
        
        # Small delay to avoid rate limiting
        if should_retrain:
            time.sleep(1)
    
    # Final save
    print(f"\n💾 Saving final results...")
    all_results = list(updated_results.values()) + failed_stocks
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
    
    if '--help' in sys.argv or '-h' in sys.argv:
        print("\nUsage: python3 weekly_update.py [options]")
        print("\nOptions:")
        print("  --retrain-all    Retrain all ML models (slower, more accurate)")
        print("  (default)        Only retrain models older than 7 days")
        print("\nExamples:")
        print("  python3 weekly_update.py                # Smart retraining")
        print("  python3 weekly_update.py --retrain-all  # Force retrain all")
        print()
    else:
        weekly_update(retrain_all=retrain_all)
