"""
Daily Update Master Script
- Runs all three daily update modules
- Price data -> Sentiment -> Fundamental data
- Incremental updates only
"""
import sys
from datetime import datetime


def run_daily_updates(symbols=None):
    """
    Run all daily update modules in sequence
    
    Args:
        symbols: List of symbols to update (None = update all from analysis_results.json)
    """
    print("\n" + "="*70)
    print("MASTER DAILY UPDATE - ALL MODULES")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Import modules
    from update_price_data import update_price_data
    from update_sentiment import update_sentiment_analysis
    from update_fundamental_data import update_fundamental_data
    
    # Module 1: Update price data (fastest, no browser needed)
    print("\n" + "🔷"*35)
    print("MODULE 1: PRICE DATA UPDATE")
    print("🔷"*35)
    
    try:
        update_price_data(symbols)
        print("✅ Price data update completed\n")
    except Exception as e:
        print(f"❌ Price data update failed: {e}\n")
    
    # Module 2: Update sentiment (requires news scraping)
    print("\n" + "🔷"*35)
    print("MODULE 2: SENTIMENT ANALYSIS UPDATE")
    print("🔷"*35)
    
    try:
        update_sentiment_analysis(symbols, max_articles=10)
        print("✅ Sentiment analysis update completed\n")
    except Exception as e:
        print(f"❌ Sentiment analysis update failed: {e}\n")
    
    # Module 3: Update fundamental data (requires browser scraping)
    print("\n" + "🔷"*35)
    print("MODULE 3: FUNDAMENTAL DATA UPDATE")
    print("🔷"*35)
    
    try:
        update_fundamental_data(symbols)
        print("✅ Fundamental data update completed\n")
    except Exception as e:
        print(f"❌ Fundamental data update failed: {e}\n")
    
    # Final summary
    print("\n" + "="*70)
    print("MASTER DAILY UPDATE COMPLETE")
    print("="*70)
    print("\nAll three modules executed:")
    print("  1. ✅ Price data update")
    print("  2. ✅ Sentiment analysis update")
    print("  3. ✅ Fundamental data update")
    print("\nOutput files:")
    print("  • sentiment_results.json - Latest sentiment scores")
    print("  • fundamental_results.json - Latest fundamental data")
    print("  • analysis_results.json - Price data updated in database")
    print("="*70 + "\n")


if __name__ == '__main__':
    # Check for command line arguments
    if len(sys.argv) > 1:
        symbols = [s.strip().upper() for s in sys.argv[1:]]
        print(f"Updating specific symbols: {', '.join(symbols)}\n")
        run_daily_updates(symbols)
    else:
        # Update all stocks from analysis_results.json
        print("Updating all stocks from analysis_results.json\n")
        run_daily_updates()
