"""
Daily Fundamental Data Update
- Updates fundamental analysis (P/E, P/B, ROE, etc.)
- Scrapes latest financial data from NepalAlpha
- Incremental update strategy
"""
import json
import os
from datetime import datetime
from nepsealpha_scraper import NepalAlphaScraper
from fundamental_analyzer import FundamentalAnalyzer
from stock_tracker import StockTracker


def update_fundamental_data(symbols=None):
    """
    Update fundamental data for all analyzed stocks
    - Scrapes latest financial ratios
    - Updates fundamental analysis scores
    
    Args:
        symbols: List of symbols to update (None = update all from analysis_results.json)
    """
    print("\n" + "="*70)
    print("FUNDAMENTAL DATA UPDATE")
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
    scraper = NepalAlphaScraper(headless=True)
    analyzer = FundamentalAnalyzer()
    tracker = StockTracker()
    
    # Track statistics
    success_count = 0
    error_count = 0
    fundamental_results = {}
    
    try:
        # Update each stock
        for i, symbol in enumerate(symbols, 1):
            print(f"\n[{i}/{len(symbols)}] Updating fundamental data: {symbol}")
            
            try:
                # Scrape fundamental data
                print(f"  🔍 Scraping fundamental data...")
                fundamental_data = scraper.scrape_fundamental_data(symbol)
                
                if not fundamental_data or fundamental_data.get('error'):
                    error_msg = fundamental_data.get('error', 'No data available') if fundamental_data else 'No data available'
                    print(f"  ⚠️ {error_msg}")
                    tracker.mark_processed(symbol, status='no_data', error=error_msg)
                    continue
                
                # Analyze fundamental data
                print(f"  📊 Analyzing fundamentals...")
                analysis = analyzer.comprehensive_analysis(fundamental_data)
                
                # Display key metrics
                print(f"  ✅ P/E Ratio: {fundamental_data.get('pe_ratio', 'N/A')}")
                print(f"  ✅ P/B Ratio: {fundamental_data.get('pb_ratio', 'N/A')}")
                print(f"  ✅ ROE: {fundamental_data.get('roe', 'N/A')}%")
                print(f"  ✅ EPS: {fundamental_data.get('eps', 'N/A')}")
                print(f"  📈 Overall Score: {analysis.get('overall_score', 0):.1f}/100")
                print(f"  🏷️ Rating: {analysis.get('overall_rating', 'N/A')}")
                
                # Store results
                fundamental_results[symbol] = {
                    'data': fundamental_data,
                    'analysis': analysis,
                    'timestamp': datetime.now().isoformat()
                }
                
                success_count += 1
                
                # Mark as processed
                tracker.mark_processed(symbol, status='success')
                
            except Exception as e:
                print(f"  ❌ Error: {str(e)[:80]}...")
                tracker.mark_processed(symbol, status='failed', error=str(e))
                error_count += 1
            
            # Small delay to avoid overwhelming the server
            if i < len(symbols):  # Don't delay after last stock
                import time
                time.sleep(1)
    
    finally:
        # Close scraper
        scraper.close_driver()
    
    # Save fundamental results
    output_file = 'fundamental_results.json'
    with open(output_file, 'w') as f:
        json.dump(fundamental_results, f, indent=2)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"FUNDAMENTAL DATA UPDATE COMPLETE")
    print(f"{'='*70}")
    print(f"Total stocks processed: {len(symbols)}")
    print(f"Successful: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}\n")
    
    # Print tracker summary
    tracker.print_summary()


if __name__ == '__main__':
    import sys
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        symbols = [s.strip().upper() for s in sys.argv[1:]]
        update_fundamental_data(symbols)
    else:
        # Update all stocks from analysis_results.json
        update_fundamental_data()
