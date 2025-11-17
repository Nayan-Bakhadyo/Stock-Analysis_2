"""
Fetch all stocks from NEPSE and ShareSansar today's price endpoint
"""
import json
import pandas as pd
from datetime import datetime
from symbol_scraper import ShareSansarSymbolScraper
from data_fetcher import NepseDataFetcher
from stock_tracker import StockTracker


def fetch_all_stocks_from_sharesansar():
    """
    Fetch all stocks from ShareSansar today's price page
    
    Returns:
        DataFrame with columns: symbol, company_name, ltp, sector
    """
    print("\n" + "="*70)
    print("FETCHING ALL STOCKS FROM SHARESANSAR")
    print("="*70)
    
    scraper = ShareSansarSymbolScraper(headless=True)
    
    try:
        scraper.setup_driver()
        df = scraper.scrape_all_symbols()
        
        if not df.empty:
            print(f"\n✅ Successfully fetched {len(df)} stocks from ShareSansar")
            print(f"Sample stocks: {', '.join(df['symbol'].head(10).tolist())}")
            
            # Save to CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_file = f"all_stocks_sharesansar_{timestamp}.csv"
            df.to_csv(csv_file, index=False)
            print(f"✅ Saved to: {csv_file}")
            
            return df
        else:
            print("❌ No stocks found")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"❌ Error fetching from ShareSansar: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()
        
    finally:
        scraper.close_driver()


def fetch_all_stocks_from_nepse():
    """
    Fetch all stocks from NEPSE database
    
    Returns:
        DataFrame with columns: symbol, latest_price, latest_date
    """
    print("\n" + "="*70)
    print("FETCHING ALL STOCKS FROM NEPSE DATABASE")
    print("="*70)
    
    try:
        fetcher = NepseDataFetcher()
        
        # Get all symbols from database
        symbols = fetcher.get_all_symbols()
        
        if not symbols:
            print("❌ No symbols found in database")
            return pd.DataFrame()
        
        print(f"✅ Found {len(symbols)} stocks in NEPSE database")
        
        # Get latest price for each stock
        stocks_data = []
        
        for symbol in symbols:
            try:
                # Get latest price from database
                price_history = fetcher.get_stock_price_history(symbol, days=1)
                
                if not price_history.empty:
                    latest = price_history.iloc[-1]
                    stocks_data.append({
                        'symbol': symbol,
                        'latest_price': float(latest['close']),
                        'latest_date': str(latest['date']),
                        'volume': float(latest.get('volume', 0)),
                        'open': float(latest.get('open', 0)),
                        'high': float(latest.get('high', 0)),
                        'low': float(latest.get('low', 0))
                    })
                    
            except Exception as e:
                print(f"  ⚠️ Error for {symbol}: {e}")
                continue
        
        df = pd.DataFrame(stocks_data)
        
        if not df.empty:
            print(f"\n✅ Successfully fetched price data for {len(df)} stocks")
            print(f"Sample stocks: {', '.join(df['symbol'].head(10).tolist())}")
            
            # Save to CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_file = f"all_stocks_nepse_{timestamp}.csv"
            df.to_csv(csv_file, index=False)
            print(f"✅ Saved to: {csv_file}")
            
            return df
        else:
            print("❌ No price data found")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"❌ Error fetching from NEPSE: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def merge_and_save_all_stocks(sharesansar_df, nepse_df):
    """
    Merge ShareSansar and NEPSE data and save combined list
    
    Args:
        sharesansar_df: DataFrame from ShareSansar
        nepse_df: DataFrame from NEPSE
    """
    print("\n" + "="*70)
    print("MERGING AND SAVING ALL STOCKS")
    print("="*70)
    
    # Start with ShareSansar data (has live prices)
    if not sharesansar_df.empty:
        merged = sharesansar_df.copy()
        
        # Add NEPSE data if available
        if not nepse_df.empty:
            # Merge on symbol
            merged = merged.merge(
                nepse_df[['symbol', 'latest_date', 'volume', 'open', 'high', 'low']], 
                on='symbol', 
                how='outer'
            )
            
            # Use NEPSE price if ShareSansar LTP is missing
            merged['price'] = merged.apply(
                lambda row: row['ltp'] if pd.notna(row.get('ltp')) else row.get('latest_price', 0),
                axis=1
            )
        else:
            merged['price'] = merged['ltp']
    
    elif not nepse_df.empty:
        # Only NEPSE data available
        merged = nepse_df.copy()
        merged['price'] = merged['latest_price']
    
    else:
        print("❌ No data to merge")
        return None
    
    # Clean up and sort
    merged = merged.sort_values('symbol')
    
    # Save to multiple formats
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. CSV file
    csv_file = f"all_nepse_stocks_{timestamp}.csv"
    merged.to_csv(csv_file, index=False)
    print(f"✅ Saved CSV: {csv_file}")
    
    # 2. JSON file
    json_file = f"all_nepse_stocks_{timestamp}.json"
    merged_dict = merged.to_dict('records')
    with open(json_file, 'w') as f:
        json.dump(merged_dict, f, indent=2)
    print(f"✅ Saved JSON: {json_file}")
    
    # 3. Simple symbols list
    symbols_file = "all_symbols.txt"
    with open(symbols_file, 'w') as f:
        for symbol in merged['symbol'].tolist():
            f.write(f"{symbol}\n")
    print(f"✅ Saved symbols list: {symbols_file}")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total stocks: {len(merged)}")
    print(f"With prices: {len(merged[merged['price'] > 0])}")
    print(f"Average price: NPR {merged[merged['price'] > 0]['price'].mean():.2f}")
    print(f"Highest price: NPR {merged['price'].max():.2f}")
    print(f"Lowest price: NPR {merged[merged['price'] > 0]['price'].min():.2f}")
    print(f"{'='*70}\n")
    
    return merged


def update_tracker_with_all_stocks(merged_df, tracker):
    """
    Update tracker with all available stocks (mark as 'available')
    
    Args:
        merged_df: DataFrame with all stocks
        tracker: StockTracker instance
    """
    if merged_df is None or merged_df.empty:
        return
    
    print("\n" + "="*70)
    print("UPDATING STOCK TRACKER")
    print("="*70)
    
    # Get list of all symbols
    all_symbols = merged_df['symbol'].tolist()
    
    # Save to a separate file for reference
    tracker_data = {
        'last_updated': datetime.now().isoformat(),
        'total_stocks_available': len(all_symbols),
        'all_symbols': all_symbols,
        'processed_count': len([s for s in all_symbols if tracker.is_processed(s, days_old=30)]),
        'unprocessed_count': len(tracker.get_unprocessed(all_symbols, days_old=30))
    }
    
    with open('all_stocks_tracker.json', 'w') as f:
        json.dump(tracker_data, f, indent=2)
    
    print(f"✅ Total stocks available: {len(all_symbols)}")
    print(f"✅ Already processed (last 30 days): {tracker_data['processed_count']}")
    print(f"✅ Unprocessed stocks: {tracker_data['unprocessed_count']}")
    print(f"✅ Tracker data saved to: all_stocks_tracker.json")
    print("="*70 + "\n")


def main():
    """Main function"""
    print("\n" + "="*70)
    print("FETCH ALL NEPSE STOCKS")
    print("="*70)
    print("Sources: ShareSansar (live prices) + NEPSE Database")
    print("="*70)
    
    # Initialize tracker
    tracker = StockTracker()
    
    # Fetch from ShareSansar (live prices)
    sharesansar_df = fetch_all_stocks_from_sharesansar()
    
    # Fetch from NEPSE database
    nepse_df = fetch_all_stocks_from_nepse()
    
    # Merge and save
    merged_df = merge_and_save_all_stocks(sharesansar_df, nepse_df)
    
    # Update tracker
    update_tracker_with_all_stocks(merged_df, tracker)
    
    print("\n✅ Done! All stocks fetched and saved.")
    print("Files created:")
    print("  - all_nepse_stocks_TIMESTAMP.csv (full data)")
    print("  - all_nepse_stocks_TIMESTAMP.json (full data)")
    print("  - all_symbols.txt (symbols only)")
    print("  - all_stocks_tracker.json (processing status)")
    print()


if __name__ == '__main__':
    main()
