"""
Refresh Website Prices Without Full Analysis
- Fetches latest prices from database
- Updates analysis_results.json
- Regenerates website HTML
"""
import json
import sqlite3
import subprocess
from datetime import datetime


def refresh_prices():
    """Update prices in analysis_results.json from database"""
    
    print("\n" + "="*70)
    print("REFRESHING WEBSITE PRICES")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    # Load existing analysis results
    try:
        with open('analysis_results.json', 'r') as f:
            results = json.load(f)
        print(f"✓ Loaded {len(results)} stocks from analysis_results.json")
    except FileNotFoundError:
        print("❌ No analysis_results.json found. Run initial analysis first.")
        return
    
    # Connect to database
    try:
        conn = sqlite3.connect('data/nepse_stocks.db')
        print("✓ Connected to database\n")
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        return
    
    # Fetch latest prices for all stocks
    symbols = [stock['symbol'] for stock in results if not stock.get('error')]
    
    if not symbols:
        print("❌ No valid stocks found in analysis results")
        conn.close()
        return
    
    placeholders = ','.join(['?' for _ in symbols])
    query = f"""
        SELECT symbol, date, close 
        FROM (
            SELECT symbol, date, close,
                   ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date DESC) as rn
            FROM price_history
            WHERE symbol IN ({placeholders})
        )
        WHERE rn = 1
    """
    
    cursor = conn.execute(query, symbols)
    latest_prices = {row[0]: {'date': row[1], 'price': row[2]} for row in cursor.fetchall()}
    conn.close()
    
    print(f"✓ Fetched latest prices for {len(latest_prices)} stocks\n")
    
    # Update prices in results
    updated_count = 0
    for stock in results:
        if stock.get('error'):
            continue
            
        symbol = stock['symbol']
        if symbol in latest_prices:
            old_price = stock.get('price_data', {}).get('latest_price', 'N/A')
            new_price = latest_prices[symbol]['price']
            new_date = latest_prices[symbol]['date']
            
            # Update price_data
            if 'price_data' not in stock:
                stock['price_data'] = {}
            stock['price_data']['latest_price'] = new_price
            stock['price_data']['latest_date'] = new_date
            
            # Update fundamentals
            if 'fundamentals' in stock:
                stock['fundamentals']['current_price'] = new_price
            
            # Update insights
            if 'insights' in stock:
                stock['insights']['current_price'] = new_price
            
            # Update trading_insights
            if 'trading_insights' in stock:
                stock['trading_insights']['current_price'] = new_price
            
            # Update recommendations
            if 'recommendations' in stock:
                for rec_type in ['short_term', 'medium_term', 'long_term']:
                    if rec_type in stock['recommendations']:
                        stock['recommendations'][rec_type]['current_price'] = new_price
            
            print(f"  {symbol:8s} {old_price:>8} → {new_price:>8}  ({new_date})")
            updated_count += 1
    
    # Save updated results
    with open('analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Updated {updated_count} stock prices in analysis_results.json")
    
    # Regenerate website
    print("\n" + "="*70)
    print("REGENERATING WEBSITE")
    print("="*70 + "\n")
    
    try:
        result = subprocess.run(['python3', 'generate_website.py'], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        print("\n✓ Website regenerated successfully!")
        print("✓ Open website/index.html to view updated prices")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to regenerate website: {e}")
        print(e.stderr)
    
    print("\n" + "="*70)
    print("REFRESH COMPLETE")
    print("="*70)


if __name__ == "__main__":
    refresh_prices()
