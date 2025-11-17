"""Quick status checker for analysis progress"""
import sqlite3
import config
import os
import json

def check_status():
    """Check database and file status"""
    
    print("\n" + "="*70)
    print("ANALYSIS PROGRESS STATUS")
    print("="*70)
    
    # Check database
    if os.path.exists(config.DB_PATH):
        conn = sqlite3.connect(config.DB_PATH)
        
        # Check price history
        cursor = conn.cursor()
        cursor.execute("SELECT symbol, COUNT(*) FROM price_history GROUP BY symbol")
        price_counts = cursor.fetchall()
        
        print("\n📊 PRICE HISTORY IN DATABASE:")
        if price_counts:
            for symbol, count in price_counts:
                print(f"  ✓ {symbol}: {count:,} records")
        else:
            print("  ⚠️  No price data yet")
        
        # Check news
        cursor.execute("SELECT symbol, COUNT(*) FROM news_cache GROUP BY symbol")
        news_counts = cursor.fetchall()
        
        print("\n📰 NEWS ARTICLES IN DATABASE:")
        if news_counts:
            for symbol, count in news_counts:
                print(f"  ✓ {symbol}: {count} articles")
        else:
            print("  ⚠️  No news data yet")
        
        conn.close()
    else:
        print("\n⚠️  Database not found")
    
    # Check JSON output
    json_file = "analysis_results.json"
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        print(f"\n📁 ANALYSIS OUTPUT FILE:")
        print(f"  ✓ {json_file} exists")
        print(f"  ✓ Contains {len(data)} stocks")
        
        for stock in data:
            print(f"\n  Stock: {stock['symbol']}")
            if 'price_data' in stock and stock['price_data']:
                print(f"    ✓ Price data: {len(stock['price_data'])} records")
            if 'news' in stock and stock['news']:
                print(f"    ✓ News: {len(stock['news'])} articles")
            if 'fundamentals' in stock and stock['fundamentals']:
                print(f"    ✓ Fundamentals: Available")
            if 'candlestick_patterns' in stock:
                patterns = stock['candlestick_patterns']
                if patterns:
                    print(f"    ✓ Candlestick patterns: {len(patterns)} found")
    else:
        print(f"\n⚠️  {json_file} not found yet (analysis still running)")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    check_status()
