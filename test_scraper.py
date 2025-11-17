"""
Quick test script for ShareSansar scraper
Tests basic functionality without full scraping
"""
from sharesansar_scraper import ShareSansarScraper
import time

def test_scraper():
    """Test the scraper with minimal data"""
    print("\n" + "="*70)
    print("Testing ShareSansar Scraper")
    print("="*70 + "\n")
    
    # Create scraper with visible browser
    print("→ Initializing scraper (browser will open)...")
    scraper = ShareSansarScraper(headless=False)
    
    try:
        # Test with NABIL, only 50 rows
        symbol = "NABIL"
        max_rows = 50
        
        print(f"→ Testing with {symbol}, max {max_rows} rows...")
        print("→ This will take about 15-20 seconds...\n")
        
        df = scraper.scrape_stock(symbol, max_rows)
        
        if not df.empty:
            print("\n" + "="*70)
            print("✓ SUCCESS! Scraper is working!")
            print("="*70 + "\n")
            
            print(f"Scraped {len(df)} transactions for {symbol}")
            print("\nFirst 5 rows:")
            print(df.head())
            
            print("\nColumn names:")
            print(df.columns.tolist())
            
            print("\n" + "="*70)
            print("The scraper is ready for production use!")
            print("="*70)
            
        else:
            print("\n" + "="*70)
            print("⚠ No data returned")
            print("="*70 + "\n")
            print("Possible reasons:")
            print("1. Market is closed (try during trading hours)")
            print("2. Stock symbol not found")
            print("3. Network issues")
            
    except Exception as e:
        print(f"\n✗ Error during test: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Chrome browser is installed")
        print("2. Check your internet connection")
        print("3. Try again during market hours")
        
    finally:
        print("\n→ Closing browser...")
        scraper.close_driver()
        print("✓ Test complete!")

if __name__ == "__main__":
    test_scraper()
