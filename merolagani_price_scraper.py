"""
MeroLagani Price History Scraper using Selenium
Scrapes historical price data from MeroLagani
"""
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import pandas as pd
import time
from datetime import datetime
import sqlite3
import config


class MeroLaganiPriceScraper:
    """Scraper for MeroLagani price history data"""
    
    def __init__(self, headless=True):
        """
        Initialize the scraper
        
        Args:
            headless: Run browser in headless mode (no GUI)
        """
        self.base_url = "https://merolagani.com/CompanyDetail.aspx?symbol="
        self.driver = None
        self.headless = headless
        
    def setup_driver(self):
        """Setup Chrome WebDriver with options"""
        chrome_options = Options()
        
        if self.headless:
            chrome_options.add_argument("--headless")
        
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.maximize_window()
        print("✓ Chrome driver initialized")
    
    def close_driver(self):
        """Close the WebDriver"""
        if self.driver:
            self.driver.quit()
            print("✓ Browser closed")
    
    def scrape_price_history(self, symbol, days=365):
        """
        Scrape price history for a stock
        
        Args:
            symbol: Stock symbol (e.g., 'NABIL')
            days: Number of days (not used, gets all available)
            
        Returns:
            pandas.DataFrame with price history
        """
        try:
            print(f"\n{'='*70}")
            print(f"Scraping price history for {symbol.upper()}")
            print(f"{'='*70}\n")
            
            # Setup driver if not already done
            if not self.driver:
                self.setup_driver()
            
            # Navigate to company page
            url = f"{self.base_url}{symbol.upper()}"
            print(f"→ Loading {url}...")
            self.driver.get(url)
            time.sleep(4)  # Wait for JavaScript to load
            
            # Handle any alerts (notification prompts)
            try:
                alert = self.driver.switch_to.alert
                alert.dismiss()
                print("✓ Dismissed notification alert")
                time.sleep(1)
            except:
                pass  # No alert present
            
            # Scroll down to load the price history section
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
            time.sleep(2)
            
            # Try to click "Price History" or "Historical Data" tab
            try:
                # Look for tabs or links
                tabs = self.driver.find_elements(By.LINK_TEXT, "Price History")
                if not tabs:
                    tabs = self.driver.find_elements(By.PARTIAL_LINK_TEXT, "Historical")
                if not tabs:
                    tabs = self.driver.find_elements(By.PARTIAL_LINK_TEXT, "Price")
                
                if tabs:
                    tabs[0].click()
                    print("✓ Clicked Price History tab")
                    time.sleep(2)
            except:
                print("⚠ No explicit tab found, looking for table on page")
            
            # Scrape all tables and find price data
            df = self._scrape_price_tables()
            
            if not df.empty:
                df['symbol'] = symbol.upper()
                df['scraped_at'] = datetime.now()
                
                # Filter to requested number of days
                if 'date' in df.columns and days < len(df):
                    df = df.head(days)
            
            return df
            
        except Exception as e:
            print(f"✗ Error scraping {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def _scrape_price_tables(self):
        """Scrape all tables and find price history data"""
        try:
            time.sleep(2)
            
            # Find all tables
            tables = self.driver.find_elements(By.TAG_NAME, "table")
            
            if not tables:
                print("✗ No tables found on page")
                return pd.DataFrame()
            
            print(f"✓ Found {len(tables)} table(s)")
            
            # Look for table with price history keywords
            price_keywords = ['date', 'open', 'high', 'low', 'close', 'ltp', 'volume', 'qty']
            
            for idx, table in enumerate(tables):
                try:
                    # Get all text from table to check if it's price data
                    table_html = table.get_attribute('outerHTML').lower()
                    
                    # Check if this table has price-related keywords
                    keyword_count = sum(1 for keyword in price_keywords if keyword in table_html)
                    
                    if keyword_count < 3:  # Need at least 3 price keywords
                        continue
                    
                    # Try to parse this table
                    rows = table.find_elements(By.TAG_NAME, "tr")
                    if len(rows) < 2:  # Need at least header + 1 data row
                        continue
                    
                    print(f"\n✓ Table {idx+1} looks like price data, attempting to parse...")
                    
                    # Get headers
                    headers = []
                    header_row = rows[0]
                    header_cells = header_row.find_elements(By.TAG_NAME, "th")
                    if not header_cells:
                        header_cells = header_row.find_elements(By.TAG_NAME, "td")
                    
                    for cell in header_cells:
                        headers.append(cell.text.strip())
                    
                    if not headers:
                        continue
                    
                    print(f"  Headers: {headers}")
                    
                    # Get data rows
                    rows_data = []
                    for row in rows[1:]:  # Skip header row
                        cells = row.find_elements(By.TAG_NAME, "td")
                        if len(cells) > 0:
                            row_data = [cell.text.strip() for cell in cells]
                            if len(row_data) == len(headers) and len(row_data) >= 5:
                                rows_data.append(row_data)
                    
                    if rows_data:
                        df = pd.DataFrame(rows_data, columns=headers)
                        print(f"✓ Successfully scraped {len(df)} records from table {idx+1}")
                        return df
                        
                except Exception as e:
                    print(f"  ⚠ Could not parse table {idx+1}: {str(e)[:100]}")
                    continue
            
            print("⚠ No valid price table found")
            return pd.DataFrame()
            
        except Exception as e:
            print(f"✗ Error scraping tables: {e}")
            return pd.DataFrame()
    
    def scrape_multiple_stocks(self, symbols, days=365):
        """
        Scrape price history for multiple stocks
        
        Args:
            symbols: List of stock symbols
            days: Number of days per stock
            
        Returns:
            pandas.DataFrame with all price data
        """
        all_data = []
        
        try:
            self.setup_driver()
            
            for i, symbol in enumerate(symbols, 1):
                print(f"\n[{i}/{len(symbols)}] Processing {symbol.upper()}...")
                
                df = self.scrape_price_history(symbol, days)
                
                if not df.empty:
                    all_data.append(df)
                    print(f"✓ {symbol.upper()}: {len(df)} records")
                else:
                    print(f"⚠ {symbol.upper()}: No data found")
                
                # Small delay between stocks
                if i < len(symbols):
                    time.sleep(3)
            
        finally:
            self.close_driver()
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"\n✓ Total price records scraped: {len(combined_df)}")
            return combined_df
        else:
            return pd.DataFrame()
    
    def save_to_database(self, df):
        """
        Save price history to database
        
        Args:
            df: DataFrame with price data
        """
        if df.empty:
            print("⚠ No data to save")
            return
        
        # Standardize column names
        column_map = {
            'Date': 'date',
            'S.N.': 'sn',
            'S.N': 'sn',
            'SN': 'sn',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Closing Price': 'close',
            'LTP': 'close',
            'Ltp': 'close',
            'Last Traded Price': 'close',
            'Volume': 'volume',
            'Qty': 'volume',
            'Quantity': 'volume',
            'Traded Shares': 'volume',
            'Prev. Close': 'prev_close',
            'Previous Close': 'prev_close',
        }
        
        df_clean = df.rename(columns=column_map)
        
        # Ensure required columns exist
        required_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        
        for col in required_cols:
            if col not in df_clean.columns:
                if col in ['open', 'high', 'low']:
                    df_clean[col] = df_clean.get('close', 0)
                elif col == 'volume':
                    df_clean[col] = 0
                elif col == 'symbol':
                    df_clean[col] = 'UNKNOWN'
        
        # Parse date
        df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
        
        # Clean numeric columns
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(
                    df_clean[col].astype(str).str.replace(',', ''), 
                    errors='coerce'
                ).fillna(0)
        
        # Add source
        df_clean['source'] = 'merolagani'
        
        # Save to database
        try:
            conn = sqlite3.connect(config.DB_PATH)
            df_clean[required_cols + ['source']].to_sql(
                'price_history', 
                conn, 
                if_exists='append', 
                index=False
            )
            conn.close()
            print(f"✓ Saved {len(df_clean)} price records to database")
        except Exception as e:
            print(f"✗ Error saving to database: {e}")


def main():
    """Interactive price scraper"""
    print("\n" + "="*70)
    print("MeroLagani Price History Scraper")
    print("="*70 + "\n")
    
    print("Options:")
    print("1. Scrape single stock")
    print("2. Scrape multiple stocks")
    print("3. Scrape top stocks (NABIL, NICA, GBIME, SBI, EBL)")
    print("4. Exit")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    scraper = MeroLaganiPriceScraper(headless=False)
    
    try:
        if choice == '1':
            symbol = input("Enter stock symbol (e.g., NABIL): ").strip().upper()
            days = int(input("Days of history (default 365): ").strip() or "365")
            
            df = scraper.scrape_price_history(symbol, days)
            
            if not df.empty:
                print("\n" + "="*70)
                print("Preview of scraped data:")
                print("="*70)
                print(df.head(10))
                
                save = input("\nSave to database? (y/n): ").strip().lower()
                if save == 'y':
                    scraper.save_to_database(df)
        
        elif choice == '2':
            symbols_input = input("Enter stock symbols (comma-separated, e.g., NABIL,NICA): ").strip()
            symbols = [s.strip().upper() for s in symbols_input.split(',')]
            days = int(input("Days per stock (default 365): ").strip() or "365")
            
            df = scraper.scrape_multiple_stocks(symbols, days)
            
            if not df.empty:
                print("\n" + "="*70)
                print("Summary:")
                print("="*70)
                print(df.groupby('symbol').size())
                
                save = input("\nSave to database? (y/n): ").strip().lower()
                if save == 'y':
                    scraper.save_to_database(df)
        
        elif choice == '3':
            top_stocks = ['NABIL', 'NICA', 'GBIME', 'SBI', 'EBL']
            days = 180  # 6 months
            
            print(f"\nScraping {len(top_stocks)} stocks with {days} days each...")
            df = scraper.scrape_multiple_stocks(top_stocks, days)
            
            if not df.empty:
                print("\n" + "="*70)
                print("Summary:")
                print("="*70)
                print(df.groupby('symbol').size())
                
                save = input("\nSave to database? (y/n): ").strip().lower()
                if save == 'y':
                    scraper.save_to_database(df)
    
    finally:
        scraper.close_driver()
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
