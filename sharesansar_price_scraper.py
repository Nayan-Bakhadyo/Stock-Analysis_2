"""
ShareSansar Price History Scraper using Selenium
Scrapes historical price data (OHLCV) from ShareSansar
"""
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import pandas as pd
import time
from datetime import datetime, timedelta
import sqlite3
import config


class ShareSansarPriceScraper:
    """Scraper for ShareSansar price history data"""
    
    def __init__(self, headless=True):
        """
        Initialize the scraper
        
        Args:
            headless: Run browser in headless mode (no GUI)
        """
        self.base_url = "https://www.sharesansar.com/company"
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
            symbol: Stock symbol (e.g., 'nabil')
            days: Number of days of history to fetch
            
        Returns:
            pandas.DataFrame with price history
        """
        # Store days parameter for use in _scrape_price_table
        self._days_limit = days
        
        try:
            print(f"\n{'='*70}")
            print(f"Scraping price history for {symbol.upper()}")
            print(f"{'='*70}\n")
            
            # Setup driver if not already done
            if not self.driver:
                self.setup_driver()
            
            # Navigate to company page (symbol in lowercase)
            url = f"{self.base_url}/{symbol.lower()}"
            print(f"→ Loading {url}...")
            self.driver.get(url)
            time.sleep(3)
            
            # Click on "Price History" tab
            try:
                price_tab = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.LINK_TEXT, "Price History"))
                )
                price_tab.click()
                print("✓ Clicked Price History tab")
                time.sleep(2)
            except:
                print("⚠ Price History tab not found, already on history page")
            
            # Find and click "View More" or pagination to get more data
            try:
                # Look for table length selector - try different patterns
                length_select = None
                for selector_name in ["myTableCPriceHistory_length", "myTable_length", "DataTables_Table_0_length"]:
                    try:
                        length_select = self.driver.find_element(By.NAME, selector_name)
                        break
                    except:
                        continue
                
                if length_select:
                    select = Select(length_select)
                    
                    # Select 50 rows per page
                    select.select_by_value("50")
                    print(f"✓ Set table to show 50 rows per page")
                    time.sleep(2)
                else:
                    print("⚠ Table length selector not found by name")
            except Exception as e:
                print(f"⚠ Could not set table length: {e}")
            
            # Try to click "Show All" or "View All" button if exists
            try:
                show_all_buttons = self.driver.find_elements(By.PARTIAL_LINK_TEXT, "Show All")
                if not show_all_buttons:
                    show_all_buttons = self.driver.find_elements(By.PARTIAL_LINK_TEXT, "View All")
                if not show_all_buttons:
                    show_all_buttons = self.driver.find_elements(By.PARTIAL_LINK_TEXT, "All")
                
                if show_all_buttons:
                    show_all_buttons[0].click()
                    print("✓ Clicked 'Show All' button")
                    time.sleep(3)
            except:
                pass
            
            # Scrape the price table
            df = self._scrape_price_table()
            
            if not df.empty:
                df['symbol'] = symbol.upper()
                df['scraped_at'] = datetime.now()
                
                # Filter to requested number of days
                if 'date' in df.columns and days < len(df):
                    df = df.head(days)
            
            return df
            
        except Exception as e:
            print(f"✗ Error scraping {symbol}: {e}")
            return pd.DataFrame()
    
    def _scrape_price_table(self):
        """Scrape the price history table with pagination support"""
        try:
            # Wait for any table to be present
            time.sleep(2)
            
            # Try to find table by different methods
            tables = self.driver.find_elements(By.TAG_NAME, "table")
            
            if not tables:
                print("✗ No tables found on page")
                return pd.DataFrame()
            
            print(f"✓ Found {len(tables)} table(s)")
            
            # Find the price table
            price_table = None
            table_index = None
            headers = []
            
            for idx, table in enumerate(tables):
                try:
                    # Get table headers
                    thead = table.find_element(By.TAG_NAME, "thead")
                    header_row = thead.find_element(By.TAG_NAME, "tr")
                    table_headers = [th.text.strip() for th in header_row.find_elements(By.TAG_NAME, "th")]
                    
                    if not table_headers:
                        continue
                    
                    # Check if this looks like a price table
                    price_keywords = ['date', 'open', 'high', 'low', 'ltp', 'close', 'qty']
                    if any(keyword in ' '.join(table_headers).lower() for keyword in price_keywords):
                        print(f"✓ Table {idx+1} headers: {table_headers}")
                        price_table = table
                        table_index = idx
                        headers = table_headers
                        break
                        
                except Exception as e:
                    continue
            
            if not price_table:
                print("⚠ No price table found")
                return pd.DataFrame()
            
            # Scrape all pages (or until we reach desired number of days)
            all_rows = []
            page = 1
            target_rows = self._days_limit if hasattr(self, '_days_limit') and self._days_limit and self._days_limit < 9999 else None  # None means scrape all
            
            while True:
                print(f"→ Scraping page {page}...")
                
                # Get current page rows
                tbody = price_table.find_element(By.TAG_NAME, "tbody")
                rows = tbody.find_elements(By.TAG_NAME, "tr")
                
                if not rows:
                    break
                
                page_rows = 0
                for row in rows:
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if len(cells) > 0:
                        row_data = [cell.text.strip() for cell in cells]
                        if len(row_data) == len(headers):
                            all_rows.append(row_data)
                            page_rows += 1
                            
                            # Stop if we've reached target number of rows
                            if target_rows and len(all_rows) >= target_rows:
                                break
                
                print(f"  ✓ Found {page_rows} rows on page {page}")
                
                # Check if we've reached target
                if target_rows and len(all_rows) >= target_rows:
                    print(f"  → Reached target of {target_rows} rows, stopping")
                    break
                
                # Try to find and click "Next" button
                try:
                    # Look for pagination buttons
                    next_button = None
                    
                    # Try different selectors for next button
                    selectors = [
                        (By.CLASS_NAME, "paginate_button.next"),
                        (By.ID, "myTable_next"),
                        (By.XPATH, "//a[contains(@class, 'next') and not(contains(@class, 'disabled'))]"),
                        (By.PARTIAL_LINK_TEXT, "Next"),
                        (By.XPATH, "//li[@class='next']/a"),
                    ]
                    
                    for by, value in selectors:
                        try:
                            next_button = self.driver.find_element(by, value)
                            # Check if button is not disabled
                            if 'disabled' not in next_button.get_attribute('class'):
                                break
                            else:
                                next_button = None
                        except:
                            continue
                    
                    if next_button:
                        # Scroll into view and click
                        self.driver.execute_script("arguments[0].scrollIntoView();", next_button)
                        time.sleep(0.3)
                        next_button.click()
                        time.sleep(1)  # Reduced from 2 to 1 second
                        page += 1
                        
                        # Re-find the table after pagination
                        tables = self.driver.find_elements(By.TAG_NAME, "table")
                        if table_index < len(tables):
                            price_table = tables[table_index]
                    else:
                        print("  → No more pages")
                        break
                        
                except Exception as e:
                    print(f"  → Pagination ended: {e}")
                    break
            
            # Create DataFrame if we have data
            if all_rows:
                df = pd.DataFrame(all_rows, columns=headers)
                
                # Trim to exact number of rows if target specified
                if target_rows and len(df) > target_rows:
                    df = df.head(target_rows)
                
                print(f"✓ Scraped {len(df)} total price records across {page} page(s)")
                return df
            else:
                print("⚠ No data found")
                return pd.DataFrame()
            
        except Exception as e:
            print(f"✗ Error scraping price table: {e}")
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
                    time.sleep(2)
            
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
        
        # Standardize column names (case-insensitive)
        column_map = {
            'Date': 'date',
            'S.N.': 'sn',
            'S.N': 'sn',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Closing Price': 'close',
            'LTP': 'close',
            'Ltp': 'close',  # ShareSansar uses "Ltp"
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
        
        # Save to database
        try:
            conn = sqlite3.connect(config.DB_PATH)
            df_clean[required_cols].to_sql(
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
    print("ShareSansar Price History Scraper")
    print("="*70 + "\n")
    
    print("Options:")
    print("1. Scrape single stock")
    print("2. Scrape multiple stocks")
    print("3. Scrape top stocks (NABIL, NICA, GBIME, SBI, EBL)")
    print("4. Exit")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    scraper = ShareSansarPriceScraper(headless=False)
    
    try:
        if choice == '1':
            symbol = input("Enter stock symbol (e.g., nabil): ").strip().lower()
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
            symbols_input = input("Enter stock symbols (comma-separated, e.g., nabil,nica): ").strip()
            symbols = [s.strip().lower() for s in symbols_input.split(',')]
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
            top_stocks = ['nabil', 'nica', 'gbime', 'sbi', 'ebl']
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
