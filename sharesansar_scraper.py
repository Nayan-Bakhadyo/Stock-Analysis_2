"""
ShareSansar Floorsheet Scraper using Selenium
Handles JavaScript-rendered content and Select2 dropdowns
"""
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
import pandas as pd
import time
from datetime import datetime
import sqlite3
import config


class ShareSansarScraper:
    """Scraper for ShareSansar floorsheet data"""
    
    def __init__(self, headless=True):
        """
        Initialize the scraper
        
        Args:
            headless: Run browser in headless mode (no GUI)
        """
        self.base_url = "https://www.sharesansar.com/floorsheet"
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
    
    def set_table_length(self, length=500):
        """
        Set the number of rows to display in the table
        
        Args:
            length: Number of rows (50, 100, 200, 500)
        """
        try:
            # Wait for the length dropdown to be present
            length_select = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.NAME, "myTable_length"))
            )
            
            # Use Select to choose the option
            select = Select(length_select)
            select.select_by_value(str(length))
            
            print(f"✓ Set table length to {length} rows")
            time.sleep(2)  # Wait for table to reload
            
        except Exception as e:
            print(f"⚠ Could not set table length: {e}")
    
    def select_stock_symbol(self, symbol):
        """
        Select a stock symbol using Select2 dropdown
        
        Args:
            symbol: Stock symbol (e.g., 'NABIL')
        """
        try:
            # Click on the Select2 container to open dropdown
            select2_container = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, ".select2-selection.select2-selection--single"))
            )
            select2_container.click()
            print(f"✓ Opened stock selector dropdown")
            time.sleep(1)
            
            # Wait for search box to appear and type the symbol
            search_box = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".select2-search__field"))
            )
            search_box.send_keys(symbol)
            print(f"✓ Typed '{symbol}' in search box")
            time.sleep(1)
            
            # Wait for results and find exact match
            results = WebDriverWait(self.driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".select2-results__option"))
            )
            
            # Find exact match for the symbol
            exact_match = None
            for result in results:
                result_text = result.text.strip()
                # Extract the symbol part (usually format: "Company Name / SYMBOL")
                if " / " in result_text:
                    # Format: "Company Name / SYMBOL"
                    symbol_part = result_text.split(" / ")[-1].strip()
                    if symbol_part == symbol:
                        exact_match = result
                        print(f"✓ Found exact match: {result_text}")
                        break
                # Also check if text contains symbol in format "SYMBOL - Company"
                elif result_text.startswith(symbol + " ") or result_text == symbol:
                    exact_match = result
                    print(f"✓ Found exact match: {result_text}")
                    break
            
            # If no exact match, use first result
            if exact_match:
                exact_match.click()
            else:
                print(f"⚠ No exact match found, using first result: {results[0].text}")
                results[0].click()
            
            print(f"✓ Selected {symbol}")
            time.sleep(1)
            
            # Click the Search button to load floorsheet data
            search_button = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.ID, "btn_flsheet_submit"))
            )
            search_button.click()
            print(f"✓ Clicked Search button")
            time.sleep(3)  # Wait for table to load with data
            
        except Exception as e:
            print(f"✗ Error selecting stock symbol: {e}")
            raise
    
    def scrape_floorsheet_table(self):
        """
        Scrape the floorsheet table data
        
        Returns:
            pandas.DataFrame with floorsheet data
        """
        try:
            # Wait for table to be present
            table = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "myTable"))
            )
            
            # Get table headers
            headers = []
            header_row = table.find_element(By.TAG_NAME, "thead").find_element(By.TAG_NAME, "tr")
            for th in header_row.find_elements(By.TAG_NAME, "th"):
                headers.append(th.text.strip())
            
            print(f"✓ Found headers: {headers}")
            
            # Get table rows
            rows_data = []
            tbody = table.find_element(By.TAG_NAME, "tbody")
            rows = tbody.find_elements(By.TAG_NAME, "tr")
            
            print(f"✓ Found {len(rows)} rows")
            
            for row in rows:
                cells = row.find_elements(By.TAG_NAME, "td")
                if len(cells) > 0:
                    row_data = [cell.text.strip() for cell in cells]
                    # Only add if we have the expected number of columns
                    if len(row_data) == len(headers):
                        rows_data.append(row_data)
                    else:
                        print(f"⚠ Skipping row with {len(row_data)} columns (expected {len(headers)})")
            
            # Check if we have valid data
            if not rows_data:
                print("⚠ No valid transaction rows found (market may be closed)")
                return pd.DataFrame()
            
            # Create DataFrame
            df = pd.DataFrame(rows_data, columns=headers)
            
            print(f"✓ Scraped {len(df)} transactions")
            return df
            
        except Exception as e:
            print(f"✗ Error scraping table: {e}")
            return pd.DataFrame()
    
    def scrape_stock(self, symbol, max_rows=500):
        """
        Scrape floorsheet data for a specific stock
        
        Args:
            symbol: Stock symbol (e.g., 'NABIL')
            max_rows: Maximum rows to fetch (50, 100, 200, 500)
            
        Returns:
            pandas.DataFrame with floorsheet data
        """
        try:
            print(f"\n{'='*70}")
            print(f"Scraping floorsheet for {symbol}")
            print(f"{'='*70}\n")
            
            # Setup driver if not already done
            if not self.driver:
                self.setup_driver()
            
            # Navigate to floorsheet page
            print(f"→ Loading {self.base_url}...")
            self.driver.get(self.base_url)
            time.sleep(3)  # Wait for page to fully load
            
            # Select stock symbol first
            self.select_stock_symbol(symbol)
            
            # Set table length after search (this is when the table appears)
            self.set_table_length(max_rows)
            
            # Scrape the table
            df = self.scrape_floorsheet_table()
            
            if not df.empty:
                df['symbol'] = symbol
                df['scraped_at'] = datetime.now()
            
            return df
            
        except Exception as e:
            print(f"✗ Error scraping {symbol}: {e}")
            return pd.DataFrame()
    
    def scrape_all_stocks(self, symbols, max_rows=500):
        """
        Scrape floorsheet data for multiple stocks
        
        Args:
            symbols: List of stock symbols
            max_rows: Maximum rows per stock
            
        Returns:
            pandas.DataFrame with all floorsheet data
        """
        all_data = []
        
        try:
            self.setup_driver()
            
            for i, symbol in enumerate(symbols, 1):
                print(f"\n[{i}/{len(symbols)}] Processing {symbol}...")
                
                df = self.scrape_stock(symbol, max_rows)
                
                if not df.empty:
                    all_data.append(df)
                    print(f"✓ {symbol}: {len(df)} transactions")
                else:
                    print(f"⚠ {symbol}: No data found")
                
                # Small delay between stocks
                if i < len(symbols):
                    time.sleep(2)
            
        finally:
            self.close_driver()
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"\n✓ Total transactions scraped: {len(combined_df)}")
            return combined_df
        else:
            return pd.DataFrame()
    
    def save_to_database(self, df):
        """
        Save floorsheet data to database
        
        Args:
            df: DataFrame with floorsheet data
        """
        if df.empty:
            print("⚠ No data to save")
            return
        
        # Standardize column names
        column_map = {
            'Contract No': 'contract_no',
            'Contract No.': 'contract_no',
            'Transaction No.': 'contract_no',
            'Symbol': 'symbol',
            'Buyer': 'buyer_broker',
            'Buyer Broker': 'buyer_broker',
            'Seller': 'seller_broker',
            'Sell Broker': 'seller_broker',
            'Seller Broker': 'seller_broker',
            'Quantity': 'quantity',
            'Share Quantity': 'quantity',
            'Rate': 'rate',
            'Rate (Rs)': 'rate',
            'Amount': 'amount',
            'Amount (in Rs)': 'amount',
            'Traded Date': 'trade_date',
        }
        
        df_clean = df.rename(columns=column_map)
        
        # Ensure required columns exist
        required_cols = ['contract_no', 'symbol', 'buyer_broker', 'seller_broker', 
                        'quantity', 'rate', 'amount']
        
        for col in required_cols:
            if col not in df_clean.columns:
                df_clean[col] = None
        
        # Add date and source
        if 'trade_date' in df_clean.columns:
            # Use the actual trade date if available
            df_clean['date'] = pd.to_datetime(df_clean['trade_date'], errors='coerce').dt.date
        else:
            df_clean['date'] = datetime.now().date()
        
        df_clean['source'] = 'sharesansar'
        
        # Clean numeric columns
        for col in ['quantity', 'rate', 'amount']:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(
                    df_clean[col].astype(str).str.replace(',', ''), 
                    errors='coerce'
                ).fillna(0)
        
        # Save to database
        try:
            conn = sqlite3.connect(config.DB_PATH)
            df_clean[required_cols + ['date', 'source']].to_sql(
                'floorsheet_data', 
                conn, 
                if_exists='append', 
                index=False
            )
            conn.close()
            print(f"✓ Saved {len(df_clean)} transactions to database")
        except Exception as e:
            print(f"✗ Error saving to database: {e}")


def main():
    """Interactive scraper"""
    print("\n" + "="*70)
    print("ShareSansar Floorsheet Scraper")
    print("="*70 + "\n")
    
    print("Options:")
    print("1. Scrape single stock")
    print("2. Scrape multiple stocks")
    print("3. Scrape top stocks (NABIL, NICA, GBIME, SBI, EBL)")
    print("4. Exit")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    scraper = ShareSansarScraper(headless=False)  # Set to True for no GUI
    
    try:
        if choice == '1':
            symbol = input("Enter stock symbol (e.g., NABIL): ").strip().upper()
            max_rows = int(input("Max rows (50/100/200/500): ").strip() or "500")
            
            df = scraper.scrape_stock(symbol, max_rows)
            
            if not df.empty:
                print("\n" + "="*70)
                print("Preview of scraped data:")
                print("="*70)
                print(df.head(10))
                
                save = input("\nSave to database? (y/n): ").strip().lower()
                if save == 'y':
                    scraper.save_to_database(df)
        
        elif choice == '2':
            symbols_input = input("Enter stock symbols (comma-separated): ").strip()
            symbols = [s.strip().upper() for s in symbols_input.split(',')]
            max_rows = int(input("Max rows per stock (50/100/200/500): ").strip() or "500")
            
            df = scraper.scrape_all_stocks(symbols, max_rows)
            
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
            max_rows = 200
            
            print(f"\nScraping {len(top_stocks)} stocks with max {max_rows} rows each...")
            df = scraper.scrape_all_stocks(top_stocks, max_rows)
            
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
