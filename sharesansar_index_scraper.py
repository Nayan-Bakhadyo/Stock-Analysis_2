"""
ShareSansar Index History Scraper using Selenium
Scrapes historical index data (NEPSE Index and Sector-wise Indices) from ShareSansar
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
from pathlib import Path


class ShareSansarIndexScraper:
    """Scraper for ShareSansar index history data"""
    
    # Available indices on ShareSansar
    INDICES = {
        'NEPSE': 'NEPSE Index',
        'SENSITIVE': 'Sensitive Index',
        'FLOAT': 'Float Index', 
        'SENSITIVEFLOAT': 'Sensitive Float Index',
        'BANKING': 'Banking SubIndex',
        'DEVBANK': 'Development Bank Index',
        'FINANCE': 'Finance Index',
        'HOTELS': 'Hotels And Tourism Index',
        'HYDROPOWER': 'HydroPower Index',
        'INVESTMENT': 'Investment Index',
        'LIFEINSURANCE': 'Life Insurance Index',
        'MANUFACTURING': 'Manufacturing And Processing Index',
        'MICROFINANCE': 'Microfinance Index',
        'MUTUAL': 'Mutual Fund Index',
        'NONLIFEINSURANCE': 'Non Life Insurance Index',
        'OTHERS': 'Others Index',
        'TRADING': 'Trading Index',
    }
    
    def __init__(self, headless=True):
        """
        Initialize the scraper
        
        Args:
            headless: Run browser in headless mode (no GUI)
        """
        self.base_url = "https://www.sharesansar.com/index-history-data"
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
    
    def list_available_indices(self):
        """List all available indices that can be scraped"""
        print("\n📊 Available Indices:")
        print("-" * 50)
        for code, name in self.INDICES.items():
            print(f"  {code:<20} → {name}")
        print("-" * 50)
        return list(self.INDICES.keys())
    
    def scrape_index_history(self, index_code='NEPSE', from_date=None, to_date=None, days=None):
        """
        Scrape index history
        
        Args:
            index_code: Index code (e.g., 'NEPSE', 'BANKING', 'HYDROPOWER')
            from_date: Start date (YYYY-MM-DD format)
            to_date: End date (YYYY-MM-DD format)
            days: Number of days of history if dates not specified (default: None = all history)
            
        Returns:
            pandas.DataFrame with index history
        """
        # Calculate dates if not provided
        if to_date is None:
            to_date = datetime.now().strftime('%Y-%m-%d')
        if from_date is None:
            if days is not None:
                from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            else:
                # No limit - go back as far as possible (NEPSE started ~1994)
                from_date = '1994-01-01'
        
        index_name = self.INDICES.get(index_code.upper(), index_code)
        
        try:
            print(f"\n{'='*70}")
            print(f"Scraping index history for {index_name}")
            print(f"Date range: {from_date} to {to_date}")
            print(f"{'='*70}\n")
            
            # Setup driver if not already done
            if not self.driver:
                self.setup_driver()
            
            # Navigate to index history page
            print(f"→ Loading {self.base_url}...")
            self.driver.get(self.base_url)
            time.sleep(3)
            
            # Select the index from dropdown
            try:
                # Find the index dropdown - look for select element
                index_select = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "select"))
                )
                
                # Try to find by name or id if multiple selects
                selects = self.driver.find_elements(By.TAG_NAME, "select")
                print(f"✓ Found {len(selects)} dropdown(s)")
                
                for sel in selects:
                    options = sel.find_elements(By.TAG_NAME, "option")
                    option_texts = [opt.text for opt in options[:5]]
                    print(f"  Options: {option_texts}...")
                    
                    # Check if this is the index selector
                    if any('NEPSE' in opt.text or 'Index' in opt.text for opt in options):
                        select = Select(sel)
                        # Select by visible text
                        select.select_by_visible_text(index_name)
                        print(f"✓ Selected index: {index_name}")
                        time.sleep(1)
                        break
                        
            except Exception as e:
                print(f"⚠ Could not select index: {e}")
            
            # Set date range
            try:
                # Find date inputs
                date_inputs = self.driver.find_elements(By.CSS_SELECTOR, "input[type='date'], input.datepicker, input[name*='date']")
                print(f"✓ Found {len(date_inputs)} date input(s)")
                
                # Also try to find by placeholder or id
                if len(date_inputs) < 2:
                    date_inputs = self.driver.find_elements(By.CSS_SELECTOR, "input[type='text']")
                    date_inputs = [inp for inp in date_inputs if 'date' in inp.get_attribute('name').lower() or 
                                   'date' in (inp.get_attribute('id') or '').lower() or
                                   'date' in (inp.get_attribute('placeholder') or '').lower()]
                
                if len(date_inputs) >= 2:
                    # Clear and set from date
                    date_inputs[0].clear()
                    date_inputs[0].send_keys(from_date)
                    print(f"✓ Set from date: {from_date}")
                    
                    # Clear and set to date
                    date_inputs[1].clear()
                    date_inputs[1].send_keys(to_date)
                    print(f"✓ Set to date: {to_date}")
                    time.sleep(1)
                    
            except Exception as e:
                print(f"⚠ Could not set dates: {e}")
            
            # Click Search button
            try:
                search_btn = self.driver.find_element(By.CSS_SELECTOR, "button[type='submit'], input[type='submit'], button.btn-primary, button:contains('Search')")
                search_btn.click()
                print("✓ Clicked Search button")
                time.sleep(3)
            except:
                # Try finding by text
                try:
                    buttons = self.driver.find_elements(By.TAG_NAME, "button")
                    for btn in buttons:
                        if 'search' in btn.text.lower() or 'submit' in btn.text.lower():
                            btn.click()
                            print("✓ Clicked Search button")
                            time.sleep(3)
                            break
                except Exception as e:
                    print(f"⚠ Could not click search: {e}")
            
            # Try to show more entries - prefer 50 rows per page
            try:
                length_selects = self.driver.find_elements(By.CSS_SELECTOR, "select[name*='length']")
                if length_selects:
                    select = Select(length_selects[0])
                    # Prefer 50 rows per page as requested, then try others
                    for value in ['50', '100', '500', '-1']:
                        try:
                            select.select_by_value(value)
                            print(f"✓ Set table to show {value} rows per page")
                            time.sleep(2)
                            break
                        except:
                            continue
            except Exception as e:
                print(f"⚠ Could not set table length: {e}")
            
            # Scrape the table
            df = self._scrape_index_table()
            
            if not df.empty:
                df['index_code'] = index_code.upper()
                df['index_name'] = index_name
                df['scraped_at'] = datetime.now()
            
            return df
            
        except Exception as e:
            print(f"✗ Error scraping {index_name}: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def _scrape_index_table(self):
        """Scrape the index history table with pagination support"""
        try:
            time.sleep(2)
            
            # Find tables
            tables = self.driver.find_elements(By.TAG_NAME, "table")
            
            if not tables:
                print("✗ No tables found on page")
                return pd.DataFrame()
            
            print(f"✓ Found {len(tables)} table(s)")
            
            # Find the index table
            index_table = None
            headers = []
            
            for idx, table in enumerate(tables):
                try:
                    thead = table.find_element(By.TAG_NAME, "thead")
                    header_row = thead.find_element(By.TAG_NAME, "tr")
                    table_headers = [th.text.strip() for th in header_row.find_elements(By.TAG_NAME, "th")]
                    
                    if not table_headers:
                        continue
                    
                    # Check if this looks like an index table
                    index_keywords = ['open', 'high', 'low', 'close', 'date', 'turnover', 'change']
                    if any(keyword in ' '.join(table_headers).lower() for keyword in index_keywords):
                        print(f"✓ Table {idx+1} headers: {table_headers}")
                        index_table = table
                        headers = table_headers
                        break
                        
                except Exception as e:
                    continue
            
            if not index_table:
                print("⚠ No index table found")
                return pd.DataFrame()
            
            # Scrape all pages
            all_rows = []
            page = 1
            
            while True:
                print(f"→ Scraping page {page}...")
                
                try:
                    tbody = index_table.find_element(By.TAG_NAME, "tbody")
                    rows = tbody.find_elements(By.TAG_NAME, "tr")
                    
                    if not rows:
                        print(f"  No rows on page {page}")
                        break
                    
                    page_data = []
                    for row in rows:
                        cells = row.find_elements(By.TAG_NAME, "td")
                        if cells:
                            row_data = [cell.text.strip() for cell in cells]
                            page_data.append(row_data)
                    
                    if not page_data:
                        break
                    
                    print(f"  ✓ Got {len(page_data)} rows from page {page}")
                    all_rows.extend(page_data)
                    
                    # Try to go to next page
                    try:
                        next_btn = self.driver.find_element(By.CSS_SELECTOR, ".paginate_button.next:not(.disabled)")
                        if 'disabled' in next_btn.get_attribute('class'):
                            break
                        next_btn.click()
                        time.sleep(2)
                        page += 1
                        
                        # Re-find the table after pagination
                        tables = self.driver.find_elements(By.TAG_NAME, "table")
                        for t in tables:
                            try:
                                thead = t.find_element(By.TAG_NAME, "thead")
                                if thead:
                                    index_table = t
                                    break
                            except:
                                continue
                    except:
                        # No more pages
                        break
                        
                except Exception as e:
                    print(f"  ⚠ Error on page {page}: {e}")
                    break
            
            print(f"\n✓ Total rows scraped: {len(all_rows)}")
            
            if not all_rows:
                return pd.DataFrame()
            
            # Create DataFrame
            df = pd.DataFrame(all_rows, columns=headers[:len(all_rows[0])] if headers else None)
            
            # Clean and standardize columns
            df = self._clean_dataframe(df)
            
            return df
            
        except Exception as e:
            print(f"✗ Error scraping table: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def _clean_dataframe(self, df):
        """Clean and standardize the DataFrame"""
        if df.empty:
            return df
        
        # Standardize column names
        column_mapping = {
            's.n.': 'sn',
            'sn': 'sn',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'ltp': 'close',
            'change': 'change',
            'per change (%)': 'pct_change',
            '% change': 'pct_change',
            'turnover': 'turnover',
            'date': 'date',
        }
        
        df.columns = [column_mapping.get(col.lower().strip(), col.lower().strip().replace(' ', '_')) 
                      for col in df.columns]
        
        # Remove S.N. column if exists
        if 'sn' in df.columns:
            df = df.drop('sn', axis=1)
        
        # Convert numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'change', 'pct_change', 'turnover']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
        
        # Convert date
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.sort_values('date', ascending=False).reset_index(drop=True)
        
        return df
    
    def scrape_all_indices(self, days=None, save_to_db=True):
        """
        Scrape history for all available indices
        
        Args:
            days: Number of days of history (None = all available history)
            save_to_db: Whether to save to SQLite database
            
        Returns:
            Dict of DataFrames {index_code: df}
        """
        results = {}
        
        try:
            self.setup_driver()
            
            for index_code in self.INDICES.keys():
                print(f"\n{'='*70}")
                print(f"Scraping {index_code}...")
                print(f"{'='*70}")
                
                df = self.scrape_index_history(index_code, days=days)
                
                if not df.empty:
                    results[index_code] = df
                    print(f"✓ Got {len(df)} rows for {index_code}")
                    
                    if save_to_db:
                        self.save_to_database(df)
                else:
                    print(f"✗ No data for {index_code}")
                
                time.sleep(2)  # Be nice to the server
            
        finally:
            self.close_driver()
        
        return results
    
    def save_to_database(self, df, db_path=None):
        """Save index data to SQLite database"""
        if df.empty:
            return
        
        if db_path is None:
            db_path = Path(config.DATA_DIR) / "nepse_stocks.db"
        
        conn = sqlite3.connect(db_path)
        
        # Create index_history table if not exists
        conn.execute("""
            CREATE TABLE IF NOT EXISTS index_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                index_code TEXT NOT NULL,
                index_name TEXT,
                date DATE NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                change REAL,
                pct_change REAL,
                turnover REAL,
                scraped_at TIMESTAMP,
                UNIQUE(index_code, date)
            )
        """)
        
        # Insert data using iloc for proper indexing
        rows_inserted = 0
        for idx in range(len(df)):
            row = df.iloc[idx]
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO index_history 
                    (index_code, index_name, date, open, high, low, close, change, pct_change, turnover, scraped_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row['index_code'] if 'index_code' in df.columns else None,
                    row['index_name'] if 'index_name' in df.columns else None,
                    str(row['date']) if 'date' in df.columns else None,
                    float(row['open']) if 'open' in df.columns and pd.notna(row['open']) else None,
                    float(row['high']) if 'high' in df.columns and pd.notna(row['high']) else None,
                    float(row['low']) if 'low' in df.columns and pd.notna(row['low']) else None,
                    float(row['close']) if 'close' in df.columns and pd.notna(row['close']) else None,
                    float(row['change']) if 'change' in df.columns and pd.notna(row['change']) else None,
                    float(row['pct_change']) if 'pct_change' in df.columns and pd.notna(row['pct_change']) else None,
                    float(row['turnover']) if 'turnover' in df.columns and pd.notna(row['turnover']) else None,
                    str(row['scraped_at']) if 'scraped_at' in df.columns else None,
                ))
                rows_inserted += 1
            except Exception as e:
                print(f"  ⚠ Error inserting row: {e}")
                continue
        
        conn.commit()
        conn.close()
        
        print(f"✓ Saved {rows_inserted} rows to database")
    
    def get_index_data(self, index_code='NEPSE', days=None, db_path=None):
        """
        Get index data from database
        
        Args:
            index_code: Index code
            days: Number of recent days (None for all)
            
        Returns:
            pandas.DataFrame
        """
        if db_path is None:
            db_path = Path(config.DATA_DIR) / "nepse_stocks.db"
        
        conn = sqlite3.connect(db_path)
        
        query = """
            SELECT * FROM index_history
            WHERE index_code = ?
            ORDER BY date DESC
        """
        
        if days:
            query += f" LIMIT {days}"
        
        df = pd.read_sql_query(query, conn, params=(index_code.upper(),))
        conn.close()
        
        return df


def main():
    """Main function to run the scraper"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ShareSansar Index History Scraper')
    parser.add_argument('--index', type=str, default='NEPSE', 
                        help='Index code (NEPSE, BANKING, HYDROPOWER, etc.)')
    parser.add_argument('--all', action='store_true',
                        help='Scrape ALL available indices (17 total)')
    parser.add_argument('--days', type=int, default=None,
                        help='Number of days of history (default: None = all available)')
    parser.add_argument('--from-date', type=str,
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--to-date', type=str,
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--list', action='store_true',
                        help='List available indices')
    parser.add_argument('--no-headless', action='store_true',
                        help='Show browser window (for debugging)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save to database')
    
    args = parser.parse_args()
    
    # Handle index scraping
    scraper = ShareSansarIndexScraper(headless=not args.no_headless)
    
    if args.list:
        scraper.list_available_indices()
        return
    
    try:
        if args.all:
            print(f"\n{'='*70}")
            print(f"🚀 SCRAPING ALL 17 INDICES (This may take a while...)")
            print(f"{'='*70}")
            results = scraper.scrape_all_indices(days=args.days, save_to_db=not args.no_save)
            print(f"\n{'='*70}")
            print(f"📊 SUMMARY - All Indices Scraped")
            print(f"{'='*70}")
            total_rows = 0
            for code, df in results.items():
                print(f"  {code:<20}: {len(df):>6} rows")
                total_rows += len(df)
            print(f"  {'-'*30}")
            print(f"  {'TOTAL':<20}: {total_rows:>6} rows")
            print(f"{'='*70}")
        else:
            scraper.setup_driver()
            df = scraper.scrape_index_history(
                args.index,
                from_date=args.from_date,
                to_date=args.to_date,
                days=args.days
            )
            
            if not df.empty:
                print(f"\n✓ Scraped {len(df)} rows for {args.index}")
                print(f"\nSample data (first 10 rows):")
                print(df.head(10))
                print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")
                
                if not args.no_save:
                    scraper.save_to_database(df)
            else:
                print(f"\n✗ No data scraped for {args.index}")
                
    finally:
        scraper.close_driver()


if __name__ == '__main__':
    main()
