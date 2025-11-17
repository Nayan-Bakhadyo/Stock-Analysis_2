"""
Get all NEPSE company symbols from ShareSansar
"""
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import pandas as pd
import time
import sqlite3
import config


class ShareSansarSymbolScraper:
    """Scrape all company symbols from ShareSansar"""
    
    def __init__(self, headless=True):
        self.base_url = "https://www.sharesansar.com/today-share-price"
        self.driver = None
        self.headless = headless
    
    def setup_driver(self):
        """Setup Chrome WebDriver"""
        chrome_options = Options()
        
        if self.headless:
            chrome_options.add_argument("--headless=new")
        
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36")
        
        self.driver = webdriver.Chrome(options=chrome_options)
        print("✓ Chrome driver initialized")
    
    def close_driver(self):
        """Close the WebDriver"""
        if self.driver:
            self.driver.quit()
    
    def scrape_all_symbols(self) -> pd.DataFrame:
        """
        Scrape all company symbols and basic info
        
        Returns:
            DataFrame with columns: symbol, company_name, sector, ltp
        """
        if not self.driver:
            self.setup_driver()
        
        print(f"\n→ Loading {self.base_url}...")
        self.driver.get(self.base_url)
        time.sleep(3)
        
        # Wait for table to load
        try:
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "table"))
            )
            print("✓ Page loaded")
        except:
            print("✗ Timeout waiting for table")
            return pd.DataFrame()
        
        # Set table to show all entries
        try:
            # Look for entries dropdown
            select_elem = self.driver.find_element(By.NAME, "myTable_length")
            from selenium.webdriver.support.ui import Select
            select = Select(select_elem)
            
            # Try to select "All" or maximum entries
            try:
                select.select_by_visible_text("All")
                print("✓ Set table to show all entries")
            except:
                # Try maximum available option
                options = [opt.text for opt in select.options]
                max_option = max([int(opt) for opt in options if opt.isdigit()], default=100)
                select.select_by_visible_text(str(max_option))
                print(f"✓ Set table to show {max_option} entries")
            
            time.sleep(2)
        except Exception as e:
            print(f"⚠️ Could not change table length: {e}")
        
        # Extract data from table
        companies = []
        
        try:
            # Find all rows (there might be multiple tables, use all rows)
            rows = self.driver.find_elements(By.TAG_NAME, "tr")
            print(f"→ Found {len(rows)} rows total")
            
            for row in rows:
                try:
                    cells = row.find_elements(By.TAG_NAME, "td")
                    
                    if len(cells) >= 4:
                        # Extract data
                        # Column 0: S.N.
                        # Column 1: Symbol
                        # Column 2: Conf. (some number)
                        # Column 3: LTP
                        sn = cells[0].text.strip()
                        symbol = cells[1].text.strip()
                        
                        # Get LTP
                        ltp_text = cells[3].text.strip()
                        try:
                            ltp = float(ltp_text.replace(',', ''))
                        except:
                            ltp = None
                        
                        # Try to get company name if available
                        company_name = symbol  # Default to symbol
                        
                        if symbol and symbol.isupper() and len(symbol) <= 10:  # Valid symbol
                            companies.append({
                                'sn': sn,
                                'symbol': symbol,
                                'company_name': company_name,
                                'ltp': ltp,
                                'sector': ""
                            })
                
                except Exception as e:
                    continue
            
            print(f"✓ Extracted {len(companies)} companies")
            
        except Exception as e:
            print(f"✗ Error extracting data: {e}")
            import traceback
            traceback.print_exc()
        
        df = pd.DataFrame(companies)
        
        # Remove duplicates (keep first occurrence)
        if not df.empty:
            df = df.drop_duplicates(subset=['symbol'], keep='first')
            print(f"✓ After removing duplicates: {len(df)} unique companies")
        
        return df
    
    def save_to_database(self, df: pd.DataFrame):
        """Save company symbols to database"""
        if df.empty:
            print("⚠️ No data to save")
            return
        
        conn = sqlite3.connect(config.DB_PATH)
        
        # Create companies table if not exists
        conn.execute("""
            CREATE TABLE IF NOT EXISTS companies (
                symbol TEXT PRIMARY KEY,
                company_name TEXT,
                sector TEXT,
                ltp REAL,
                last_updated TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Save data
        df[['symbol', 'company_name', 'sector', 'ltp']].to_sql(
            'companies', 
            conn, 
            if_exists='replace', 
            index=False
        )
        
        conn.commit()
        conn.close()
        
        print(f"✓ Saved {len(df)} companies to database")
    
    def get_all_symbols(self, save_to_db: bool = True) -> list:
        """
        Get all NEPSE stock symbols
        
        Args:
            save_to_db: Save to database
            
        Returns:
            List of stock symbols
        """
        df = self.scrape_all_symbols()
        
        if save_to_db and not df.empty:
            self.save_to_database(df)
        
        return df['symbol'].tolist() if not df.empty else []


if __name__ == "__main__":
    print("Scraping all NEPSE company symbols from ShareSansar")
    print("=" * 60)
    
    scraper = ShareSansarSymbolScraper(headless=True)
    
    try:
        # Get all symbols
        symbols = scraper.get_all_symbols(save_to_db=True)
        
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"Total companies: {len(symbols)}")
        print(f"\nFirst 20 symbols:")
        for i, symbol in enumerate(symbols[:20], 1):
            print(f"  {i}. {symbol}")
        
        if len(symbols) > 20:
            print(f"  ... and {len(symbols) - 20} more")
        
    finally:
        scraper.close_driver()
