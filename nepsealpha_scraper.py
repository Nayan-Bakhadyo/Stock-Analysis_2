"""Scraper for NepalAlpha fundamental data"""
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import re
from typing import Dict, Optional


class NepalAlphaScraper:
    """Scrape fundamental data from NepalAlpha"""
    
    def __init__(self, headless: bool = True):
        self.base_url = "https://nepsealpha.com"
        self.driver = None
        self.headless = headless
    
    def init_driver(self):
        """Initialize Chrome WebDriver"""
        if self.driver:
            return
        
        options = webdriver.ChromeOptions()
        if self.headless:
            options.add_argument('--headless=new')  # Use new headless mode
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        self.driver = webdriver.Chrome(options=options)
        
        # Remove webdriver flag
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        print("✓ Chrome driver initialized")
    
    def close(self):
        """Close browser"""
        if self.driver:
            self.driver.quit()
            self.driver = None
    
    def _clean_number(self, text: str) -> Optional[float]:
        """Clean and convert text to number"""
        if not text or text.strip() in ['N/A', '-', '']:
            return None
        
        try:
            # Remove currency symbols, commas, percent signs
            cleaned = re.sub(r'[Rs,₹NPR%\s]', '', text.strip())
            
            # Handle 'Cr' (Crore), 'M' (Million), 'K' (Thousand), 'B' (Billion)
            multipliers = {
                'Cr': 10000000,     # 1 Crore = 10 million
                'M': 1000000,       # 1 Million
                'K': 1000,          # 1 Thousand
                'B': 1000000000,    # 1 Billion
                'Arb': 1000000000   # 1 Arab = 1 Billion
            }
            
            for suffix, multiplier in multipliers.items():
                if suffix in cleaned:
                    cleaned = cleaned.replace(suffix, '')
                    return float(cleaned) * multiplier
            
            # Handle parentheses (negative numbers)
            if '(' in cleaned and ')' in cleaned:
                cleaned = cleaned.replace('(', '-').replace(')', '')
            
            return float(cleaned)
        except:
            return None
    
    def scrape_fundamental_data(self, symbol: str) -> Dict:
        """
        Scrape fundamental data for a stock symbol
        
        Args:
            symbol: Stock symbol (e.g., 'IGI')
            
        Returns:
            Dictionary with fundamental metrics
        """
        
        if not self.driver:
            self.init_driver()
        
        fundamental_data = {
            'symbol': symbol.upper(),
            'current_price': None,
            'market_cap': None,
            'pe_ratio': None,
            'pb_ratio': None,
            'eps': None,
            'book_value_per_share': None,
            'roe': None,
            'dividend_yield': None,
            'annual_dividend': None,
            'debt_to_equity': None,
            'current_ratio': None,
            'net_income': None,
            'shareholders_equity': None,
            'total_debt': None,
            'current_assets': None,
            'current_liabilities': None,
            'total_shares': None,
            'paid_up_capital': None,
            '52_week_high': None,
            '52_week_low': None,
            'sector': None,
            'scraped_from': 'nepsealpha.com',
            'scrape_timestamp': None
        }
        
        try:
            # Navigate to stock info page
            url = f"{self.base_url}/stocks/{symbol.upper()}/info"
            print(f"\n→ Loading {url}...")
            self.driver.get(url)
            time.sleep(5)  # Wait longer for JavaScript to load
            
            # Scroll to load lazy content
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            self.driver.execute_script("window.scrollTo(0, 0);")
            time.sleep(1)
            
            # Wait for content to load
            try:
                WebDriverWait(self.driver, 15).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                print("✓ Page loaded")
            except TimeoutException:
                print("⚠️ Page load timeout")
                return fundamental_data
            
            # Get current price (usually displayed prominently)
            try:
                # Try multiple selectors for price
                price_selectors = [
                    "//span[contains(@class, 'price')]",
                    "//div[contains(@class, 'current-price')]",
                    "//h1[contains(@class, 'price')]",
                    "//strong[contains(text(), 'Rs')]",
                ]
                
                for selector in price_selectors:
                    try:
                        price_elem = self.driver.find_element(By.XPATH, selector)
                        price_text = price_elem.text
                        price = self._clean_number(price_text)
                        if price and price > 0:
                            fundamental_data['current_price'] = price
                            print(f"  ✓ Current Price: Rs {price}")
                            break
                    except:
                        continue
            except Exception as e:
                print(f"  ⚠️ Could not find price: {e}")
            
            # Extract data from tables or divs
            # NepalAlpha typically shows data in tables or key-value pairs
            
            # Try to find all text elements and extract key-value pairs
            try:
                # Get all text content
                page_text = self.driver.find_element(By.TAG_NAME, "body").text
                
                # Parse common patterns FIRST (more accurate)
                self._extract_from_text(page_text, fundamental_data)
                
            except Exception as e:
                print(f"  ⚠️ Error extracting data: {e}")
            
            # Don't extract from tables as they might have historical data
            # The text extraction is more accurate for current values
            
            # Print what we found
            print(f"\n📊 Extracted Fundamental Data for {symbol}:")
            for key, value in fundamental_data.items():
                if value is not None and key not in ['symbol', 'scraped_from', 'scrape_timestamp']:
                    print(f"  • {key}: {value}")
            
            import datetime
            fundamental_data['scrape_timestamp'] = datetime.datetime.now().isoformat()
            
        except Exception as e:
            print(f"✗ Error scraping fundamental data: {e}")
            import traceback
            traceback.print_exc()
        
        return fundamental_data
    
    def _extract_from_text(self, text: str, data: Dict):
        """Extract metrics from page text using regex patterns"""
        
        patterns = {
            'pe_ratio': r'PERatioTTM\s+([0-9.,]+)',  # Match PERatioTTM specifically
            'pb_ratio': r'PBRatio\s+([0-9.,]+)\s+\(3-5\s+Yrs',  # Match PBRatio specifically
            'eps': r'EPSTTM\s+([0-9.,]+)',  # Match EPSTTM specifically
            'book_value_per_share': r'Book\s+Value\s+NPR\s+([0-9.,]+)',  # Match Book Value NPR
            'roe': r'ROETTM\s+([0-9.,]+)\s*%',  # Match ROETTM
            'market_cap': r'Market\s+Capitalization\s+NPR\s+([0-9.,]+)',
            'paid_up_capital': r'Paid\s+Up\s+Capital\s+NPR\s+([0-9.,]+)',
            'current_price': r'^\s*([0-9.,]+)\s+\([-+]?[0-9.,]+%?\)\s*$',  # Price at start of line
            'sector': r'Sector:\s+([^\n]+)',
            '52_week_high': r'52\s+Weeks\s+High/Low.*NPR\s+([0-9.,]+)\s*/\s*NPR',
            '52_week_low': r'52\s+Weeks\s+High/Low.*NPR\s+[0-9.,]+\s*/\s*NPR\s+([0-9.,]+)',
        }
        
        for key, pattern in patterns.items():
            try:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    if key == 'sector':
                        data[key] = match.group(1).strip()
                    else:
                        value = self._clean_number(match.group(1))
                        if value is not None:
                            data[key] = value
            except Exception as e:
                continue
    
    def _extract_from_table(self, table, data: Dict):
        """Extract data from HTML table"""
        
        try:
            rows = table.find_elements(By.TAG_NAME, "tr")
            
            for row in rows:
                try:
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if len(cells) >= 2:
                        label = cells[0].text.strip().lower()
                        value_text = cells[1].text.strip()
                        
                        # Map label to data key
                        if 'p/e' in label or 'pe ratio' in label:
                            data['pe_ratio'] = self._clean_number(value_text)
                        elif 'p/b' in label or 'pb ratio' in label:
                            data['pb_ratio'] = self._clean_number(value_text)
                        elif 'eps' in label:
                            data['eps'] = self._clean_number(value_text)
                        elif 'book value' in label:
                            data['book_value_per_share'] = self._clean_number(value_text)
                        elif 'roe' in label or 'return on equity' in label:
                            data['roe'] = self._clean_number(value_text)
                        elif 'market cap' in label:
                            data['market_cap'] = self._clean_number(value_text)
                        elif 'dividend yield' in label:
                            data['dividend_yield'] = self._clean_number(value_text)
                        elif 'debt' in label and 'equity' in label:
                            data['debt_to_equity'] = self._clean_number(value_text)
                        elif 'current ratio' in label:
                            data['current_ratio'] = self._clean_number(value_text)
                        elif 'sector' in label:
                            data['sector'] = value_text
                except:
                    continue
        except:
            pass
    
    def __enter__(self):
        """Context manager entry"""
        self.init_driver()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


if __name__ == "__main__":
    # Test scraper
    print("Testing NepalAlpha Scraper\n")
    
    with NepalAlphaScraper(headless=False) as scraper:
        # Test with IGI
        data = scraper.scrape_fundamental_data('IGI')
        
        print(f"\n{'='*60}")
        print(f"FUNDAMENTAL DATA: {data['symbol']}")
        print(f"{'='*60}")
        
        if data['current_price']:
            print(f"Current Price: Rs {data['current_price']:.2f}")
        if data['pe_ratio']:
            print(f"P/E Ratio: {data['pe_ratio']:.2f}")
        if data['pb_ratio']:
            print(f"P/B Ratio: {data['pb_ratio']:.2f}")
        if data['eps']:
            print(f"EPS: Rs {data['eps']:.2f}")
        if data['market_cap']:
            print(f"Market Cap: Rs {data['market_cap']:,.0f}")
        if data['roe']:
            print(f"ROE: {data['roe']:.2f}%")
        
        print(f"\nSource: {data['scraped_from']}")
        print(f"Scraped at: {data['scrape_timestamp']}")
