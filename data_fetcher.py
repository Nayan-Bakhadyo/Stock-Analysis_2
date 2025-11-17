"""Data fetcher module for NEPSE stock data - Web Scraping Based"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time
import json
from pathlib import Path
import re
import config


class NepseDataFetcher:
    """Fetch historical and real-time data from NEPSE via web scraping"""
    
    def __init__(self):
        self.sources = config.DATA_SOURCES
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        })
    
    def get_company_list(self) -> pd.DataFrame:
        """Scrape list of all companies listed on NEPSE"""
        try:
            # Try MeroLagani first (has comprehensive company list)
            url = self.sources['merolagani']['company_list']
            print(f"Fetching company list from {url}")
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            companies = []
            
            # Find table with company listings
            table = soup.find('table', {'class': 'table'}) or soup.find('table')
            
            if table:
                rows = table.find_all('tr')[1:]  # Skip header
                
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 2:
                        symbol = cols[0].get_text(strip=True)
                        name = cols[1].get_text(strip=True) if len(cols) > 1 else ''
                        
                        if symbol:
                            companies.append({
                                'symbol': symbol,
                                'name': name,
                                'source': 'merolagani'
                            })
            
            df = pd.DataFrame(companies)
            
            if not df.empty:
                # Save to database
                self._save_to_db(df, 'companies')
                print(f"✓ Found {len(df)} companies")
            else:
                # Load from database if scraping failed
                df = self._load_from_db('companies')
            
            return df
            
        except Exception as e:
            print(f"Error fetching company list: {e}")
            # Try to load from database
            return self._load_from_db('companies')
    
    def get_stock_price_history(self, symbol: str, days: int = None) -> pd.DataFrame:
        """Get historical price data for a stock - checks database first
        
        Args:
            symbol: Stock symbol
            days: Number of days to fetch. None = all available data
        """
        try:
            # First, try to load from database
            df = self._load_price_history_from_db(symbol, days)
            
            if not df.empty:
                print(f"✓ Loaded {len(df)} days of price data for {symbol} from database")
                return df
            
            # If no data in database, try scraping
            print(f"⚠️ No data in database for {symbol}, attempting to scrape...")
            
            # Try NepsAlpha first (Priority 1)
            df = self._scrape_nepsealpha_history(symbol, days)
            
            if df.empty:
                # Try MeroLagani as backup
                df = self._scrape_merolagani_history(symbol, days)
            
            if df.empty:
                # Try ShareSansar as last resort
                df = self._scrape_sharesansar_history(symbol, days)
            
            if not df.empty:
                # Standardize and save
                df = self._standardize_price_data(df)
                df['symbol'] = symbol
                self._save_to_db(df, 'price_history', if_exists='append')
                print(f"✓ Fetched {len(df)} days of price data for {symbol}")
            
            return df
            
        except Exception as e:
            print(f"Error fetching price history: {e}")
            return self._load_price_history_from_db(symbol, days)
    
    def _scrape_nepsealpha_history(self, symbol: str, days: int) -> pd.DataFrame:
        """Scrape historical data from NepsAlpha"""
        try:
            url = self.sources['nepsealpha']['stock_detail'].format(symbol=symbol)
            response = self.session.get(url, timeout=30)
            
            if response.status_code != 200:
                return pd.DataFrame()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for price history table or chart data
            # This is a placeholder - actual implementation depends on site structure
            data = []
            
            # Try to find historical data table
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                for row in rows[1:]:  # Skip header
                    cols = row.find_all('td')
                    if len(cols) >= 6:
                        try:
                            data.append({
                                'date': cols[0].get_text(strip=True),
                                'open': self._parse_float(cols[1].get_text(strip=True)),
                                'high': self._parse_float(cols[2].get_text(strip=True)),
                                'low': self._parse_float(cols[3].get_text(strip=True)),
                                'close': self._parse_float(cols[4].get_text(strip=True)),
                                'volume': self._parse_float(cols[5].get_text(strip=True))
                            })
                        except:
                            continue
            
            return pd.DataFrame(data)
            
        except Exception as e:
            print(f"NepsAlpha scraping error: {e}")
            return pd.DataFrame()
    
    def _scrape_merolagani_history(self, symbol: str, days: int) -> pd.DataFrame:
        """Scrape historical data from MeroLagani"""
        try:
            url = self.sources['merolagani']['stock_detail'].format(symbol=symbol)
            response = self.session.get(url, timeout=30)
            
            if response.status_code != 200:
                return pd.DataFrame()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            data = []
            
            # Find price history section
            price_section = soup.find('div', {'id': 'graph1'}) or soup.find('div', {'class': 'card-header'})
            
            if price_section:
                # Look for data in script tags or hidden elements
                scripts = soup.find_all('script')
                for script in scripts:
                    if 'chartData' in script.text or 'priceData' in script.text:
                        # Extract JSON data from script
                        try:
                            # This is a placeholder - actual parsing depends on site structure
                            pass
                        except:
                            continue
            
            return pd.DataFrame(data)
            
        except Exception as e:
            print(f"MeroLagani scraping error: {e}")
            return pd.DataFrame()
    
    def _scrape_sharesansar_history(self, symbol: str, days: int) -> pd.DataFrame:
        """Scrape historical data from ShareSansar"""
        try:
            url = self.sources['sharesansar']['stock_detail'].format(symbol=symbol.lower())
            response = self.session.get(url, timeout=30)
            
            if response.status_code != 200:
                return pd.DataFrame()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            data = []
            
            # Find historical price table
            tables = soup.find_all('table', {'class': 'table'})
            for table in tables:
                rows = table.find_all('tr')
                for row in rows[1:]:
                    cols = row.find_all('td')
                    if len(cols) >= 6:
                        try:
                            data.append({
                                'date': cols[0].get_text(strip=True),
                                'close': self._parse_float(cols[1].get_text(strip=True)),
                                'high': self._parse_float(cols[2].get_text(strip=True)),
                                'low': self._parse_float(cols[3].get_text(strip=True)),
                                'open': self._parse_float(cols[1].get_text(strip=True)),  # Use close as open if not available
                                'volume': self._parse_float(cols[5].get_text(strip=True)) if len(cols) > 5 else 0
                            })
                        except:
                            continue
            
            return pd.DataFrame(data)
            
        except Exception as e:
            print(f"ShareSansar scraping error: {e}")
            return pd.DataFrame()
    
    def get_live_price(self, symbol: str) -> Dict:
        """Scrape current live trading price"""
        try:
            # Try NepsAlpha first
            price_data = self._scrape_nepsealpha_live(symbol)
            
            if not price_data:
                # Try MeroLagani
                price_data = self._scrape_merolagani_live(symbol)
            
            if not price_data:
                # Try ShareSansar
                price_data = self._scrape_sharesansar_live(symbol)
            
            return price_data
            
        except Exception as e:
            print(f"Error fetching live price: {e}")
            return {}
    
    def _scrape_nepsealpha_live(self, symbol: str) -> Dict:
        """Get live price from NepsAlpha"""
        try:
            url = self.sources['nepsealpha']['stock_detail'].format(symbol=symbol)
            response = self.session.get(url, timeout=30)
            
            if response.status_code != 200:
                return {}
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract live price data from the page
            # This is a placeholder - actual selectors depend on site structure
            price_element = soup.find('span', {'class': 'ltp'}) or soup.find('div', {'class': 'price'})
            
            if price_element:
                ltp = self._parse_float(price_element.get_text(strip=True))
                
                return {
                    'symbol': symbol,
                    'ltp': ltp,
                    'open': ltp,  # Placeholder
                    'high': ltp,
                    'low': ltp,
                    'close': ltp,
                    'volume': 0,
                    'timestamp': datetime.now().isoformat(),
                    'source': 'nepsealpha'
                }
            
            return {}
            
        except Exception as e:
            print(f"NepsAlpha live price error: {e}")
            return {}
    
    def _scrape_merolagani_live(self, symbol: str) -> Dict:
        """Get live price from MeroLagani"""
        try:
            url = self.sources['merolagani']['stock_detail'].format(symbol=symbol)
            response = self.session.get(url, timeout=30)
            
            if response.status_code != 200:
                return {}
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find LTP and other details
            price_div = soup.find('div', {'id': 'top_detail'}) or soup.find('div', {'class': 'company-detail'})
            
            if price_div:
                # Extract LTP
                ltp_element = price_div.find('span', {'id': 'ltp'}) or price_div.find('strong')
                
                if ltp_element:
                    ltp = self._parse_float(ltp_element.get_text(strip=True))
                    
                    return {
                        'symbol': symbol,
                        'ltp': ltp,
                        'open': ltp,
                        'high': ltp,
                        'low': ltp,
                        'close': ltp,
                        'volume': 0,
                        'timestamp': datetime.now().isoformat(),
                        'source': 'merolagani'
                    }
            
            return {}
            
        except Exception as e:
            print(f"MeroLagani live price error: {e}")
            return {}
    
    def _scrape_sharesansar_live(self, symbol: str) -> Dict:
        """Get live price from ShareSansar"""
        try:
            url = self.sources['sharesansar']['stock_detail'].format(symbol=symbol.lower())
            response = self.session.get(url, timeout=30)
            
            if response.status_code != 200:
                return {}
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find price information
            price_section = soup.find('div', {'class': 'company-overview'})
            
            if price_section:
                ltp_element = price_section.find('span', {'class': 'ltp'})
                
                if ltp_element:
                    ltp = self._parse_float(ltp_element.get_text(strip=True))
                    
                    return {
                        'symbol': symbol,
                        'ltp': ltp,
                        'open': ltp,
                        'high': ltp,
                        'low': ltp,
                        'close': ltp,
                        'volume': 0,
                        'timestamp': datetime.now().isoformat(),
                        'source': 'sharesansar'
                    }
            
            return {}
            
        except Exception as e:
            print(f"ShareSansar live price error: {e}")
            return {}
    
    def get_market_summary(self) -> Dict:
        """Scrape overall market summary"""
        try:
            # Try MeroLagani market page
            url = self.sources['merolagani']['market']
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            summary = {
                'source': 'merolagani',
                'timestamp': datetime.now().isoformat()
            }
            
            # Extract market summary data
            # This is a placeholder - actual selectors depend on site structure
            index_element = soup.find('div', {'class': 'nepse-index'})
            
            if index_element:
                summary['nepse_index'] = index_element.get_text(strip=True)
            
            return summary
            
        except Exception as e:
            print(f"Error fetching market summary: {e}")
            return {'error': str(e)}
    
    def get_top_gainers(self, limit: int = 10) -> pd.DataFrame:
        """Scrape top gaining stocks"""
        try:
            url = self.sources['merolagani'].get('top_gainers', '')
            
            if not url:
                return pd.DataFrame()
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            gainers = []
            
            # Find gainers table
            table = soup.find('table', {'class': 'table'}) or soup.find('table')
            
            if table:
                rows = table.find_all('tr')[1:limit+1]
                
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 3:
                        gainers.append({
                            'symbol': cols[0].get_text(strip=True),
                            'ltp': self._parse_float(cols[1].get_text(strip=True)),
                            'change': cols[2].get_text(strip=True)
                        })
            
            return pd.DataFrame(gainers)
            
        except Exception as e:
            print(f"Error fetching top gainers: {e}")
            return pd.DataFrame()
    
    def get_top_losers(self, limit: int = 10) -> pd.DataFrame:
        """Scrape top losing stocks"""
        try:
            url = self.sources['merolagani'].get('top_losers', '')
            
            if not url:
                return pd.DataFrame()
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            losers = []
            
            # Find losers table
            table = soup.find('table', {'class': 'table'}) or soup.find('table')
            
            if table:
                rows = table.find_all('tr')[1:limit+1]
                
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 3:
                        losers.append({
                            'symbol': cols[0].get_text(strip=True),
                            'ltp': self._parse_float(cols[1].get_text(strip=True)),
                            'change': cols[2].get_text(strip=True)
                        })
            
            return pd.DataFrame(losers)
            
        except Exception as e:
            print(f"Error fetching top losers: {e}")
            return pd.DataFrame()
    
    def scrape_fundamentals(self, symbol: str) -> Dict:
        """Scrape fundamental data from multiple sources"""
        fundamentals = {
            'symbol': symbol,
            'scraped_at': datetime.now().isoformat()
        }
        
        # Try NepsAlpha
        nepsealpha_data = self._scrape_nepsealpha_fundamentals(symbol)
        fundamentals.update(nepsealpha_data)
        
        # Try MeroLagani
        merolagani_data = self._scrape_merolagani_fundamentals(symbol)
        fundamentals.update(merolagani_data)
        
        # Try ShareSansar
        sharesansar_data = self._scrape_sharesansar_fundamentals(symbol)
        fundamentals.update(sharesansar_data)
        
        return fundamentals
    
    def _scrape_nepsealpha_fundamentals(self, symbol: str) -> Dict:
        """Scrape fundamentals from NepsAlpha"""
        try:
            url = self.sources['nepsealpha']['stock_detail'].format(symbol=symbol)
            response = self.session.get(url, timeout=30)
            
            if response.status_code != 200:
                return {}
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            data = {}
            
            # Extract EPS, P/E, Book Value, etc.
            # This is a placeholder - actual selectors depend on site structure
            
            return data
            
        except Exception as e:
            print(f"Error scraping NepsAlpha fundamentals: {e}")
            return {}
    
    def _scrape_merolagani_fundamentals(self, symbol: str) -> Dict:
        """Scrape fundamentals from MeroLagani"""
        try:
            url = self.sources['merolagani']['stock_detail'].format(symbol=symbol)
            response = self.session.get(url, timeout=30)
            
            if response.status_code != 200:
                return {}
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            data = {}
            
            # Extract fundamental metrics
            # Look for key financial ratios
            metrics_section = soup.find('div', {'class': 'financial-info'})
            
            if metrics_section:
                # Extract P/E, EPS, Book Value, Market Cap, etc.
                pass
            
            return data
            
        except Exception as e:
            print(f"Error scraping MeroLagani fundamentals: {e}")
            return {}
    
    def _scrape_sharesansar_fundamentals(self, symbol: str) -> Dict:
        """Scrape fundamentals from ShareSansar"""
        try:
            url = self.sources['sharesansar']['stock_detail'].format(symbol=symbol.lower())
            response = self.session.get(url, timeout=30)
            
            if response.status_code != 200:
                return {}
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            data = {}
            
            # Extract financial ratios
            # This is a placeholder
            
            return data
            
        except Exception as e:
            print(f"Error scraping ShareSansar fundamentals: {e}")
            return {}
    
    def _parse_float(self, value: str) -> float:
        """Parse string to float, handling commas and special characters"""
        try:
            # Remove commas, percentage signs, and other non-numeric characters
            cleaned = re.sub(r'[^\d.-]', '', value)
            return float(cleaned) if cleaned else 0.0
        except:
            return 0.0
    
    def _standardize_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names across different data sources"""
        # Map various possible column names to standard names
        column_mapping = {
            'businessDate': 'date',
            'tradingDate': 'date',
            'Date': 'date',
            'date': 'date',
            'openPrice': 'open',
            'Open': 'open',
            'open': 'open',
            'highPrice': 'high',
            'High': 'high',
            'high': 'high',
            'lowPrice': 'low',
            'Low': 'low',
            'low': 'low',
            'closePrice': 'close',
            'Close': 'close',
            'ltp': 'close',
            'close': 'close',
            'totalTradeQuantity': 'volume',
            'Volume': 'volume',
            'volume': 'volume',
        }
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Ensure we have essential columns
        essential_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        for col in essential_cols:
            if col not in df.columns:
                if col == 'date':
                    df[col] = datetime.now().date()
                else:
                    df[col] = 0
        
        # Convert date to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Ensure numeric columns are float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df[essential_cols]
    
    def _save_to_db(self, df: pd.DataFrame, table_name: str, if_exists: str = 'replace'):
        """Save DataFrame to SQLite database"""
        try:
            conn = sqlite3.connect(config.DB_PATH)
            df.to_sql(table_name, conn, if_exists=if_exists, index=False)
            conn.close()
        except Exception as e:
            print(f"Error saving to database: {e}")
    
    def _load_from_db(self, table_name: str) -> pd.DataFrame:
        """Load DataFrame from SQLite database"""
        try:
            conn = sqlite3.connect(config.DB_PATH)
            df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
            conn.close()
            return df
        except Exception as e:
            print(f"Error loading from database: {e}")
            return pd.DataFrame()
    
    def _load_price_history_from_db(self, symbol: str, days: int = None) -> pd.DataFrame:
        """Load price history from database
        
        Args:
            symbol: Stock symbol
            days: Number of days to fetch. None = all available data
        """
        try:
            conn = sqlite3.connect(config.DB_PATH)
            
            if days is None:
                # Fetch all available data
                query = f"""
                    SELECT * FROM price_history 
                    WHERE UPPER(symbol) = UPPER('{symbol}')
                    ORDER BY date ASC
                """
            else:
                # Fetch limited days
                cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
                query = f"""
                    SELECT * FROM price_history 
                    WHERE UPPER(symbol) = UPPER('{symbol}') 
                    AND date >= '{cutoff_date}'
                    ORDER BY date ASC
                """
            
            df = pd.read_sql(query, conn)
            conn.close()
            
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
            
            return df
        except Exception as e:
            print(f"Error loading price history from database: {e}")
            return pd.DataFrame()
    
    def bulk_update_prices(self, symbols: List[str], delay: float = 2.0):
        """Fetch and update prices for multiple stocks with rate limiting"""
        results = {}
        
        for symbol in symbols:
            try:
                print(f"Fetching data for {symbol}...")
                df = self.get_stock_price_history(symbol)
                results[symbol] = df
                time.sleep(delay)  # Respect rate limits - important for web scraping!
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
                results[symbol] = None
        
        return results
    
    def get_floorsheet_data(self, symbol: Optional[str] = None, days: int = 1) -> pd.DataFrame:
        """
        Scrape floorsheet data (broker-level transactions)
        
        Args:
            symbol: Stock symbol (optional - if None, gets all stocks)
            days: Number of days to fetch (default: 1)
        
        Returns:
            DataFrame with broker transaction details
        """
        try:
            # Try ShareSansar first (has detailed floorsheet)
            df = self._scrape_sharesansar_floorsheet(symbol, days)
            
            if df.empty:
                # Try MeroLagani as backup
                df = self._scrape_merolagani_floorsheet(symbol, days)
            
            if not df.empty:
                # Save to database
                df['scraped_at'] = datetime.now()
                self._save_to_db(df, 'floorsheet_data', if_exists='append')
                print(f"✓ Fetched {len(df)} floorsheet transactions")
            else:
                # Load from database
                df = self._load_floorsheet_from_db(symbol, days)
            
            return df
            
        except Exception as e:
            print(f"Error fetching floorsheet data: {e}")
            return self._load_floorsheet_from_db(symbol, days)
    
    def _scrape_sharesansar_floorsheet(self, symbol: Optional[str], days: int) -> pd.DataFrame:
        """Scrape floorsheet from ShareSansar"""
        try:
            url = self.sources['sharesansar']['floorsheet']
            
            # Add symbol filter if provided
            if symbol:
                url = f"{url}?stock={symbol}"
            
            response = self.session.get(url, timeout=30)
            
            if response.status_code != 200:
                return pd.DataFrame()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            transactions = []
            
            # Find floorsheet table
            table = soup.find('table', {'id': 'floorsheet'}) or soup.find('table', {'class': 'table'})
            
            if table:
                rows = table.find_all('tr')[1:]  # Skip header
                
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 8:
                        try:
                            transactions.append({
                                'date': cols[0].get_text(strip=True),
                                'contract_no': cols[1].get_text(strip=True),
                                'symbol': cols[2].get_text(strip=True),
                                'buyer_broker': cols[3].get_text(strip=True),
                                'seller_broker': cols[4].get_text(strip=True),
                                'quantity': self._parse_float(cols[5].get_text(strip=True)),
                                'rate': self._parse_float(cols[6].get_text(strip=True)),
                                'amount': self._parse_float(cols[7].get_text(strip=True)),
                                'source': 'sharesansar'
                            })
                        except Exception as e:
                            continue
            
            df = pd.DataFrame(transactions)
            
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            return df
            
        except Exception as e:
            print(f"ShareSansar floorsheet scraping error: {e}")
            return pd.DataFrame()
    
    def _scrape_merolagani_floorsheet(self, symbol: Optional[str], days: int) -> pd.DataFrame:
        """Scrape floorsheet from MeroLagani"""
        try:
            # MeroLagani floorsheet URL
            url = 'https://merolagani.com/Floorsheet.aspx'
            
            response = self.session.get(url, timeout=30)
            
            if response.status_code != 200:
                return pd.DataFrame()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            transactions = []
            
            # Find floorsheet table
            table = soup.find('table', {'class': 'table'}) or soup.find('table')
            
            if table:
                rows = table.find_all('tr')[1:]
                
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 7:
                        try:
                            stock_symbol = cols[1].get_text(strip=True)
                            
                            # Filter by symbol if provided
                            if symbol and stock_symbol != symbol:
                                continue
                            
                            transactions.append({
                                'date': datetime.now().date(),  # MeroLagani may not show date
                                'contract_no': cols[0].get_text(strip=True),
                                'symbol': stock_symbol,
                                'buyer_broker': cols[2].get_text(strip=True),
                                'seller_broker': cols[3].get_text(strip=True),
                                'quantity': self._parse_float(cols[4].get_text(strip=True)),
                                'rate': self._parse_float(cols[5].get_text(strip=True)),
                                'amount': self._parse_float(cols[6].get_text(strip=True)),
                                'source': 'merolagani'
                            })
                        except Exception as e:
                            continue
            
            df = pd.DataFrame(transactions)
            
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
            
            return df
            
        except Exception as e:
            print(f"MeroLagani floorsheet scraping error: {e}")
            return pd.DataFrame()
    
    def _load_floorsheet_from_db(self, symbol: Optional[str], days: int) -> pd.DataFrame:
        """Load floorsheet data from database"""
        try:
            conn = sqlite3.connect(config.DB_PATH)
            cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            if symbol:
                query = f"""
                    SELECT * FROM floorsheet_data 
                    WHERE symbol = '{symbol}' 
                    AND date >= '{cutoff_date}'
                    ORDER BY date DESC
                """
            else:
                query = f"""
                    SELECT * FROM floorsheet_data 
                    WHERE date >= '{cutoff_date}'
                    ORDER BY date DESC
                """
            
            df = pd.read_sql(query, conn)
            conn.close()
            
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
            
            return df
        except Exception as e:
            print(f"Error loading floorsheet from database: {e}")
            return pd.DataFrame()
    
    def get_broker_stats(self, broker_code: str, days: int = 30) -> Dict:
        """Get statistics for a specific broker"""
        try:
            floorsheet = self.get_floorsheet_data(days=days)
            
            if floorsheet.empty:
                return {}
            
            # Filter for this broker
            broker_buys = floorsheet[floorsheet['buyer_broker'] == broker_code]
            broker_sells = floorsheet[floorsheet['seller_broker'] == broker_code]
            
            total_bought = broker_buys['quantity'].sum()
            total_sold = broker_sells['quantity'].sum()
            
            # Most traded stocks
            top_stocks_bought = broker_buys.groupby('symbol')['quantity'].sum().sort_values(ascending=False).head(5)
            top_stocks_sold = broker_sells.groupby('symbol')['quantity'].sum().sort_values(ascending=False).head(5)
            
            return {
                'broker_code': broker_code,
                'total_bought': int(total_bought),
                'total_sold': int(total_sold),
                'net_position': int(total_bought - total_sold),
                'buy_transactions': len(broker_buys),
                'sell_transactions': len(broker_sells),
                'top_stocks_bought': top_stocks_bought.to_dict(),
                'top_stocks_sold': top_stocks_sold.to_dict()
            }
            
        except Exception as e:
            print(f"Error fetching broker stats: {e}")
            return {}


if __name__ == "__main__":
    # Test the data fetcher
    fetcher = NepseDataFetcher()
    
    print("Testing NEPSE Data Fetcher (Web Scraping)")
    print("=" * 60)
    
    print("\n1. Fetching company list...")
    companies = fetcher.get_company_list()
    if not companies.empty:
        print(f"✓ Found {len(companies)} companies")
        print(companies.head())
    
    print("\n2. Fetching live price for NABIL...")
    live_price = fetcher.get_live_price("NABIL")
    if live_price:
        print(f"✓ NABIL LTP: {live_price.get('ltp', 'N/A')}")
    
    print("\n3. Fetching price history for NABIL...")
    history = fetcher.get_stock_price_history("NABIL", days=30)
    if not history.empty:
        print(f"✓ Fetched {len(history)} days of data")
        print(history.head())
    
    print("\n4. Fetching top gainers...")
    gainers = fetcher.get_top_gainers(limit=5)
    if not gainers.empty:
        print(f"✓ Top {len(gainers)} gainers:")
        print(gainers)
    
    print("\nNote: Some data may be empty if websites change structure or are unavailable.")
    print("The system will fall back to cached database data when available.")
