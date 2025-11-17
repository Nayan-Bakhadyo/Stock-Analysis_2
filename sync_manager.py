"""
Sync Manager for Incremental Data Updates
Intelligently fetches only new data since last update
"""
import sqlite3
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import config
from sharesansar_price_scraper import ShareSansarPriceScraper
from sharesansar_news_scraper import ShareSansarNewsScraper


class SyncManager:
    """Manage incremental data synchronization"""
    
    def __init__(self):
        self.db_path = config.DB_PATH
        self._init_sync_tables()
    
    def _init_sync_tables(self):
        """Initialize sync tracking tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Table to track last sync times
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sync_history (
                symbol TEXT PRIMARY KEY,
                last_price_update TEXT,
                last_news_update TEXT,
                last_fundamental_update TEXT,
                price_records_count INTEGER DEFAULT 0,
                news_articles_count INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Table to cache news articles
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS news_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                title TEXT NOT NULL,
                url TEXT UNIQUE,
                content TEXT,
                published_date TEXT,
                sentiment_score REAL,
                sentiment_label TEXT,
                scraped_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, title)
            )
        """)
        
        # Index for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sync_symbol ON sync_history(symbol)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_news_symbol_date 
            ON news_cache(symbol, published_date DESC)
        """)
        
        conn.commit()
        conn.close()
    
    def get_last_update_info(self, symbol: str) -> Dict:
        """Get last update information for a symbol"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT last_price_update, last_news_update, last_fundamental_update,
                   price_records_count, news_articles_count
            FROM sync_history
            WHERE symbol = ?
        """, (symbol.upper(),))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'last_price_update': row[0],
                'last_news_update': row[1],
                'last_fundamental_update': row[2],
                'price_records': row[3] or 0,
                'news_articles': row[4] or 0
            }
        else:
            return {
                'last_price_update': None,
                'last_news_update': None,
                'last_fundamental_update': None,
                'price_records': 0,
                'news_articles': 0
            }
    
    def get_latest_price_date(self, symbol: str) -> Optional[datetime]:
        """Get the most recent price date in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT MAX(date) FROM price_history
            WHERE UPPER(symbol) = UPPER(?)
        """, (symbol,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row and row[0]:
            return datetime.strptime(row[0], '%Y-%m-%d')
        return None
    
    def sync_price_history(self, symbol: str, force_full: bool = False, limit_rows: int = None) -> Dict:
        """
        Intelligently sync price history - only fetch new data
        
        Args:
            symbol: Stock symbol
            force_full: Force full scrape instead of incremental
            limit_rows: Limit number of rows for testing (e.g., 50)
            
        Returns:
            Dict with sync results
        """
        symbol = symbol.upper()
        update_info = self.get_last_update_info(symbol)
        latest_date = self.get_latest_price_date(symbol)
        
        limit_msg = f" (limited to {limit_rows} rows for testing)" if limit_rows else ""
        print(f"\n🔄 Syncing price history for {symbol}...{limit_msg}")
        
        if not force_full and latest_date:
            days_old = (datetime.now() - latest_date).days
            print(f"  ℹ️ Latest data: {latest_date.strftime('%Y-%m-%d')} ({days_old} days old)")
            
            if days_old == 0:
                print(f"  ✓ Already up-to-date (today's data exists)")
                return {
                    'status': 'up_to_date',
                    'records_added': 0,
                    'total_records': update_info['price_records']
                }
            elif days_old <= 7:
                print(f"  → Fetching only new data (incremental update)")
                # Only scrape recent data - more efficient
                new_records = self._scrape_incremental_prices(symbol, days=days_old + 5)
            else:
                print(f"  → Data is stale, performing full sync")
                new_records = self._scrape_all_prices(symbol, limit_rows=limit_rows)
        else:
            print(f"  → No existing data, performing full sync")
            new_records = self._scrape_all_prices(symbol, limit_rows=limit_rows)
        
        # Update sync history
        self._update_sync_history(symbol, 'price', len(new_records))
        
        return {
            'status': 'updated',
            'records_added': len(new_records),
            'total_records': update_info['price_records'] + len(new_records)
        }
    
    def _scrape_incremental_prices(self, symbol: str, days: int = 30) -> List[Dict]:
        """Scrape only recent price data"""
        scraper = ShareSansarPriceScraper(headless=True)
        
        try:
            # Scrape recent data
            df = scraper.scrape_price_history(symbol, days=days)
            
            if df.empty:
                return []
            
            # Rename Date column if needed
            if 'Date' in df.columns and 'date' not in df.columns:
                df = df.rename(columns={'Date': 'date'})
            
            # Filter out data that already exists in DB
            latest_date = self.get_latest_price_date(symbol)
            if latest_date:
                df['date'] = pd.to_datetime(df['date'])
                df = df[df['date'] > latest_date]
            
            # Save to database
            if not df.empty:
                self._save_price_data(df, symbol)
                print(f"  ✓ Added {len(df)} new price records")
                return df.to_dict('records')
            else:
                print(f"  ℹ️ No new price data available")
                return []
                
        finally:
            scraper.close_driver()
    
    def _scrape_all_prices(self, symbol: str, limit_rows: int = None) -> List[Dict]:
        """Scrape all available price history"""
        scraper = ShareSansarPriceScraper(headless=True)
        
        try:
            # For testing, limit to specific number of days/rows
            days = 50 if limit_rows == 50 else 9999
            df = scraper.scrape_price_history(symbol, days=days)
            
            if df.empty:
                return []
            
            # If limit specified, take only first N rows
            if limit_rows:
                df = df.head(limit_rows)
            
            # Save to database
            self._save_price_data(df, symbol)
            print(f"  ✓ Saved {len(df)} price records")
            return df.to_dict('records')
                
        finally:
            scraper.close_driver()
    
    def _save_price_data(self, df: pd.DataFrame, symbol: str):
        """Save price data to database"""
        conn = sqlite3.connect(self.db_path)
        
        # Ensure symbol column exists
        if 'symbol' not in df.columns:
            df['symbol'] = symbol.upper()
        
        # Drop S.N. column if it exists (not in schema)
        if 'S.N.' in df.columns:
            df = df.drop(columns=['S.N.'])
        
        # Rename columns to match database schema
        column_mapping = {
            'Ltp': 'close',
            'ltp': 'close',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Date': 'date',
            'Qty': 'volume',
            'qty': 'volume',
            'Turnover': 'turnover',
            '% Change': 'change_percent'
        }
        
        # Apply column mapping
        df = df.rename(columns=column_mapping)
        
        # Clean numeric columns - remove commas and convert to numeric
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                # Remove commas and convert to numeric
                df[col] = df[col].astype(str).str.replace(',', '').astype(float)
        
        # Clean date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        
        # Ensure required columns exist and drop extras not in schema
        required_cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
        df = df[[col for col in required_cols if col in df.columns]]
        
        # Save to database (ignore duplicates)
        df.to_sql('price_history', conn, if_exists='append', index=False)
        
        conn.close()
    
    def sync_news(self, symbol: str, max_articles: int = 10) -> Dict:
        """
        Intelligently sync news articles - only fetch new ones
        
        Args:
            symbol: Stock symbol
            max_articles: Maximum articles to fetch
            
        Returns:
            Dict with sync results
        """
        symbol = symbol.upper()
        
        print(f"\n📰 Syncing news for {symbol}...")
        
        # Get existing news titles to avoid duplicates
        existing_titles = self._get_existing_news_titles(symbol)
        print(f"  ℹ️ Found {len(existing_titles)} existing articles in cache")
        
        # Scrape news
        scraper = ShareSansarNewsScraper(headless=True)
        
        try:
            # Get news data
            news_data = scraper.scrape_company_news(symbol, max_articles=max_articles)
            
            if not news_data:
                print(f"  ℹ️ No news articles found")
                return {'status': 'no_data', 'articles_added': 0}
            
            # Filter out duplicates
            new_articles = [
                article for article in news_data
                if article['title'] not in existing_titles
            ]
            
            if not new_articles:
                print(f"  ✓ Already up-to-date (no new articles)")
                return {'status': 'up_to_date', 'articles_added': 0}
            
            # Save new articles to cache
            self._save_news_to_cache(symbol, new_articles)
            
            # Update sync history
            self._update_sync_history(symbol, 'news', len(new_articles))
            
            print(f"  ✓ Added {len(new_articles)} new articles")
            
            return {
                'status': 'updated',
                'articles_added': len(new_articles),
                'total_articles': len(existing_titles) + len(new_articles)
            }
                
        finally:
            scraper.close_driver()
    
    def _get_existing_news_titles(self, symbol: str) -> set:
        """Get existing news article titles from cache"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT title FROM news_cache
            WHERE symbol = ?
        """, (symbol.upper(),))
        
        titles = {row[0] for row in cursor.fetchall()}
        conn.close()
        
        return titles
    
    def _save_news_to_cache(self, symbol: str, articles: List[Dict]):
        """Save news articles to cache"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for article in articles:
            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO news_cache 
                    (symbol, title, url, content, published_date, sentiment_score, sentiment_label)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol.upper(),
                    article.get('title', ''),
                    article.get('url', ''),
                    article.get('content', ''),
                    article.get('date', ''),
                    article.get('sentiment_score', 0),
                    article.get('sentiment_label', 'NEUTRAL')
                ))
            except Exception as e:
                print(f"  ⚠️ Error saving article: {e}")
                continue
        
        conn.commit()
        conn.close()
    
    def get_cached_news(self, symbol: str, days: int = 180, limit: int = None) -> List[Dict]:
        """Get cached news articles for a symbol"""
        conn = sqlite3.connect(self.db_path)
        
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        query = """
            SELECT title, url, content, published_date, sentiment_score, sentiment_label
            FROM news_cache
            WHERE symbol = ? AND published_date >= ?
            ORDER BY published_date DESC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        df = pd.read_sql(query, conn, params=(symbol.upper(), cutoff_date))
        conn.close()
        
        if df.empty:
            return []
        
        return df.to_dict('records')
    
    def _update_sync_history(self, symbol: str, data_type: str, records_count: int):
        """Update sync history after successful sync"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        current_time = datetime.now().isoformat()
        
        if data_type == 'price':
            cursor.execute("""
                INSERT INTO sync_history (symbol, last_price_update, price_records_count, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    last_price_update = ?,
                    price_records_count = price_records_count + ?,
                    updated_at = ?
            """, (symbol.upper(), current_time, records_count, current_time,
                  current_time, records_count, current_time))
        
        elif data_type == 'news':
            cursor.execute("""
                INSERT INTO sync_history (symbol, last_news_update, news_articles_count, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    last_news_update = ?,
                    news_articles_count = news_articles_count + ?,
                    updated_at = ?
            """, (symbol.upper(), current_time, records_count, current_time,
                  current_time, records_count, current_time))
        
        conn.commit()
        conn.close()
    
    def bulk_sync(self, symbols: List[str], sync_price: bool = True, sync_news: bool = True) -> Dict:
        """
        Sync multiple symbols efficiently
        
        Args:
            symbols: List of stock symbols
            sync_price: Whether to sync price data
            sync_news: Whether to sync news data
            
        Returns:
            Summary of sync operations
        """
        results = {
            'success': [],
            'failed': [],
            'total_price_records': 0,
            'total_news_articles': 0
        }
        
        print(f"\n🔄 Starting bulk sync for {len(symbols)} symbols...")
        print(f"   Price: {'✓' if sync_price else '✗'} | News: {'✓' if sync_news else '✗'}")
        
        for i, symbol in enumerate(symbols, 1):
            print(f"\n[{i}/{len(symbols)}] Processing {symbol}...")
            
            try:
                if sync_price:
                    price_result = self.sync_price_history(symbol)
                    results['total_price_records'] += price_result.get('records_added', 0)
                
                if sync_news:
                    news_result = self.sync_news(symbol)
                    results['total_news_articles'] += news_result.get('articles_added', 0)
                
                results['success'].append(symbol)
                time.sleep(2)  # Rate limiting
                
            except Exception as e:
                print(f"  ✗ Error syncing {symbol}: {e}")
                results['failed'].append(symbol)
                continue
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"SYNC SUMMARY")
        print(f"{'='*60}")
        print(f"✓ Successful: {len(results['success'])}/{len(symbols)}")
        print(f"✗ Failed: {len(results['failed'])}")
        print(f"📊 Total price records added: {results['total_price_records']}")
        print(f"📰 Total news articles added: {results['total_news_articles']}")
        
        return results


if __name__ == "__main__":
    # Test sync manager
    sync = SyncManager()
    
    # Test single symbol sync
    print("Testing Sync Manager")
    print("=" * 60)
    
    symbol = "IGI"
    
    # Sync price history
    price_result = sync.sync_price_history(symbol)
    print(f"\nPrice sync result: {price_result}")
    
    # Sync news
    news_result = sync.sync_news(symbol)
    print(f"\nNews sync result: {news_result}")
    
    # Get cached news
    cached_news = sync.get_cached_news(symbol, days=90)
    print(f"\n✓ Found {len(cached_news)} cached articles")
    
    # Show sync info
    info = sync.get_last_update_info(symbol)
    print(f"\nSync Info for {symbol}:")
    print(f"  Last price update: {info['last_price_update']}")
    print(f"  Last news update: {info['last_news_update']}")
    print(f"  Total price records: {info['price_records']}")
    print(f"  Total news articles: {info['news_articles']}")
