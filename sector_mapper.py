"""
Sector Mapper - Maps stock symbols to their sectors
Scrapes from ShareSansar sectorwise-share-price page
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
from pathlib import Path
import config
import re
from typing import Dict, List, Optional


# Map sector names to index codes (used in index_history table)
SECTOR_TO_INDEX = {
    'Commercial Bank': 'BANKING',
    'Development Bank': 'DEVBANK',
    'Finance': 'FINANCE',
    'Microfinance': 'MICROFINANCE',
    'Life Insurance': 'LIFEINSURANCE',
    'Non-Life Insurance': 'NONLIFEINSURANCE',
    'Hydropower': 'HYDROPOWER',
    'Manufacturing and Processing': 'MANUFACTURING',
    'Hotel & Tourism': 'HOTELS',
    'Trading': 'TRADING',
    'Investment': 'INVESTMENT',
    'Others': 'OTHERS',
    'Mutual Fund': 'MUTUAL',
    'Corporate Debentures': None,  # No index for this
    'Promoter Share': None,  # No index for this
    'Government Bonds': None,  # No index for this
}


class SectorMapper:
    """Maps stock symbols to their sectors"""
    
    def __init__(self):
        self.url = "https://www.sharesansar.com/sectorwise-share-price"
        self.db_path = Path(config.DATA_DIR) / "nepse_stocks.db"
        
    def scrape_sectors(self) -> pd.DataFrame:
        """
        Scrape sector information from ShareSansar
        
        Returns:
            DataFrame with columns: symbol, sector, sector_index_code
        """
        print(f"\n{'='*70}")
        print("Scraping sector information from ShareSansar")
        print(f"{'='*70}\n")
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            
            print(f"→ Fetching {self.url}...")
            response = requests.get(self.url, headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            sector_data = []
            current_sector = None
            
            # Find all sector headers and their tables
            # The page structure has h3/h4 headers followed by tables
            
            # Get all text content to find sector patterns
            for element in soup.find_all(['h3', 'h4', 'h5', 'div', 'table']):
                # Check if this is a sector header
                if element.name in ['h3', 'h4', 'h5']:
                    text = element.get_text(strip=True)
                    for sector_name in SECTOR_TO_INDEX.keys():
                        if sector_name.lower() in text.lower():
                            current_sector = sector_name
                            break
                
                # Check if this div contains sector header
                if element.name == 'div':
                    heading = element.find(['h3', 'h4', 'h5', 'strong'])
                    if heading:
                        text = heading.get_text(strip=True)
                        for sector_name in SECTOR_TO_INDEX.keys():
                            if sector_name.lower() in text.lower():
                                current_sector = sector_name
                                break
                
                # Extract symbols from tables
                if element.name == 'table' and current_sector:
                    rows = element.find_all('tr')
                    for row in rows:
                        cells = row.find_all('td')
                        if cells and len(cells) >= 2:
                            # Symbol is usually in the second cell (after S.N.)
                            # or first cell if no S.N.
                            for cell in cells[:3]:  # Check first 3 cells
                                link = cell.find('a')
                                if link:
                                    symbol = link.get_text(strip=True).upper()
                                    # Validate symbol (should be alphanumeric, max 10 chars)
                                    if symbol and re.match(r'^[A-Z0-9]{1,15}$', symbol):
                                        sector_data.append({
                                            'symbol': symbol,
                                            'sector': current_sector,
                                            'sector_index_code': SECTOR_TO_INDEX.get(current_sector)
                                        })
                                        break
            
            # Create DataFrame and remove duplicates
            df = pd.DataFrame(sector_data)
            if not df.empty:
                df = df.drop_duplicates(subset='symbol', keep='first')
                df = df.sort_values('symbol').reset_index(drop=True)
            
            print(f"✓ Found {len(df)} stocks with sector information")
            
            # Print summary by sector
            if not df.empty:
                print(f"\n📊 Stocks per sector:")
                sector_counts = df.groupby('sector').size().sort_values(ascending=False)
                for sector, count in sector_counts.items():
                    idx_code = SECTOR_TO_INDEX.get(sector, 'N/A')
                    print(f"  {sector:<35}: {count:>4} stocks (Index: {idx_code})")
            
            return df
            
        except Exception as e:
            print(f"✗ Error scraping sectors: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def save_to_database(self, df: pd.DataFrame):
        """Save sector data to database and update companies table"""
        if df.empty:
            print("⚠ No data to save")
            return
        
        conn = sqlite3.connect(self.db_path)
        
        # Create stock_sectors table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS stock_sectors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT UNIQUE NOT NULL,
                sector TEXT,
                sector_index_code TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert/update data
        rows_inserted = 0
        for idx in range(len(df)):
            row = df.iloc[idx]
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO stock_sectors 
                    (symbol, sector, sector_index_code, updated_at)
                    VALUES (?, ?, ?, datetime('now'))
                """, (
                    row['symbol'],
                    row['sector'],
                    row['sector_index_code'],
                ))
                rows_inserted += 1
            except Exception as e:
                print(f"  ⚠ Error inserting {row['symbol']}: {e}")
                continue
        
        conn.commit()
        print(f"✓ Saved {rows_inserted} sector records to stock_sectors table")
        
        # Update companies table with sector info
        try:
            # Check if companies table exists
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='companies'")
            if cursor.fetchone():
                # Add sector columns if not exist
                try:
                    conn.execute("ALTER TABLE companies ADD COLUMN sector TEXT")
                except:
                    pass
                
                try:
                    conn.execute("ALTER TABLE companies ADD COLUMN sector_index_code TEXT")
                except:
                    pass
                
                # Update companies with sector info
                result = conn.execute("""
                    UPDATE companies 
                    SET sector = (SELECT sector FROM stock_sectors WHERE stock_sectors.symbol = companies.symbol),
                        sector_index_code = (SELECT sector_index_code FROM stock_sectors WHERE stock_sectors.symbol = companies.symbol)
                    WHERE symbol IN (SELECT symbol FROM stock_sectors)
                """)
                conn.commit()
                print(f"✓ Updated {result.rowcount} companies with sector info")
        except Exception as e:
            print(f"⚠ Could not update companies table: {e}")
        
        conn.close()
    
    def get_sector(self, symbol: str) -> Optional[Dict]:
        """Get sector info for a specific symbol"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT sector, sector_index_code FROM stock_sectors WHERE symbol = ?",
            (symbol.upper(),)
        )
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'sector': result[0],
                'sector_index_code': result[1]
            }
        return None
    
    def get_symbols_by_sector(self, sector: str) -> List[str]:
        """Get all symbols in a specific sector"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT symbol FROM stock_sectors WHERE sector = ?",
            (sector,)
        )
        symbols = [row[0] for row in cursor.fetchall()]
        conn.close()
        return symbols
    
    def get_all_sectors(self) -> pd.DataFrame:
        """Get all sector mappings from database"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            "SELECT symbol, sector, sector_index_code FROM stock_sectors ORDER BY sector, symbol",
            conn
        )
        conn.close()
        return df


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Map stock symbols to sectors')
    parser.add_argument('--scrape', action='store_true', help='Scrape and update sector data')
    parser.add_argument('--symbol', type=str, help='Get sector for a specific symbol')
    parser.add_argument('--list', action='store_true', help='List all sector mappings')
    
    args = parser.parse_args()
    
    mapper = SectorMapper()
    
    if args.scrape:
        df = mapper.scrape_sectors()
        if not df.empty:
            mapper.save_to_database(df)
            print(f"\n✓ Sector mapping complete!")
    
    elif args.symbol:
        info = mapper.get_sector(args.symbol)
        if info:
            print(f"\n{args.symbol.upper()}:")
            print(f"  Sector: {info['sector']}")
            print(f"  Index Code: {info['sector_index_code']}")
        else:
            print(f"\n✗ No sector info found for {args.symbol}")
    
    elif args.list:
        df = mapper.get_all_sectors()
        if not df.empty:
            print(f"\n📊 All Sector Mappings ({len(df)} stocks):")
            print(df.to_string())
        else:
            print("\n⚠ No sector data in database. Run with --scrape first.")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
