"""Real data integration adapter for NEPSE
Supports multiple data sources: CSV files, APIs, manual imports
"""
import pandas as pd
import requests
from datetime import datetime
import sqlite3
import config
from pathlib import Path


class RealDataAdapter:
    """Adapter for integrating real NEPSE data from various sources"""
    
    def __init__(self):
        self.db_path = config.DB_PATH
    
    def import_from_csv(self, csv_path: str, data_type: str):
        """
        Import data from CSV file
        
        Args:
            csv_path: Path to CSV file
            data_type: 'price_history', 'floorsheet', or 'companies'
        """
        print(f"Importing {data_type} from {csv_path}...")
        
        df = pd.read_csv(csv_path)
        
        # Standardize column names based on data type
        if data_type == 'price_history':
            df = self._standardize_price_data(df)
        elif data_type == 'floorsheet':
            df = self._standardize_floorsheet_data(df)
        elif data_type == 'companies':
            df = self._standardize_company_data(df)
        
        # Save to database
        conn = sqlite3.connect(self.db_path)
        df.to_sql(data_type, conn, if_exists='append', index=False)
        conn.close()
        
        print(f"✓ Imported {len(df)} rows to {data_type}")
        return df
    
    def import_from_excel(self, excel_path: str, data_type: str, sheet_name: str = 0):
        """Import data from Excel file"""
        print(f"Importing {data_type} from {excel_path}...")
        
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        
        # Standardize and save
        if data_type == 'price_history':
            df = self._standardize_price_data(df)
        elif data_type == 'floorsheet':
            df = self._standardize_floorsheet_data(df)
        elif data_type == 'companies':
            df = self._standardize_company_data(df)
        
        conn = sqlite3.connect(self.db_path)
        df.to_sql(data_type, conn, if_exists='append', index=False)
        conn.close()
        
        print(f"✓ Imported {len(df)} rows to {data_type}")
        return df
    
    def fetch_from_nepse_api(self, endpoint: str, auth_token: str = None):
        """
        Fetch data from NEPSE API (if you have access)
        
        Args:
            endpoint: API endpoint URL
            auth_token: Authentication token if required
        """
        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json'
        }
        
        if auth_token:
            headers['Authorization'] = f'Bearer {auth_token}'
        
        try:
            response = requests.get(endpoint, headers=headers, timeout=15, verify=False)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Fetched data from {endpoint}")
                return data
            else:
                print(f"✗ Failed: {response.status_code}")
                return None
        except Exception as e:
            print(f"✗ Error: {e}")
            return None
    
    def _standardize_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize price data columns"""
        
        # Common column name mappings
        column_map = {
            # Various date formats
            'Date': 'date',
            'DATE': 'date',
            'TradingDate': 'date',
            'BusinessDate': 'date',
            'business_date': 'date',
            
            # Symbol
            'Symbol': 'symbol',
            'SYMBOL': 'symbol',
            'StockSymbol': 'symbol',
            'scrip': 'symbol',
            
            # OHLC
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'LTP': 'close',
            'LastTradedPrice': 'close',
            'close_price': 'close',
            'open_price': 'open',
            'high_price': 'high',
            'low_price': 'low',
            
            # Volume
            'Volume': 'volume',
            'VOLUME': 'volume',
            'TradedVolume': 'volume',
            'SharesTraded': 'volume',
            'total_traded_quantity': 'volume',
        }
        
        # Rename columns
        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
        
        # Ensure required columns exist
        required = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                if col in ['open', 'high', 'low', 'close']:
                    df[col] = df.get('close', 0)  # Use close as fallback
                elif col == 'volume':
                    df[col] = 0
                elif col == 'date':
                    df[col] = datetime.now().date()
                elif col == 'symbol':
                    df[col] = 'UNKNOWN'
        
        # Convert types
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df[required]
    
    def _standardize_floorsheet_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize floorsheet data columns"""
        
        column_map = {
            'Date': 'date',
            'ContractNo': 'contract_no',
            'ContractNumber': 'contract_no',
            'Symbol': 'symbol',
            'BuyerBroker': 'buyer_broker',
            'BuyerBrokerNo': 'buyer_broker',
            'SellerBroker': 'seller_broker',
            'SellerBrokerNo': 'seller_broker',
            'Quantity': 'quantity',
            'Rate': 'rate',
            'Price': 'rate',
            'Amount': 'amount',
            'TotalAmount': 'amount',
        }
        
        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
        
        # Required columns
        required = ['date', 'contract_no', 'symbol', 'buyer_broker', 'seller_broker', 
                   'quantity', 'rate', 'amount']
        
        for col in required:
            if col not in df.columns:
                if col == 'date':
                    df[col] = datetime.now().date()
                elif col in ['contract_no', 'symbol', 'buyer_broker', 'seller_broker']:
                    df[col] = 'UNKNOWN'
                else:
                    df[col] = 0
        
        # Add source column
        df['source'] = 'import'
        
        # Convert types
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0)
        df['rate'] = pd.to_numeric(df['rate'], errors='coerce').fillna(0)
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
        
        return df[required + ['source']]
    
    def _standardize_company_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize company list data"""
        
        column_map = {
            'Symbol': 'symbol',
            'CompanyName': 'name',
            'Name': 'name',
            'Sector': 'sector',
            'Industry': 'sector',
        }
        
        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
        
        required = ['symbol', 'name']
        for col in required:
            if col not in df.columns:
                df[col] = 'UNKNOWN'
        
        if 'sector' not in df.columns:
            df['sector'] = 'Other'
        
        df['source'] = 'import'
        
        return df[['symbol', 'name', 'sector', 'source']]
    
    def get_data_sources_guide(self):
        """Print guide for getting real NEPSE data"""
        print("""
╔══════════════════════════════════════════════════════════════════╗
║          NEPSE Real Data Integration Guide                       ║
╚══════════════════════════════════════════════════════════════════╝

📊 WHERE TO GET REAL NEPSE DATA:

1. NEPSE Official Website (www.nepalstock.com.np)
   - Download historical data (may require registration)
   - Format: Usually Excel/CSV
   - Includes: Price history, floorsheet, market data

2. Brokerage Firms
   - Most brokers provide historical data to clients
   - Contact your broker for data access
   - Often available through their trading platforms

3. Financial Data Providers
   - MeroLagani.com (may have premium data services)
   - ShareSansar.com (may have downloadable reports)
   - NepsAlpha.com (check for data export features)

4. NEPSE Mobile App / Trading Software
   - Many apps allow data export
   - Check export/download features in your trading app

5. Manual Data Collection
   - Daily screenshots/records
   - Compile into Excel/CSV manually

📝 SUPPORTED FILE FORMATS:

✓ CSV (.csv)
✓ Excel (.xlsx, .xls)
✓ JSON (from APIs)

🔧 HOW TO IMPORT YOUR DATA:

# From CSV
adapter = RealDataAdapter()
adapter.import_from_csv('my_price_data.csv', 'price_history')
adapter.import_from_csv('my_floorsheet.csv', 'floorsheet')

# From Excel
adapter.import_from_excel('nepse_data.xlsx', 'price_history', sheet_name='Prices')
adapter.import_from_excel('nepse_data.xlsx', 'floorsheet', sheet_name='Floorsheet')

📋 REQUIRED CSV COLUMNS:

Price History CSV:
- date, symbol, open, high, low, close, volume

Floorsheet CSV:
- date, contract_no, symbol, buyer_broker, seller_broker, quantity, rate, amount

Companies CSV:
- symbol, name, sector (optional)

💡 The import functions will automatically map common column name variations!

        """)


def main():
    """Interactive data import tool"""
    adapter = RealDataAdapter()
    
    print("\n" + "="*70)
    print("NEPSE Real Data Import Tool")
    print("="*70 + "\n")
    
    print("Options:")
    print("1. Import CSV file")
    print("2. Import Excel file")
    print("3. Show data sources guide")
    print("4. Exit")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == '1':
        file_path = input("CSV file path: ").strip()
        data_type = input("Data type (price_history/floorsheet/companies): ").strip()
        
        if Path(file_path).exists():
            adapter.import_from_csv(file_path, data_type)
        else:
            print(f"✗ File not found: {file_path}")
    
    elif choice == '2':
        file_path = input("Excel file path: ").strip()
        data_type = input("Data type (price_history/floorsheet/companies): ").strip()
        sheet = input("Sheet name (or press Enter for first sheet): ").strip() or 0
        
        if Path(file_path).exists():
            adapter.import_from_excel(file_path, data_type, sheet)
        else:
            print(f"✗ File not found: {file_path}")
    
    elif choice == '3':
        adapter.get_data_sources_guide()
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
