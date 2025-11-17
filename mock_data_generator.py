"""Mock data generator for testing NEPSE analysis features"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import sqlite3
import config


class MockDataGenerator:
    """Generate realistic mock data for testing"""
    
    def __init__(self):
        self.symbols = [
            'NABIL', 'NICA', 'GBIME', 'EBL', 'SBI', 'NBL', 'SCB', 'HBL',
            'ADBL', 'BOKL', 'NCCB', 'SBL', 'KBL', 'PCBL', 'PRVU',
            'UPPER', 'NHPC', 'CHCL', 'RADHI', 'API', 'SHIVM', 'NGPL'
        ]
        
        self.broker_codes = [f'B{str(i).zfill(3)}' for i in range(1, 51)]
    
    def generate_all_mock_data(self):
        """Generate all types of mock data"""
        print("🔧 Generating mock data for testing...\n")
        
        # Generate company list
        print("1. Generating company list...")
        companies = self.generate_companies()
        self._save_to_db(companies, 'companies')
        print(f"   ✓ Created {len(companies)} companies")
        
        # Generate price history for each stock
        print("\n2. Generating price history...")
        for symbol in self.symbols[:10]:  # First 10 stocks
            price_data = self.generate_price_history(symbol, days=365)
            self._save_to_db(price_data, 'price_history', if_exists='append')
        print(f"   ✓ Created price history for {len(self.symbols[:10])} stocks")
        
        # Generate floorsheet data
        print("\n3. Generating floorsheet data...")
        floorsheet = self.generate_floorsheet_data(days=30)
        self._save_to_db(floorsheet, 'floorsheet_data')
        print(f"   ✓ Created {len(floorsheet)} floorsheet transactions")
        
        print("\n✅ Mock data generation complete!")
        print(f"📁 Database: {config.DB_PATH}")
    
    def generate_companies(self) -> pd.DataFrame:
        """Generate mock company list"""
        companies = []
        
        sectors = ['Commercial Banks', 'Development Banks', 'Finance', 'Insurance', 
                   'Hydropower', 'Manufacturing', 'Hotels', 'Others']
        
        for symbol in self.symbols:
            companies.append({
                'symbol': symbol,
                'name': f'{symbol} Limited',
                'sector': random.choice(sectors),
                'source': 'mock'
            })
        
        return pd.DataFrame(companies)
    
    def generate_price_history(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """Generate realistic price history with trends"""
        
        # Starting price based on symbol
        base_price = hash(symbol) % 500 + 200  # 200-700 range
        
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # Generate price with trend and noise
        trend = np.linspace(0, 0.2, days)  # Slight upward trend
        noise = np.random.normal(0, 0.02, days).cumsum()  # Random walk
        
        prices = base_price * (1 + trend + noise)
        
        data = []
        for i, date in enumerate(dates):
            close = prices[i]
            
            # Add daily variance
            daily_var = close * 0.02
            
            data.append({
                'symbol': symbol,
                'date': date.date(),
                'open': close + np.random.uniform(-daily_var, daily_var),
                'high': close + abs(np.random.uniform(0, daily_var)),
                'low': close - abs(np.random.uniform(0, daily_var)),
                'close': close,
                'volume': int(np.random.uniform(10000, 100000))
            })
        
        df = pd.DataFrame(data)
        
        # Ensure high >= low and close within range
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        
        return df
    
    def generate_floorsheet_data(self, days: int = 30, 
                                 transactions_per_day: int = 500) -> pd.DataFrame:
        """Generate mock floorsheet data with realistic broker patterns"""
        
        transactions = []
        
        for day in range(days):
            date = datetime.now().date() - timedelta(days=days-day)
            
            for _ in range(transactions_per_day):
                symbol = random.choice(self.symbols[:15])  # Focus on top 15 stocks
                
                # Create broker concentration for some stocks (manipulation pattern)
                if symbol in ['NABIL', 'NICA', 'GBIME']:
                    # Healthy distribution - many brokers
                    buyer = random.choice(self.broker_codes)
                    seller = random.choice(self.broker_codes)
                elif symbol in ['RADHI', 'API']:
                    # High concentration - few dominant brokers (manipulation risk)
                    dominant_brokers = self.broker_codes[:3]
                    buyer = random.choice(dominant_brokers) if random.random() > 0.3 else random.choice(self.broker_codes)
                    seller = random.choice(dominant_brokers) if random.random() > 0.3 else random.choice(self.broker_codes)
                else:
                    # Normal distribution
                    buyer = random.choice(self.broker_codes[:20])
                    seller = random.choice(self.broker_codes[:20])
                
                # Generate price around base price
                base_price = hash(symbol) % 500 + 200
                price_variance = base_price * 0.05
                rate = base_price + np.random.uniform(-price_variance, price_variance)
                
                quantity = int(np.random.uniform(10, 500))
                
                transactions.append({
                    'date': date,
                    'contract_no': f'C{day}{_:04d}',
                    'symbol': symbol,
                    'buyer_broker': buyer,
                    'seller_broker': seller,
                    'quantity': quantity,
                    'rate': round(rate, 2),
                    'amount': round(quantity * rate, 2),
                    'source': 'mock'
                })
        
        return pd.DataFrame(transactions)
    
    def generate_smart_money_pattern(self, symbol: str, pattern: str = 'accumulation') -> pd.DataFrame:
        """Generate specific broker patterns for testing
        
        Args:
            symbol: Stock symbol
            pattern: 'accumulation', 'distribution', or 'manipulation'
        """
        transactions = []
        days = 30
        
        # Define institutional brokers (large volume)
        institutional_brokers = self.broker_codes[:5]
        retail_brokers = self.broker_codes[5:30]
        
        base_price = hash(symbol) % 500 + 200
        
        for day in range(days):
            date = datetime.now().date() - timedelta(days=days-day)
            
            # Number of transactions per day
            num_transactions = random.randint(50, 150)
            
            for _ in range(num_transactions):
                if pattern == 'accumulation':
                    # Institutions consistently buying
                    if random.random() > 0.4:  # 60% institutional
                        buyer = random.choice(institutional_brokers)
                        seller = random.choice(retail_brokers)
                        quantity = int(np.random.uniform(200, 1000))  # Large volumes
                    else:
                        buyer = random.choice(retail_brokers)
                        seller = random.choice(retail_brokers)
                        quantity = int(np.random.uniform(10, 100))
                
                elif pattern == 'distribution':
                    # Institutions consistently selling
                    if random.random() > 0.4:
                        buyer = random.choice(retail_brokers)
                        seller = random.choice(institutional_brokers)
                        quantity = int(np.random.uniform(200, 1000))
                    else:
                        buyer = random.choice(retail_brokers)
                        seller = random.choice(retail_brokers)
                        quantity = int(np.random.uniform(10, 100))
                
                elif pattern == 'manipulation':
                    # One or two brokers dominating both sides
                    manipulator = self.broker_codes[0]
                    accomplice = self.broker_codes[1]
                    
                    if random.random() > 0.3:  # 70% from manipulators
                        buyer = random.choice([manipulator, accomplice])
                        seller = random.choice([manipulator, accomplice])
                        quantity = int(np.random.uniform(500, 2000))
                    else:
                        buyer = random.choice(retail_brokers)
                        seller = random.choice(retail_brokers)
                        quantity = int(np.random.uniform(10, 50))
                
                rate = base_price + np.random.uniform(-base_price*0.03, base_price*0.03)
                
                transactions.append({
                    'date': date,
                    'contract_no': f'C{day}{_:04d}',
                    'symbol': symbol,
                    'buyer_broker': buyer,
                    'seller_broker': seller,
                    'quantity': quantity,
                    'rate': round(rate, 2),
                    'amount': round(quantity * rate, 2),
                    'source': 'mock'
                })
        
        return pd.DataFrame(transactions)
    
    def _save_to_db(self, df: pd.DataFrame, table_name: str, if_exists: str = 'replace'):
        """Save DataFrame to SQLite database"""
        try:
            conn = sqlite3.connect(config.DB_PATH)
            df.to_sql(table_name, conn, if_exists=if_exists, index=False)
            conn.close()
        except Exception as e:
            print(f"Error saving to database: {e}")


if __name__ == "__main__":
    generator = MockDataGenerator()
    generator.generate_all_mock_data()
    
    print("\n" + "="*70)
    print("Mock data is ready! You can now test all features:")
    print("="*70)
    print("\n📊 Try these commands:")
    print("  python3 main.py analyze NABIL")
    print("  python3 main.py broker NABIL --days 30")
    print("  python3 main.py compare NABIL NICA GBIME")
    print("  python3 main.py market")
    print("\n💡 Tip: Run this script anytime to regenerate fresh mock data")
