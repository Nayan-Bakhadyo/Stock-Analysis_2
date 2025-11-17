# NEPSE Stock Analysis - Quick Start Guide

## 🚀 Current Status

✅ **All features working with mock data**  
⚠️ **Real web scraping needs refinement** (website structures vary)

## 📊 Available Commands

```bash
# Full stock analysis (includes broker metrics)
python3 main.py analyze NABIL

# Broker-specific analysis
python3 main.py broker NABIL --days 30

# Compare multiple stocks
python3 main.py compare NABIL NICA GBIME

# Market overview
python3 main.py market
```

## 🔧 Using Mock Data (Current Setup)

The application is currently using **mock data** for testing. To regenerate fresh data:

```bash
python3 mock_data_generator.py
```

This creates realistic data including:
- **22 stocks** with price history
- **15,000 floorsheet transactions** across 30 days
- **Realistic broker patterns** (concentration, accumulation, etc.)
- Some stocks with **manipulation patterns** for testing

## 🌐 Switching to Real Data

When you're ready to use real NEPSE data, you have two options:

### Option 1: Manual Data Import
If you have access to NEPSE data (CSV/Excel):

```python
import pandas as pd
import sqlite3

# Import your price data
price_data = pd.read_csv('your_price_data.csv')
conn = sqlite3.connect('data/nepse_stocks.db')
price_data.to_sql('price_history', conn, if_exists='append', index=False)
conn.close()

# Same for floorsheet data
floorsheet_data = pd.read_csv('your_floorsheet.csv')
conn = sqlite3.connect('data/nepse_stocks.db')
floorsheet_data.to_sql('floorsheet_data', conn, if_exists='append', index=False)
conn.close()
```

### Option 2: Refine Web Scrapers
The web scrapers in `data_fetcher.py` need customization for each website:

**Current issues:**
- **NepsAlpha**: Blocking (403) - needs better headers or authentication
- **NEPSE Official**: SSL certificate issue
- **MeroLagani & ShareSansar**: Accessible but need HTML structure inspection

**To fix scrapers:**

1. **Inspect website HTML:**
```bash
curl -A "Mozilla/5.0" https://merolagani.com/Companylisting.aspx > test.html
# Then manually check the HTML structure
```

2. **Update selectors in `data_fetcher.py`:**
```python
# Example: Update table selectors to match actual HTML
table = soup.find('table', {'id': 'actual-table-id'})
# or
table = soup.find('div', {'class': 'actual-class-name'})
```

3. **Test individual scrapers:**
```python
from data_fetcher import NepseDataFetcher
fetcher = NepseDataFetcher()

# Test specific method
result = fetcher._scrape_merolagani_history('NABIL', 30)
print(result)
```

## 📁 Data Structure

The SQLite database (`data/nepse_stocks.db`) has these tables:

### `price_history`
| Column | Type | Description |
|--------|------|-------------|
| symbol | TEXT | Stock symbol (e.g., 'NABIL') |
| date | DATE | Trading date |
| open | FLOAT | Opening price |
| high | FLOAT | Highest price |
| low | FLOAT | Lowest price |
| close | FLOAT | Closing price |
| volume | INTEGER | Trading volume |

### `floorsheet_data`
| Column | Type | Description |
|--------|------|-------------|
| date | DATE | Transaction date |
| contract_no | TEXT | Contract number |
| symbol | TEXT | Stock symbol |
| buyer_broker | TEXT | Buyer broker code |
| seller_broker | TEXT | Seller broker code |
| quantity | INTEGER | Number of shares |
| rate | FLOAT | Price per share |
| amount | FLOAT | Total transaction value |
| source | TEXT | Data source |

### `companies`
| Column | Type | Description |
|--------|------|-------------|
| symbol | TEXT | Stock symbol |
| name | TEXT | Company name |
| sector | TEXT | Industry sector (optional) |
| source | TEXT | Data source |

## 🎯 Broker Analysis Features

The broker analysis includes:

1. **Concentration Analysis** - Detect if few brokers dominate
2. **Smart Money Tracking** - Institutional accumulation/distribution
3. **Buy/Sell Pressure** - VWAP and imbalance ratios
4. **Liquidity Analysis** - Trade safety assessment
5. **Manipulation Detection** - Red flags and risk scoring

See `BROKER_ANALYSIS.md` for detailed documentation.

## 🔍 Example Output Interpretation

```bash
python3 main.py broker NABIL --days 30
```

**What to look for:**

- **Overall Score > 80**: Excellent - safe to trade
- **Manipulation Risk > 70**: CRITICAL - avoid this stock
- **Smart Money = BULLISH**: Institutions accumulating (good sign)
- **Liquidity > 60**: Safe for medium to large positions
- **Top broker > 60%**: Warning - possible manipulation

## 💡 Tips for Using Mock Data

While testing with mock data:

1. **Experiment with different stocks** - Some have manipulation patterns:
   - `NABIL, NICA, GBIME`: Healthy broker distribution
   - `RADHI, API`: High concentration (manipulation risk)

2. **Test all time horizons:**
```bash
python3 main.py analyze NABIL --horizon short
python3 main.py analyze NABIL --horizon medium
python3 main.py analyze NABIL --horizon long
```

3. **Export analysis results:**
```bash
python3 main.py analyze NABIL --export
# Creates: NABIL_analysis_YYYYMMDD_HHMMSS.json
```

4. **Compare multiple stocks:**
```bash
python3 main.py compare NABIL NICA GBIME EBL
```

## 🐛 Troubleshooting

**"No data available"**
→ Run `python3 mock_data_generator.py` to create mock data

**"No such table" error**
→ Database not initialized. Run mock data generator.

**Import errors**
→ Install dependencies: `pip install -r requirements.txt`

**Slow performance**
→ Reduce analysis period: `--days 7` instead of `--days 30`

## 🚧 Next Steps for Production

1. **API Integration**: If NEPSE provides official APIs, integrate them
2. **Scraper Refinement**: Customize scrapers for each data source
3. **Data Validation**: Add data quality checks and cleaning
4. **Caching Strategy**: Implement smart caching to reduce requests
5. **Real-time Updates**: Add scheduler for automatic data updates
6. **Error Handling**: Robust fallbacks when data sources fail

## 📖 Documentation

- `README.md` - General project overview
- `BROKER_ANALYSIS.md` - Detailed broker analysis guide
- `QUICKSTART.md` - Installation and basic usage
- `requirements.txt` - All Python dependencies

## 🎉 You're Ready!

Start exploring:
```bash
# Analyze your first stock
python3 main.py analyze NABIL

# Check for manipulation
python3 main.py broker RADHI --days 30

# Find best opportunities
python3 main.py compare NABIL NICA GBIME EBL SBI
```

Happy trading! 📈🚀
