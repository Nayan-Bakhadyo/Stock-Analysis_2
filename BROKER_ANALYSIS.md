# Broker Analysis Feature - NEPSE Stock Analysis

## Overview

The **Broker Analysis** module is a powerful feature specifically designed for the NEPSE (Nepal Stock Exchange) market that analyzes broker-level trading patterns to detect market manipulation, track institutional money flows, assess liquidity, and identify trading risks.

## Key Features

### 1. **Broker Concentration Analysis** 🎯
Identifies if trading activity is dominated by a few brokers, which can indicate manipulation risk.

**Metrics Calculated:**
- **Top Broker Percentages**: % of volume controlled by top 1, 3, and 5 brokers
- **HHI (Herfindahl-Hirschman Index)**: Industry-standard concentration measure
  - < 1500: Competitive/Healthy
  - 1500-2500: Moderate concentration
  - \> 2500: High concentration (manipulation risk)
- **Concentration Risk Score**: 0-100 (higher = more risk)

**What it detects:**
- Single broker controlling >60% of volume = **CRITICAL**
- Top 3 brokers controlling >80% = **WARNING**
- Coordinated "pump-and-dump" schemes

### 2. **Smart Money Tracking** 💼
Tracks institutional and large-volume broker activity to identify informed trading.

**Identifies:**
- **Accumulation**: Large brokers consistently buying (bullish signal)
- **Distribution**: Large brokers consistently selling (bearish signal)
- **Net Institutional Flow**: Overall institutional buying vs selling pressure

**Signals Generated:**
- STRONG BULLISH - Institutions Accumulating
- BULLISH - Net Institutional Buying  
- NEUTRAL - Balanced Activity
- BEARISH - Net Institutional Selling
- STRONG BEARISH - Institutions Distributing

### 3. **Buy/Sell Pressure Analysis** ⚖️
Measures immediate market sentiment through volume-weighted price analysis.

**Metrics:**
- **VWAP (Volume Weighted Average Price)**: Institutional execution benchmark
- **Buy/Sell Imbalance Ratio**: 
  - \> 1.5 = Strong buying pressure
  - < 0.67 = Strong selling pressure
- **Aggressive vs Passive Trading**: Transactions above/below VWAP

### 4. **Liquidity & Market Depth** 💧
Assesses how easily you can enter/exit positions without moving the price.

**Evaluates:**
- Total trading volume
- Number of unique brokers
- Transaction frequency
- Price volatility
- Average transaction size

**Liquidity Classification:**
- EXCELLENT (80+ score): Safe for large positions
- GOOD (60-79): Safe for medium positions
- MODERATE (40-59): Caution for large positions
- LOW (20-39): High risk - small positions only
- VERY LOW (<20): Very high risk - avoid

### 5. **Manipulation Risk Detection** 🚨
Combines all metrics to identify potential market manipulation.

**Red Flags:**
- Single broker dominance (>60%)
- Extremely high HHI (>3000)
- Same brokers appearing as both top buyers and sellers
- Low liquidity + high concentration
- Coordinated buying/selling patterns

**Risk Levels:**
- CRITICAL (70+): Do not trade
- HIGH (50-69): Trade with extreme caution
- MODERATE (30-49): Small positions only
- LOW (15-29): Monitor closely
- VERY LOW (<15): Healthy market activity

## How to Use

### Command Line Interface

#### Analyze Specific Stock's Broker Activity
```bash
python main.py broker NABIL --days 30
```

#### Market-Wide Broker Analysis
```bash
python main.py broker --days 7
```

#### Standard Stock Analysis (includes broker metrics)
```bash
python main.py analyze NABIL
```

### Python API

```python
from broker_analyzer import BrokerAnalyzer
from data_fetcher import NepseDataFetcher

# Initialize
fetcher = NepseDataFetcher()
analyzer = BrokerAnalyzer()

# Get floorsheet data
floorsheet = fetcher.get_floorsheet_data('NABIL', days=30)

# Comprehensive analysis
analysis = analyzer.comprehensive_broker_analysis(floorsheet, 'NABIL', days=30)

# Access results
print(f"Overall Score: {analysis['overall_broker_score']['overall_score']}")
print(f"Manipulation Risk: {analysis['manipulation_risk']['risk_level']}")
print(f"Smart Money: {analysis['smart_money_flow']['smart_money_signal']}")
print(f"Liquidity: {analysis['liquidity_metrics']['liquidity_classification']}")
```

## Understanding the Output

### Overall Broker Score (0-100)
Weighted combination of all broker metrics:
- **Concentration (25%)**: Lower concentration = better (inverted)
- **Smart Money (25%)**: Institutional accumulation = higher score
- **Pressure (20%)**: Buy pressure = higher score
- **Liquidity (20%)**: Higher liquidity = higher score  
- **Manipulation Risk (10%)**: Lower risk = higher score (inverted)

**Grades:**
- A+ (90-100): Excellent - strong buy candidate
- A/A- (80-89): Very good - buy
- B+/B (70-79): Good/Average - hold or small buy
- B- (65-69): Below average - caution
- C/C+ (50-64): Poor - avoid or sell
- D/F (<50): Very poor - strong avoid

### Integration with Profitability Analysis

When you run `python main.py analyze NABIL`, the broker analysis is **automatically included** with 15% weight in the profitability calculation:

**Standard Weights:**
- Technical: 30%
- Fundamental: 25%
- Sentiment: 25%
- Momentum: 20%

**With Broker Analysis:**
- Technical: 25.5% (30% × 0.85)
- Fundamental: 21.25% (25% × 0.85)
- Sentiment: 21.25% (25% × 0.85)
- Momentum: 17% (20% × 0.85)
- **Broker: 15%** ← New!

**Manipulation Risk Penalties:**
- Critical Risk (70+): Probability reduced by **50%**
- High Risk (50-69): Probability reduced by **25%**

**Liquidity Adjustments:**
- Low Liquidity (<40): Position size reduced by **50%**
- Moderate Liquidity (40-59): Position size reduced by **25%**

## Data Sources

Broker data is scraped from:
1. **ShareSansar Floorsheet** (Primary): https://www.sharesansar.com/floorsheet
2. **MeroLagani Floorsheet** (Backup): https://merolagani.com/Floorsheet.aspx

**Required Data Fields:**
- Date
- Contract Number
- Stock Symbol
- Buyer Broker Code
- Seller Broker Code
- Quantity
- Rate (Price)
- Amount

## Configuration

All thresholds are configurable in `config.py`:

```python
BROKER_ANALYSIS_CONFIG = {
    'manipulation_thresholds': {
        'single_broker_critical': 60,  # %
        'single_broker_warning': 45,   # %
        'top_3_critical': 80,          # %
        'top_3_warning': 70,           # %
        'hhi_critical': 2500,
        'hhi_warning': 1500,
    },
    'smart_money_thresholds': {
        'large_volume_percentile': 0.80,  # Top 20%
        'accumulation_ratio': 1.2,
        'distribution_ratio': 0.83,
    },
    'liquidity_thresholds': {
        'high_volume': 50000,      # shares
        'medium_volume': 20000,
        'low_volume': 5000,
        # ... more thresholds
    }
}
```

## Use Cases

### 1. Avoid Manipulation Traps
```bash
# Before buying a stock that's been "pumped"
python main.py broker XYZ
```
**Look for:** High manipulation risk, single broker dominance, low liquidity

### 2. Follow Smart Money
```bash
# Check if institutions are accumulating
python main.py broker NABIL
```
**Look for:** BULLISH smart money signal, positive net institutional flow

### 3. Assess Trade Safety
```bash
# Before taking a large position
python main.py broker GBIME
```
**Look for:** High liquidity score, many unique brokers, low manipulation risk

### 4. Market-Wide Scan
```bash
# Find healthy stocks to trade
python main.py broker --days 7
```
**Look for:** High overall scores (80+), low manipulation risk, good liquidity

## Warnings & Limitations

⚠️ **Important Considerations:**

1. **Data Availability**: Floorsheet data may not always be available. The system caches data in SQLite database.

2. **Website Changes**: Web scraping relies on website structure. If ShareSansar/MeroLagani change their layout, scrapers may need updates.

3. **Rate Limiting**: Respect rate limits - default delay is 2 seconds between requests.

4. **Historical Context**: Broker patterns should be analyzed over time (30 days recommended). Single-day data can be misleading.

5. **Not Legal Advice**: This tool identifies *potential* manipulation patterns. It's not evidence of illegal activity.

6. **NEPSE-Specific**: This feature is designed specifically for NEPSE market structure and may not apply to other markets.

## Advanced Examples

### Check Specific Broker Activity
```python
from data_fetcher import NepseDataFetcher

fetcher = NepseDataFetcher()
broker_stats = fetcher.get_broker_stats('B001', days=30)

print(f"Broker B001:")
print(f"Total Bought: {broker_stats['total_bought']:,}")
print(f"Total Sold: {broker_stats['total_sold']:,}")
print(f"Net Position: {broker_stats['net_position']:,}")
print(f"Top Stocks: {broker_stats['top_stocks_bought']}")
```

### Custom Thresholds
```python
from broker_analyzer import BrokerAnalyzer
import config

# Temporarily adjust thresholds
config.BROKER_ANALYSIS_CONFIG['manipulation_thresholds']['single_broker_critical'] = 50

analyzer = BrokerAnalyzer()
# ... run analysis with custom thresholds
```

### Export Broker Analysis
```python
import json
from broker_analyzer import BrokerAnalyzer
from data_fetcher import NepseDataFetcher

fetcher = NepseDataFetcher()
analyzer = BrokerAnalyzer()

floorsheet = fetcher.get_floorsheet_data('NABIL', days=30)
analysis = analyzer.comprehensive_broker_analysis(floorsheet, 'NABIL', 30)

# Save to file
with open('NABIL_broker_analysis.json', 'w') as f:
    json.dump(analysis, f, indent=2)
```

## Troubleshooting

**Issue**: "No floorsheet data available"
- **Solution**: Check if websites are accessible. Data is cached in database - may use older data.

**Issue**: Broker analysis takes long time
- **Solution**: Reduce `--days` parameter or analyze single stock instead of market-wide.

**Issue**: Different results from manual verification
- **Solution**: Check date ranges match. Floorsheet data can have delays.

**Issue**: Import errors
- **Solution**: Install dependencies: `pip install beautifulsoup4 lxml requests pandas`

## Future Enhancements

Planned features:
- [ ] Broker reputation scoring based on historical accuracy
- [ ] Pattern detection (head-and-shoulders in broker behavior)
- [ ] Real-time alerts for manipulation patterns
- [ ] Broker network analysis (identifying coordinated groups)
- [ ] Machine learning for manipulation prediction

## Credits

Developed specifically for NEPSE market analysis. Inspired by:
- Market microstructure research
- HHI concentration index from economics
- VWAP analysis used by institutional traders
- Floorsheet analysis practices in Nepali stock forums

---

**Remember**: Always combine broker analysis with technical, fundamental, and sentiment analysis for best results. No single metric guarantees success!
