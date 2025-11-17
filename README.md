# NEPSE Stock Analysis System

A comprehensive Python application for analyzing stocks listed on the Nepal Stock Exchange (NEPSE). This system provides deep insights through technical analysis, fundamental analysis, sentiment analysis, **broker manipulation detection**, and generates trading recommendations with profitability probability calculations.

## 🌟 Features

### 🕵️ **Broker Analysis & Manipulation Detection** ⭐ NEW
- **Broker Concentration Analysis**: HHI (Herfindahl Index), top broker percentages
- **Smart Money Tracking**: Accumulation/distribution pattern detection
- **Buy/Sell Pressure Analysis**: VWAP efficiency, imbalance ratios
- **Liquidity Analysis**: Transaction volume, broker diversity, market depth
- **Manipulation Risk Detection**: Red flag identification, risk scoring
- **Broker Trading Patterns**: Individual broker behavior tracking

### 📊 **Multi-Source Data Management**
- **Mock Data System**: Realistic test data generation (22 stocks, 365 days, 15,000 transactions)
- **CSV/Excel Import**: Flexible import from any data source you have access to
- **Database Storage**: SQLite database for efficient data management
- **Multiple Data Sources**: Support for NEPSE, broker reports, manual imports

### 📈 **Technical Analysis**
- **Moving Averages**: SMA, EMA (7, 20, 50-day periods)
- **Momentum Indicators**: RSI, Stochastic Oscillator
- **Trend Indicators**: MACD with signal line and histogram
- **Volatility**: Bollinger Bands, ATR
- **Volume Analysis**: On-Balance Volume (OBV)
- **Pattern Recognition**: Head & Shoulders, Double Top/Bottom, Triangles
- **Support/Resistance Levels**: Automatic identification

### 💰 **Fundamental Analysis**
- **Valuation Ratios**: P/E Ratio, P/B Ratio
- **Profitability**: ROE, EPS Growth
- **Dividend Analysis**: Dividend Yield
- **Financial Health**: Debt-to-Equity, Current Ratio
- **Comparative Analysis**: Multi-stock comparison

### 📰 **Sentiment Analysis**
- News scraping from multiple Nepali stock portals:
  - ShareSansar
  - MeroLagani
  - Nepali Paisa
- Natural Language Processing for sentiment scoring
- VADER and TextBlob sentiment engines
- Sentiment trend tracking (improving/declining/stable)

### 🎯 **Trading Insights**
- **Profitability Probability**: ML-based probability calculation (now includes broker analysis)
- **Risk-Reward Analysis**: Automatic calculation with support/resistance
- **Entry/Exit Points**: Conservative and aggressive entry points
- **Position Sizing**: Risk-based position recommendations (liquidity-adjusted)
- **Stop Loss & Take Profit**: Automatic level calculation
- **Time Horizons**: Short-term (1-7 days), Medium-term (1-4 weeks), Long-term (1-3 months)
- **Manipulation Penalties**: Reduces profitability for stocks with manipulation risk

### 📋 **Recommendations**
- Buy/Sell/Hold signals with confidence levels
- Multi-factor weighted scoring system
- Strength and weakness identification
- Risk warnings and alerts (including manipulation warnings)

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- **ChromeDriver** (for live scraping - optional)
- Chrome browser (for live scraping - optional)

### Setup Steps

1. **Install Python dependencies**:
```bash
pip install pandas numpy beautifulsoup4 requests lxml colorama tabulate
pip install vaderSentiment textblob newspaper3k scikit-learn

# For live web scraping (optional):
pip install selenium
```

2. **Install ChromeDriver** (macOS - for live scraping):
```bash
brew install chromedriver
```

For other platforms, download from: https://chromedriver.chromium.org/

3. **Create data directory**:
```bash
mkdir -p data
```

### Quick Start (Using Mock Data)

```bash
# Generate realistic test data
python3 mock_data_generator.py

# Analyze a stock
python3 main.py analyze NABIL

# Broker analysis
python3 main.py broker NABIL --days 30
```

### Using Live Data (Selenium Scraper)

```bash
# Test the scraper
python3 test_scraper.py

# Or run interactive scraper
python3 sharesansar_scraper.py
```

### Importing CSV/Excel Data

```bash
# See data sources guide
python3 main.py sources

# Import your data
python3 main.py import your_data.csv price
```

## 📖 Usage

### Command Line Interface

#### 1. Analyze a Single Stock

```bash
# Basic analysis
python main.py analyze NABIL

# Medium-term analysis
python main.py analyze NABIL --horizon medium

# Detailed analysis with all metrics
python main.py analyze NABIL --detailed

# Export results to JSON
python main.py analyze NABIL --export
```

#### 2. Compare Multiple Stocks

```bash
# Compare banking stocks
python main.py compare NABIL NICA GBIME SCB

# Long-term comparison
python main.py compare NABIL NICA --horizon long
```

#### 3. Market Overview

```bash
# View market summary with top gainers/losers
python main.py market
```

### Programmatic Usage

```python
from trading_insights import TradingInsightsEngine

# Initialize the engine
engine = TradingInsightsEngine()

# Analyze a stock
result = engine.calculate_profitability_probability(
    symbol="NABIL",
    time_horizon="short"
)

# Display results
print(f"Profitability Probability: {result['profitability_probability']:.2f}%")
print(f"Recommendation: {result['recommendation']['action']}")
print(f"Entry Point: NPR {result['entry_points']['aggressive']:.2f}")
print(f"Stop Loss: NPR {result['stop_loss']:.2f}")
print(f"Take Profit: NPR {result['take_profit']:.2f}")
```

### Module-Specific Usage

#### Data Fetcher
```python
from data_fetcher import NepseDataFetcher

fetcher = NepseDataFetcher()

# Get company list
companies = fetcher.get_company_list()

# Get historical data
price_data = fetcher.get_stock_price_history("NABIL", days=365)

# Get live price
live_price = fetcher.get_live_price("NABIL")
```

#### Technical Analysis
```python
from technical_analyzer import TechnicalAnalyzer

analyzer = TechnicalAnalyzer()

# Comprehensive analysis
result = analyzer.comprehensive_analysis(price_dataframe)

print(f"RSI: {result['indicators']['rsi']:.2f}")
print(f"MACD Signal: {result['signals']['macd_signal']}")
print(f"Trend: {result['signals']['trend']}")
```

#### Fundamental Analysis
```python
from fundamental_analyzer import FundamentalAnalyzer

analyzer = FundamentalAnalyzer()

# Analyze fundamentals
stock_data = {
    'symbol': 'NABIL',
    'current_price': 1000,
    'eps': 50,
    'book_value_per_share': 400,
    # ... more data
}

result = analyzer.comprehensive_analysis(stock_data)
print(f"Overall Rating: {result['overall_rating']}")
print(f"P/E Ratio: {result['ratios']['pe_ratio']['value']:.2f}")
```

#### Sentiment Analysis
```python
from sentiment_analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer()

# Get sentiment for a stock
sentiment = analyzer.get_aggregate_sentiment("NABIL", days=7)

print(f"Overall Sentiment: {sentiment['overall_sentiment']}")
print(f"Positive Articles: {sentiment['positive_count']}")
print(f"Sentiment Trend: {sentiment['sentiment_trend']}")
```

## 📊 Output Example

```
======================================================================
                    NEPSE STOCK ANALYSIS: NABIL
======================================================================

📊 BASIC INFORMATION
┌─────────────────┬─────────────────────────┐
│ Symbol          │ NABIL                   │
│ Current Price   │ NPR 1050.00             │
│ Analysis Date   │ 2024-11-15 14:30:00     │
│ Time Horizon    │ SHORT                   │
└─────────────────┴─────────────────────────┘

💰 PROFITABILITY ANALYSIS
Profitability Probability: 72.50%
Confidence Level: High

[██████████████████████████████████████░░░░░░░░░] 72.5%

📈 TRADING RECOMMENDATION
Action: BUY
Confidence: Medium-High

Reasoning:
  • Strong technical indicators
  • Positive market sentiment
  • Favorable risk-reward ratio

⚖️ RISK-REWARD ANALYSIS
┌──────────────────────┬──────────┐
│ Risk-Reward Ratio    │ 2.50     │
│ Potential Profit     │ 7.50%    │
│ Potential Loss       │ 3.00%    │
│ Nearest Support      │ NPR 1020 │
│ Nearest Resistance   │ NPR 1130 │
└──────────────────────┴──────────┘

🎯 ENTRY & EXIT STRATEGY
┌─────────────────────┬──────────────┐
│ Entry (Aggressive)  │ NPR 1050.00  │
│ Entry (Conservative)│ NPR 1020.00  │
│ Target 1            │ NPR 1130.00  │
│ Target 2            │ NPR 1207.50  │
│ Stop Loss           │ NPR 997.50   │
│ Take Profit         │ NPR 1207.50  │
└─────────────────────┴──────────────┘

💡 KEY INSIGHTS
  ✅ Strong technical setup (Score: 75/100)
  📊 Current trend: Uptrend
  ✅ Strong fundamentals (Score: 68/100)
  💪 Attractive P/E ratio (20.00)
  📰 Market sentiment: POSITIVE (12 articles)
  📈 5-day momentum: up 3.50%

⚠️ WARNINGS & RISKS
  ✅ No major warnings identified
```

## ⚙️ Configuration

### Environment Variables (`.env`)

```bash
# NEPSE API Configuration
NEPSE_BASE_URL=https://www.nepalstock.com.np
NEPSE_API_URL=https://newweb.nepalstock.com.np/api

# Sentiment Analysis
SENTIMENT_MODEL=cardiffnlp/twitter-roberta-base-sentiment

# Database
DB_PATH=data/nepse_stocks.db

# Analysis Parameters
DEFAULT_LOOKBACK_DAYS=365
MIN_DATA_POINTS=30
```

### Custom Configuration (`config.py`)

You can customize:
- Technical indicator periods
- Fundamental analysis thresholds
- Signal weights for probability calculation
- Risk parameters (stop loss, take profit)

## 📁 Project Structure

```
Stock_Analysis/
├── main.py                    # Main CLI application
├── config.py                  # Configuration settings
├── data_fetcher.py           # NEPSE data fetching module
├── sentiment_analyzer.py      # News scraping and sentiment analysis
├── fundamental_analyzer.py    # Fundamental analysis module
├── technical_analyzer.py      # Technical analysis module
├── trading_insights.py        # Trading insights engine
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variables template
├── README.md                 # This file
├── data/                     # Database and cached data
├── logs/                     # Application logs
└── reports/                  # Exported analysis reports
```

## 🎓 Understanding the Analysis

### Profitability Probability Score

The probability score (0-100%) represents the likelihood of a profitable trade based on:
- **Technical Analysis (30%)**: Chart patterns, indicators, and trends
- **Fundamental Analysis (25%)**: Financial health and valuation
- **Sentiment Analysis (25%)**: Market sentiment from news
- **Momentum Analysis (20%)**: Price and volume momentum

### Recommendation Levels

| Probability | Risk-Reward | Recommendation |
|-------------|-------------|----------------|
| ≥70%        | ≥2.0        | STRONG BUY     |
| ≥60%        | ≥1.5        | BUY            |
| ≥50%        | ≥1.0        | HOLD/ACCUMULATE|
| ≥40%        | Any         | HOLD           |
| ≥30%        | Any         | SELL           |
| <30%        | Any         | STRONG SELL    |

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Activate virtual environment

2. **API Connection Issues**
   - Check internet connection
   - Verify NEPSE API is accessible
   - Check firewall settings

3. **Insufficient Data**
   - Some stocks may not have enough historical data
   - Try different stocks with longer trading history

4. **Sentiment Analysis Errors**
   - News websites may change their structure
   - Internet connection required for scraping

## ⚠️ Disclaimer

**This tool is for educational and informational purposes only.**

- Not financial advice
- Past performance doesn't guarantee future results
- Always do your own research (DYOR)
- Consult with a licensed financial advisor
- Trade at your own risk
- The developers are not responsible for any financial losses

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional technical indicators
- More news sources
- Machine learning models for better predictions
- Real-time alerts
- Portfolio tracking
- Backtesting capabilities

## 📄 License

This project is open source and available under the MIT License.

## 📧 Support

For issues, questions, or suggestions:
- Create an issue on GitHub
- Contact the development team

## 🙏 Acknowledgments

- NEPSE for market data
- ShareSansar, MeroLagani for news sources
- Open-source Python community

---

**Happy Trading! 📈💰**

*Remember: The stock market is risky. Never invest more than you can afford to lose.*
# Stock-Analytics
