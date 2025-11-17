"""Configuration settings for NEPSE Stock Analysis"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
REPORTS_DIR = BASE_DIR / "reports"

# Create directories if they don't exist
for directory in [DATA_DIR, LOGS_DIR, REPORTS_DIR]:
    directory.mkdir(exist_ok=True)

# NEPSE Data Sources - Web Scraping Only (No API)
DATA_SOURCES = {
    'nepsealpha': {
        'url': 'https://nepsealpha.com',
        'stock_detail': 'https://nepsealpha.com/stock/{symbol}',
        'market': 'https://nepsealpha.com/trading',
        'priority': 1  # Primary source
    },
    'merolagani': {
        'url': 'https://merolagani.com',
        'stock_detail': 'https://merolagani.com/CompanyDetail.aspx?symbol={symbol}',
        'market': 'https://merolagani.com/LatestMarket.aspx',
        'company_list': 'https://merolagani.com/Companylisting.aspx',
        'top_gainers': 'https://merolagani.com/TopGainers.aspx',
        'top_losers': 'https://merolagani.com/TopLosers.aspx',
        'priority': 2
    },
    'sharesansar': {
        'url': 'https://www.sharesansar.com',
        'stock_detail': 'https://www.sharesansar.com/company/{symbol}',
        'market': 'https://www.sharesansar.com/today-share-price',
        'floorsheet': 'https://www.sharesansar.com/floorsheet',
        'news': 'https://www.sharesansar.com/category/market',
        'priority': 2
    },
    'nepse_official': {
        'url': 'https://www.nepalstock.com.np',
        'market': 'https://www.nepalstock.com.np/market',
        'priority': 3
    }
}

# News Sources for Nepali Stock Market
NEWS_SOURCES = {
    'sharesansar': 'https://www.sharesansar.com',
    'merolagani': 'https://merolagani.com',
    'nepsealpha': 'https://nepsealpha.com',
    'nepse_official': 'https://www.nepalstock.com.np',
}

# Sentiment Analysis
SENTIMENT_CONFIG = {
    'model': os.getenv('SENTIMENT_MODEL', 'cardiffnlp/twitter-roberta-base-sentiment'),
    'batch_size': 8,
    'max_length': 512,
}

# Technical Analysis Parameters
TECHNICAL_INDICATORS = {
    'short_ma': 7,
    'medium_ma': 20,
    'long_ma': 50,
    'rsi_period': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'bb_period': 20,
    'bb_std': 2,
}

# Fundamental Analysis Thresholds
FUNDAMENTAL_THRESHOLDS = {
    'pe_ratio': {
        'undervalued': 15,
        'fair': 25,
        'overvalued': 35,
    },
    'pb_ratio': {
        'undervalued': 1.0,
        'fair': 2.0,
        'overvalued': 3.0,
    },
    'eps_growth': {
        'poor': 0,
        'good': 10,
        'excellent': 20,
    },
    'dividend_yield': {
        'poor': 0,
        'good': 2,
        'excellent': 5,
    }
}

# Trading Signal Weights
SIGNAL_WEIGHTS = {
    'technical': 0.30,
    'fundamental': 0.25,
    'sentiment': 0.25,
    'momentum': 0.20,
}

# Broker Analysis Configuration
BROKER_ANALYSIS_CONFIG = {
    'manipulation_thresholds': {
        'single_broker_critical': 60,  # % - Single broker > 60% is critical
        'single_broker_warning': 45,   # % - Single broker > 45% is warning
        'top_3_critical': 80,          # % - Top 3 brokers > 80% is critical
        'top_3_warning': 70,           # % - Top 3 brokers > 70% is warning
        'hhi_critical': 2500,          # HHI > 2500 = high concentration
        'hhi_warning': 1500,           # HHI > 1500 = moderate concentration
    },
    'smart_money_thresholds': {
        'large_volume_percentile': 0.80,  # Top 20% brokers by volume
        'accumulation_ratio': 1.2,         # Buy/Sell > 1.2 = accumulating
        'distribution_ratio': 0.83,        # Buy/Sell < 0.83 = distributing
        'min_transactions': 5,             # Minimum transactions to be considered
    },
    'liquidity_thresholds': {
        'high_volume': 50000,              # Shares
        'medium_volume': 20000,
        'low_volume': 5000,
        'many_brokers': 30,                # Number of unique brokers
        'some_brokers': 15,
        'few_brokers': 5,
        'high_frequency': 100,             # Transactions per day
        'medium_frequency': 50,
        'low_frequency': 10,
    },
    'pressure_thresholds': {
        'strong_buy_ratio': 1.5,           # Imbalance > 1.5 = strong buying
        'moderate_buy_ratio': 1.2,         # Imbalance > 1.2 = moderate buying
        'strong_sell_ratio': 0.67,         # Imbalance < 0.67 = strong selling
        'moderate_sell_ratio': 0.83,       # Imbalance < 0.83 = moderate selling
    }
}

# Broker Signal Weights (for overall broker score)
BROKER_SIGNAL_WEIGHTS = {
    'concentration': 0.25,        # Lower concentration is better
    'smart_money': 0.25,          # Institutional accumulation
    'pressure': 0.20,             # Buy/sell pressure
    'liquidity': 0.20,            # Market depth
    'manipulation_risk': 0.10,    # Manipulation detection (inverted)
}

# Risk Parameters
RISK_CONFIG = {
    'max_position_size': 0.10,  # 10% of portfolio
    'stop_loss': 0.05,  # 5% stop loss
    'take_profit': 0.15,  # 15% take profit
}

# Database
DB_PATH = os.getenv('DB_PATH', str(DATA_DIR / 'nepse_stocks.db'))

# Analysis Parameters
DEFAULT_LOOKBACK_DAYS = int(os.getenv('DEFAULT_LOOKBACK_DAYS', 365))
MIN_DATA_POINTS = int(os.getenv('MIN_DATA_POINTS', 30))
