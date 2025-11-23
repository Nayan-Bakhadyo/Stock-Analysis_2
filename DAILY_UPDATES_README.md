# Daily Update Scripts

This directory contains modular scripts for incremental daily updates of stock data.

## Overview

The original `daily_update.py` has been broken down into three focused modules:

1. **update_price_data.py** - Price history updates
2. **update_sentiment.py** - News & sentiment analysis
3. **update_fundamental_data.py** - Financial ratios & fundamentals

Plus a master script:
- **update_all.py** - Runs all three modules in sequence

## Scripts

### 1. update_price_data.py

**Purpose:** Incrementally sync price history data

**Features:**
- Fast incremental updates (only fetches new data)
- Checks last update date and syncs only missing days
- No browser automation needed
- Lightweight and quick

**Usage:**
```bash
# Update all stocks from analysis_results.json
python3 update_price_data.py

# Update specific stocks
python3 update_price_data.py NABIL NICA SCB
```

**Output:**
- Updates database (`data/nepse_stocks.db`)
- Console output with sync statistics

### 2. update_sentiment.py

**Purpose:** Incrementally sync news articles and analyze sentiment

**Features:**
- Fetches only new news articles (avoids duplicates)
- Simple keyword-based sentiment analysis
- Caches articles in database
- Returns sentiment score (-1 to 1) and label (Positive/Neutral/Negative)

**Usage:**
```bash
# Update all stocks from analysis_results.json
python3 update_sentiment.py

# Update specific stocks
python3 update_sentiment.py NABIL NICA SCB
```

**Output:**
- `sentiment_results.json` - Sentiment scores for all stocks
- Updates news cache in database

### 3. update_fundamental_data.py

**Purpose:** Update fundamental financial data

**Features:**
- Scrapes latest financial ratios from NepalAlpha
- Updates P/E, P/B, ROE, EPS, etc.
- Comprehensive fundamental analysis
- Overall score (0-100) and rating

**Usage:**
```bash
# Update all stocks from analysis_results.json
python3 update_fundamental_data.py

# Update specific stocks
python3 update_fundamental_data.py NABIL NICA SCB
```

**Output:**
- `fundamental_results.json` - Latest fundamental data and analysis

### 4. update_all.py (Master Script)

**Purpose:** Run all three update modules in sequence

**Features:**
- Executes all updates in optimal order
- Error handling for each module
- Comprehensive summary at the end

**Usage:**
```bash
# Update all stocks from analysis_results.json
python3 update_all.py

# Update specific stocks
python3 update_all.py NABIL NICA SCB
```

**Output:**
- All three JSON files updated
- Database updated with latest data

## Incremental Update Strategy

### Price Data (update_price_data.py)
- Checks latest date in database
- If data is < 7 days old: fetches only missing days
- If data is > 7 days old: full sync
- If today's data exists: skip (already up-to-date)

### Sentiment (update_sentiment.py)
- Fetches max 10 new articles per stock
- Compares article titles to avoid duplicates
- Only new articles are saved to cache
- Sentiment calculated from all cached articles (last 180 days)

### Fundamental Data (update_fundamental_data.py)
- Scrapes latest data from NepalAlpha
- Always fetches fresh data (fundamentals change less frequently)
- 1-second delay between stocks to avoid overwhelming server

## Comparison with daily_update.py

| Feature | daily_update.py | New Modular Scripts |
|---------|-----------------|---------------------|
| **ML Training** | Yes (can be disabled) | No (ML disabled by default) |
| **Modularity** | Single monolithic script | 3 separate focused modules |
| **Flexibility** | All-or-nothing | Run individual modules |
| **Speed** | Slower (full analysis) | Faster (incremental only) |
| **Debugging** | Harder (everything together) | Easier (isolated modules) |

## When to Use What

### Use update_price_data.py when:
- You only need latest prices (no sentiment/fundamentals)
- Quick daily check for price changes
- Building price-based technical indicators

### Use update_sentiment.py when:
- News-driven trading strategy
- Want to track sentiment changes over time
- Checking for market-moving news

### Use update_fundamental_data.py when:
- Quarterly earnings released
- Company announces dividends/bonuses
- Financial ratios need updating

### Use update_all.py when:
- Daily comprehensive update
- Keeping all data fresh
- Automated daily cron job

## Recommended Schedule

```bash
# Daily (every morning before market opens)
python3 update_all.py

# OR run individually as needed:

# Multiple times per day (fast)
python3 update_price_data.py

# Once per day
python3 update_sentiment.py

# Weekly or when fundamentals change
python3 update_fundamental_data.py
```

## Automation with Cron

```bash
# Edit crontab
crontab -e

# Add daily update at 9 AM
0 9 * * * cd /path/to/Stock_Analysis && python3 update_all.py >> logs/daily_update.log 2>&1
```

## Output Files

| File | Content | Updated By |
|------|---------|-----------|
| `sentiment_results.json` | Sentiment scores | update_sentiment.py |
| `fundamental_results.json` | Financial ratios | update_fundamental_data.py |
| `data/nepse_stocks.db` | Price history & news cache | All scripts |
| `analysis_results.json` | Full analysis (if exists) | Not updated by these scripts |

## Migration from daily_update.py

The new scripts provide the same functionality as `daily_update.py` but with:
- **No ML overhead** - Faster execution
- **Better modularity** - Run only what you need
- **Easier debugging** - Isolated failures
- **More flexibility** - Customize update frequency per module

You can still use `daily_update.py` if you need ML predictions, but for pure data updates, the new modular scripts are recommended.
