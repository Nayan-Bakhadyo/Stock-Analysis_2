# Analysis Data Sources & Methodology

## Complete Breakdown of Analysis Components

This document explains **exactly** what data is used for each component of the stock analysis and how the final probability is calculated.

---

## 1. TECHNICAL ANALYSIS (42.00/100 for IGI)

### Data Source
- **Price History**: 223 days of OHLCV (Open, High, Low, Close, Volume) data from database
- Source: ShareSansar price scraper → SQLite database

### Indicators Calculated

#### Moving Averages
- **SMA Short** (20-day): Simple Moving Average over 20 days
- **SMA Medium** (50-day): Simple Moving Average over 50 days  
- **SMA Long** (200-day): Simple Moving Average over 200 days
- **Purpose**: Identify trend direction and support/resistance

#### Momentum Indicators
- **RSI (14-period)**: Relative Strength Index
  - Measures overbought (>70) / oversold (<30) conditions
  - Formula: `RSI = 100 - (100 / (1 + RS))` where RS = Avg Gain / Avg Loss
  
- **MACD**: Moving Average Convergence Divergence
  - MACD Line = EMA(12) - EMA(26)
  - Signal Line = EMA(9) of MACD Line
  - Histogram = MACD Line - Signal Line
  - **Purpose**: Identify trend changes and momentum

- **Stochastic Oscillator**: %K and %D lines
  - Formula: `%K = 100 * ((Close - Lowest Low) / (Highest High - Lowest Low))`
  - **Purpose**: Identify reversal points

#### Volatility Indicators
- **Bollinger Bands**: Upper, Middle (SMA 20), Lower bands
  - Upper = SMA + (2 × Standard Deviation)
  - Lower = SMA - (2 × Standard Deviation)
  - **Purpose**: Identify volatility and potential breakouts

- **ATR (Average True Range)**: Measures volatility
  - True Range = Max of (High-Low, High-PrevClose, PrevClose-Low)
  - ATR = 14-period average of True Range

#### Volume Indicators
- **OBV (On-Balance Volume)**: Cumulative volume indicator
  - If Close > PrevClose: OBV += Volume
  - If Close < PrevClose: OBV -= Volume
  - **Purpose**: Confirm price trends with volume

### Pattern Detection
- **Head and Shoulders**: Reversal pattern (bearish detected for IGI)
- **Double Top/Bottom**: Reversal patterns
- **Ascending/Descending Triangle**: Continuation patterns
- **Support & Resistance**: Local minima/maxima levels

### Technical Score Calculation
```python
Score = (
    Moving Average Signal (30%) +
    RSI Signal (25%) +
    MACD Signal (20%) +
    Stochastic Signal (15%) +
    Pattern Score (10%)
) × 100
```

**IGI Result**: 42/100 (Bearish pattern, sideways trend)

---

## 2. FUNDAMENTAL ANALYSIS (62.50/100 for IGI)

### Data Source
- **Currently**: Mock/Generated data (not real financial statements)
- **Future**: Should come from company quarterly reports, annual reports

### Metrics Calculated

#### Valuation Ratios
- **P/E Ratio (Price-to-Earnings)**: Current Price / EPS
  - IGI: 20.00 (within acceptable range)
  - Thresholds: <15 undervalued, 15-25 fair, >25 overvalued
  
- **P/B Ratio (Price-to-Book)**: Current Price / Book Value per Share
  - IGI: 2.50 (slightly high)
  - Thresholds: <1.5 undervalued, 1.5-3 fair, >3 overvalued

#### Profitability Metrics
- **EPS (Earnings Per Share)**: Net Income / Total Shares
  - IGI: NPR 20.45
  
- **ROE (Return on Equity)**: (Net Income / Shareholders' Equity) × 100
  - IGI: 12.5%
  - Measures how efficiently company uses shareholders' money

#### Growth Metrics
- **EPS Growth**: ((Current EPS - Previous EPS) / Previous EPS) × 100
  - Measures year-over-year earnings growth

#### Financial Health
- **Debt-to-Equity**: Total Debt / Shareholders' Equity
  - IGI: 0.30 (low debt, good)
  - Measures financial leverage
  
- **Current Ratio**: Current Assets / Current Liabilities
  - Measures liquidity (ability to pay short-term obligations)

#### Dividend Metrics
- **Dividend Yield**: (Annual Dividend / Current Price) × 100
  - IGI: 9.73% cash dividend announced

### Fundamental Score Calculation
```python
Score = Average of:
  - P/E Evaluation (1.0 if undervalued, 0.1 if overvalued)
  - P/B Evaluation (1.0 if undervalued, 0.1 if overvalued)
  - Dividend Yield Evaluation (1.0 if >6%, 0.1 if <1%)
  - EPS Growth Evaluation (1.0 if >20%, 0.1 if negative)
× 100
```

**IGI Result**: 62.50/100 (Fair valuation, good dividend)

---

## 3. SENTIMENT ANALYSIS (70.40/100 for IGI)

### Data Source
- **ShareSansar News Scraper**: Real news articles with full content
- **Method**: Selenium-based web scraping

### Process

#### 1. Article Collection
- Navigate to `https://www.sharesansar.com/company/igi`
- Click "News" tab
- Extract all article links with `/newsdetail/` in URL
- **IGI Found**: 10 news articles

#### 2. Content Extraction
For each article:
- Open article URL in new browser tab
- Extract full paragraph content (up to 2000 chars)
- Extract publication date from URL
- Close tab and return

#### 3. Sentiment Analysis
Uses **TWO** NLP libraries for accuracy:

**VADER (Valence Aware Dictionary and sEntiment Reasoner)**
- Optimized for social media/news text
- Returns compound score: -1 (very negative) to +1 (very positive)

**TextBlob**
- Pattern-based sentiment analysis
- Returns polarity: -1 (negative) to +1 (positive)

**Combined Score**: `(VADER compound + TextBlob polarity) / 2`

#### 4. Recency Weighting
Newer articles have higher impact:
- **0-7 days old**: Weight = 1.0 (100%)
- **8-30 days old**: Weight = 0.7 (70%)
- **31-90 days old**: Weight = 0.4 (40%)
- **91-180 days old**: Weight = 0.2 (20%)
- **180+ days old**: Weight = 0.1 (10%)

#### 5. Aggregate Sentiment
```python
Weighted Average = Σ(Sentiment Score × Recency Weight) / Σ(Recency Weight)
```

### IGI Sentiment Results
- **Total Articles**: 10
- **Positive**: 9 (90%) - dividend announcements, profit growth (156% Q4 surge, 19.5% Q1)
- **Neutral**: 1 (10%) - general industry analysis
- **Negative**: 0 (0%)
- **Average Score**: 0.408
- **Weighted Score**: 0.333
- **Overall**: POSITIVE

### Sentiment Score Calculation
```python
# Normalize to 0-100 scale
normalized_score = ((weighted_average + 1) / 2) × 100

# If no articles found
if total_articles == 0:
    sentiment_score = 50  # Neutral
    weight = 0  # Excluded from final calculation
```

**IGI Result**: 70.40/100 (Strong positive sentiment)

---

## 4. MOMENTUM ANALYSIS (35.54/100 for IGI)

### Data Source
- **Price History**: Same 223 days from database
- Uses only `close` prices and calculates returns

### Calculations

#### Short-term Momentum (5-day)
```python
return_5d = (close[-1] - close[-6]) / close[-6]
```
- Measures price change over last 5 days
- **IGI**: -3.08% (down)

#### Medium-term Momentum (20-day)
```python
return_20d = (close[-1] - close[-21]) / close[-21]
```
- Measures price change over last 20 days

#### Volatility Penalty
```python
volatility = std_dev(daily_returns)
```
- Higher volatility reduces momentum score
- Indicates unstable price movement

#### Combined Momentum Score
```python
momentum = (return_5d × 0.4) + (return_20d × 0.4) - (volatility × 0.2)

# Normalize using tanh function to 0-1 range
normalized = (tanh(momentum × 10) + 1) / 2

# Scale to 0-100
momentum_score = normalized × 100
```

**IGI Result**: 35.54/100 (Weak momentum, recent decline)

---

## 5. BROKER ANALYSIS (50.00/100 for IGI)

### Data Source
- **Floorsheet Data**: Transaction-level broker buy/sell data
- Source: ShareSansar scraper
- **IGI Status**: NO DATA AVAILABLE → Default neutral score (50/100)

### What It Would Analyze (if data available)

#### Smart Money Detection
- Identify brokers with >90% accuracy in timing
- Track their recent buy/sell patterns
- Score: +20 if smart money buying, -20 if selling

#### Concentration Analysis
- Detect if few brokers control >70% volume
- Calculate Herfindahl-Hirschman Index (HHI)
- Flag manipulation risk if concentrated

#### Manipulation Detection
- **Wash Trading**: Same broker buy/sell at similar price
- **Pump & Dump**: Sudden volume spike then decline
- **Circular Trading**: Repeated transfers between same brokers
- **Risk Score**: 0-100 (0=clean, 100=high manipulation)

#### Liquidity Analysis
- Average daily volume
- Bid-ask spread
- Number of active brokers

### Broker Score (when available)
```python
Score = (
    Smart Money Signal (40%) +
    Liquidity Score (30%) +
    Anti-Manipulation Score (30%)
) × 100
```

**IGI Result**: 50/100 (Neutral - no data)

---

## FINAL PROBABILITY CALCULATION

### Weight Distribution

Default weights:
```python
Technical:    25%
Fundamental:  25%
Sentiment:    20%
Momentum:     15%
Broker:       15%
```

When broker analysis included (with data):
```python
Technical:    21.25% (25% × 0.85)
Fundamental:  21.25% (25% × 0.85)
Sentiment:    17.00% (20% × 0.85)
Momentum:     12.75% (15% × 0.85)
Broker:       15.00%
```

When sentiment data missing:
```python
# Set sentiment weight to 0
# Redistribute remaining weights proportionally
```

**IGI Actual Weights Used**:
```python
Technical:    0.2500 (25%)
Fundamental:  0.2500 (25%)
Sentiment:    0.2000 (20%)
Momentum:     0.1500 (15%)
Broker:       0.1500 (15%)
```

### Calculation Formula

```python
profitability_probability = (
    technical_score × technical_weight +
    fundamental_score × fundamental_weight +
    sentiment_score × sentiment_weight +
    momentum_score × momentum_weight +
    broker_score × broker_weight
) × 100
```

### IGI Example Calculation

```python
probability = (
    (42.00/100 × 0.25) +      # Technical:    10.50%
    (62.50/100 × 0.25) +      # Fundamental:  15.625%
    (70.40/100 × 0.20) +      # Sentiment:    14.08%
    (35.54/100 × 0.15) +      # Momentum:     5.331%
    (50.00/100 × 0.15)        # Broker:       7.50%
) × 100

= (0.105 + 0.15625 + 0.1408 + 0.05331 + 0.075) × 100
= 0.53036 × 100
= 53.036%
```

### Manipulation Risk Adjustment
If broker analysis detects manipulation:
```python
if manipulation_risk >= 70:
    probability × 0.5  # -50% penalty
elif manipulation_risk >= 50:
    probability × 0.75  # -25% penalty
```

### Time Horizon Adjustment
```python
volatility_penalty = {
    'short':  -5 × volatility × 100  # Most affected
    'medium': -2 × volatility × 100  # Moderately affected
    'long':    0                     # Not affected
}

final_probability = probability + volatility_penalty
```

**IGI Final Result**: 44.72% (after adjustments)

---

## RISK-REWARD ANALYSIS

### Support & Resistance Calculation
- **Support**: Local minimum prices from price history
- **Resistance**: Local maximum prices from price history

### Nearest Levels
```python
nearest_support = max(support_levels where level < current_price)
                  default: current_price × 0.95

nearest_resistance = min(resistance_levels where level > current_price)
                     default: current_price × 1.05
```

**IGI**:
- Support: NPR 396.20
- Resistance: NPR 570.91

### Risk-Reward Ratio
```python
potential_profit = nearest_resistance - current_price
potential_loss = current_price - nearest_support

risk_reward_ratio = potential_profit / potential_loss
```

**IGI**:
- Potential Profit: 570.91 - 409 = 161.91 (39.59%)
- Potential Loss: 409 - 396.20 = 12.80 (3.13%)
- **Ratio**: 161.91 / 12.80 = **12.65:1**

---

## TRADING RECOMMENDATION

### Decision Logic

```python
if probability >= 70 and risk_reward >= 2.5:
    recommendation = "STRONG BUY"
elif probability >= 60 and risk_reward >= 2:
    recommendation = "BUY"
elif probability >= 40:
    recommendation = "HOLD"
elif probability >= 30:
    recommendation = "SELL"
else:
    recommendation = "STRONG SELL"
```

### Confidence Level
```python
if probability >= 70 or probability <= 30:
    confidence = "High"
elif probability >= 60 or probability <= 40:
    confidence = "Medium"
else:
    confidence = "Low"
```

**IGI Recommendation**:
- Probability: 44.72% → **HOLD**
- Confidence: Low-Medium
- Reasoning: Positive sentiment, favorable R:R ratio, but weak momentum

---

## POSITION SIZING

### Kelly Criterion (Modified)

```python
# Conservative Kelly Criterion
kelly_fraction = (probability/100 - (1-probability/100)) / risk_reward_ratio

# Apply conservative factor (25%)
recommended_position = min(kelly_fraction × 0.25, 0.10)  # Max 10%
```

**IGI**:
- Probability: 44.72%
- R:R Ratio: 12.65
- **Recommended**: 4.47% of portfolio
- **Maximum**: 10% (risk limit)

---

## DATA QUALITY NOTES

### Real Data Sources
✅ **Price History**: Real (ShareSansar scraper → 556 records)
✅ **News Sentiment**: Real (ShareSansar scraper → 10 articles with full content)
✅ **Technical Indicators**: Real (calculated from actual price data)

### Mock/Missing Data
⚠️ **Fundamental Metrics**: Currently mock data (need real company financials)
⚠️ **Broker Analysis**: No floorsheet data for IGI (need to scrape)

### To Improve Accuracy
1. Integrate real company financial statements
2. Scrape floorsheet data for broker analysis
3. Add more news sources (MeroLagani, ShareSansar forums)
4. Include macroeconomic indicators (NEPSE index, sector performance)

---

## Summary: What Drives the Analysis

| Component | Weight | IGI Score | Contribution | Data Source |
|-----------|--------|-----------|--------------|-------------|
| **Technical** | 25% | 42.00 | 10.50% | Real price data (223 days) |
| **Fundamental** | 25% | 62.50 | 15.63% | Mock data (P/E, P/B, dividends) |
| **Sentiment** | 20% | 70.40 | 14.08% | Real news (10 articles, 90% positive) |
| **Momentum** | 15% | 35.54 | 5.33% | Real price returns (5-day: -3.08%) |
| **Broker** | 15% | 50.00 | 7.50% | No data (neutral default) |
| **TOTAL** | 100% | - | **53.04%** | - |
| **After Adjustments** | - | - | **44.72%** | Volatility penalty applied |

**Final Recommendation**: HOLD with 4.47% position size
