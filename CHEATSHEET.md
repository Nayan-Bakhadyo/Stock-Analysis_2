# NEPSE Stock Analysis - Command Cheat Sheet

## 🚀 Quick Commands

### Setup & Installation
```bash
# Initial setup (one time)
./setup.sh

# Activate environment (every session)
source venv/bin/activate          # macOS/Linux
venv\Scripts\activate             # Windows

# Update dependencies
pip install --upgrade -r requirements.txt
```

### Basic Analysis Commands

```bash
# Analyze a single stock
python main.py analyze NABIL

# Market overview
python main.py market

# Compare stocks
python main.py compare NABIL NICA GBIME

# Get help
python main.py --help
python main.py analyze --help
```

### Analysis Options

```bash
# Time horizons
python main.py analyze NABIL --horizon short    # 1-7 days (default)
python main.py analyze NABIL --horizon medium   # 1-4 weeks
python main.py analyze NABIL --horizon long     # 1-3 months

# Detailed analysis
python main.py analyze NABIL --detailed

# Export to JSON
python main.py analyze NABIL --export

# Combined options
python main.py analyze NABIL --horizon medium --detailed --export
```

### Stock Symbols Quick Reference

```bash
# Banking (Commercial)
NABIL NICA SCB HBL EBL GBIME NBL SBI NIB
ADBL SANIMA MEGA PCBL CZBIL KBL BOKL

# Development Banks
SHINE SAPDBL JBNL

# Hydropower
UPPER CHCL NHPC

# Insurance
NICL SICL HGI

# Hotels
OHL TRHPR
```

## 📊 Common Analysis Patterns

### 1. Daily Market Check
```bash
# Quick market overview
python main.py market
```

### 2. Research Before Buying
```bash
# Detailed analysis of target stock
python main.py analyze NABIL --detailed

# Compare with competitors
python main.py compare NABIL NICA SCB --horizon medium

# Export for record keeping
python main.py analyze NABIL --export
```

### 3. Portfolio Review
```bash
# Analyze each holding
python main.py analyze NABIL
python main.py analyze NICA
python main.py analyze GBIME

# Or compare all at once
python main.py compare NABIL NICA GBIME
```

### 4. Sector Analysis
```bash
# Banking sector
python main.py compare NABIL NICA SCB HBL EBL

# Hydropower sector
python main.py compare UPPER CHCL NHPC

# Insurance sector
python main.py compare NICL SICL HGI
```

## 🔍 Interpreting Results

### Profitability Probability
- **75-100%**: Very High → Consider Strong Buy
- **60-74%**: High → Consider Buy
- **40-59%**: Medium → Hold/Monitor
- **25-39%**: Low → Consider Sell
- **0-24%**: Very Low → Strong Sell Signal

### Risk-Reward Ratio
- **>3.0**: Excellent
- **2.0-3.0**: Very Good
- **1.5-2.0**: Good
- **1.0-1.5**: Fair
- **<1.0**: Poor (avoid)

### Recommendation Actions
- **STRONG BUY**: High probability + Great R:R
- **BUY**: Good probability + Good R:R
- **HOLD/ACCUMULATE**: Medium probability
- **HOLD**: Neutral signals
- **SELL**: Weak signals
- **STRONG SELL**: Very weak signals

## 💡 Pro Tips

### Best Practices
```bash
# 1. Always check market overview first
python main.py market

# 2. Analyze individual stocks
python main.py analyze SYMBOL --detailed

# 3. Compare with sector peers
python main.py compare SYMBOL PEER1 PEER2

# 4. Check different time horizons
python main.py analyze SYMBOL --horizon short
python main.py analyze SYMBOL --horizon medium
python main.py analyze SYMBOL --horizon long

# 5. Export important analyses
python main.py analyze SYMBOL --export
```

### Workflow Examples

**For Day Trading**
```bash
python main.py market                           # Check market
python main.py compare GAINER1 GAINER2 GAINER3 # Analyze gainers
python main.py analyze SYMBOL --horizon short   # Quick analysis
```

**For Swing Trading**
```bash
python main.py analyze SYMBOL --horizon medium --detailed
python main.py compare SYMBOL PEER1 PEER2
```

**For Long-term Investment**
```bash
python main.py analyze SYMBOL --horizon long --detailed --export
python main.py compare SYMBOL1 SYMBOL2 SYMBOL3 --horizon long
```

## 🛠️ Maintenance Commands

```bash
# Check Python version
python --version

# List installed packages
pip list

# Update specific package
pip install --upgrade pandas

# Clear cache (if needed)
rm -rf data/*.db     # macOS/Linux
del data\*.db        # Windows

# View logs
cat logs/*.log       # macOS/Linux
type logs\*.log      # Windows
```

## 🐛 Troubleshooting Quick Fixes

```bash
# Virtual environment not activated
source venv/bin/activate          # macOS/Linux
venv\Scripts\activate             # Windows

# Module not found
pip install -r requirements.txt

# Permission denied
chmod +x setup.sh                 # macOS/Linux

# Clear Python cache
find . -type d -name __pycache__ -exec rm -rf {} +  # macOS/Linux
```

## 📱 Integration Examples

### Save to Variable (Python)
```python
from trading_insights import TradingInsightsEngine

engine = TradingInsightsEngine()
result = engine.calculate_profitability_probability("NABIL", "short")

print(f"Probability: {result['profitability_probability']:.2f}%")
print(f"Action: {result['recommendation']['action']}")
```

### Batch Processing
```bash
# Create a script for multiple stocks
for stock in NABIL NICA GBIME SCB; do
    python main.py analyze $stock --export
    sleep 5  # Rate limiting
done
```

## 📋 Output File Locations

```bash
data/                 # Database and cached data
  └── nepse_stocks.db

logs/                 # Application logs
  └── nepse_analysis_YYYYMMDD.log

reports/              # Exported analyses
  └── SYMBOL_analysis_YYYYMMDD_HHMMSS.json
```

## ⚡ Keyboard Shortcuts (in terminal)

```bash
Ctrl+C    # Stop current command
Ctrl+L    # Clear screen
↑ / ↓     # Navigate command history
Tab       # Auto-complete
Ctrl+R    # Search command history
```

## 🎯 Common Use Cases

### Morning Routine
```bash
python main.py market
python main.py compare NABIL NICA SCB HBL EBL
```

### Research New Stock
```bash
python main.py analyze NEWSYMBOL --detailed
python main.py compare NEWSYMBOL COMPETITOR1 COMPETITOR2
```

### Portfolio Check
```bash
python main.py compare HOLDING1 HOLDING2 HOLDING3
```

### Export for Records
```bash
python main.py analyze SYMBOL --export
# File saved to reports/SYMBOL_analysis_*.json
```

---

## 💡 Remember

- **Always activate virtual environment first**
- **Check market overview before individual analysis**
- **Compare stocks within same sector**
- **Use appropriate time horizon for your strategy**
- **Export important analyses for record keeping**
- **This is educational - not financial advice**

**Happy Trading! 📈💰**
