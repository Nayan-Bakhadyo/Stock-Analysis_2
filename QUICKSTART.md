# Quick Start Guide - NEPSE Stock Analysis

## 🚀 Getting Started in 5 Minutes

### Step 1: Install Python
Make sure you have Python 3.8+ installed:
```bash
python --version
```

### Step 2: Run Setup Script

#### On macOS/Linux:
```bash
chmod +x setup.sh
./setup.sh
```

#### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
mkdir data logs reports
copy .env.example .env
```

### Step 3: Activate Virtual Environment

#### On macOS/Linux:
```bash
source venv/bin/activate
```

#### On Windows:
```bash
venv\Scripts\activate
```

### Step 4: Run Your First Analysis

```bash
# Analyze a single stock
python main.py analyze NABIL

# View market overview
python main.py market

# Compare multiple stocks
python main.py compare NABIL NICA GBIME
```

## 📊 Common Stock Symbols (NEPSE)

### Commercial Banks
- NABIL - Nabil Bank Limited
- NICA - NIC Asia Bank Limited
- SCB - Standard Chartered Bank Nepal Limited
- HBL - Himalayan Bank Limited
- EBL - Everest Bank Limited
- GBIME - Global IME Bank Limited
- ADBL - Agriculture Development Bank Limited
- NIB - Nepal Investment Bank Limited
- SBI - Nepal SBI Bank Limited
- NBL - Nepal Bank Limited
- SANIMA - Sanima Bank Limited
- MEGA - Mega Bank Nepal Limited
- PCBL - Prime Commercial Bank Limited
- CZBIL - Citizen Bank International Limited
- KBL - Kumari Bank Limited
- BOKL - Bank of Kathmandu Limited

### Development Banks
- SHINE - Shine Resunga Development Bank Limited
- SAPDBL - Saptakoshi Development Bank Limited
- JBNL - Jyoti Bikash Bank Limited

### Finance Companies
- GFCL - Goodwill Finance Limited
- CFCL - Central Finance Limited

### Insurance Companies
- NICL - Nepal Insurance Company Limited
- SICL - Shikhar Insurance Company Limited
- HGI - Himalayan General Insurance Limited

### Hydropower
- UPPER - Upper Tamakoshi Hydropower Limited
- CHCL - Chilime Hydropower Company Limited
- NHPC - National Hydropower Company Limited

### Hotels
- OHL - Oriental Hotels Limited
- TRHPR - Taragaon Regency Hotel Limited

## 💡 Quick Tips

### 1. Finding Good Stocks
```bash
# Get market overview to see top gainers
python main.py market
```

### 2. Detailed Analysis
```bash
# Get detailed analysis with all metrics
python main.py analyze NABIL --detailed
```

### 3. Export Results
```bash
# Save analysis to JSON file
python main.py analyze NABIL --export
```

### 4. Time Horizons
```bash
# Short-term (1-7 days) - default
python main.py analyze NABIL --horizon short

# Medium-term (1-4 weeks)
python main.py analyze NABIL --horizon medium

# Long-term (1-3 months)
python main.py analyze NABIL --horizon long
```

## 🎯 Understanding the Output

### Profitability Probability
- **75-100%**: Very High - Strong Buy Signal
- **60-74%**: High - Buy Signal
- **40-59%**: Medium - Hold/Neutral
- **25-39%**: Low - Sell Signal
- **0-24%**: Very Low - Strong Sell Signal

### Risk-Reward Ratio
- **>2.0**: Excellent - High reward vs risk
- **1.5-2.0**: Good - Favorable ratio
- **1.0-1.5**: Fair - Balanced
- **<1.0**: Poor - Risk outweighs reward

### Technical Score
- **70-100**: Strong technical signals
- **40-69**: Moderate signals
- **0-39**: Weak technical signals

## ⚠️ Important Notes

1. **Paper Trading First**: Test the system with paper trading before using real money
2. **DYOR**: Always do your own research
3. **Risk Management**: Never invest more than you can afford to lose
4. **Diversification**: Don't put all eggs in one basket
5. **Stop Loss**: Always use stop loss orders

## 🐛 Troubleshooting

### "Module not found" Error
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

### "Insufficient Data" Error
- Stock may be newly listed
- Try a different stock with longer trading history
- Check internet connection

### API Connection Issues
- Check internet connection
- NEPSE website may be temporarily down
- Try again later

## 📚 Learn More

- Read the full `README.md` for comprehensive documentation
- Check individual module files for advanced usage
- Explore `config.py` to customize settings

## 🆘 Get Help

If you encounter issues:
1. Check the troubleshooting section
2. Review error messages carefully
3. Ensure all dependencies are installed
4. Verify Python version compatibility

---

**Happy Trading! 📈💰**

*Remember: This tool is for educational purposes. Not financial advice.*
