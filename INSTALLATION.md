# Installation Instructions

## System Requirements

- **Python**: 3.8 or higher
- **Operating System**: macOS, Linux, or Windows
- **RAM**: Minimum 4GB (8GB recommended)
- **Disk Space**: At least 500MB free
- **Internet**: Required for data fetching and news scraping

## Installation Methods

### Method 1: Automated Setup (Recommended)

#### macOS/Linux:
```bash
# Navigate to project directory
cd Stock_Analysis

# Make setup script executable
chmod +x setup.sh

# Run setup
./setup.sh
```

#### Windows:
```cmd
# Navigate to project directory
cd Stock_Analysis

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir data logs reports

# Copy environment file
copy .env.example .env
```

### Method 2: Manual Setup

1. **Create Virtual Environment**
```bash
python -m venv venv
```

2. **Activate Virtual Environment**

macOS/Linux:
```bash
source venv/bin/activate
```

Windows:
```cmd
venv\Scripts\activate
```

3. **Upgrade pip**
```bash
pip install --upgrade pip
```

4. **Install Dependencies**
```bash
pip install -r requirements.txt
```

5. **Create Directories**
```bash
mkdir -p data logs reports
```

6. **Setup Environment**
```bash
cp .env.example .env
```

## Verify Installation

Run this command to verify everything is installed correctly:

```bash
python -c "import pandas, numpy, requests, bs4; print('✅ All dependencies installed successfully!')"
```

## Optional: TA-Lib Installation

TA-Lib is an optional dependency for advanced technical analysis. It can be tricky to install.

### macOS:
```bash
brew install ta-lib
pip install TA-Lib
```

### Ubuntu/Debian:
```bash
sudo apt-get install ta-lib
pip install TA-Lib
```

### Windows:
Download the appropriate wheel file from:
https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib

Then install:
```bash
pip install TA_Lib‑0.4.XX‑cpXX‑cpXX‑win_amd64.whl
```

**Note**: The system works without TA-Lib using the `ta` library as a fallback.

## Post-Installation

### 1. Configure Environment Variables (Optional)

Edit the `.env` file:
```bash
nano .env  # or use your preferred editor
```

### 2. Test the Installation

```bash
# Test basic functionality
python main.py --help

# Test with market overview
python main.py market
```

### 3. Run Your First Analysis

```bash
python main.py analyze NABIL
```

## Troubleshooting Installation Issues

### Issue: "Python not found"

**Solution**: Install Python from https://www.python.org/downloads/

### Issue: "pip not found"

**Solution**:
```bash
python -m ensurepip --upgrade
```

### Issue: "Permission denied" on setup.sh

**Solution**:
```bash
chmod +x setup.sh
```

### Issue: Module import errors

**Solution**:
1. Ensure virtual environment is activated
2. Reinstall dependencies:
```bash
pip install --upgrade -r requirements.txt
```

### Issue: SSL Certificate errors

**Solution**:
```bash
pip install --upgrade certifi
```

### Issue: NumPy/Pandas installation failures

**Solution**: Install build dependencies first

macOS:
```bash
xcode-select --install
```

Ubuntu/Debian:
```bash
sudo apt-get install python3-dev build-essential
```

## Updating the Application

To update to the latest version:

```bash
# Activate virtual environment
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Update dependencies
pip install --upgrade -r requirements.txt

# Pull latest code (if using git)
git pull origin main
```

## Uninstallation

To completely remove the application:

```bash
# Deactivate virtual environment
deactivate

# Remove project directory
rm -rf Stock_Analysis  # macOS/Linux
rmdir /s Stock_Analysis  # Windows
```

## Need Help?

If you encounter any issues during installation:

1. Check the error message carefully
2. Ensure your Python version is 3.8+
3. Make sure you're in the correct directory
4. Try running with `sudo` (macOS/Linux) or as Administrator (Windows) if permission issues occur
5. Check your internet connection for downloading packages

## Next Steps

After successful installation:

1. Read `QUICKSTART.md` for quick start guide
2. Check `README.md` for comprehensive documentation
3. Explore `examples.py` for usage examples
4. Start analyzing stocks!

---

**Congratulations! You're ready to analyze NEPSE stocks! 🎉**
