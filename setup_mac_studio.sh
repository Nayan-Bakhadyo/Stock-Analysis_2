#!/bin/bash
# Mac Studio M1 Max Setup Script for Stock Analysis
# Environment: Stock_Prediction

set -e

echo "=================================================="
echo "🚀 Stock Analysis - Mac Studio Setup"
echo "=================================================="

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "📦 Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "✅ Homebrew already installed"
fi

# Check Python version
if ! command -v python3.11 &> /dev/null; then
    echo "📦 Installing Python 3.11..."
    brew install python@3.11
else
    echo "✅ Python 3.11 already installed"
fi

# Create conda environment if using conda
if command -v conda &> /dev/null; then
    echo ""
    echo "🐍 Setting up Conda environment: Stock_Prediction"
    
    # Check if environment exists
    if conda env list | grep -q "Stock_Prediction"; then
        echo "✅ Stock_Prediction environment exists"
        echo "   Activating..."
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate Stock_Prediction
    else
        echo "📦 Creating Stock_Prediction environment..."
        conda create -n Stock_Prediction python=3.11 -y
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate Stock_Prediction
    fi
else
    echo ""
    echo "🐍 Setting up venv: Stock_Prediction"
    
    if [ ! -d "Stock_Prediction" ]; then
        python3.11 -m venv Stock_Prediction
    fi
    source Stock_Prediction/bin/activate
fi

echo ""
echo "📦 Installing dependencies..."

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with MPS support
echo "   Installing PyTorch (with MPS/Metal GPU support)..."
pip install torch torchvision torchaudio

# Install xLSTM
echo "   Installing xLSTM..."
pip install xlstm

# Install other requirements
echo "   Installing other requirements..."
pip install -r requirements.txt

# Verify MPS
echo ""
echo "🔍 Verifying MPS (Metal GPU) support..."
python3 -c "import torch; mps = torch.backends.mps.is_available(); print(f'MPS Available: {mps}')"

# Install Chrome and ChromeDriver for scrapers
echo ""
echo "🌐 Setting up Chrome and ChromeDriver..."
if ! command -v chromedriver &> /dev/null; then
    brew install --cask google-chrome 2>/dev/null || echo "Chrome already installed"
    brew install chromedriver 2>/dev/null || echo "ChromeDriver already installed"
    xattr -d com.apple.quarantine $(which chromedriver) 2>/dev/null || true
fi

echo ""
echo "=================================================="
echo "✅ Setup Complete!"
echo "=================================================="
echo ""
echo "To activate the environment:"
if command -v conda &> /dev/null; then
    echo "   conda activate Stock_Prediction"
else
    echo "   source Stock_Prediction/bin/activate"
fi
echo ""
echo "Quick test:"
echo "   python3 xlstm_stock_forecaster.py NABIL --market --fast"
echo ""
echo "Run all predictions:"
echo "   python3 run_all_predictions.py"
echo ""
echo "Run with maximum accuracy (takes longer):"
echo "   python3 run_all_predictions.py --n-models 5 --epochs 100 --hidden-size 512"
echo ""
