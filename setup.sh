#!/bin/bash
# Setup script for NEPSE Stock Analysis

echo "🚀 Setting up NEPSE Stock Analysis System..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "✅ Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create directories
echo "📁 Creating necessary directories..."
mkdir -p data logs reports

# Copy environment file
echo "⚙️  Setting up environment configuration..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✅ Created .env file. Please update it with your configuration."
else
    echo "ℹ️  .env file already exists."
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "📖 Next steps:"
echo "   1. Activate the virtual environment:"
echo "      source venv/bin/activate"
echo ""
echo "   2. Edit .env file with your configuration (optional)"
echo ""
echo "   3. Run your first analysis:"
echo "      python main.py analyze NABIL"
echo ""
echo "   4. View market overview:"
echo "      python main.py market"
echo ""
echo "   5. Compare stocks:"
echo "      python main.py compare NABIL NICA GBIME"
echo ""
echo "Happy Trading! 📈💰"
