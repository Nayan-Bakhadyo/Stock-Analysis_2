#!/bin/bash
# Update ML Predictions Website

echo "🔄 Updating ML Predictions Website..."

# Source and destination paths
SOURCE_JSON="/Users/Nayan/Documents/Business/Stock_Analysis/stock_predictions.json"
DEST_DIR="/Users/Nayan/Documents/Business/Stock_Analysis/stock_website"

# Check if source file exists
if [ ! -f "$SOURCE_JSON" ]; then
    echo "❌ Error: stock_predictions.json not found!"
    echo "   Please run the predictor first: python3 stock_predictor.py <SYMBOL>"
    exit 1
fi

# Copy the JSON file
cp "$SOURCE_JSON" "$DEST_DIR/"

echo "✅ ML predictions data updated!"
echo "📊 Number of stocks: $(python3 -c "import json; data=json.load(open('$SOURCE_JSON')); print(len(data.get('stocks', {})))")"
echo ""
echo "🌐 Open the website:"
echo "   file://$DEST_DIR/ml_predictions.html"
echo ""
echo "💡 To add more stocks, run:"
echo "   python3 stock_predictor.py <SYMBOL>"
echo "   Then run this script again to update the website"
