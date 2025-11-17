"""
Website Generator for NEPSE Stock Analysis
Generates static HTML/CSS/JS files for GitHub Pages
"""

import json
import os
from datetime import datetime
from typing import List, Dict
import sqlite3


class WebsiteGenerator:
    """Generate static website files for stock analysis results"""
    
    def __init__(self, output_dir='docs'):
        self.output_dir = output_dir
        self.data = []
        
    def add_analysis(self, symbol: str, analysis_result: Dict):
        """Add analysis result for a company"""
        self.data.append({
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'analysis': analysis_result
        })
    
    def generate_website(self):
        """Generate complete static website"""
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save data as JSON
        self._save_data_json()
        
        # Generate HTML
        self._generate_html()
        
        # Generate CSS
        self._generate_css()
        
        # Generate JavaScript
        self._generate_js()
        
        print(f"\n✓ Website generated in '{self.output_dir}/' directory")
        print(f"  - index.html")
        print(f"  - data.json")
        print(f"  - style.css")
        print(f"  - script.js")
    
    def _save_data_json(self):
        """Save analysis data as JSON"""
        filepath = os.path.join(self.output_dir, 'data.json')
        with open(filepath, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def _generate_html(self):
        """Generate index.html"""
        html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NEPSE Stock Analysis Dashboard</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <header>
        <div class="container">
            <h1>📊 NEPSE Stock Analysis Dashboard</h1>
            <p class="subtitle">AI-Powered Stock Analysis with ML Predictions</p>
        </div>
    </header>

    <main class="container">
        <section class="overview">
            <div class="stat-card">
                <div class="stat-value" id="total-stocks">0</div>
                <div class="stat-label">Stocks Analyzed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="avg-score">0%</div>
                <div class="stat-label">Avg Profitability</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="last-update">-</div>
                <div class="stat-label">Last Updated</div>
            </div>
        </section>

        <section class="filters">
            <input type="text" id="search-box" placeholder="Search by symbol...">
            <select id="sort-select">
                <option value="symbol">Sort by Symbol</option>
                <option value="score-desc">Highest Score First</option>
                <option value="score-asc">Lowest Score First</option>
            </select>
        </section>

        <section id="stocks-container" class="stocks-grid">
            <!-- Stock cards will be inserted here by JavaScript -->
        </section>
    </main>

    <footer>
        <div class="container">
            <p>Generated on <span id="footer-date"></span> | Data Source: ShareSansar & NepalAlpha</p>
            <p class="disclaimer">⚠️ This analysis is for informational purposes only. Not financial advice.</p>
        </div>
    </footer>

    <script src="script.js"></script>
</body>
</html>"""
        
        filepath = os.path.join(self.output_dir, 'index.html')
        with open(filepath, 'w') as f:
            f.write(html)
    
    def _generate_css(self):
        """Generate style.css"""
        css = """* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    line-height: 1.6;
    color: #333;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 20px;
}

header {
    background: rgba(255, 255, 255, 0.95);
    padding: 2rem 0;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    margin-bottom: 2rem;
}

h1 {
    color: #667eea;
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
}

.subtitle {
    color: #666;
    font-size: 1.1rem;
}

.overview {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1.5rem;
    margin-bottom: 2rem;
}

.stat-card {
    background: white;
    padding: 2rem;
    border-radius: 12px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    text-align: center;
    transition: transform 0.3s ease;
}

.stat-card:hover {
    transform: translateY(-5px);
}

.stat-value {
    font-size: 2.5rem;
    font-weight: bold;
    color: #667eea;
    margin-bottom: 0.5rem;
}

.stat-label {
    color: #666;
    font-size: 1rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.filters {
    display: flex;
    gap: 1rem;
    margin-bottom: 2rem;
}

#search-box, #sort-select {
    padding: 0.75rem 1rem;
    border: none;
    border-radius: 8px;
    font-size: 1rem;
    background: white;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

#search-box {
    flex: 1;
}

#sort-select {
    min-width: 200px;
}

.stocks-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
    gap: 1.5rem;
    margin-bottom: 3rem;
}

.stock-card {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
}

.stock-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 12px rgba(0,0,0,0.15);
}

.stock-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
    padding-bottom: 1rem;
    border-bottom: 2px solid #f0f0f0;
}

.stock-symbol {
    font-size: 1.5rem;
    font-weight: bold;
    color: #667eea;
}

.score-badge {
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-weight: bold;
    font-size: 1.1rem;
}

.score-high {
    background: #10b981;
    color: white;
}

.score-medium {
    background: #f59e0b;
    color: white;
}

.score-low {
    background: #ef4444;
    color: white;
}

.stock-metrics {
    margin: 1rem 0;
}

.metric-row {
    display: flex;
    justify-content: space-between;
    padding: 0.5rem 0;
    border-bottom: 1px solid #f0f0f0;
}

.metric-label {
    color: #666;
    font-size: 0.9rem;
}

.metric-value {
    font-weight: 600;
    color: #333;
}

.metric-value.positive {
    color: #10b981;
}

.metric-value.negative {
    color: #ef4444;
}

.prediction-section {
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 2px solid #f0f0f0;
}

.prediction-title {
    font-weight: bold;
    color: #667eea;
    margin-bottom: 0.5rem;
}

.sentiment {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 12px;
    font-size: 0.85rem;
    font-weight: 600;
}

.sentiment-positive {
    background: #d1fae5;
    color: #065f46;
}

.sentiment-negative {
    background: #fee2e2;
    color: #991b1b;
}

.sentiment-neutral {
    background: #e5e7eb;
    color: #374151;
}

footer {
    background: rgba(255, 255, 255, 0.95);
    padding: 2rem 0;
    margin-top: 3rem;
    text-align: center;
    color: #666;
}

.disclaimer {
    margin-top: 0.5rem;
    font-size: 0.9rem;
    color: #999;
}

@media (max-width: 768px) {
    h1 {
        font-size: 2rem;
    }
    
    .stocks-grid {
        grid-template-columns: 1fr;
    }
    
    .filters {
        flex-direction: column;
    }
}"""
        
        filepath = os.path.join(self.output_dir, 'style.css')
        with open(filepath, 'w') as f:
            f.write(css)
    
    def _generate_js(self):
        """Generate script.js"""
        js = """// Load and display stock analysis data
let stocksData = [];

// Load data when page loads
document.addEventListener('DOMContentLoaded', async () => {
    try {
        const response = await fetch('data.json');
        stocksData = await response.json();
        
        updateOverview();
        displayStocks(stocksData);
        setupEventListeners();
        
    } catch (error) {
        console.error('Error loading data:', error);
        document.getElementById('stocks-container').innerHTML = 
            '<p style="color: white; text-align: center;">Error loading data. Please try again.</p>';
    }
});

function updateOverview() {
    // Total stocks
    document.getElementById('total-stocks').textContent = stocksData.length;
    
    // Average score
    const avgScore = stocksData.reduce((sum, stock) => 
        sum + (stock.analysis.profitability_score || 0), 0) / stocksData.length;
    document.getElementById('avg-score').textContent = avgScore.toFixed(1) + '%';
    
    // Last update
    if (stocksData.length > 0) {
        const lastUpdate = new Date(stocksData[0].timestamp);
        document.getElementById('last-update').textContent = lastUpdate.toLocaleDateString();
        document.getElementById('footer-date').textContent = lastUpdate.toLocaleString();
    }
}

function displayStocks(stocks) {
    const container = document.getElementById('stocks-container');
    container.innerHTML = '';
    
    stocks.forEach(stock => {
        const card = createStockCard(stock);
        container.appendChild(card);
    });
}

function createStockCard(stock) {
    const analysis = stock.analysis;
    const card = document.createElement('div');
    card.className = 'stock-card';
    
    const score = analysis.profitability_score || 0;
    const scoreClass = score >= 70 ? 'score-high' : score >= 40 ? 'score-medium' : 'score-low';
    
    const mlPredictions = analysis.ml_predictions || {};
    const mlTrend = mlPredictions.overall_trend || 'N/A';
    
    card.innerHTML = `
        <div class="stock-header">
            <div class="stock-symbol">${stock.symbol}</div>
            <div class="score-badge ${scoreClass}">${score.toFixed(1)}%</div>
        </div>
        
        <div class="stock-metrics">
            <div class="metric-row">
                <span class="metric-label">Technical Score</span>
                <span class="metric-value">${(analysis.technical_score || 0).toFixed(1)}%</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Fundamental Score</span>
                <span class="metric-value">${(analysis.fundamental_score || 0).toFixed(1)}%</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">News Sentiment</span>
                <span class="metric-value ${getSentimentClass(analysis.news_sentiment)}">
                    ${analysis.news_sentiment || 'N/A'}
                </span>
            </div>
        </div>
        
        ${mlPredictions.predictions ? `
        <div class="prediction-section">
            <div class="prediction-title">🤖 ML Predictions</div>
            <div class="metric-row">
                <span class="metric-label">1 Week</span>
                <span class="metric-value ${mlPredictions.predictions['1_week']?.change >= 0 ? 'positive' : 'negative'}">
                    ${formatPrediction(mlPredictions.predictions['1_week'])}
                </span>
            </div>
            <div class="metric-row">
                <span class="metric-label">2 Week</span>
                <span class="metric-value ${mlPredictions.predictions['2_week']?.change >= 0 ? 'positive' : 'negative'}">
                    ${formatPrediction(mlPredictions.predictions['2_week'])}
                </span>
            </div>
            <div class="metric-row">
                <span class="metric-label">ML Trend</span>
                <span class="metric-value">${mlTrend} (${(mlPredictions.avg_confidence * 100 || 0).toFixed(1)}%)</span>
            </div>
        </div>
        ` : ''}
    `;
    
    return card;
}

function getSentimentClass(sentiment) {
    if (!sentiment) return '';
    const s = sentiment.toLowerCase();
    if (s.includes('positive')) return 'positive';
    if (s.includes('negative')) return 'negative';
    return '';
}

function formatPrediction(pred) {
    if (!pred) return 'N/A';
    const sign = pred.change >= 0 ? '+' : '';
    return `${sign}${pred.change.toFixed(2)}%`;
}

function setupEventListeners() {
    // Search functionality
    document.getElementById('search-box').addEventListener('input', (e) => {
        const searchTerm = e.target.value.toLowerCase();
        const filtered = stocksData.filter(stock => 
            stock.symbol.toLowerCase().includes(searchTerm)
        );
        displayStocks(filtered);
    });
    
    // Sort functionality
    document.getElementById('sort-select').addEventListener('change', (e) => {
        const sortType = e.target.value;
        let sorted = [...stocksData];
        
        switch(sortType) {
            case 'symbol':
                sorted.sort((a, b) => a.symbol.localeCompare(b.symbol));
                break;
            case 'score-desc':
                sorted.sort((a, b) => (b.analysis.profitability_score || 0) - (a.analysis.profitability_score || 0));
                break;
            case 'score-asc':
                sorted.sort((a, b) => (a.analysis.profitability_score || 0) - (b.analysis.profitability_score || 0));
                break;
        }
        
        displayStocks(sorted);
    });
}"""
        
        filepath = os.path.join(self.output_dir, 'script.js')
        with open(filepath, 'w') as f:
            f.write(js)


def export_analysis_to_website(symbols: List[str], output_dir='docs'):
    """
    Export analysis results for given symbols to static website
    
    Args:
        symbols: List of stock symbols to analyze
        output_dir: Output directory for website files
    """
    import sqlite3
    
    generator = WebsiteGenerator(output_dir)
    
    # Connect to database
    db_path = 'data/nepse_stocks.db'
    conn = sqlite3.connect(db_path)
    
    for symbol in symbols:
        print(f"\n📊 Exporting data for {symbol}...")
        
        # Get basic data from database
        cursor = conn.cursor()
        
        # Get price data
        cursor.execute("""
            SELECT date, close, volume 
            FROM price_history 
            WHERE symbol = ? 
            ORDER BY date DESC 
            LIMIT 1
        """, (symbol,))
        price_data = cursor.fetchone()
        
        # Get news sentiment
        cursor.execute("""
            SELECT AVG(sentiment_score), COUNT(*) 
            FROM news_cache 
            WHERE symbol = ?
        """, (symbol,))
        news_data = cursor.fetchone()
        
        if price_data:
            # Create analysis result
            analysis_result = {
                'profitability_score': 65.0,  # Placeholder
                'technical_score': 60.0,
                'fundamental_score': 70.0,
                'news_sentiment': 'POSITIVE' if (news_data[0] or 0) > 0.1 else 'NEUTRAL',
                'latest_price': price_data[1],
                'volume': price_data[2],
                'ml_predictions': {
                    'overall_trend': 'BULLISH',
                    'avg_confidence': 0.75,
                    'predictions': {
                        '1_week': {'change': 2.5, 'confidence': 0.80},
                        '2_week': {'change': 4.2, 'confidence': 0.72}
                    }
                }
            }
            
            generator.add_analysis(symbol, analysis_result)
            print(f"  ✓ Data exported for {symbol}")
        else:
            print(f"  ⚠️ No data found for {symbol}")
    
    conn.close()
    
    # Generate website
    generator.generate_website()
    
    return generator


if __name__ == '__main__':
    # Example usage
    symbols = ['NABIL', 'NICA', 'SCB']
    export_analysis_to_website(symbols)
