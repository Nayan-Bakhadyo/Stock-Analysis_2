"""
Static Website Generator for Stock Analysis Results
Creates interactive HTML/CSS/JS website with charts and visualizations
"""

import json
import os
from datetime import datetime


class StockWebsiteGenerator:
    """Generate static website from analysis results"""
    
    def __init__(self, json_file='analysis_results.json', output_dir='stock_website'):
        self.json_file = json_file
        self.output_dir = output_dir
        self.data = []
        
    def load_data(self):
        """Load analysis results from JSON"""
        with open(self.json_file, 'r') as f:
            self.data = json.load(f)
        
        # Sort by profitability (we'll calculate from data)
        for stock in self.data:
            # Calculate profitability score from available data
            score = 0
            if stock.get('news'):
                sentiment = stock['news'].get('avg_sentiment', 0)
                score += (sentiment + 1) * 25  # Convert -1 to 1 range to 0-50
            if stock.get('candlestick_patterns'):
                bullish = sum(1 for p in stock['candlestick_patterns'] if p['type'] == 'Bullish')
                bearish = sum(1 for p in stock['candlestick_patterns'] if p['type'] == 'Bearish')
                score += (bullish - bearish) * 5
            if stock.get('price_data'):
                score += 25  # Base score for having data
                
            stock['profitability_score'] = max(0, min(100, score))
        
        # Sort by profitability score (highest first)
        self.data.sort(key=lambda x: x.get('profitability_score', 0), reverse=True)
        
        print(f"✓ Loaded {len(self.data)} stocks from {self.json_file}")
    
    def generate_website(self):
        """Generate complete website"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        self._generate_html()
        self._generate_css()
        self._generate_js()
        self._copy_data()
        
        print(f"\n✓ Website generated in '{self.output_dir}/' directory")
        print(f"  → Open {self.output_dir}/index.html in your browser")
    
    def _copy_data(self):
        """Copy JSON data to website folder"""
        import shutil
        dest = os.path.join(self.output_dir, 'data.json')
        
        # Save sorted data
        with open(dest, 'w') as f:
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
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
</head>
<body>
    <div class="header-bg"></div>
    
    <header>
        <div class="container">
            <h1>📈 NEPSE Stock Analysis</h1>
            <p class="subtitle">Professional Stock Analysis with AI-Powered Insights</p>
        </div>
    </header>

    <main class="container">
        <!-- Overview Stats -->
        <section class="overview-stats">
            <div class="stat-card pulse">
                <div class="stat-icon">📊</div>
                <div class="stat-value" id="total-stocks">0</div>
                <div class="stat-label">Stocks Analyzed</div>
            </div>
            <div class="stat-card pulse">
                <div class="stat-icon">📈</div>
                <div class="stat-value" id="bullish-count">0</div>
                <div class="stat-label">Bullish Signals</div>
            </div>
            <div class="stat-card pulse">
                <div class="stat-icon">💰</div>
                <div class="stat-value" id="avg-score">0%</div>
                <div class="stat-label">Avg Score</div>
            </div>
            <div class="stat-card pulse">
                <div class="stat-icon">📰</div>
                <div class="stat-value" id="total-news">0</div>
                <div class="stat-label">News Articles</div>
            </div>
        </section>

        # Charts Section -->
        <section class="charts-section">
            <div class="chart-container">
                <h3>📊 Profitability Comparison</h3>
                <canvas id="profitabilityChart"></canvas>
            </div>
            <div class="chart-container">
                <h3>🎯 Sentiment Analysis</h3>
                <canvas id="sentimentChart"></canvas>
            </div>
        </section>

        <!-- ML Predictions Section -->
        <section class="ml-predictions-overview">
            <h2>🤖 AI/ML Price Predictions</h2>
            <div id="ml-predictions-container" class="ml-predictions-grid">
                <!-- Populated by JavaScript -->
            </div>
        </section>

        <!-- Filters and Search -->
        <section class="controls">
            <div class="search-box">
                <input type="text" id="search-input" placeholder="🔍 Search stocks...">
            </div>
            <div class="filter-group">
                <select id="sort-select">
                    <option value="profitability-desc">Highest Profitability</option>
                    <option value="profitability-asc">Lowest Profitability</option>
                    <option value="name-asc">Name (A-Z)</option>
                    <option value="price-desc">Highest Price</option>
                </select>
                <select id="filter-sentiment">
                    <option value="all">All Sentiments</option>
                    <option value="positive">Positive Only</option>
                    <option value="neutral">Neutral Only</option>
                    <option value="negative">Negative Only</option>
                </select>
            </div>
        </section>

        <!-- Stock Cards -->
        <section id="stocks-container" class="stocks-grid">
            <!-- Populated by JavaScript -->
        </section>
    </main>

    <footer>
        <div class="container">
            <p>Generated on <span id="generation-date"></span></p>
            <p class="disclaimer">⚠️ For informational purposes only. Not financial advice.</p>
        </div>
    </footer>

    <script src="script.js"></script>
</body>
</html>"""
        
        with open(os.path.join(self.output_dir, 'index.html'), 'w') as f:
            f.write(html)
    
    def _generate_css(self):
        """Generate style.css with modern, appealing design"""
        css = """* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

:root {
    --primary: #667eea;
    --secondary: #764ba2;
    --success: #10b981;
    --danger: #ef4444;
    --warning: #f59e0b;
    --info: #3b82f6;
    --dark: #1f2937;
    --light: #f9fafb;
    --gray: #6b7280;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
    line-height: 1.6;
    color: var(--dark);
    background: #f8fafc;
    min-height: 100vh;
    position: relative;
}

.header-bg {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: 300px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    z-index: -1;
}

.container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 0 20px;
}

header {
    padding: 3rem 0 2rem;
    text-align: center;
    color: white;
}

h1 {
    font-size: 3rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
}

.subtitle {
    font-size: 1.2rem;
    opacity: 0.95;
    font-weight: 300;
}

main {
    margin-top: 2rem;
}

/* Overview Stats */
.overview-stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1.5rem;
    margin-bottom: 2rem;
}

.stat-card {
    background: white;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
}

.stat-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 40px rgba(0,0,0,0.15);
}

.stat-card.pulse {
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.02); }
}

.stat-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
}

.stat-value {
    font-size: 2.5rem;
    font-weight: 700;
    color: var(--primary);
    margin-bottom: 0.5rem;
}

.stat-label {
    color: var(--gray);
    font-size: 0.95rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
}

/* Charts Section */
.charts-section {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
    gap: 2rem;
    margin-bottom: 2rem;
}

.chart-container {
    background: white;
    border-radius: 16px;
    padding: 2rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
}

.chart-container h3 {
    margin-bottom: 1.5rem;
    color: var(--dark);
    font-size: 1.3rem;
}

/* ML Predictions Section */
.ml-predictions-overview {
    margin-bottom: 2rem;
}

.ml-predictions-overview h2 {
    color: var(--dark);
    font-size: 1.8rem;
    margin-bottom: 1.5rem;
    text-align: center;
}

.ml-predictions-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
    gap: 1.5rem;
}

.ml-prediction-card {
    background: white;
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    border-left: 4px solid var(--primary);
}

.ml-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
}

.ml-stock-name {
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--primary);
}

.ml-trend-badge {
    padding: 0.4rem 0.9rem;
    border-radius: 20px;
    font-size: 0.9rem;
    font-weight: 700;
}

.ml-trend-bullish {
    background: #d1fae5;
    color: #065f46;
}

.ml-trend-bearish {
    background: #fee2e2;
    color: #991b1b;
}

.ml-predictions-list {
    display: grid;
    gap: 0.75rem;
}

.ml-prediction-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem;
    background: #f9fafb;
    border-radius: 10px;
}

.ml-period {
    font-weight: 600;
    color: var(--gray);
}

.ml-change {
    font-weight: 700;
    font-size: 1.1rem;
}

.ml-change.positive {
    color: var(--success);
}

.ml-change.negative {
    color: var(--danger);
}

.ml-confidence {
    font-size: 0.85rem;
    color: var(--gray);
    margin-top: 0.25rem;
}

/* Controls */
.controls {
    background: white;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 2rem;
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
}

.search-box {
    flex: 1;
    min-width: 250px;
}

#search-input {
    width: 100%;
    padding: 0.75rem 1rem;
    border: 2px solid #e5e7eb;
    border-radius: 10px;
    font-size: 1rem;
    transition: all 0.3s ease;
}

#search-input:focus {
    outline: none;
    border-color: var(--primary);
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

.filter-group {
    display: flex;
    gap: 1rem;
}

.filter-group select {
    padding: 0.75rem 1rem;
    border: 2px solid #e5e7eb;
    border-radius: 10px;
    font-size: 1rem;
    background: white;
    cursor: pointer;
    transition: all 0.3s ease;
}

.filter-group select:focus {
    outline: none;
    border-color: var(--primary);
}

/* Stock Cards */
.stocks-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
    gap: 2rem;
    margin-bottom: 3rem;
}

.stock-card {
    background: white;
    border-radius: 16px;
    padding: 2rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.stock-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, var(--primary), var(--secondary));
}

.stock-card:hover {
    transform: translateY(-8px);
    box-shadow: 0 20px 40px rgba(0,0,0,0.15);
}

.stock-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 2px solid #f3f4f6;
}

.stock-symbol {
    font-size: 1.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.score-badge {
    padding: 0.6rem 1.2rem;
    border-radius: 25px;
    font-weight: 700;
    font-size: 1.1rem;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
}

.score-high {
    background: linear-gradient(135deg, #10b981, #059669);
    color: white;
}

.score-medium {
    background: linear-gradient(135deg, #f59e0b, #d97706);
    color: white;
}

.score-low {
    background: linear-gradient(135deg, #ef4444, #dc2626);
    color: white;
}

.stock-metrics {
    margin: 1.5rem 0;
}

.metric-row {
    display: flex;
    justify-content: space-between;
    padding: 0.75rem 0;
    border-bottom: 1px solid #f3f4f6;
}

.metric-row:last-child {
    border-bottom: none;
}

.metric-label {
    color: var(--gray);
    font-size: 0.95rem;
    font-weight: 500;
}

.metric-value {
    font-weight: 700;
    color: var(--dark);
    font-size: 1.05rem;
}

.metric-value.positive {
    color: var(--success);
}

.metric-value.negative {
    color: var(--danger);
}

.sentiment-badge {
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: 15px;
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

.patterns-section {
    margin-top: 1.5rem;
    padding-top: 1.5rem;
    border-top: 2px solid #f3f4f6;
}

.patterns-title {
    font-weight: 700;
    color: var(--primary);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.pattern-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
}

.pattern-tag {
    padding: 0.4rem 0.8rem;
    border-radius: 12px;
    font-size: 0.85rem;
    font-weight: 600;
    transition: all 0.2s ease;
}

.pattern-tag:hover {
    transform: scale(1.05);
}

.pattern-bullish {
    background: #d1fae5;
    color: #065f46;
}

.pattern-bearish {
    background: #fee2e2;
    color: #991b1b;
}

.pattern-neutral {
    background: #e0e7ff;
    color: #3730a3;
}

footer {
    background: rgba(255, 255, 255, 0.98);
    padding: 2rem 0;
    margin-top: 4rem;
    text-align: center;
    box-shadow: 0 -5px 20px rgba(0,0,0,0.05);
}

footer p {
    color: var(--gray);
    margin: 0.5rem 0;
}

.disclaimer {
    font-size: 0.9rem;
    color: var(--warning);
    font-weight: 600;
}

@media (max-width: 768px) {
    h1 {
        font-size: 2rem;
    }
    
    .charts-section {
        grid-template-columns: 1fr;
    }
    
    .stocks-grid {
        grid-template-columns: 1fr;
    }
    
    .controls {
        flex-direction: column;
    }
    
    .filter-group {
        flex-direction: column;
    }
}"""
        
        with open(os.path.join(self.output_dir, 'style.css'), 'w') as f:
            f.write(css)
    
    def _generate_js(self):
        """Generate script.js with charts and interactivity"""
        js = """// Stock Analysis Dashboard
let stocksData = [];
let profitabilityChart = null;
let sentimentChart = null;

// Load data on page load
document.addEventListener('DOMContentLoaded', async () => {
    try {
        const response = await fetch('data.json');
        stocksData = await response.json();
        
        updateOverview();
        createCharts();
        displayMLPredictions();
        displayStocks(stocksData);
        setupEventListeners();
        
        document.getElementById('generation-date').textContent = new Date().toLocaleString();
        
    } catch (error) {
        console.error('Error loading data:', error);
        document.getElementById('stocks-container').innerHTML = 
            '<p style="color: white; text-align: center; grid-column: 1/-1;">Error loading data. Please check console.</p>';
    }
});

function updateOverview() {
    document.getElementById('total-stocks').textContent = stocksData.length;
    
    const bullishCount = stocksData.filter(s => 
        s.candlestick_patterns && 
        s.candlestick_patterns.filter(p => p.type === 'Bullish').length > 
        s.candlestick_patterns.filter(p => p.type === 'Bearish').length
    ).length;
    document.getElementById('bullish-count').textContent = bullishCount;
    
    const avgScore = stocksData.reduce((sum, s) => sum + (s.profitability_score || 0), 0) / stocksData.length;
    document.getElementById('avg-score').textContent = avgScore.toFixed(1) + '%';
    
    const totalNews = stocksData.reduce((sum, s) => sum + (s.news?.total_articles || 0), 0);
    document.getElementById('total-news').textContent = totalNews;
}

function createCharts() {
    // Profitability Chart
    const profCtx = document.getElementById('profitabilityChart');
    profitabilityChart = new Chart(profCtx, {
        type: 'bar',
        data: {
            labels: stocksData.map(s => s.symbol),
            datasets: [{
                label: 'Profitability Score',
                data: stocksData.map(s => s.profitability_score || 0),
                backgroundColor: stocksData.map(s => {
                    const score = s.profitability_score || 0;
                    if (score >= 70) return 'rgba(16, 185, 129, 0.8)';
                    if (score >= 40) return 'rgba(245, 158, 11, 0.8)';
                    return 'rgba(239, 68, 68, 0.8)';
                }),
                borderRadius: 8,
                borderSkipped: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    padding: 12,
                    borderRadius: 8
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: { color: 'rgba(0, 0, 0, 0.05)' }
                },
                x: {
                    grid: { display: false }
                }
            }
        }
    });
    
    // Sentiment Chart
    const sentCtx = document.getElementById('sentimentChart');
    const sentimentCounts = {
        'POSITIVE': 0,
        'NEUTRAL': 0,
        'NEGATIVE': 0
    };
    
    stocksData.forEach(s => {
        if (s.news && s.news.sentiment_label) {
            sentimentCounts[s.news.sentiment_label]++;
        }
    });
    
    sentimentChart = new Chart(sentCtx, {
        type: 'doughnut',
        data: {
            labels: ['Positive', 'Neutral', 'Negative'],
            datasets: [{
                data: [sentimentCounts.POSITIVE, sentimentCounts.NEUTRAL, sentimentCounts.NEGATIVE],
                backgroundColor: [
                    'rgba(16, 185, 129, 0.8)',
                    'rgba(156, 163, 175, 0.8)',
                    'rgba(239, 68, 68, 0.8)'
                ],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 20,
                        font: { size: 14 }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    padding: 12,
                    borderRadius: 8
                }
            }
        }
    });
}

function displayMLPredictions() {
    const container = document.getElementById('ml-predictions-container');
    container.innerHTML = '';
    
    // Filter stocks that have ML predictions
    const stocksWithML = stocksData.filter(s => s.ml_predictions);
    
    if (stocksWithML.length === 0) {
        container.innerHTML = '<p style="grid-column: 1/-1; text-align: center; color: #6b7280;">No ML predictions available yet. Run analysis with ML model to see forecasts.</p>';
        return;
    }
    
    stocksWithML.forEach(stock => {
        const mlData = stock.ml_predictions;
        const card = document.createElement('div');
        card.className = 'ml-prediction-card';
        
        const trendClass = mlData.overall_trend === 'BULLISH' ? 'ml-trend-bullish' : 'ml-trend-bearish';
        
        card.innerHTML = `
            <div class="ml-card-header">
                <div class="ml-stock-name">${stock.symbol}</div>
                <div class="ml-trend-badge ${trendClass}">${mlData.overall_trend || 'N/A'}</div>
            </div>
            <div class="ml-predictions-list">
                ${mlData.predictions && Object.keys(mlData.predictions).length > 0 ? 
                    Object.entries(mlData.predictions).map(([period, pred]) => {
                        const change = pred.change || 0;
                        const changeClass = change >= 0 ? 'positive' : 'negative';
                        const sign = change >= 0 ? '+' : '';
                        return `
                            <div class="ml-prediction-item">
                                <span class="ml-period">${period.replace('_', ' ')}</span>
                                <div style="text-align: right;">
                                    <div class="ml-change ${changeClass}">${sign}${change.toFixed(2)}%</div>
                                    <div class="ml-confidence">${(pred.confidence * 100).toFixed(0)}% confidence</div>
                                </div>
                            </div>
                        `;
                    }).join('')
                : '<p style="color: #6b7280; text-align: center;">No prediction data available</p>'}
            </div>
        `;
        
        container.appendChild(card);
    });
}

function displayStocks(stocks) {
    const container = document.getElementById('stocks-container');
    container.innerHTML = '';
    
    if (stocks.length === 0) {
        container.innerHTML = '<p style="grid-column: 1/-1; text-align: center; color: white;">No stocks match your filters.</p>';
        return;
    }
    
    stocks.forEach(stock => {
        const card = createStockCard(stock);
        container.appendChild(card);
    });
}

function createStockCard(stock) {
    const card = document.createElement('div');
    card.className = 'stock-card';
    
    const score = stock.profitability_score || 0;
    const scoreClass = score >= 70 ? 'score-high' : score >= 40 ? 'score-medium' : 'score-low';
    
    const priceData = stock.price_data || {};
    const news = stock.news || {};
    const fundamentals = stock.fundamentals || {};
    const patterns = stock.candlestick_patterns || [];
    
    const bullishPatterns = patterns.filter(p => p.type === 'Bullish');
    const bearishPatterns = patterns.filter(p => p.type === 'Bearish');
    const neutralPatterns = patterns.filter(p => p.type === 'Neutral');
    
    card.innerHTML = `
        <div class="stock-header">
            <div class="stock-symbol">${stock.symbol}</div>
            <div class="score-badge ${scoreClass}">${score.toFixed(1)}%</div>
        </div>
        
        <div class="stock-metrics">
            ${priceData.latest_price ? `
            <div class="metric-row">
                <span class="metric-label">Latest Price</span>
                <span class="metric-value">Rs. ${priceData.latest_price.toFixed(2)}</span>
            </div>
            ` : ''}
            
            ${priceData.total_days ? `
            <div class="metric-row">
                <span class="metric-label">Historical Data</span>
                <span class="metric-value">${priceData.total_days} days</span>
            </div>
            ` : ''}
            
            ${news.sentiment_label ? `
            <div class="metric-row">
                <span class="metric-label">News Sentiment</span>
                <span class="sentiment-badge sentiment-${news.sentiment_label.toLowerCase()}">${news.sentiment_label}</span>
            </div>
            ` : ''}
            
            ${news.total_articles ? `
            <div class="metric-row">
                <span class="metric-label">News Articles</span>
                <span class="metric-value">${news.total_articles}</span>
            </div>
            ` : ''}
            
            ${fundamentals['P/E Ratio'] ? `
            <div class="metric-row">
                <span class="metric-label">P/E Ratio</span>
                <span class="metric-value">${fundamentals['P/E Ratio']}</span>
            </div>
            ` : ''}
            
            ${fundamentals['P/B Ratio'] ? `
            <div class="metric-row">
                <span class="metric-label">P/B Ratio</span>
                <span class="metric-value">${fundamentals['P/B Ratio']}</span>
            </div>
            ` : ''}
        </div>
        
        ${patterns.length > 0 ? `
        <div class="patterns-section">
            <div class="patterns-title">🕯️ Candlestick Patterns (${patterns.length})</div>
            <div class="pattern-tags">
                ${bullishPatterns.slice(0, 3).map(p => 
                    `<span class="pattern-tag pattern-bullish" title="${p.description}">${p.pattern}</span>`
                ).join('')}
                ${bearishPatterns.slice(0, 3).map(p => 
                    `<span class="pattern-tag pattern-bearish" title="${p.description}">${p.pattern}</span>`
                ).join('')}
                ${neutralPatterns.slice(0, 2).map(p => 
                    `<span class="pattern-tag pattern-neutral" title="${p.description}">${p.pattern}</span>`
                ).join('')}
                ${patterns.length > 8 ? `<span class="pattern-tag pattern-neutral">+${patterns.length - 8} more</span>` : ''}
            </div>
        </div>
        ` : ''}
        
        ${stock.ml_predictions && stock.ml_predictions.predictions ? `
        <div class="patterns-section">
            <div class="patterns-title">🤖 ML Price Forecast</div>
            <div class="metric-row">
                <span class="metric-label">Trend</span>
                <span class="metric-value">${stock.ml_predictions.overall_trend || 'N/A'}</span>
            </div>
            ${Object.entries(stock.ml_predictions.predictions).slice(0, 2).map(([period, pred]) => {
                const change = pred.change || 0;
                const sign = change >= 0 ? '+' : '';
                const changeClass = change >= 0 ? 'positive' : 'negative';
                return `
                <div class="metric-row">
                    <span class="metric-label">${period.replace('_', ' ')}</span>
                    <span class="metric-value ${changeClass}">${sign}${change.toFixed(2)}%</span>
                </div>
                `;
            }).join('')}
        </div>
        ` : ''}
    `;
    
    return card;
}

function setupEventListeners() {
    // Search
    document.getElementById('search-input').addEventListener('input', filterStocks);
    
    // Sort
    document.getElementById('sort-select').addEventListener('change', filterStocks);
    
    // Sentiment filter
    document.getElementById('filter-sentiment').addEventListener('change', filterStocks);
}

function filterStocks() {
    const searchTerm = document.getElementById('search-input').value.toLowerCase();
    const sortBy = document.getElementById('sort-select').value;
    const sentimentFilter = document.getElementById('filter-sentiment').value;
    
    let filtered = stocksData.filter(stock => {
        // Search filter
        const matchesSearch = stock.symbol.toLowerCase().includes(searchTerm);
        
        // Sentiment filter
        let matchesSentiment = true;
        if (sentimentFilter !== 'all' && stock.news) {
            matchesSentiment = stock.news.sentiment_label?.toLowerCase() === sentimentFilter;
        }
        
        return matchesSearch && matchesSentiment;
    });
    
    // Sort
    filtered.sort((a, b) => {
        switch(sortBy) {
            case 'profitability-desc':
                return (b.profitability_score || 0) - (a.profitability_score || 0);
            case 'profitability-asc':
                return (a.profitability_score || 0) - (b.profitability_score || 0);
            case 'name-asc':
                return a.symbol.localeCompare(b.symbol);
            case 'price-desc':
                return (b.price_data?.latest_price || 0) - (a.price_data?.latest_price || 0);
            default:
                return 0;
        }
    });
    
    displayStocks(filtered);
}"""
        
        with open(os.path.join(self.output_dir, 'script.js'), 'w') as f:
            f.write(js)


if __name__ == '__main__':
    import sys
    
    json_file = sys.argv[1] if len(sys.argv) > 1 else 'analysis_results.json'
    
    if not os.path.exists(json_file):
        print(f"❌ Error: {json_file} not found!")
        print(f"   Please run the analysis first to generate the JSON file.")
        sys.exit(1)
    
    generator = StockWebsiteGenerator(json_file)
    generator.load_data()
    generator.generate_website()
    
    print(f"\n{'='*60}")
    print("✅ Website Generation Complete!")
    print(f"{'='*60}")
    print(f"\nTo view the website:")
    print(f"  1. Open: stock_website/index.html")
    print(f"  2. Or run: python3 -m http.server 8000 --directory stock_website")
    print(f"     Then visit: http://localhost:8000")
