"""
Enhanced Stock Website Generator
Displays comprehensive trading insights including profitability probability, risk-reward, ML predictions
"""

import json
import os
from datetime import datetime


class StockWebsiteGenerator:
    def __init__(self, json_file='analysis_results.json', output_dir='stock_website'):
        self.json_file = json_file
        self.output_dir = output_dir
        self.data = []
    
    def load_data(self):
        """Load analysis results from JSON"""
        with open(self.json_file, 'r') as f:
            self.data = json.load(f)
        
        # Sort by profitability probability
        for stock in self.data:
            insights = stock.get('trading_insights', {})
            stock['profitability_score'] = insights.get('profitability_probability', 0)
        
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
        with open(os.path.join(self.output_dir, 'data.json'), 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def _generate_html(self):
        """Generate index.html with embedded data"""
        # Embed the data directly in HTML to avoid CORS issues
        import json
        data_json = json.dumps(self.data)
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NEPSE Stock Analysis Dashboard</title>
    <link rel="stylesheet" href="style.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
</head>
<body>
    <header>
        <div class="container">
            <h1>📈 NEPSE Stock Analysis Dashboard</h1>
            <p class="subtitle">AI-Powered Trading Insights & Predictions</p>
            <p class="update-time" id="last-updated"></p>
        </div>
    </header>

    <main class="container">
        <!-- Summary Cards -->
        <div class="summary-section">
            <div class="summary-card">
                <div class="summary-icon">📊</div>
                <div class="summary-value" id="total-stocks">0</div>
                <div class="summary-label">Stocks Analyzed</div>
            </div>
            <div class="summary-card">
                <div class="summary-icon">📈</div>
                <div class="summary-value" id="buy-count">0</div>
                <div class="summary-label">Buy Signals</div>
            </div>
            <div class="summary-card">
                <div class="summary-icon">📉</div>
                <div class="summary-value" id="sell-count">0</div>
                <div class="summary-label">Sell Signals</div>
            </div>
        </div>

        <!-- Filters and Search -->
        <div class="filters-section">
            <div class="search-box">
                <input type="text" id="search-input" placeholder="🔍 Search stocks by symbol..." />
            </div>
            <div class="filter-controls">
                <select id="sort-select">
                    <option value="profitability-desc">Sort: Profitability (High to Low)</option>
                    <option value="profitability-asc">Sort: Profitability (Low to High)</option>
                    <option value="profit-potential-desc">Sort: Profit Potential (High to Low)</option>
                    <option value="profit-potential-asc">Sort: Profit Potential (Low to High)</option>
                    <option value="symbol-asc">Sort: Symbol (A-Z)</option>
                    <option value="symbol-desc">Sort: Symbol (Z-A)</option>
                    <option value="price-desc">Sort: Price (High to Low)</option>
                    <option value="price-asc">Sort: Price (Low to High)</option>
                </select>
                <select id="filter-recommendation">
                    <option value="all">All Recommendations</option>
                    <option value="buy">Buy Only</option>
                    <option value="sell">Sell Only</option>
                    <option value="hold">Hold Only</option>
                </select>
                <select id="filter-profitability">
                    <option value="all">All Profitability</option>
                    <option value="high">High (>70%)</option>
                    <option value="medium">Medium (40-70%)</option>
                    <option value="low">Low (<40%)</option>
                </select>
            </div>
        </div>

        <!-- Stock Cards Grid -->
        <div id="stock-cards-container"></div>
        <div id="no-results" style="display: none; text-align: center; padding: 3rem; color: #6b7280;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📭</div>
            <div style="font-size: 1.2rem;">No stocks match your filters</div>
        </div>
        
        <!-- Stock Detail Modal -->
        <div id="stock-modal" class="modal">
            <div class="modal-content">
                <span class="modal-close">&times;</span>
                <div id="modal-body"></div>
            </div>
        </div>
    </main>

    <footer>
        <div class="container">
            <p>⚠️ For informational purposes only. Not financial advice. Always do your own research.</p>
            <p>Data updated: <span id="footer-timestamp"></span></p>
        </div>
    </footer>

    <!-- Embedded data to avoid CORS issues -->
    <script>
        const STOCK_DATA = {data_json};
    </script>
    <script src="script.js"></script>
</body>
</html>
"""
        with open(os.path.join(self.output_dir, 'index.html'), 'w') as f:
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
    color: #2c3e50;
    background: #f8fafc;
}

.container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 0 20px;
}

/* Header */
header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 3rem 0;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.subtitle {
    font-size: 1.2rem;
    opacity: 0.9;
    margin-bottom: 0.5rem;
}

.update-time {
    font-size: 0.9rem;
    opacity: 0.8;
}

/* Main Content */
main {
    padding: 3rem 0;
}

#stock-cards-container {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
    gap: 1.5rem;
    margin-top: 2rem;
}

/* Summary Section */
.summary-section {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1.5rem;
    margin-bottom: 2rem;
}

.summary-card {
    background: white;
    padding: 2rem;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    text-align: center;
    transition: transform 0.2s;
}

.summary-card:hover {
    transform: translateY(-4px);
}

.summary-icon {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
}

.summary-value {
    font-size: 2.5rem;
    font-weight: 700;
    color: #667eea;
    margin-bottom: 0.3rem;
}

.summary-label {
    font-size: 0.9rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Stock Card - Compact Version */
.stock-card {
    background: white;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
    cursor: pointer;
}

.stock-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 16px rgba(0,0,0,0.15);
}

.stock-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1.5rem;
}

.stock-title {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.stock-symbol {
    font-size: 1.5rem;
    font-weight: 700;
}

.stock-price {
    font-size: 1.3rem;
    font-weight: 700;
}

.recommendation-badge {
    display: inline-block;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    background: rgba(255,255,255,0.2);
    font-size: 0.9rem;
    font-weight: 600;
}

.recommendation-badge.buy {
    background: #10b981;
}

.recommendation-badge.sell {
    background: #ef4444;
}

.recommendation-badge.hold {
    background: #f59e0b;
}

/* Filters Section */
.filters-section {
    background: white;
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    margin-bottom: 2rem;
    display: grid;
    gap: 1rem;
}

.search-box input {
    width: 100%;
    padding: 0.75rem 1rem;
    border: 2px solid #e5e7eb;
    border-radius: 8px;
    font-size: 1rem;
    transition: border-color 0.2s;
}

.search-box input:focus {
    outline: none;
    border-color: #667eea;
}

.filter-controls {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
}

.filter-controls select {
    padding: 0.75rem 1rem;
    border: 2px solid #e5e7eb;
    border-radius: 8px;
    font-size: 0.95rem;
    background: white;
    cursor: pointer;
    transition: border-color 0.2s;
}

.filter-controls select:focus {
    outline: none;
    border-color: #667eea;
}

/* Probability Section - Compact */
.probability-section {
    padding: 1.5rem;
    background: linear-gradient(to right, #f3f4f6, #e5e7eb);
}

.probability-value {
    font-size: 2rem;
    font-weight: 700;
    color: #111827;
    margin-bottom: 0.5rem;
}

.probability-bar {
    height: 8px;
    background: #d1d5db;
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 0.5rem;
}

.probability-fill {
    height: 100%;
    background: linear-gradient(to right, #ef4444, #f59e0b, #10b981);
    border-radius: 6px;
    transition: width 1s ease;
}

.confidence-level {
    font-size: 0.85rem;
    color: #4b5563;
}

/* Quick Stats - Compact */
.quick-stats {
    padding: 1rem;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
    background: #f9fafb;
}

.stat-item {
    text-align: center;
    padding: 0.75rem;
    background: white;
    border-radius: 8px;
}

.stat-label {
    font-size: 0.75rem;
    color: #6b7280;
    margin-bottom: 0.25rem;
}

.stat-value {
    font-size: 1.1rem;
    font-weight: 700;
    color: #111827;
}

.stat-value.positive {
    color: #10b981;
}

.stat-value.negative {
    color: #ef4444;
}

/* Modal */
.modal {
    display: none;
    position: fixed;
    z-index: 1000;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    overflow: auto;
    background-color: rgba(0,0,0,0.5);
    backdrop-filter: blur(4px);
}

.modal.active {
    display: block;
}

.modal-content {
    background-color: #fefefe;
    margin: 2% auto;
    padding: 0;
    border-radius: 16px;
    width: 90%;
    max-width: 1200px;
    max-height: 90vh;
    overflow-y: auto;
    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
}

.modal-close {
    color: #aaa;
    position: absolute;
    right: 1rem;
    top: 1rem;
    font-size: 28px;
    font-weight: bold;
    cursor: pointer;
    z-index: 10;
    background: rgba(255,255,255,0.9);
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
}

#modal-body {
    position: relative;
}

.modal-close:hover,
.modal-close:focus {
    color: #000;
}

/* Risk Reward Section */
.risk-reward {
    padding: 2rem;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1.5rem;
}

.metric-box {
    padding: 1.5rem;
    background: #f9fafb;
    border-radius: 12px;
    border-left: 4px solid #667eea;
}

.metric-label {
    font-size: 0.85rem;
    color: #6b7280;
    margin-bottom: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #111827;
}

.metric-value.positive {
    color: #10b981;
}

.metric-value.negative {
    color: #ef4444;
}

/* Entry/Exit Section */
.entry-exit {
    padding: 2rem;
    background: #f9fafb;
}

.section-title {
    font-size: 1.3rem;
    font-weight: 600;
    color: #111827;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.entry-exit-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1rem;
}

.entry-exit-item {
    padding: 1rem;
    background: white;
    border-radius: 8px;
    text-align: center;
}

.entry-exit-label {
    font-size: 0.8rem;
    color: #6b7280;
    margin-bottom: 0.5rem;
}

.entry-exit-value {
    font-size: 1.3rem;
    font-weight: 700;
    color: #667eea;
}

/* Scores Section */
.scores-section {
    padding: 2rem;
}

.scores-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1.5rem;
}

.score-item {
    background: #f9fafb;
    padding: 1.5rem;
    border-radius: 12px;
}

.score-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.8rem;
}

.score-name {
    font-weight: 600;
    color: #374151;
}

.score-number {
    font-size: 1.4rem;
    font-weight: 700;
    color: #667eea;
}

.score-bar-container {
    height: 8px;
    background: #e5e7eb;
    border-radius: 4px;
    overflow: hidden;
}

.score-bar-fill {
    height: 100%;
    background: linear-gradient(to right, #ef4444, #f59e0b, #10b981);
    border-radius: 4px;
    transition: width 1s ease;
}

/* ML Predictions */
.ml-predictions {
    padding: 2rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}

.prediction-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-top: 1.5rem;
}

.prediction-card {
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(10px);
    padding: 1.5rem;
    border-radius: 12px;
    text-align: center;
}

.prediction-horizon {
    font-size: 0.9rem;
    opacity: 0.9;
    margin-bottom: 0.5rem;
}

.prediction-price {
    font-size: 1.8rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}

.prediction-change {
    font-size: 1.1rem;
    font-weight: 600;
}

.prediction-change.positive {
    color: #86efac;
}

.prediction-change.negative {
    color: #fca5a5;
}

/* Insights & Warnings */
.insights-warnings {
    padding: 2rem;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 2rem;
}

.insights-box, .warnings-box {
    padding: 1.5rem;
    border-radius: 12px;
}

.insights-box {
    background: #ecfdf5;
    border-left: 4px solid #10b981;
}

.warnings-box {
    background: #fef2f2;
    border-left: 4px solid #ef4444;
}

.box-title {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.insight-item, .warning-item {
    padding: 0.5rem 0;
    border-bottom: 1px solid rgba(0,0,0,0.05);
}

.insight-item:last-child, .warning-item:last-child {
    border-bottom: none;
}

/* Candlestick Patterns */
.candlestick-patterns {
    padding: 2rem;
    background: #f9fafb;
}

/* Detailed Analysis Sections */
.detailed-analysis {
    padding: 2rem;
    background: #f8fafc;
    border-top: 1px solid #e5e7eb;
}

.analysis-details-section {
    margin-bottom: 2rem;
    padding: 1.5rem;
    background: white;
    border-radius: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.analysis-details-section:last-child {
    margin-bottom: 0;
}

.estimated-data-warning {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.875rem 1rem;
    margin-bottom: 1.25rem;
    background: #fef3c7;
    border: 1px solid #f59e0b;
    border-radius: 8px;
    color: #92400e;
}

.estimated-data-warning .warning-icon {
    font-size: 1.25rem;
    flex-shrink: 0;
}

.estimated-data-warning .warning-text {
    font-size: 0.875rem;
    font-weight: 500;
    line-height: 1.4;
}

.analysis-details-title {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 1rem;
    color: #1f2937;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.analysis-details-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
}

.detail-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem;
    background: #f9fafb;
    border-radius: 6px;
    border-left: 3px solid #667eea;
}

.detail-label {
    font-size: 0.85rem;
    color: #6b7280;
    font-weight: 500;
}

.detail-value {
    font-size: 0.95rem;
    font-weight: 600;
    color: #1f2937;
}

.detail-value.positive {
    color: #10b981;
}

.detail-value.negative {
    color: #ef4444;
}

.detail-value.trend-bullish {
    color: #10b981;
    background: #ecfdf5;
    padding: 0.25rem 0.75rem;
    border-radius: 12px;
}

.detail-value.trend-bearish {
    color: #ef4444;
    background: #fef2f2;
    padding: 0.25rem 0.75rem;
    border-radius: 12px;
}

.detail-value.trend-neutral {
    color: #f59e0b;
    background: #fef3c7;
    padding: 0.25rem 0.75rem;
    border-radius: 12px;
}

.detail-value.sentiment-positive {
    color: #10b981;
    background: #ecfdf5;
    padding: 0.25rem 0.75rem;
    border-radius: 12px;
    font-weight: 700;
}

.detail-value.sentiment-negative {
    color: #ef4444;
    background: #fef2f2;
    padding: 0.25rem 0.75rem;
    border-radius: 12px;
    font-weight: 700;
}

.detail-value.sentiment-neutral {
    color: #6b7280;
    background: #f3f4f6;
    padding: 0.25rem 0.75rem;
    border-radius: 12px;
    font-weight: 700;
}

/* Last Updated Section */
.last-updated-section {
    padding: 1rem 2rem;
    background: #f3f4f6;
    border-top: 1px solid #e5e7eb;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.85rem;
    color: #6b7280;
}

.last-updated-icon {
    font-size: 1.2rem;
}

.last-updated-text {
    font-weight: 500;
}

.patterns-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1rem;
    margin-top: 1rem;
}

.pattern-card {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #667eea;
}

.pattern-name {
    font-weight: 600;
    color: #111827;
    margin-bottom: 0.3rem;
}

.pattern-type {
    font-size: 0.85rem;
    color: #6b7280;
}

.pattern-type.bullish {
    color: #10b981;
}

.pattern-type.bearish {
    color: #ef4444;
}

/* Footer */
footer {
    background: #1f2937;
    color: white;
    padding: 2rem 0;
    text-align: center;
    margin-top: 3rem;
}

footer p {
    margin: 0.5rem 0;
    opacity: 0.8;
}

/* Responsive */
@media (max-width: 768px) {
    header h1 {
        font-size: 1.8rem;
    }
    
    .stock-symbol, .stock-price {
        font-size: 1.5rem;
    }
    
    .probability-value {
        font-size: 2rem;
    }
}
"""
        with open(os.path.join(self.output_dir, 'style.css'), 'w') as f:
            f.write(css)
    
    def _generate_js(self):
        """Generate script.js"""
        js = """// Load and display stock analysis data

// Use embedded data (no fetch needed - works with file:// protocol)
let allStocks = [];
let filteredStocks = [];

function loadData() {
    try {
        allStocks = STOCK_DATA;
        filteredStocks = [...allStocks];
        displayStocks(filteredStocks);
        updateTimestamps(allStocks);
        updateSummary(allStocks);
        initializeFilters();
    } catch (error) {
        console.error('Error loading data:', error);
    }
}

function initializeFilters() {
    const searchInput = document.getElementById('search-input');
    const sortSelect = document.getElementById('sort-select');
    const filterRecommendation = document.getElementById('filter-recommendation');
    const filterProfitability = document.getElementById('filter-profitability');
    
    searchInput.addEventListener('input', applyFilters);
    sortSelect.addEventListener('change', applyFilters);
    filterRecommendation.addEventListener('change', applyFilters);
    filterProfitability.addEventListener('change', applyFilters);
}

function applyFilters() {
    const searchTerm = document.getElementById('search-input').value.toLowerCase();
    const sortBy = document.getElementById('sort-select').value;
    const recFilter = document.getElementById('filter-recommendation').value;
    const profFilter = document.getElementById('filter-profitability').value;
    
    // Start with all stocks
    filteredStocks = [...allStocks];
    
    // Apply search filter
    if (searchTerm) {
        filteredStocks = filteredStocks.filter(stock => 
            stock.symbol.toLowerCase().includes(searchTerm)
        );
    }
    
    // Apply recommendation filter
    if (recFilter !== 'all') {
        filteredStocks = filteredStocks.filter(stock => {
            const action = stock.trading_insights?.recommendation?.action || '';
            return action.toLowerCase().includes(recFilter);
        });
    }
    
    // Apply profitability filter
    if (profFilter !== 'all') {
        filteredStocks = filteredStocks.filter(stock => {
            const prob = stock.trading_insights?.profitability_probability || 0;
            if (profFilter === 'high') return prob > 70;
            if (profFilter === 'medium') return prob >= 40 && prob <= 70;
            if (profFilter === 'low') return prob < 40;
            return true;
        });
    }
    
    // Apply sorting
    filteredStocks.sort((a, b) => {
        const aInsights = a.trading_insights || {};
        const bInsights = b.trading_insights || {};
        
        switch(sortBy) {
            case 'profitability-desc':
                return (bInsights.profitability_probability || 0) - (aInsights.profitability_probability || 0);
            case 'profitability-asc':
                return (aInsights.profitability_probability || 0) - (bInsights.profitability_probability || 0);
            case 'profit-potential-desc':
                const bProfit = bInsights.risk_reward_ratio?.potential_profit_percent || 0;
                const aProfit = aInsights.risk_reward_ratio?.potential_profit_percent || 0;
                return bProfit - aProfit;
            case 'profit-potential-asc':
                const aProfitAsc = aInsights.risk_reward_ratio?.potential_profit_percent || 0;
                const bProfitAsc = bInsights.risk_reward_ratio?.potential_profit_percent || 0;
                return aProfitAsc - bProfitAsc;
            case 'symbol-asc':
                return a.symbol.localeCompare(b.symbol);
            case 'symbol-desc':
                return b.symbol.localeCompare(a.symbol);
            case 'price-desc':
                const bPrice = bInsights.current_price || b.price_data?.latest_price || 0;
                const aPrice = aInsights.current_price || a.price_data?.latest_price || 0;
                return bPrice - aPrice;
            case 'price-asc':
                const aPriceAsc = aInsights.current_price || a.price_data?.latest_price || 0;
                const bPriceAsc = bInsights.current_price || b.price_data?.latest_price || 0;
                return aPriceAsc - bPriceAsc;
            default:
                return 0;
        }
    });
    
    displayStocks(filteredStocks);
    
    // Show/hide no results message
    const noResults = document.getElementById('no-results');
    const container = document.getElementById('stock-cards-container');
    if (filteredStocks.length === 0) {
        container.style.display = 'none';
        noResults.style.display = 'block';
    } else {
        container.style.display = 'grid';
        noResults.style.display = 'none';
    }
}

function updateSummary(stocks) {
    document.getElementById('total-stocks').textContent = stocks.length;
    
    const buys = stocks.filter(s => s.trading_insights?.recommendation?.action?.includes('BUY')).length;
    const sells = stocks.filter(s => s.trading_insights?.recommendation?.action?.includes('SELL')).length;
    
    document.getElementById('buy-count').textContent = buys;
    document.getElementById('sell-count').textContent = sells;
}

function displayStocks(stocks) {
    const container = document.getElementById('stock-cards-container');
    container.innerHTML = '';
    
    stocks.forEach(stock => {
        if (stock.error) {
            container.innerHTML += createErrorCard(stock);
        } else {
            container.innerHTML += createCompactCard(stock);
        }
    });
    
    // Add click handlers for modals
    document.querySelectorAll('.stock-card').forEach((card, index) => {
        card.addEventListener('click', () => showStockDetail(stocks[index]));
    });
}

function createCompactCard(stock) {
    const insights = stock.trading_insights || {};
    const recommendation = insights.recommendation || {};
    const action = recommendation.action || 'HOLD';
    const probability = insights.profitability_probability || 0;
    const confidence = insights.confidence_level || 'Unknown';
    const currentPrice = insights.current_price || stock.price_data?.latest_price || 0;
    const riskReward = insights.risk_reward_ratio || {};
    const rrRatio = riskReward.ratio || 0;
    // Use the correct field name from risk_reward_ratio
    const potentialProfit = riskReward.potential_profit_percent || insights.potential_profit_pct || 0;
    const scores = stock.scores || {};
    
    return `
        <div class="stock-card" data-symbol="${stock.symbol}">
            <div class="stock-header">
                <div class="stock-title">
                    <div class="stock-symbol">${stock.symbol}</div>
                    <div class="stock-price">NPR ${currentPrice.toFixed(2)}</div>
                </div>
            </div>
            
            <div class="probability-section">
                <div class="probability-value">${probability.toFixed(1)}%</div>
                <div class="probability-bar">
                    <div class="probability-fill" style="width: ${probability}%"></div>
                </div>
                <div class="confidence-level">
                    <span class="recommendation-badge ${action.toLowerCase()}">${action}</span>
                    • ${confidence}
                </div>
            </div>
            
            <div class="quick-stats">
                <div class="stat-item">
                    <div class="stat-label">R:R Ratio</div>
                    <div class="stat-value">${rrRatio.toFixed(2)}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Potential</div>
                    <div class="stat-value positive">+${potentialProfit.toFixed(1)}%</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Technical</div>
                    <div class="stat-value">${(scores.technical || 0).toFixed(0)}/100</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Sentiment</div>
                    <div class="stat-value">${(scores.sentiment || 0).toFixed(0)}/100</div>
                </div>
            </div>
        </div>
    `;
}

function showStockDetail(stock) {
    const modal = document.getElementById('stock-modal');
    const modalBody = document.getElementById('modal-body');
    
    modalBody.innerHTML = createDetailedView(stock);
    modal.classList.add('active');
}

// Close modal
document.addEventListener('DOMContentLoaded', () => {
    const modal = document.getElementById('stock-modal');
    const closeBtn = document.querySelector('.modal-close');
    
    closeBtn.onclick = () => modal.classList.remove('active');
    window.onclick = (event) => {
        if (event.target === modal) {
            modal.classList.remove('active');
        }
    };
});

function createDetailedView(stock) {
    const insights = stock.trading_insights || {};
    const technicalAnalysis = stock.technical_analysis || {};
    const scores = stock.scores || {};
    const mlPredictions = stock.ml_predictions;
    
    // Recommendation
    const recommendation = insights.recommendation || {};
    const action = recommendation.action || 'HOLD';
    const probability = insights.profitability_probability || 0;
    const confidence = insights.confidence_level || 'Unknown';
    
    // Risk Reward
    const riskReward = insights.risk_reward_ratio || {};
    const rrRatio = riskReward.ratio || 0;
    // Use correct field names from risk_reward_ratio object
    const potentialProfit = riskReward.potential_profit_percent || insights.potential_profit_pct || 0;
    const potentialLoss = riskReward.potential_loss_percent || insights.potential_loss_pct || 0;
    
    // Entry/Exit
    const entryPoints = insights.entry_points || {};
    const exitPoints = insights.exit_points || {};
    const stopLoss = insights.stop_loss || 0;
    const takeProfit = insights.take_profit || 0;
    
    // Current price
    const currentPrice = insights.current_price || stock.price_data?.latest_price || 0;
    
    // Candlestick patterns
    const patterns = stock.candlestick_patterns || [];
    
    // Key insights and warnings
    const keyInsights = stock.key_insights || [];
    const warnings = stock.warnings || [];
    
    return `
        <div class="stock-card">
            <!-- Header -->
            <div class="stock-header">
                <div class="stock-title">
                    <div class="stock-symbol">${stock.symbol}</div>
                    <div class="stock-price">NPR ${currentPrice.toFixed(2)}</div>
                </div>
                <div class="recommendation-badge ${action.toLowerCase()}">
                    ${action} • ${recommendation.confidence || 'Medium'}
                </div>
            </div>
            
            <!-- Profitability Probability -->
            <div class="probability-section">
                <div class="probability-title">Profitability Probability</div>
                <div class="probability-value">${probability.toFixed(1)}%</div>
                <div class="probability-bar">
                    <div class="probability-fill" style="width: ${probability}%"></div>
                </div>
                <div class="confidence-level">Confidence: ${confidence}</div>
            </div>
            
            <!-- Risk Reward -->
            <div class="risk-reward">
                <div class="metric-box">
                    <div class="metric-label">Risk-Reward Ratio</div>
                    <div class="metric-value">${rrRatio.toFixed(2)}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Potential Profit</div>
                    <div class="metric-value positive">+${potentialProfit.toFixed(2)}%</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Potential Loss</div>
                    <div class="metric-value negative">-${potentialLoss.toFixed(2)}%</div>
                </div>
            </div>
            
            <!-- Entry/Exit Points -->
            <div class="entry-exit">
                <h3 class="section-title">🎯 Entry & Exit Strategy</h3>
                <div class="entry-exit-grid">
                    <div class="entry-exit-item">
                        <div class="entry-exit-label">Entry (Aggressive)</div>
                        <div class="entry-exit-value">NPR ${(entryPoints.aggressive || 0).toFixed(2)}</div>
                    </div>
                    <div class="entry-exit-item">
                        <div class="entry-exit-label">Entry (Conservative)</div>
                        <div class="entry-exit-value">NPR ${(entryPoints.conservative || 0).toFixed(2)}</div>
                    </div>
                    <div class="entry-exit-item">
                        <div class="entry-exit-label">Stop Loss</div>
                        <div class="entry-exit-value">NPR ${stopLoss.toFixed(2)}</div>
                    </div>
                    <div class="entry-exit-item">
                        <div class="entry-exit-label">Take Profit</div>
                        <div class="entry-exit-value">NPR ${takeProfit.toFixed(2)}</div>
                    </div>
                </div>
            </div>
            
            <!-- Analysis Scores -->
            <div class="scores-section">
                <h3 class="section-title">📊 Analysis Scores</h3>
                <div class="scores-grid">
                    ${createScoreItem('Technical Analysis', scores.technical || 0)}
                    ${createScoreItem('Fundamental Analysis', scores.fundamental || 0)}
                    ${createScoreItem('Sentiment Analysis', scores.sentiment || 0)}
                    ${createScoreItem('Momentum Analysis', scores.momentum || 0)}
                </div>
            </div>
            
            <!-- Detailed Analysis Information -->
            <div class="detailed-analysis">
                ${createTechnicalAnalysisDetails(stock)}
                ${createFundamentalAnalysisDetails(stock)}
                ${createSentimentAnalysisDetails(stock)}
            </div>
            
            <!-- Last Updated -->
            <div class="last-updated-section">
                <span class="last-updated-icon">🕐</span>
                <span class="last-updated-text">Last Updated: ${formatTimestamp(stock.timestamp)}</span>
            </div>
            
            ${mlPredictions ? createMLPredictionsSection(mlPredictions) : ''}
            
            ${patterns.length > 0 ? createCandlestickPatternsSection(patterns) : ''}
            
            <!-- Insights & Warnings -->
            <div class="insights-warnings">
                ${keyInsights.length > 0 ? createInsightsSection(keyInsights) : ''}
                ${warnings.length > 0 ? createWarningsSection(warnings) : ''}
            </div>
        </div>
    `;
}

function createScoreItem(name, score) {
    return `
        <div class="score-item">
            <div class="score-header">
                <span class="score-name">${name}</span>
                <span class="score-number">${score.toFixed(1)}</span>
            </div>
            <div class="score-bar-container">
                <div class="score-bar-fill" style="width: ${score}%"></div>
            </div>
        </div>
    `;
}

function createMLPredictionsSection(mlPredictions) {
    if (!mlPredictions) return '';
    
    // Handle both old weekly and new 7-day prediction formats
    let predictions = [];
    const trendAnalysis = mlPredictions.trend_analysis || {};
    
    // NEW FORMAT: Check for 'days' object (7-day predictions)
    const daysObj = mlPredictions.days;
    
    if (daysObj && typeof daysObj === 'object') {
        // New 7-day format
        ['day_1', 'day_2', 'day_3', 'day_4', 'day_5', 'day_6', 'day_7'].forEach((key, index) => {
            if (daysObj[key]) {
                const pred = daysObj[key];
                predictions.push({
                    period: `Day ${index + 1}`,
                    predicted_price: pred.predicted_price,
                    price_change_pct: pred.price_change_pct,
                    confidence: pred.confidence_score || 85,
                    target_date: pred.target_date,
                    trend: pred.trend
                });
            }
        });
    } else {
        // OLD FORMAT: Check for horizons/weeks object (weekly predictions)
        const predObj = mlPredictions.horizons || mlPredictions.predictions || {};
        
        if (typeof predObj === 'object' && !Array.isArray(predObj)) {
            ['1_week', '2_week', '4_week', '6_week'].forEach(key => {
                if (predObj[key]) {
                    const pred = predObj[key];
                    predictions.push({
                        period: key.replace('_', ' ').toUpperCase(),
                        predicted_price: pred.predicted_price,
                        price_change_pct: pred.price_change_pct,
                        confidence: pred.confidence_score || 85,
                        target_date: pred.target_date,
                        trend: pred.trend
                    });
                }
            });
        } else if (Array.isArray(predObj)) {
            predictions = predObj;
        }
    }
    
    if (predictions.length === 0) return '';
    
    let predictionCards = predictions.map(pred => {
        const change = pred.price_change_pct || pred.predicted_change_pct || 0;
        const changeClass = change >= 0 ? 'positive' : 'negative';
        const trendIcon = change >= 0 ? '📈' : '📉';
        const price = pred.predicted_price || 0;
        const confidence = pred.confidence || pred.confidence_score || 85;
        
        return `
            <div class="prediction-card">
                <div class="prediction-horizon">${pred.period || ''}</div>
                <div style="font-size: 0.8rem; opacity: 0.8; margin-bottom: 0.5rem;">${pred.target_date || ''}</div>
                <div class="prediction-price">NPR ${price.toFixed(2)}</div>
                <div class="prediction-change ${changeClass}">${change >= 0 ? '+' : ''}${change.toFixed(2)}% ${trendIcon}</div>
                <div style="font-size: 0.85rem; opacity: 0.9; margin-top: 0.5rem;">
                    Confidence: ${confidence.toFixed(1)}%
                </div>
            </div>
        `;
    }).join('');
    
    return `
        <div class="ml-predictions">
            <h3 class="section-title" style="color: white;">🔮 ML Price Predictions</h3>
            <div style="margin-bottom: 1rem; opacity: 0.9;">
                Overall Trend: <strong>${trendAnalysis.overall_trend || 'N/A'}</strong>
                ${trendAnalysis.avg_predicted_change ? ` • Avg Change: <strong>${trendAnalysis.avg_predicted_change >= 0 ? '+' : ''}${trendAnalysis.avg_predicted_change.toFixed(2)}%</strong>` : ''}
            </div>
            <div class="prediction-grid">
                ${predictionCards}
            </div>
        </div>
    `;
}

function createCandlestickPatternsSection(patterns) {
    const patternCards = patterns.map(pattern => {
        const typeClass = pattern.type === 'Bullish' ? 'bullish' : pattern.type === 'Bearish' ? 'bearish' : '';
        return `
            <div class="pattern-card">
                <div class="pattern-name">${pattern.pattern}</div>
                <div class="pattern-type ${typeClass}">${pattern.type} • ${(pattern.confidence * 100).toFixed(0)}% confidence</div>
                <div style="font-size: 0.8rem; color: #6b7280; margin-top: 0.3rem;">${pattern.description || ''}</div>
            </div>
        `;
    }).join('');
    
    return `
        <div class="candlestick-patterns">
            <h3 class="section-title">🕯️ Candlestick Patterns (Last 10 Days)</h3>
            <div class="patterns-grid">
                ${patternCards}
            </div>
        </div>
    `;
}

function createInsightsSection(insights) {
    const items = insights.map(insight => 
        `<div class="insight-item">${insight}</div>`
    ).join('');
    
    return `
        <div class="insights-box">
            <div class="box-title">💡 Key Insights</div>
            ${items}
        </div>
    `;
}

function createWarningsSection(warnings) {
    const items = warnings.map(warning => 
        `<div class="warning-item">${warning}</div>`
    ).join('');
    
    return `
        <div class="warnings-box">
            <div class="box-title">⚠️ Warnings & Risks</div>
            ${items}
        </div>
    `;
}

function formatTimestamp(timestamp) {
    if (!timestamp) return 'N/A';
    const date = new Date(timestamp);
    return date.toLocaleString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

function createTechnicalAnalysisDetails(stock) {
    const technical = stock.technical_analysis || {};
    const indicators = technical.indicators || {};
    const signals = technical.signals || {};
    const momentum = technical.momentum || {};
    const supportResistance = technical.support_resistance || {};
    
    // Get support and resistance levels
    const support = supportResistance.support ? supportResistance.support[0] : 0;
    const resistance = supportResistance.resistance ? supportResistance.resistance[0] : 0;
    
    return `
        <div class="analysis-details-section">
            <h4 class="analysis-details-title">🔧 Technical Analysis Details</h4>
            <div class="analysis-details-grid">
                <div class="detail-item">
                    <span class="detail-label">Trend</span>
                    <span class="detail-value trend-${(signals.trend || 'neutral').toLowerCase()}">${signals.trend || 'N/A'}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">RSI (14)</span>
                    <span class="detail-value">${(indicators.rsi || 0).toFixed(2)}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">MACD</span>
                    <span class="detail-value">${(indicators.macd || 0).toFixed(2)}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">MACD Signal</span>
                    <span class="detail-value">${signals.macd_signal || 'N/A'}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">SMA 20</span>
                    <span class="detail-value">NPR ${(indicators.sma_medium || 0).toFixed(2)}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">SMA 50</span>
                    <span class="detail-value">NPR ${(indicators.sma_long || 0).toFixed(2)}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Support Level</span>
                    <span class="detail-value">NPR ${support.toFixed(2)}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Resistance Level</span>
                    <span class="detail-value">NPR ${resistance.toFixed(2)}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">ATR (Volatility)</span>
                    <span class="detail-value">${(indicators.atr || 0).toFixed(2)}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">5-Day Change</span>
                    <span class="detail-value ${momentum.price_change_5d >= 0 ? 'positive' : 'negative'}">${momentum.price_change_5d >= 0 ? '+' : ''}${(momentum.price_change_5d || 0).toFixed(2)}%</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">20-Day Change</span>
                    <span class="detail-value ${momentum.price_change_20d >= 0 ? 'positive' : 'negative'}">${momentum.price_change_20d >= 0 ? '+' : ''}${(momentum.price_change_20d || 0).toFixed(2)}%</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Technical Score</span>
                    <span class="detail-value">${(stock.scores?.technical || 0).toFixed(1)}/100</span>
                </div>
            </div>
        </div>
    `;
}

function createFundamentalAnalysisDetails(stock) {
    const fundamentals = stock.fundamentals || {};
    const ratios = fundamentals.ratios || {};
    const insights = stock.trading_insights || {};
    const isEstimated = fundamentals.is_estimated === true;
    
    // Extract values from ratios (which have {value, score, interpretation} structure)
    const peRatio = ratios.pe_ratio?.value || 0;
    const pbRatio = ratios.pb_ratio?.value || 0;
    const dividendYield = ratios.dividend_yield?.value || 0;
    const epsGrowth = ratios.eps_growth?.value || 0;
    const roe = ratios.roe || 0;
    const debtToEquity = ratios.debt_to_equity || 0;
    const currentRatio = ratios.current_ratio || 0;
    
    // Get top-level fundamental data
    const currentPrice = fundamentals.current_price || 0;
    const marketCap = fundamentals.market_cap || 0;
    
    // Calculate EPS from PE ratio if available
    const eps = peRatio > 0 && currentPrice > 0 ? currentPrice / peRatio : 0;
    
    // Calculate book value from PB ratio if available
    const bookValue = pbRatio > 0 && currentPrice > 0 ? currentPrice / pbRatio : 0;
    
    // Warning banner for estimated data
    const estimatedWarning = isEstimated ? `
        <div class="estimated-data-warning">
            <span class="warning-icon">⚠️</span>
            <span class="warning-text">This fundamental data is estimated. Real data could not be fetched from external sources.</span>
        </div>
    ` : '';
    
    return `
        <div class="analysis-details-section">
            <h4 class="analysis-details-title">💼 Fundamental Analysis Details</h4>
            ${estimatedWarning}
            <div class="analysis-details-grid">
                <div class="detail-item">
                    <span class="detail-label">P/E Ratio</span>
                    <span class="detail-value">${peRatio.toFixed(2)}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">P/B Ratio</span>
                    <span class="detail-value">${pbRatio.toFixed(2)}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">EPS</span>
                    <span class="detail-value">NPR ${eps.toFixed(2)}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">EPS Growth</span>
                    <span class="detail-value ${epsGrowth > 0 ? 'positive' : epsGrowth < 0 ? 'negative' : ''}">${epsGrowth > 0 ? '+' : ''}${epsGrowth.toFixed(2)}%</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Book Value</span>
                    <span class="detail-value">NPR ${bookValue.toFixed(2)}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">ROE</span>
                    <span class="detail-value">${roe.toFixed(2)}%</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Dividend Yield</span>
                    <span class="detail-value">${dividendYield.toFixed(2)}%</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Debt/Equity</span>
                    <span class="detail-value">${debtToEquity.toFixed(2)}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Current Ratio</span>
                    <span class="detail-value">${currentRatio.toFixed(2)}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Market Cap</span>
                    <span class="detail-value">${marketCap ? formatMarketCap(marketCap) : 'N/A'}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Overall Rating</span>
                    <span class="detail-value">${fundamentals.overall_rating || 'N/A'}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Fundamental Score</span>
                    <span class="detail-value">${(stock.scores?.fundamental || fundamentals.overall_score || 0).toFixed(1)}/100</span>
                </div>
            </div>
        </div>
    `;
}

function createSentimentAnalysisDetails(stock) {
    const news = stock.news || {};
    const sentiment = stock.sentiment_details || {};
    
    // Get sentiment from news object or sentiment_details
    const sentimentLabel = news.sentiment_label || sentiment.sentiment_label || 'NEUTRAL';
    const avgSentiment = news.avg_sentiment || sentiment.overall_sentiment || 0;
    const totalArticles = news.total_articles || sentiment.total_articles || sentiment.articles_analyzed || 0;
    
    // Check if we have any article data
    const hasArticles = totalArticles > 0;
    
    // Calculate sentiment score (0-100 scale)
    let sentimentScore = 50; // Default neutral
    if (typeof avgSentiment === 'number') {
        // If it's already 0-1 scale, convert to 0-100
        sentimentScore = avgSentiment <= 1 ? avgSentiment * 100 : avgSentiment;
    } else if (avgSentiment === 'positive') {
        sentimentScore = 75;
    } else if (avgSentiment === 'negative') {
        sentimentScore = 25;
    } else if (avgSentiment === 'neutral') {
        sentimentScore = 50;
    }
    
    // Count article types
    const articles = news.articles || sentiment.articles || [];
    let positiveCount = sentiment.positive_articles || 0;
    let negativeCount = sentiment.negative_articles || 0;
    let neutralCount = sentiment.neutral_articles || 0;
    
    // If not in sentiment_details, try to count from articles
    if (!sentiment.positive_articles && articles.length > 0) {
        articles.forEach(article => {
            const score = article.sentiment_score || 0;
            if (score > 0.1) positiveCount++;
            else if (score < -0.1) negativeCount++;
            else neutralCount++;
        });
    }
    
    // If no articles, show a message
    if (!hasArticles) {
        return `
            <div class="analysis-details-section">
                <h4 class="analysis-details-title">📰 Sentiment Analysis Details</h4>
                <div style="padding: 2rem; text-align: center; color: #6b7280;">
                    <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">ℹ️ No News Articles Available</p>
                    <p style="font-size: 0.9rem; opacity: 0.8;">Sentiment analysis requires news data. Run a fresh analysis to fetch latest articles.</p>
                </div>
            </div>
        `;
    }
    
    return `
        <div class="analysis-details-section">
            <h4 class="analysis-details-title">📰 Sentiment Analysis Details</h4>
            <div class="analysis-details-grid">
                <div class="detail-item">
                    <span class="detail-label">Overall Sentiment</span>
                    <span class="detail-value sentiment-${sentimentLabel.toLowerCase()}">
                        ${sentimentLabel}
                    </span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Sentiment Score</span>
                    <span class="detail-value">${sentimentScore.toFixed(1)}%</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Articles Analyzed</span>
                    <span class="detail-value">${totalArticles}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Positive Articles</span>
                    <span class="detail-value positive">${positiveCount}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Negative Articles</span>
                    <span class="detail-value negative">${negativeCount}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Neutral Articles</span>
                    <span class="detail-value">${neutralCount}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Sentiment Score</span>
                    <span class="detail-value">${(stock.scores?.sentiment || 0).toFixed(1)}/100</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Data Source</span>
                    <span class="detail-value">${articles.length} recent articles</span>
                </div>
            </div>
        </div>
    `;
}

function formatMarketCap(value) {
    if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
    if (value >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
    if (value >= 1e3) return `${(value / 1e3).toFixed(2)}K`;
    return value.toFixed(0);
}

function createErrorCard(stock) {
    return `
        <div class="stock-card">
            <div class="stock-header" style="background: #ef4444;">
                <div class="stock-title">
                    <div class="stock-symbol">${stock.symbol}</div>
                </div>
            </div>
            <div style="padding: 2rem; text-align: center; color: #6b7280;">
                <p>❌ Analysis failed: ${stock.error}</p>
            </div>
        </div>
    `;
}

function updateTimestamps(stocks) {
    if (stocks.length > 0 && stocks[0].timestamp) {
        const timestamp = new Date(stocks[0].timestamp).toLocaleString();
        document.getElementById('last-updated').textContent = `Last updated: ${timestamp}`;
        document.getElementById('footer-timestamp').textContent = timestamp;
    }
}

// Load data when page loads
document.addEventListener('DOMContentLoaded', loadData);
"""
        with open(os.path.join(self.output_dir, 'script.js'), 'w') as f:
            f.write(js)


def main():
    generator = StockWebsiteGenerator()
    
    print("\n" + "="*70)
    print("GENERATING STOCK ANALYSIS WEBSITE")
    print("="*70)
    
    generator.load_data()
    generator.generate_website()
    
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
