"""
Enhanced Stock Website Generator
Displays comprehensive trading insights including profitability probability, risk-reward, ML predictions
"""

import json
import os
from datetime import datetime


class StockWebsiteGenerator:
    def __init__(self, json_file='analysis_results.json', ml_file='stock_predictions.json', output_dir='stock_website'):
        self.json_file = json_file
        self.ml_file = ml_file
        self.output_dir = output_dir
        self.data = []
        self.ml_data = {}
    
    def load_data(self):
        """Load analysis results from JSON"""
        with open(self.json_file, 'r') as f:
            self.data = json.load(f)
        
        # Load ML predictions if available
        if os.path.exists(self.ml_file):
            with open(self.ml_file, 'r') as f:
                ml_json = json.load(f)
                self.ml_data = ml_json.get('stocks', {})
            print(f"✓ Loaded ML predictions for {len(self.ml_data)} stocks from {self.ml_file}")
        else:
            print(f"⚠ No ML predictions file found at {self.ml_file}")
        
        # Merge ML predictions into stock data
        for stock in self.data:
            symbol = stock['symbol']
            if symbol in self.ml_data:
                stock['ml_predictions'] = self.ml_data[symbol]
        
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
        
        # Copy ML predictions separately for direct access if needed
        if self.ml_data:
            with open(os.path.join(self.output_dir, 'ml_predictions.json'), 'w') as f:
                json.dump({'stocks': self.ml_data, 'last_updated': datetime.now().isoformat()}, f, indent=2)
    
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
                <select id="filter-ml-predictions">
                    <option value="all">All Stocks</option>
                    <option value="with-ml">With ML Predictions</option>
                    <option value="without-ml">Without ML Predictions</option>
                </select>
                <input type="date" id="filter-ml-date" placeholder="Filter by ML update date" style="padding: 0.5rem; border: 1px solid var(--border-color); border-radius: 0.5rem; background: var(--card-bg); color: var(--text-primary); font-size: 0.875rem;">
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
        css = """@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

:root {
    --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --accent-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    --success-gradient: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    --danger-gradient: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
    --bg-primary: #f8fafc;
    --bg-secondary: #f1f5f9;
    --card-bg: #ffffff;
    --text-primary: #1e293b;
    --text-secondary: #64748b;
    --border-color: rgba(0, 0, 0, 0.08);
    --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.04);
    --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.08);
    --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.12);
}

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    line-height: 1.6;
    color: var(--text-primary);
    background: var(--bg-primary);
    position: relative;
    overflow-x: hidden;
}

body::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: 
        radial-gradient(circle at 20% 50%, rgba(102, 126, 234, 0.03) 0%, transparent 50%),
        radial-gradient(circle at 80% 80%, rgba(245, 87, 108, 0.03) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
}

.container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 0 20px;
    position: relative;
    z-index: 1;
}

/* Animated Background */
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-20px); }
}

@keyframes glow {
    0%, 100% { opacity: 0.5; }
    50% { opacity: 1; }
}

/* Header */
header {
    background: var(--primary-gradient);
    color: white;
    padding: 4rem 0;
    position: relative;
    overflow: hidden;
}

header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 40%;
    height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
    animation: float 6s ease-in-out infinite;
}

header h1 {
    font-size: 3rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
    letter-spacing: -0.02em;
    background: linear-gradient(to right, #fff, #f0f0ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.subtitle {
    font-size: 1.3rem;
    opacity: 0.95;
    margin-bottom: 0.5rem;
    font-weight: 400;
}

.update-time {
    font-size: 0.95rem;
    opacity: 0.85;
}

/* Main Content */
main {
    padding: 3rem 0;
}

#stock-cards-container {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
    gap: 2rem;
    margin-top: 2rem;
}

/* Summary Section */
.summary-section {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 2rem;
    margin-bottom: 3rem;
}

.summary-card {
    background: var(--card-bg);
    padding: 2rem;
    border-radius: 16px;
    border: 1px solid var(--border-color);
    text-align: center;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-sm);
}

.summary-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 3px;
    background: var(--primary-gradient);
    transform: scaleX(0);
    transform-origin: left;
    transition: transform 0.3s ease;
}

.summary-card:hover::before {
    transform: scaleX(1);
}

.summary-card:hover {
    transform: translateY(-5px);
    box-shadow: var(--shadow-lg);
    border-color: rgba(102, 126, 234, 0.2);
}

.summary-card h3 {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--text-secondary);
    margin-bottom: 0.75rem;
    font-weight: 600;
}

.summary-card .number {
    font-size: 2.5rem;
    font-weight: 800;
    background: var(--primary-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
}

/* Animations */
@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes slideUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes pulse {
    0%, 100% {
        opacity: 1;
    }
    50% {
        opacity: 0.7;
    }
}

@keyframes shimmer {
    0% {
        background-position: -1000px 0;
    }
    100% {
        background-position: 1000px 0;
    }
}

.summary-icon {
    display: none;
}

.summary-value {
    font-size: 2.5rem;
    font-weight: 800;
    background: var(--primary-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
}

.summary-label {
    font-size: 0.85rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 500;
}

/* Stock Card */
.stock-card {
    background: var(--card-bg);
    border-radius: 16px;
    border: 1px solid var(--border-color);
    overflow: hidden;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    cursor: pointer;
    position: relative;
    animation: fadeIn 0.6s ease-out backwards;
    box-shadow: var(--shadow-sm);
}

.stock-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.05), transparent);
    transition: left 0.5s;
}

.stock-card:hover::before {
    left: 100%;
}

.stock-card:hover {
    transform: translateY(-8px);
    box-shadow: var(--shadow-lg);
    border-color: rgba(102, 126, 234, 0.3);
}

.stock-header {
    background: var(--primary-gradient);
    color: white;
    padding: 2rem;
    position: relative;
    overflow: hidden;
}

.stock-header::after {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 150%;
    height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 70%);
    animation: glow 4s ease-in-out infinite;
}

.stock-title {
    display: flex;
    justify-content: space-between;
    align-items: center;
    position: relative;
    z-index: 1;
}

.stock-symbol {
    font-size: 1.75rem;
    font-weight: 800;
    letter-spacing: -0.02em;
}

.stock-price {
    font-size: 1.5rem;
    font-weight: 700;
}

.stock-body {
    padding: 2rem;
}

.quick-stats {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 1.5rem;
    margin-bottom: 1.5rem;
}

.stat-item {
    padding: 1rem;
    background: var(--bg-secondary);
    border-radius: 12px;
    border: 1px solid var(--border-color);
    transition: all 0.3s ease;
}

.stat-item:hover {
    background: rgba(102, 126, 234, 0.05);
    border-color: rgba(102, 126, 234, 0.2);
    transform: scale(1.02);
}

.stat-label {
    font-size: 0.8rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.4rem;
}

.stat-value {
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--text-primary);
}

.stat-value.positive {
    background: var(--success-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.stat-value.negative {
    background: var(--danger-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.recommendation-badge {
    display: inline-block;
    padding: 0.6rem 1.2rem;
    border-radius: 24px;
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    position: relative;
    overflow: hidden;
}

.recommendation-badge::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
    animation: shimmer 2s infinite;
}

.recommendation-badge.buy {
    background: var(--success-gradient);
    box-shadow: 0 4px 15px rgba(17, 153, 142, 0.3);
}

.recommendation-badge.sell {
    background: var(--danger-gradient);
    box-shadow: 0 4px 15px rgba(235, 51, 73, 0.3);
}

.recommendation-badge.hold {
    background: linear-gradient(135deg, #f59e0b 0%, #f97316 100%);
    box-shadow: 0 4px 15px rgba(245, 158, 11, 0.3);
}

/* Filters Section */
.filters-section {
    background: var(--card-bg);
    padding: 2rem;
    border-radius: 16px;
    border: 1px solid var(--border-color);
    margin-bottom: 3rem;
    display: grid;
    gap: 1.5rem;
    animation: slideUp 0.5s ease-out;
    box-shadow: var(--shadow-sm);
}

.search-box input {
    width: 100%;
    padding: 1rem 1.5rem;
    border: 1px solid var(--border-color);
    background: var(--bg-secondary);
    border-radius: 12px;
    font-size: 1rem;
    color: var(--text-primary);
    transition: all 0.3s ease;
}

.search-box input:focus {
    outline: none;
    border-color: rgba(102, 126, 234, 0.5);
    background: var(--card-bg);
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

.search-box input::placeholder {
    color: var(--text-secondary);
}

.filter-controls {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1rem;
}

.filter-controls select {
    padding: 1rem 1.5rem;
    border: 1px solid var(--border-color);
    background: var(--bg-secondary);
    border-radius: 12px;
    font-size: 0.95rem;
    color: var(--text-primary);
    cursor: pointer;
    transition: all 0.3s ease;
}

.filter-controls select:focus {
    outline: none;
    border-color: rgba(102, 126, 234, 0.5);
    background: var(--card-bg);
}

.filter-controls select option {
    background: var(--card-bg);
    color: var(--text-primary);
}

/* Probability Section */
.probability-section {
    padding: 1.5rem;
    background: rgba(102, 126, 234, 0.05);
    border-radius: 12px;
    border: 1px solid rgba(102, 126, 234, 0.15);
    margin-bottom: 1.5rem;
}

/* Probability Section */
.probability-section {
    padding: 1.5rem;
    background: rgba(102, 126, 234, 0.05);
    border-radius: 12px;
    border: 1px solid rgba(102, 126, 234, 0.15);
    margin-bottom: 1.5rem;
}

.probability-value {
    font-size: 2rem;
    font-weight: 800;
    background: var(--primary-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.75rem;
}

.probability-bar {
    height: 10px;
    background: rgba(0, 0, 0, 0.05);
    border-radius: 6px;
    overflow: hidden;
    margin-bottom: 0.75rem;
    position: relative;
}

.probability-fill {
    height: 100%;
    background: linear-gradient(90deg, #ef4444, #f59e0b, #10b981);
    border-radius: 6px;
    transition: width 1.2s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 0 10px rgba(16, 185, 129, 0.5);
}

.confidence-level {
    font-size: 0.85rem;
    color: var(--text-secondary);
    font-weight: 500;
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
    background-color: rgba(0, 0, 0, 0.6);
    backdrop-filter: blur(8px);
    animation: fadeIn 0.3s ease-out;
}

.modal.active {
    display: block;
}

.modal-content {
    background: var(--card-bg);
    border: 1px solid var(--border-color);
    margin: 2% auto;
    padding: 0;
    border-radius: 20px;
    width: 90%;
    max-width: 1200px;
    max-height: 90vh;
    overflow-y: auto;
    box-shadow: var(--shadow-lg);
    animation: slideUp 0.4s ease-out;
}

.modal-close {
    color: var(--text-secondary);

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

.modal-close {
    color: var(--text-secondary);
    float: right;
    font-size: 2.5rem;
    font-weight: 300;
    line-height: 1;
    padding: 1.5rem;
    transition: all 0.3s ease;
}

.modal-close:hover,
.modal-close:focus {
    color: var(--text-primary);
    transform: rotate(90deg);
    cursor: pointer;
}

/* Modal Sections */
.risk-reward {
    padding: 2.5rem;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 2rem;
}

.metric-box {
    padding: 2rem;
    background: rgba(102, 126, 234, 0.05);
    border-radius: 16px;
    border: 1px solid rgba(102, 126, 234, 0.2);
    border-left: 4px solid;
    border-left-color: var(--primary-gradient);
    transition: all 0.3s ease;
    box-shadow: var(--shadow-sm);
}

.metric-box:hover {
    transform: translateY(-3px);
    box-shadow: var(--shadow-md);
}

.metric-label {
    font-size: 0.8rem;
    color: var(--text-secondary);
    margin-bottom: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
}

.metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: var(--text-primary);
}

.metric-value.positive {
    background: var(--success-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.metric-value.negative {
    background: var(--danger-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* Entry/Exit Section */
.entry-exit {
    padding: 2.5rem;
    background: var(--bg-secondary);
}

.section-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.entry-exit-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1.5rem;
}

.entry-exit-item {
    padding: 1.5rem;
    background: var(--card-bg);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    text-align: center;
    transition: all 0.3s ease;
    box-shadow: var(--shadow-sm);
}

.entry-exit-item:hover {
    border-color: rgba(102, 126, 234, 0.3);
    transform: scale(1.05);
    box-shadow: var(--shadow-md);
}

.entry-exit-label {
    font-size: 0.8rem;
    color: var(--text-secondary);
    margin-bottom: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.entry-exit-value {
    font-size: 1.5rem;
    font-weight: 700;
    background: var(--primary-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* Scores Section */
.scores-section {
    padding: 2.5rem;
}

.scores-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 2rem;
}

.score-item {
    background: rgba(102, 126, 234, 0.05);
    padding: 2rem;
    border-radius: 16px;
    border: 1px solid rgba(102, 126, 234, 0.1);
    transition: all 0.3s ease;
}

.score-item:hover {
    border-color: rgba(102, 126, 234, 0.3);
    transform: translateY(-3px);
}

.score-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
}

.score-name {
    font-weight: 600;
    color: var(--text-primary);
    font-size: 1.1rem;
}

.score-number {
    font-size: 1.6rem;
    font-weight: 800;
    background: var(--primary-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.score-bar-container {
    height: 10px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 6px;
    overflow: hidden;
    position: relative;
}


.score-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #ef4444, #f59e0b, #10b981);
    border-radius: 6px;
    transition: width 1.2s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 0 10px rgba(16, 185, 129, 0.4);
}

/* ML Predictions */
.ml-predictions {
    padding: 2.5rem;
    background: var(--primary-gradient);
    color: white;
    position: relative;
    overflow: hidden;
}

.ml-predictions::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 150%;
    height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
    animation: float 8s ease-in-out infinite;
}

.processing-message {
    padding: 2.5rem;
    text-align: center;
    font-size: 1.2rem;
    opacity: 0.95;
    background: rgba(255, 255, 255, 0.15);
    border-radius: 12px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    position: relative;
    z-index: 1;
}

.prediction-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1.5rem;
    margin-top: 2rem;
    position: relative;
    z-index: 1;
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
    padding: 2.5rem;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 2rem;
}

.insights-box, .warnings-box {
    padding: 2rem;
    border-radius: 16px;
    border: 1px solid;
    box-shadow: var(--shadow-sm);
}

.insights-box {
    background: rgba(16, 185, 129, 0.05);
    border-color: rgba(16, 185, 129, 0.2);
    border-left: 4px solid #10b981;
}

.warnings-box {
    background: rgba(239, 68, 68, 0.05);
    border-color: rgba(239, 68, 68, 0.2);
    border-left: 4px solid #ef4444;
}

.box-title {
    font-size: 1.2rem;
    font-weight: 700;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    color: var(--text-primary);
}

.insight-item, .warning-item {
    padding: 0.75rem 0;
    border-bottom: 1px solid rgba(0, 0, 0, 0.05);
    color: var(--text-primary);
    font-size: 0.95rem;
}

.insight-item:last-child, .warning-item:last-child {
    border-bottom: none;
}

/* Candlestick Patterns */
.candlestick-patterns {
    padding: 2.5rem;
    background: var(--bg-secondary);
}

/* Detailed Analysis Sections */
.detailed-analysis {
    padding: 2.5rem;
    background: var(--bg-secondary);
    border-top: 1px solid var(--border-color);
}

.analysis-details-section {
    margin-bottom: 2rem;
    padding: 2rem;
    background: var(--card-bg);
    border: 1px solid var(--border-color);
    border-radius: 16px;
}

.analysis-details-section:last-child {
    margin-bottom: 0;
}

.estimated-data-warning {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 1rem 1.25rem;
    margin-bottom: 1.5rem;
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid rgba(245, 158, 11, 0.3);
    border-radius: 12px;
    color: #f59e0b;
}

.estimated-data-warning .warning-icon {
    font-size: 1.25rem;
    flex-shrink: 0;
    display: none;
}

.estimated-data-warning .warning-text {
    font-size: 0.875rem;
    font-weight: 500;
    line-height: 1.4;
}

.analysis-details-title {
    font-size: 1.2rem;
    font-weight: 700;
    margin-bottom: 1.5rem;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.analysis-details-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1.5rem;
}

.detail-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem;
    background: rgba(102, 126, 234, 0.05);
    border-radius: 12px;
    border-left: 3px solid;
    border-left-color: var(--primary-gradient);
    transition: all 0.3s ease;
}

.detail-item:hover {
    background: rgba(102, 126, 234, 0.1);
    transform: translateX(3px);
}

.detail-label {
    font-size: 0.85rem;
    color: var(--text-secondary);
    font-weight: 500;
}

.detail-value {
    font-size: 1rem;
    font-weight: 700;
    color: var(--text-primary);
}

.detail-value.positive {
    background: var(--success-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.detail-value.negative {
    background: var(--danger-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.detail-value.trend-bullish {
    color: #10b981;
    background: rgba(16, 185, 129, 0.1);
    padding: 0.4rem 1rem;
    border-radius: 12px;
}

.detail-value.trend-bearish {
    color: #ef4444;
    background: rgba(239, 68, 68, 0.1);
    padding: 0.4rem 1rem;
    border-radius: 12px;
}

.detail-value.trend-neutral {
    color: #f59e0b;
    background: rgba(245, 158, 11, 0.1);
    padding: 0.4rem 1rem;
    border-radius: 12px;
}

.detail-value.sentiment-positive {
    color: #10b981;
    background: rgba(16, 185, 129, 0.15);
    padding: 0.4rem 1rem;
    border-radius: 12px;
    font-weight: 700;
}

.detail-value.sentiment-negative {
    color: #ef4444;
    background: rgba(239, 68, 68, 0.15);
    padding: 0.4rem 1rem;
    border-radius: 12px;
    font-weight: 700;
}

.detail-value.sentiment-neutral {
    color: #f59e0b;
    background: rgba(245, 158, 11, 0.15);
    padding: 0.4rem 1rem;
    border-radius: 12px;
    font-weight: 700;
}

/* Last Updated Section */
.last-updated-section {
    padding: 1.5rem 2rem;
    background: rgba(255, 255, 255, 0.02);
    border-top: 1px solid var(--border-color);
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.85rem;
    color: var(--text-secondary);
}

.last-updated-icon {
    display: none;
}

.last-updated-text {
    font-weight: 500;
}

.patterns-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1.5rem;
    margin-top: 1.5rem;
}

.pattern-card {
    background: var(--card-bg);
    padding: 1.5rem;
    border-radius: 12px;
    border: 1px solid var(--border-color);
    border-left: 4px solid;
    border-left-color: var(--primary-gradient);
    transition: all 0.3s ease;
}

.pattern-card:hover {
    border-color: rgba(102, 126, 234, 0.5);
    transform: translateY(-3px);
}

.pattern-name {
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
    font-size: 1.1rem;
}

.pattern-type {
    font-size: 0.85rem;
    color: var(--text-secondary);
}

.pattern-type.bullish {
    color: #10b981;
}

.pattern-type.bearish {
    color: #ef4444;
}

/* Footer */
footer {
    background: var(--card-bg);
    border-top: 1px solid var(--border-color);
    color: var(--text-primary);
    padding: 2rem 1rem;
    text-align: center;
    margin-top: 4rem;
}

footer p {
    margin: 0.5rem 0;
    color: var(--text-secondary);
    font-size: 0.9rem;
    max-width: 1200px;
    margin-left: auto;
    margin-right: auto;
    word-wrap: break-word;
}

footer .container {
    max-width: 100%;
    padding: 0 1rem;
}

/* Responsive */
@media (max-width: 1024px) {
    #stock-cards-container {
        grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    }
}

@media (max-width: 768px) {
    header h1 {
        font-size: 2rem;
    }
    
    .subtitle {
        font-size: 1.1rem;
    }
    
    #stock-cards-container {
        grid-template-columns: 1fr;
        gap: 1.5rem;
    }
    
    .summary-section {
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    }
    
    .stock-symbol {
        font-size: 1.5rem;
    }
    
    .stock-price {
        font-size: 1.3rem;
    }
    
    .probability-value {
        font-size: 1.8rem;
    }
    
    .quick-stats {
        grid-template-columns: 1fr;
        gap: 1rem;
    }
    
    .modal-content {
        width: 95%;
        margin: 5% auto;
    }
    
    .risk-reward, .entry-exit, .scores-section {
        padding: 1.5rem;
    }
    
    footer p {
        font-size: 0.85rem;
        padding: 0 1rem;
    }
}

@media (max-width: 480px) {
    header {
        padding: 2.5rem 0;
    }
    
    header h1 {
        font-size: 1.75rem;
    }
    
    .summary-card .number {
        font-size: 2rem;
    }
    
    .filter-controls {
        grid-template-columns: 1fr;
    }
    
    footer p {
        font-size: 0.8rem;
        line-height: 1.5;
    }
}

/* Animations on scroll */
.stock-card {
    animation-delay: calc(var(--card-index, 0) * 0.05s);
}

/* Loading animation */
@keyframes loading {
    0% {
        transform: scale(0.95);
        opacity: 0.7;
    }
    50% {
        transform: scale(1);
        opacity: 1;
    }
    100% {
        transform: scale(0.95);
        opacity: 0.7;
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
    const filterMlPredictions = document.getElementById('filter-ml-predictions');
    const filterMlDate = document.getElementById('filter-ml-date');
    
    searchInput.addEventListener('input', applyFilters);
    sortSelect.addEventListener('change', applyFilters);
    filterRecommendation.addEventListener('change', applyFilters);
    filterProfitability.addEventListener('change', applyFilters);
    filterMlPredictions.addEventListener('change', applyFilters);
    filterMlDate.addEventListener('change', applyFilters);
}

function applyFilters() {
    const searchTerm = document.getElementById('search-input').value.toLowerCase();
    const sortBy = document.getElementById('sort-select').value;
    const recFilter = document.getElementById('filter-recommendation').value;
    const profFilter = document.getElementById('filter-profitability').value;
    const mlFilter = document.getElementById('filter-ml-predictions').value;
    const mlDateFilter = document.getElementById('filter-ml-date').value;
    
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
    
    // Apply ML predictions filter
    if (mlFilter !== 'all') {
        filteredStocks = filteredStocks.filter(stock => {
            const hasMlPredictions = stock.ml_predictions && stock.ml_predictions.predictions && 
                                    (Array.isArray(stock.ml_predictions.predictions) ? stock.ml_predictions.predictions.length > 0 : 
                                     (stock.ml_predictions.predictions.dates && stock.ml_predictions.predictions.dates.length > 0));
            if (mlFilter === 'with-ml') return hasMlPredictions;
            if (mlFilter === 'without-ml') return !hasMlPredictions;
            return true;
        });
    }
    
    // Apply ML date filter
    if (mlDateFilter && mlDateFilter !== '') {
        filteredStocks = filteredStocks.filter(stock => {
            if (!stock.ml_predictions || !stock.ml_predictions.last_updated) return false;
            
            const lastUpdated = new Date(stock.ml_predictions.last_updated);
            const filterDate = new Date(mlDateFilter);
            
            // Compare only the date part (ignore time)
            return lastUpdated.toDateString() === filterDate.toDateString();
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
            
            <div class="stock-body">
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
    
    const predictions = mlPredictions.predictions;
    const tradingSignal = mlPredictions.trading_signal || {}; // Add fallback
    const model = mlPredictions.model || {}; // Add fallback
    const recentActual = mlPredictions.recent_actual;
    
    if (!predictions || !predictions.dates || !predictions.prices) {
        return `
            <div class="ml-predictions">
                <h3 class="section-title">🤖 AI Predictions</h3>
                <div class="processing-message">
                    ⏳ Processing... ML predictions will be available soon.
                </div>
            </div>
        `;
    }
    
    // Handle old format without trading_signal
    if (!tradingSignal.direction) {
        return `
            <div class="ml-predictions">
                <h3 class="section-title">🤖 AI Predictions</h3>
                <div class="processing-message">
                    ⚠️ This stock was trained with an older model version.<br>
                    Please retrain to see trading signals and performance metrics.
                </div>
            </div>
        `;
    }
    
    // Create prediction cards for 7 days
    const predictionCards = predictions.dates.map((date, index) => {
        const price = predictions.prices[index];
        const lastPrice = model.last_actual_price;
        const change = price - lastPrice;
        const changePct = (change / lastPrice) * 100;
        const changeClass = changePct >= 0 ? 'positive' : 'negative';
        const trendIcon = changePct >= 0 ? '📈' : '📉';
        
        return `
            <div class="prediction-card">
                <div class="prediction-horizon">Day ${index + 1}</div>
                <div style="font-size: 0.8rem; opacity: 0.8; margin-bottom: 0.5rem;">${date}</div>
                <div class="prediction-price">NPR ${price.toFixed(2)}</div>
                <div class="prediction-change ${changeClass}">${changePct >= 0 ? '+' : ''}${changePct.toFixed(2)}% ${trendIcon}</div>
            </div>
        `;
    }).join('');
    
    // Performance rating colors
    const mapeRating = model.performance.mape_rating;
    const mapeColor = mapeRating === 'Excellent' ? '#10b981' : 
                      mapeRating === 'Good' ? '#3b82f6' :
                      mapeRating === 'Fair' ? '#f59e0b' : '#ef4444';
    
    const signalStrength = model.performance.signal_strength;
    const signalColor = signalStrength === 'Strong' ? '#10b981' :
                       signalStrength === 'Moderate' ? '#f59e0b' : '#ef4444';
    
    const direction = tradingSignal.direction;
    const directionColor = direction === 'BULLISH' ? '#10b981' :
                          direction === 'BEARISH' ? '#ef4444' : '#6b7280';
    const directionIcon = direction === 'BULLISH' ? '🟢' :
                         direction === 'BEARISH' ? '🔴' : '⚪';
    
    return `
        <div class="ml-predictions">
            <h3 class="section-title">🤖 AI Price Predictions (Bi-LSTM Model)</h3>
            
            <!-- Trading Signal -->
            <div class="trading-signal-box" style="background: linear-gradient(135deg, ${directionColor}20, ${directionColor}10); border-left: 4px solid ${directionColor}; padding: 1.5rem; border-radius: 8px; margin-bottom: 1.5rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-size: 0.9rem; opacity: 0.8; margin-bottom: 0.5rem;">Trading Signal</div>
                        <div style="font-size: 1.8rem; font-weight: bold; color: ${directionColor};">
                            ${directionIcon} ${direction}
                        </div>
                        <div style="font-size: 1rem; margin-top: 0.5rem;">
                            <strong>${tradingSignal.recommendation}</strong> • Confidence: ${tradingSignal.confidence.toFixed(1)}%
                        </div>
                        <div style="font-size: 0.9rem; opacity: 0.8; margin-top: 0.3rem;">
                            ${tradingSignal.up_days} up days, ${tradingSignal.down_days} down days
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Model Performance -->
            <div class="model-performance" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; margin-bottom: 1.5rem;">
                <div class="performance-metric" style="background: white; padding: 1rem; border-radius: 8px; border: 1px solid rgba(59, 130, 246, 0.2);">
                    <div style="font-size: 0.85rem; opacity: 0.7; color: #6b7280;">MAPE (Accuracy)</div>
                    <div style="font-size: 1.5rem; font-weight: bold; color: ${mapeColor};">${model.performance.test_mape.toFixed(2)}%</div>
                    <div style="font-size: 0.8rem; font-weight: 600; color: ${mapeColor};">${mapeRating}</div>
                    <div style="font-size: 0.75rem; opacity: 0.7; color: #6b7280; margin-top: 0.25rem;">±Rs. ${(model.last_actual_price * model.performance.test_mape / 100).toFixed(2)}</div>
                </div>
                <div class="performance-metric" style="background: white; padding: 1rem; border-radius: 8px; border: 1px solid rgba(59, 130, 246, 0.2);">
                    <div style="font-size: 0.85rem; opacity: 0.7; color: #6b7280;">Direction Accuracy</div>
                    <div style="font-size: 1.5rem; font-weight: bold; color: ${signalColor};">${model.performance.direction_accuracy.toFixed(1)}%</div>
                    <div style="font-size: 0.8rem; font-weight: 600; color: ${signalColor};">${signalStrength}</div>
                </div>
                <div class="performance-metric" style="background: white; padding: 1rem; border-radius: 8px; border: 1px solid rgba(59, 130, 246, 0.2);">
                    <div style="font-size: 0.85rem; opacity: 0.7; color: #6b7280;">Model Type</div>
                    <div style="font-size: 1rem; font-weight: bold; margin-top: 0.5rem; color: #1f2937;">${model.architecture.toUpperCase()}</div>
                    <div style="font-size: 0.8rem; opacity: 0.7; color: #6b7280;">${model.lookback_days} days lookback</div>
                </div>
                <div class="performance-metric" style="background: white; padding: 1rem; border-radius: 8px; border: 1px solid rgba(59, 130, 246, 0.2);">
                    <div style="font-size: 0.85rem; opacity: 0.7; color: #6b7280;">Last Known Price</div>
                    <div style="font-size: 1.5rem; font-weight: bold; color: #1f2937;">NPR ${model.last_actual_price.toFixed(2)}</div>
                    <div style="font-size: 0.8rem; opacity: 0.7; color: #6b7280;">Base for predictions</div>
                </div>
            </div>
            
            <!-- Predictions Grid -->
            <div class="prediction-grid">
                ${predictionCards}
            </div>
            
            <!-- Model Info -->
            <div style="margin-top: 1.5rem; padding: 1rem; background: rgba(59, 130, 246, 0.05); border-radius: 8px; font-size: 0.85rem; opacity: 0.9;">
                <div><strong>Model Details:</strong></div>
                <div>• Architecture: ${model.architecture.toUpperCase()} with ${model.layers} layers</div>
                <div>• Training Samples: ${model.training_samples.toLocaleString()} data points</div>
                <div>• MAE: ${model.performance.test_mae.toFixed(2)} Rs | RMSE: ${model.performance.test_rmse.toFixed(2)} Rs</div>
                <div style="margin-top: 0.5rem; opacity: 0.7;">
                    <strong>Note:</strong> MAPE < 2% = Excellent, 2-5% = Good, 5-10% = Fair, >10% = Poor<br>
                    Direction Accuracy > 70% = Strong signal, 50-70% = Moderate, < 50% = Weak
                </div>
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
    // Filter out placeholder metrics (EPS growth 11.11% and ROE 20.00% appear to be placeholder data)
    const filteredInsights = insights.filter(insight => {
        const lowerInsight = insight.toLowerCase();
        // Remove if it's a generic/placeholder EPS growth or ROE message
        if (lowerInsight.includes('eps growth') && lowerInsight.includes('11.11')) {
            return false;
        }
        if (lowerInsight.includes('roe') && lowerInsight.includes('20.')) {
            return false;
        }
        return true;
    });
    
    if (filteredInsights.length === 0) return '';
    
    const items = filteredInsights.map(insight => 
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
    const epsGrowth = ratios.eps_growth?.value;
    // Don't show EPS Growth if it's the placeholder value (11.11%) or invalid
    const hasEpsGrowth = epsGrowth !== undefined && epsGrowth !== null && epsGrowth !== 0 && Math.abs(epsGrowth - 11.11) > 0.01;
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
                ${hasEpsGrowth ? `
                <div class="detail-item">
                    <span class="detail-label">EPS Growth</span>
                    <span class="detail-value ${epsGrowth > 0 ? 'positive' : epsGrowth < 0 ? 'negative' : ''}">${epsGrowth > 0 ? '+' : ''}${epsGrowth.toFixed(2)}%</span>
                </div>
                ` : ''}
                <div class="detail-item">
                    <span class="detail-label">Book Value</span>
                    <span class="detail-value">NPR ${bookValue.toFixed(2)}</span>
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
