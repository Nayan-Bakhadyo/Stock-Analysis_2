// ML Predictions Dashboard JavaScript

let stocksData = {};
let filteredStocks = [];

// Load data from stock_predictions.json
async function loadPredictions() {
    try {
        const response = await fetch('../stock_predictions.json');
        const data = await response.json();
        stocksData = data.stocks || {};
        
        // Update last updated time
        const lastUpdated = new Date(data.last_updated);
        document.getElementById('last-updated').textContent = 
            `Last Updated: ${lastUpdated.toLocaleString()}`;
        document.getElementById('footer-timestamp').textContent = 
            lastUpdated.toLocaleString();
        
        // Initialize display
        updateSummary();
        filteredStocks = Object.values(stocksData);
        renderPredictions();
        
    } catch (error) {
        console.error('Error loading predictions:', error);
        document.getElementById('predictions-container').innerHTML = `
            <div class="processing-state">
                <div style="font-size: 3rem; margin-bottom: 1rem;">⚠️</div>
                <h3>Unable to load predictions</h3>
                <p>Please check if stock_predictions.json exists</p>
            </div>
        `;
    }
}

// Update summary cards
function updateSummary() {
    const stocks = Object.values(stocksData);
    
    const bullishCount = stocks.filter(s => s.trading_signal?.direction === 'BULLISH').length;
    const bearishCount = stocks.filter(s => s.trading_signal?.direction === 'BEARISH').length;
    const neutralCount = stocks.filter(s => s.trading_signal?.direction === 'NEUTRAL').length;
    
    const avgMape = stocks.length > 0 
        ? (stocks.reduce((sum, s) => sum + (s.model?.performance?.mape || 0), 0) / stocks.length).toFixed(2)
        : 0;
    
    document.getElementById('bullish-count').textContent = bullishCount;
    document.getElementById('bearish-count').textContent = bearishCount;
    document.getElementById('neutral-count').textContent = neutralCount;
    document.getElementById('avg-accuracy').textContent = `${avgMape}%`;
}

// Render prediction cards
function renderPredictions() {
    const container = document.getElementById('predictions-container');
    const noResults = document.getElementById('no-results');
    
    if (filteredStocks.length === 0) {
        container.style.display = 'none';
        noResults.style.display = 'block';
        return;
    }
    
    container.style.display = 'grid';
    noResults.style.display = 'none';
    
    container.innerHTML = filteredStocks.map(stock => createPredictionCard(stock)).join('');
    
    // Add click listeners
    document.querySelectorAll('.prediction-card').forEach(card => {
        card.addEventListener('click', () => {
            showStockDetail(card.dataset.symbol);
        });
    });
}

// Create individual prediction card
function createPredictionCard(stock) {
    const signal = stock.trading_signal || {};
    const model = stock.model || {};
    const performance = model.performance || {};
    const predictions = stock.predictions || {};
    
    const lastPrice = model.last_actual_price || 0;
    const firstPrediction = predictions.prices?.[0] || lastPrice;
    const lastPrediction = predictions.prices?.[6] || lastPrice;
    const weekChange = ((lastPrediction - lastPrice) / lastPrice * 100).toFixed(2);
    
    return `
        <div class="prediction-card" data-symbol="${stock.symbol}">
            <div class="prediction-header">
                <div class="prediction-symbol">${stock.symbol}</div>
                <div class="prediction-price">Current: Rs. ${lastPrice.toFixed(2)}</div>
            </div>
            <div class="prediction-body">
                <span class="signal-badge ${signal.direction || 'NEUTRAL'}">
                    ${signal.direction || 'NEUTRAL'} 
                    ${signal.recommendation ? `• ${signal.recommendation}` : ''}
                </span>
                
                <div class="prediction-stats">
                    <div class="stat-item">
                        <div class="stat-label">MAPE</div>
                        <div class="stat-value ${performance.mape_rating?.toLowerCase() || ''}">
                            ${performance.mape?.toFixed(2) || 'N/A'}%
                        </div>
                        <div class="stat-label" style="margin-top: 0.25rem; font-size: 0.7rem;">
                            ${performance.mape_rating || 'Unknown'}
                        </div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Direction</div>
                        <div class="stat-value">
                            ${performance.direction_accuracy?.toFixed(1) || 'N/A'}%
                        </div>
                        <div class="stat-label" style="margin-top: 0.25rem; font-size: 0.7rem;">
                            ${performance.signal_strength || 'Unknown'}
                        </div>
                    </div>
                </div>
                
                <div class="prediction-stats" style="margin-top: 0.5rem;">
                    <div class="stat-item">
                        <div class="stat-label">Confidence</div>
                        <div class="stat-value">${signal.confidence?.toFixed(1) || 'N/A'}%</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">7-Day Change</div>
                        <div class="stat-value ${weekChange >= 0 ? 'excellent' : 'poor'}">
                            ${weekChange >= 0 ? '+' : ''}${weekChange}%
                        </div>
                    </div>
                </div>
                
                <div class="forecast-preview">
                    <div class="stat-label" style="margin-bottom: 0.5rem;">7-Day Forecast Preview</div>
                    ${createForecastPreview(predictions, lastPrice)}
                </div>
            </div>
        </div>
    `;
}

// Create forecast preview (first 3 days)
function createForecastPreview(predictions, basePrice) {
    if (!predictions.dates || !predictions.prices) {
        return '<p style="color: var(--text-secondary); font-size: 0.9rem;">No forecast available</p>';
    }
    
    return predictions.dates.slice(0, 3).map((date, i) => {
        const price = predictions.prices[i];
        const change = ((price - basePrice) / basePrice * 100).toFixed(2);
        const isPositive = change >= 0;
        
        return `
            <div class="forecast-item">
                <span class="forecast-date">${new Date(date).toLocaleDateString()}</span>
                <span class="forecast-price">Rs. ${price.toFixed(2)}</span>
                <span class="forecast-change ${isPositive ? 'positive' : 'negative'}">
                    ${isPositive ? '+' : ''}${change}%
                </span>
            </div>
        `;
    }).join('');
}

// Show detailed stock modal
function showStockDetail(symbol) {
    const stock = stocksData[symbol];
    if (!stock) return;
    
    const modal = document.getElementById('stock-modal');
    const modalBody = document.getElementById('modal-body');
    
    const signal = stock.trading_signal || {};
    const model = stock.model || {};
    const performance = model.performance || {};
    const predictions = stock.predictions || {};
    const recent = stock.recent_actual || {};
    
    modalBody.innerHTML = `
        <div style="padding: 2rem;">
            <h2 style="margin-bottom: 1.5rem; color: var(--text-primary);">${symbol} - Detailed Prediction</h2>
            
            <div class="prediction-stats" style="margin-bottom: 2rem;">
                <div class="stat-item">
                    <div class="stat-label">Current Price</div>
                    <div class="stat-value">Rs. ${model.last_actual_price?.toFixed(2) || 'N/A'}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Signal</div>
                    <div class="stat-value">${signal.direction || 'N/A'}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Recommendation</div>
                    <div class="stat-value">${signal.recommendation || 'N/A'}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Confidence</div>
                    <div class="stat-value">${signal.confidence?.toFixed(1) || 'N/A'}%</div>
                </div>
            </div>
            
            <h3 style="margin: 1.5rem 0 1rem;">Model Performance</h3>
            <div class="prediction-stats">
                <div class="stat-item">
                    <div class="stat-label">MAPE (Accuracy)</div>
                    <div class="stat-value ${performance.mape_rating?.toLowerCase() || ''}">
                        ${performance.mape?.toFixed(2) || 'N/A'}%
                    </div>
                    <div class="stat-label" style="margin-top: 0.25rem;">${performance.mape_rating || 'Unknown'}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">MAE (Error)</div>
                    <div class="stat-value">Rs. ${performance.mae?.toFixed(2) || 'N/A'}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Direction Accuracy</div>
                    <div class="stat-value">${performance.direction_accuracy?.toFixed(1) || 'N/A'}%</div>
                    <div class="stat-label" style="margin-top: 0.25rem;">${performance.signal_strength || 'Unknown'}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Architecture</div>
                    <div class="stat-value" style="font-size: 1rem;">${model.architecture?.toUpperCase() || 'N/A'}</div>
                    <div class="stat-label" style="margin-top: 0.25rem;">${model.lookback_days || 'N/A'} days lookback</div>
                </div>
            </div>
            
            <h3 style="margin: 1.5rem 0 1rem;">7-Day Price Forecast</h3>
            <div style="background: var(--bg-light); padding: 1rem; border-radius: 8px;">
                ${createFullForecast(predictions, model.last_actual_price)}
            </div>
            
            <h3 style="margin: 1.5rem 0 1rem;">Trading Insights</h3>
            <div style="background: var(--bg-light); padding: 1rem; border-radius: 8px;">
                <p><strong>Up Days:</strong> ${signal.up_days || 0} | <strong>Down Days:</strong> ${signal.down_days || 0}</p>
                <p style="margin-top: 0.5rem;"><strong>Recommendation:</strong> ${getRecommendationText(signal, performance)}</p>
            </div>
            
            <div style="margin-top: 1.5rem; padding: 1rem; background: #fef3c7; border-left: 4px solid #f59e0b; border-radius: 4px;">
                <p style="font-size: 0.9rem; color: #92400e;">
                    ⚠️ This is an ML prediction based on historical data. Actual prices may vary significantly. 
                    Always do your own research and consider multiple factors before making investment decisions.
                </p>
            </div>
        </div>
    `;
    
    modal.style.display = 'block';
}

// Create full 7-day forecast
function createFullForecast(predictions, basePrice) {
    if (!predictions.dates || !predictions.prices) {
        return '<p>No forecast data available</p>';
    }
    
    return predictions.dates.map((date, i) => {
        const price = predictions.prices[i];
        const change = ((price - basePrice) / basePrice * 100).toFixed(2);
        const isPositive = change >= 0;
        
        return `
            <div class="forecast-item" style="padding: 0.75rem 0; border-bottom: 1px solid var(--border-color);">
                <span style="font-weight: 600;">Day ${i + 1} (${new Date(date).toLocaleDateString()})</span>
                <span style="font-weight: 600; color: var(--primary-color);">Rs. ${price.toFixed(2)}</span>
                <span class="forecast-change ${isPositive ? 'positive' : 'negative'}">
                    ${isPositive ? '↑' : '↓'} ${Math.abs(change)}%
                </span>
            </div>
        `;
    }).join('');
}

// Get recommendation text
function getRecommendationText(signal, performance) {
    const rec = signal.recommendation || 'HOLD';
    const mape = performance.mape || 100;
    const dirAcc = performance.direction_accuracy || 0;
    
    let text = `<strong>${rec}</strong> - `;
    
    if (rec === 'BUY') {
        if (mape < 2 && dirAcc > 70) {
            text += 'Strong buy signal with excellent model accuracy. Consider accumulating.';
        } else if (mape < 5 && dirAcc > 50) {
            text += 'Moderate buy signal. Model shows good accuracy. Consider buying with caution.';
        } else {
            text += 'Weak buy signal. Exercise caution and verify with other indicators.';
        }
    } else if (rec === 'SELL') {
        if (mape < 2 && dirAcc > 70) {
            text += 'Strong sell signal with excellent model accuracy. Consider reducing position.';
        } else if (mape < 5 && dirAcc > 50) {
            text += 'Moderate sell signal. Model shows good accuracy. Consider selling with caution.';
        } else {
            text += 'Weak sell signal. Exercise caution and verify with other indicators.';
        }
    } else {
        text += 'Neutral signal. Model predicts sideways movement. Hold current position or wait for clearer signals.';
    }
    
    return text;
}

// Filter and sort functions
function applyFilters() {
    const searchTerm = document.getElementById('search-input').value.toLowerCase();
    const signalFilter = document.getElementById('filter-signal').value;
    const accuracyFilter = document.getElementById('filter-accuracy').value;
    const sortBy = document.getElementById('sort-select').value;
    
    // Filter
    filteredStocks = Object.values(stocksData).filter(stock => {
        // Search filter
        if (searchTerm && !stock.symbol.toLowerCase().includes(searchTerm)) {
            return false;
        }
        
        // Signal filter
        if (signalFilter !== 'all' && stock.trading_signal?.direction !== signalFilter) {
            return false;
        }
        
        // Accuracy filter
        if (accuracyFilter !== 'all') {
            const mape = stock.model?.performance?.mape || 100;
            if (accuracyFilter === 'excellent' && mape >= 2) return false;
            if (accuracyFilter === 'good' && (mape < 2 || mape >= 5)) return false;
            if (accuracyFilter === 'fair' && (mape < 5 || mape >= 10)) return false;
        }
        
        return true;
    });
    
    // Sort
    filteredStocks.sort((a, b) => {
        switch (sortBy) {
            case 'confidence-desc':
                return (b.trading_signal?.confidence || 0) - (a.trading_signal?.confidence || 0);
            case 'accuracy-asc':
                return (a.model?.performance?.mape || 100) - (b.model?.performance?.mape || 100);
            case 'symbol-asc':
                return a.symbol.localeCompare(b.symbol);
            default:
                return 0;
        }
    });
    
    renderPredictions();
}

// Modal close handler
document.querySelector('.modal-close').addEventListener('click', () => {
    document.getElementById('stock-modal').style.display = 'none';
});

window.addEventListener('click', (e) => {
    const modal = document.getElementById('stock-modal');
    if (e.target === modal) {
        modal.style.display = 'none';
    }
});

// Event listeners
document.getElementById('search-input').addEventListener('input', applyFilters);
document.getElementById('filter-signal').addEventListener('change', applyFilters);
document.getElementById('filter-accuracy').addEventListener('change', applyFilters);
document.getElementById('sort-select').addEventListener('change', applyFilters);

// Initialize
loadPredictions();
