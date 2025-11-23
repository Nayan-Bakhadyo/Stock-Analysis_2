// Load and display stock analysis data

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
