"""Trading insights engine - profitability probability and recommendations"""
import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime, timedelta
import config
from data_fetcher import NepseDataFetcher
from sharesansar_news_scraper import ShareSansarNewsScraper
from nepsealpha_scraper import NepalAlphaScraper
from fundamental_analyzer import FundamentalAnalyzer
from technical_analyzer import TechnicalAnalyzer
from broker_analyzer import BrokerAnalyzer
from sync_manager import SyncManager

# Try to import ML predictor
try:
    from ml_predictor import MLStockPredictor
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️ ML predictions disabled. Install TensorFlow: pip install tensorflow scikit-learn")


class TradingInsightsEngine:
    """Generate trading insights and profitability probability"""
    
    def __init__(self, enable_ml: bool = True):
        self.data_fetcher = NepseDataFetcher()
        self.news_scraper = ShareSansarNewsScraper(headless=True)
        self.fundamental_scraper = NepalAlphaScraper(headless=True)
        self.fundamental_analyzer = FundamentalAnalyzer()
        self.technical_analyzer = TechnicalAnalyzer()
        self.broker_analyzer = BrokerAnalyzer()
        self.enable_ml = enable_ml  # Flag to enable/disable ML
        self.ml_predictor = MLStockPredictor(lookback_days=60) if (ML_AVAILABLE and enable_ml) else None
        self.sync_manager = SyncManager()  # Add sync manager
        self.weights = config.SIGNAL_WEIGHTS
        self.risk_config = config.RISK_CONFIG
    
    def calculate_profitability_probability(self, symbol: str, 
                                           time_horizon: str = 'short',
                                           include_broker_analysis: bool = True,
                                           use_cache: bool = True,
                                           reuse_ml_model: bool = False) -> Dict:
        """
        Calculate probability of profitability
        
        Args:
            symbol: Stock symbol
            time_horizon: 'short' (1-7 days), 'medium' (1-4 weeks), 'long' (1-3 months)
            include_broker_analysis: Include broker manipulation/liquidity analysis
            use_cache: Use cached data and only fetch new updates
            reuse_ml_model: Reuse existing ML model if available (faster, for daily updates)
        """
        
        print(f"\n🔍 Analyzing {symbol}...")
        
        # Sync data (only fetch new data if use_cache=True)
        news_just_synced = False
        sync_total_articles = 0
        if use_cache:
            print("🔄 Checking for data updates...")
            self.sync_manager.sync_price_history(symbol)
            sync_result = self.sync_manager.sync_news(symbol, max_articles=10)
            news_just_synced = True  # Mark that we just synced news
            sync_total_articles = sync_result.get('total_articles', 0)
        
        # Fetch data
        print("📊 Fetching historical data...")
        price_data = self.data_fetcher.get_stock_price_history(symbol, days=None)  # Get all available data for ML
        
        if price_data.empty or len(price_data) < 50:
            return {
                'error': 'Insufficient historical data',
                'symbol': symbol,
                'probability': 0
            }
        
        # Get current price from latest data in database (last row after ASC sort)
        current_price = float(price_data['close'].iloc[-1])
        
        # Perform analyses
        print("📈 Performing technical analysis...")
        technical_analysis = self.technical_analyzer.comprehensive_analysis(price_data)
        
        print("💰 Performing fundamental analysis...")
        # Create mock fundamental data - in production, fetch from actual sources
        fundamental_data = self._get_fundamental_data(symbol, current_price)
        fundamental_analysis = self.fundamental_analyzer.comprehensive_analysis(fundamental_data)
        
        print("📰 Analyzing market sentiment...")
        # Combine cached news + fetch new articles only if needed
        if use_cache:
            # Step 1: Get cached news from database (after sync if it just happened)
            # Use a longer lookback or no limit if we just synced to get all articles
            if news_just_synced and sync_total_articles > 0:
                # Query without date restriction to get all articles after sync
                cached_news = self.sync_manager.get_cached_news(symbol, days=365*5, limit=20)
                print(f"  ℹ️ Found {len(cached_news)} articles in cache (after sync)")
            else:
                cached_news = self.sync_manager.get_cached_news(symbol, days=180)
                print(f"  ℹ️ Found {len(cached_news)} existing articles in cache")
            
            # If we just synced and have enough articles, use them directly
            if news_just_synced and len(cached_news) >= 5:
                print(f"  ✓ Using {len(cached_news)} freshly synced articles")
                all_cached_news = cached_news
            else:
                # Check if we have enough articles with sentiment scores
                articles_with_sentiment = sum(1 for a in cached_news if a.get('sentiment_score') is not None)
                
                # Step 2: Only fetch new articles if we don't have enough with sentiment
                if articles_with_sentiment < 5:
                    print(f"  → Only {articles_with_sentiment} articles have sentiment scores, fetching latest news...")
                    # Fetch and store new articles (sentiment returned, but we'll use cache)
                    self.news_scraper.get_aggregate_sentiment(symbol, max_articles=10)
                    
                    # Step 3: Get updated cached news (includes newly fetched ones)
                    all_cached_news = self.sync_manager.get_cached_news(symbol, days=180)
                else:
                    print(f"  ✓ Using {articles_with_sentiment} cached articles with sentiment scores")
                    all_cached_news = cached_news
            
            if all_cached_news and len(all_cached_news) >= 5:
                # Re-check sentiment scores after potential fetch
                articles_with_sentiment = sum(1 for a in all_cached_news if a.get('sentiment_score') is not None)
                
                if articles_with_sentiment >= 5:
                    print(f"  ✓ Analyzing {len(all_cached_news)} total articles")
                    sentiment_analysis = self._analyze_cached_sentiment(all_cached_news)
                    # Add articles array
                    sentiment_analysis['articles'] = all_cached_news
                    sentiment_analysis['total_articles'] = len(all_cached_news)
                else:
                    # Analyze all cached articles without sentiment scores
                    print(f"  → Computing sentiment for {len(all_cached_news)} articles")
                    from sentiment_analyzer import SentimentAnalyzer
                    sent_analyzer = SentimentAnalyzer()
                    sentiment_df = sent_analyzer.analyze_news_list(all_cached_news)
                    
                    # Calculate aggregate metrics
                    avg_score = sentiment_df['score'].mean() if not sentiment_df.empty else 0
                    sentiment_analysis = {
                        'overall_sentiment': avg_score,
                        'sentiment_label': 'POSITIVE' if avg_score > 0.1 else 'NEGATIVE' if avg_score < -0.1 else 'NEUTRAL',
                        'articles_analyzed': len(all_cached_news),
                        'total_articles': len(all_cached_news),
                        'positive_articles': len(sentiment_df[sentiment_df['sentiment'] == 'positive']) if not sentiment_df.empty else 0,
                        'negative_articles': len(sentiment_df[sentiment_df['sentiment'] == 'negative']) if not sentiment_df.empty else 0,
                        'neutral_articles': len(sentiment_df[sentiment_df['sentiment'] == 'neutral']) if not sentiment_df.empty else 0,
                        'articles': all_cached_news
                    }
            else:
                # Not enough cached articles - fetch fresh
                print(f"  → Not enough cached articles, fetching latest news...")
                sentiment_analysis = self.news_scraper.get_aggregate_sentiment(symbol, max_articles=10)
        else:
            # No cache - just fetch fresh
            sentiment_analysis = self.news_scraper.get_aggregate_sentiment(symbol, max_articles=10)
        
        # Broker analysis (optional but recommended)
        broker_analysis = None
        broker_score = 0.5  # Neutral default
        
        if include_broker_analysis:
            try:
                print("🏢 Analyzing broker activity...")
                floorsheet = self.data_fetcher.get_floorsheet_data(symbol, days=30)
                if not floorsheet.empty:
                    broker_analysis = self.broker_analyzer.comprehensive_broker_analysis(
                        floorsheet, symbol, days=30
                    )
                    # Normalize broker score (0-1)
                    broker_score = broker_analysis['overall_broker_score']['overall_score'] / 100
                else:
                    print("⚠️ No floorsheet data available - using neutral broker score")
            except Exception as e:
                print(f"⚠️ Broker analysis skipped: {e}")
        
        # Calculate individual scores
        technical_score = technical_analysis.get('technical_score', 0) / 100
        fundamental_score = fundamental_analysis.get('overall_score', 0) / 100
        sentiment_score = self._normalize_sentiment_score(sentiment_analysis)
        momentum_score = self._calculate_momentum_score(price_data)
        
        # Check if we have sentiment data
        has_sentiment_data = sentiment_analysis.get('total_articles', 0) > 0
        
        # Adjust weights if broker analysis is included
        if include_broker_analysis and broker_analysis:
            # Reduce other weights slightly to accommodate broker analysis
            adjusted_weights = {
                'technical': self.weights['technical'] * 0.85,
                'fundamental': self.weights['fundamental'] * 0.85,
                'sentiment': self.weights['sentiment'] * 0.85 if has_sentiment_data else 0,
                'momentum': self.weights['momentum'] * 0.85,
                'broker': 0.15  # 15% weight for broker analysis
            }
        else:
            adjusted_weights = self.weights.copy()
            adjusted_weights['broker'] = 0
            # If no sentiment data, set sentiment weight to 0
            if not has_sentiment_data:
                adjusted_weights['sentiment'] = 0
        
        # Normalize weights if sentiment is excluded
        if not has_sentiment_data:
            total_weight = sum(adjusted_weights.values())
            if total_weight > 0:
                adjusted_weights = {k: v/total_weight for k, v in adjusted_weights.items()}
        
        # Weighted combined probability
        probability = (
            technical_score * adjusted_weights['technical'] +
            fundamental_score * adjusted_weights['fundamental'] +
            sentiment_score * adjusted_weights['sentiment'] +
            momentum_score * adjusted_weights['momentum'] +
            broker_score * adjusted_weights.get('broker', 0)
        ) * 100
        
        # Apply manipulation risk penalty
        if broker_analysis:
            manipulation_risk = broker_analysis['manipulation_risk']['risk_score']
            if manipulation_risk >= 70:
                # Critical manipulation risk - heavy penalty
                probability *= 0.5
                print(f"⚠️ CRITICAL: Manipulation risk detected - probability reduced by 50%")
            elif manipulation_risk >= 50:
                # High risk - moderate penalty
                probability *= 0.75
                print(f"⚠️ WARNING: High manipulation risk - probability reduced by 25%")
        
        # Adjust based on time horizon
        probability = self._adjust_for_time_horizon(probability, time_horizon, price_data)
        
        # Calculate risk-reward ratio
        risk_reward = self._calculate_risk_reward(
            current_price, 
            technical_analysis.get('support_resistance', {}),
            price_data
        )
        
        # ML price predictions (if available) - Generate BEFORE recommendation
        ml_predictions = None
        if self.enable_ml and ML_AVAILABLE and self.ml_predictor:
            try:
                print("🔮 Generating ML price predictions...")
                
                # Check if we should reuse existing model
                if reuse_ml_model:
                    # Try to load existing model
                    if self.ml_predictor.load_model(symbol):
                        print(f"  ✓ Loaded existing ML model for {symbol}")
                        ml_predictions = self.ml_predictor.predict_future_prices(price_data, days=[1, 2, 3, 4, 5, 6, 7])
                    else:
                        print(f"  ℹ️ No existing model found, training new model...")
                        reuse_ml_model = False  # Force training
                
                # Train new model if not reusing or load failed
                if not reuse_ml_model:
                    self.ml_predictor.train_model(price_data, epochs=30, batch_size=32, validation_split=0.15)
                    ml_predictions = self.ml_predictor.predict_future_prices(price_data, days=[1, 2, 3, 4, 5, 6, 7])
                    # Save the trained model for this stock
                    self.ml_predictor.save_model(symbol)
                
                # Add trend analysis
                if ml_predictions and 'error' not in ml_predictions:
                    trend_analysis = self.ml_predictor.get_trend_analysis(ml_predictions)
                    ml_predictions['trend_analysis'] = trend_analysis
                    print(f"  ✓ ML predictions: {trend_analysis.get('overall_trend', 'N/A')} trend")
                else:
                    ml_predictions = None
                    
            except Exception as e:
                print(f"  ⚠️ ML prediction failed: {e}")
                ml_predictions = None
        
        # Generate recommendation (now has access to ml_predictions)
        recommendation = self._generate_recommendation(
            probability, 
            risk_reward, 
            technical_analysis,
            fundamental_analysis,
            sentiment_analysis,
            broker_analysis,
            ml_predictions  # Pass ML predictions for trend consideration
        )
        
        # Calculate position size
        position_size = self._calculate_position_size(
            probability, 
            risk_reward,
            broker_analysis
        )
        
        # Identify entry and exit points
        entry_exit = self._identify_entry_exit_points(
            current_price,
            technical_analysis,
            risk_reward
        )
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'profitability_probability': round(probability, 2),
            'confidence_level': self._get_confidence_level(probability),
            'time_horizon': time_horizon,
            'recommendation': recommendation,
            'risk_reward_ratio': risk_reward,
            'position_size': position_size,
            'entry_points': entry_exit['entry'],
            'exit_points': entry_exit['exit'],
            'stop_loss': entry_exit['stop_loss'],
            'take_profit': entry_exit['take_profit'],
            'scores': {
                'technical': round(technical_score * 100, 2),
                'fundamental': round(fundamental_score * 100, 2),
                'sentiment': round(sentiment_score * 100, 2),
                'momentum': round(momentum_score * 100, 2)
            },
            'key_insights': self._generate_key_insights(
                technical_analysis,
                fundamental_analysis,
                sentiment_analysis,
                price_data
            ),
            'warnings': self._identify_warnings(
                technical_analysis,
                fundamental_analysis,
                sentiment_analysis
            ),
            'analysis_timestamp': datetime.now().isoformat(),
            # Additional data for visualization
            'component_scores': {
                'technical': round(technical_score * 100, 2),
                'fundamental': round(fundamental_score * 100, 2),
                'sentiment': round(sentiment_score * 100, 2),
                'momentum': round(momentum_score * 100, 2),
                'broker': round(broker_score * 100, 2)
            },
            'weights_used': adjusted_weights,
            'technical_analysis': technical_analysis,
            'fundamental_analysis': fundamental_analysis,  # Save complete fundamental analysis
            'sentiment_details': sentiment_analysis,
            'broker_analysis': broker_analysis,
            'ml_predictions': ml_predictions,  # ML price predictions
            'data_points': len(price_data),
            'broker_trades': len(self.data_fetcher.get_floorsheet_data(symbol, days=30)) if include_broker_analysis else 0,
            'probability': round(probability, 2),
            'potential_profit_pct': round((risk_reward['potential_profit'] / current_price * 100), 2) if risk_reward.get('potential_profit') else 0,
            'potential_loss_pct': round((risk_reward['potential_loss'] / current_price * 100), 2) if risk_reward.get('potential_loss') else 0,
        }
    
    def _analyze_cached_sentiment(self, cached_news: List[Dict]) -> Dict:
        """Analyze sentiment from cached news articles"""
        from datetime import datetime, timedelta
        
        if not cached_news:
            return {'overall_sentiment': 0.5, 'sentiment_label': 'NEUTRAL', 'articles_analyzed': 0}
        
        # Calculate weighted sentiment with date-based weighting
        weighted_scores = []
        now = datetime.now()
        
        for article in cached_news:
            score = article.get('sentiment_score', 0)
            pub_date = article.get('published_date', '')
            
            # Calculate recency weight
            try:
                if pub_date:
                    article_date = datetime.strptime(pub_date, '%Y-%m-%d')
                    days_old = (now - article_date).days
                    recency_weight = max(0.1, 1.0 - (days_old / 180))  # Linear decay over 180 days
                else:
                    recency_weight = 0.5
            except:
                recency_weight = 0.5
            
            weighted_scores.append(score * recency_weight)
        
        # Calculate overall sentiment
        avg_sentiment = np.mean(weighted_scores) if weighted_scores else 0.5
        
        # Determine label
        if avg_sentiment > 0.1:
            label = 'POSITIVE'
        elif avg_sentiment < -0.1:
            label = 'NEGATIVE'
        else:
            label = 'NEUTRAL'
        
        # Convert to 0-100 scale
        sentiment_score = ((avg_sentiment + 1) / 2) * 100
        
        return {
            'overall_sentiment': sentiment_score / 100,
            'sentiment_label': label,
            'articles_analyzed': len(cached_news),
            'total_articles': len(cached_news),
            'positive_articles': sum(1 for a in cached_news if a.get('sentiment_score', 0) > 0.1),
            'negative_articles': sum(1 for a in cached_news if a.get('sentiment_score', 0) < -0.1),
            'neutral_articles': sum(1 for a in cached_news if -0.1 <= a.get('sentiment_score', 0) <= 0.1),
            'articles': cached_news
        }
    
    def _get_fundamental_data(self, symbol: str, current_price: float) -> Dict:
        """Get real fundamental data from NepalAlpha or use estimates"""
        try:
            print("  → Fetching fundamental data from NepalAlpha...")
            real_data = self.fundamental_scraper.scrape_fundamental_data(symbol)
            
            # Close the browser immediately after scraping to avoid "late close" issue
            self.fundamental_scraper.close()
            
            # If we got real data, use it and fill in missing values with estimates
            if real_data and real_data.get('pe_ratio'):
                print(f"  ✓ Using real fundamental data from NepalAlpha")
                
                # Calculate derived values if missing
                # Use the current_price from our database, not from NepalAlpha
                real_data['current_price'] = current_price
                
                eps = real_data.get('eps')
                if not eps and real_data.get('pe_ratio') and real_data.get('pe_ratio') > 0:
                    eps = current_price / real_data['pe_ratio']
                    real_data['eps'] = eps
                
                book_value = real_data.get('book_value_per_share')
                if not book_value and real_data.get('pb_ratio') and real_data.get('pb_ratio') > 0:
                    book_value = current_price / real_data['pb_ratio']
                    real_data['book_value_per_share'] = book_value
                
                # Fill missing values with estimates
                real_data['symbol'] = symbol
                real_data['annual_dividend'] = real_data.get('annual_dividend') or 0
                real_data['eps'] = real_data.get('eps') or (current_price * 0.05)
                real_data['previous_eps'] = real_data.get('previous_eps') or (real_data['eps'] * 0.9)
                real_data['book_value_per_share'] = real_data.get('book_value_per_share') or (current_price * 0.4)
                real_data['net_income'] = real_data.get('net_income') or 5000000
                real_data['shareholders_equity'] = real_data.get('shareholders_equity') or 25000000
                real_data['total_debt'] = real_data.get('total_debt') or 10000000
                real_data['current_assets'] = real_data.get('current_assets') or 50000000
                real_data['current_liabilities'] = real_data.get('current_liabilities') or 30000000
                real_data['market_cap'] = real_data.get('market_cap') or (current_price * 10000000)
                real_data['is_estimated'] = False  # Real data from NepalAlpha
                
                return real_data
            else:
                print("  ⚠️ Could not fetch real data, using estimates")
                raise Exception("No real data available")
                
        except Exception as e:
            print(f"  ⚠️ Using estimated fundamental data: {e}")
            # Fallback to estimates
            return {
                'symbol': symbol,
                'current_price': current_price,
                'eps': current_price * 0.05,
                'previous_eps': current_price * 0.045,
                'book_value_per_share': current_price * 0.4,
                'annual_dividend': current_price * 0.03,
                'net_income': 5000000,
                'shareholders_equity': 25000000,
                'total_debt': 10000000,
                'current_assets': 50000000,
                'current_liabilities': 30000000,
                'market_cap': current_price * 10000000,
                'is_estimated': True  # Estimated data (fallback)
            }
    
    def _normalize_sentiment_score(self, sentiment_analysis: Dict) -> float:
        """Convert sentiment analysis to 0-1 score"""
        sentiment = sentiment_analysis.get('overall_sentiment', 'neutral')
        avg_score = sentiment_analysis.get('average_score', 0)
        article_count = sentiment_analysis.get('total_articles', 0)
        
        # If no articles, return neutral score with very low confidence
        if article_count == 0:
            return 0.5  # Neutral, but this will be weighted low in final calculation
        
        # Convert -1 to 1 scale to 0 to 1 scale
        normalized = (avg_score + 1) / 2
        
        # Weight by number of articles
        confidence_factor = min(article_count / 10, 1.0)  # Max confidence at 10+ articles
        
        return normalized * (0.7 + 0.3 * confidence_factor)
    
    def _calculate_momentum_score(self, price_data: pd.DataFrame) -> float:
        """Calculate momentum score from price data"""
        if len(price_data) < 20:
            return 0.5
        
        close = price_data['close']
        
        # Calculate returns over different periods
        return_5d = (close.iloc[-1] - close.iloc[-6]) / close.iloc[-6] if len(close) > 5 else 0
        return_20d = (close.iloc[-1] - close.iloc[-21]) / close.iloc[-21] if len(close) > 20 else 0
        
        # Volatility
        volatility = close.pct_change().std()
        
        # Combine factors
        momentum = (return_5d * 0.4 + return_20d * 0.4 - volatility * 0.2)
        
        # Normalize to 0-1
        normalized = (np.tanh(momentum * 10) + 1) / 2
        
        return max(0, min(1, normalized))
    
    def _adjust_for_time_horizon(self, probability: float, time_horizon: str, 
                                 price_data: pd.DataFrame) -> float:
        """Adjust probability based on time horizon"""
        volatility = price_data['close'].pct_change().std()
        
        adjustments = {
            'short': -5 * volatility * 100,  # Short term more affected by volatility
            'medium': -2 * volatility * 100,
            'long': 0  # Long term less affected
        }
        
        adjusted = probability + adjustments.get(time_horizon, 0)
        return max(0, min(100, adjusted))
    
    def _calculate_risk_reward(self, current_price: float, 
                               support_resistance: Dict,
                               price_data: pd.DataFrame) -> Dict:
        """Calculate risk-reward ratio"""
        
        # Get support and resistance levels
        support_levels = support_resistance.get('support', [])
        resistance_levels = support_resistance.get('resistance', [])
        
        # Find nearest support and resistance
        nearest_support = max([s for s in support_levels if s < current_price], 
                            default=current_price * 0.95)
        nearest_resistance = min([r for r in resistance_levels if r > current_price], 
                                default=current_price * 1.05)
        
        # Calculate potential profit and loss
        potential_profit = nearest_resistance - current_price
        potential_loss = current_price - nearest_support
        
        # Risk-reward ratio
        ratio = potential_profit / potential_loss if potential_loss > 0 else 0
        
        return {
            'ratio': round(ratio, 2),
            'potential_profit_percent': round((potential_profit / current_price) * 100, 2),
            'potential_loss_percent': round((potential_loss / current_price) * 100, 2),
            'nearest_support': round(nearest_support, 2),
            'nearest_resistance': round(nearest_resistance, 2)
        }
    
    def _generate_recommendation(self, probability: float, risk_reward: Dict,
                                technical_analysis: Dict, fundamental_analysis: Dict,
                                sentiment_analysis: Dict, broker_analysis: Dict = None,
                                ml_predictions: Dict = None) -> Dict:
        """Generate detailed trading recommendation"""
        
        rr_ratio = risk_reward.get('ratio', 0)
        
        # Check ML predictions trend if available
        ml_trend_bearish = False
        ml_avg_change = 0
        if ml_predictions and 'trend_analysis' in ml_predictions:
            trend = ml_predictions['trend_analysis']
            ml_avg_change = trend.get('avg_predicted_change', 0)
            overall_trend = trend.get('overall_trend', '')
            
            # If ML predicts consistent decline, adjust recommendation
            if ml_avg_change < -2 or 'DOWN' in overall_trend.upper():
                ml_trend_bearish = True
        
        # Check broker manipulation risk first
        manipulation_warning = False
        if broker_analysis:
            manip_risk = broker_analysis['manipulation_risk']['risk_score']
            if manip_risk >= 70:
                # Override recommendation due to manipulation risk
                return {
                    'action': 'AVOID - High Manipulation Risk',
                    'confidence': 'Critical',
                    'reasoning': [
                        '🚨 CRITICAL: High broker manipulation risk detected',
                        f"Manipulation risk score: {manip_risk}/100",
                        *broker_analysis['manipulation_risk']['risk_flags'][:3],
                        'Recommendation: Do not trade this stock'
                    ]
                }
            elif manip_risk >= 50:
                manipulation_warning = True
        
        # Determine action - adjust for ML trend
        if ml_trend_bearish:
            # Downgrade recommendation if ML predicts decline
            if probability >= 70 and rr_ratio >= 2:
                action = 'HOLD' if ml_avg_change < -5 else 'BUY'
                confidence = 'Medium'
            elif probability >= 60 and rr_ratio >= 1.5:
                action = 'HOLD'
                confidence = 'Medium'
            elif probability >= 50:
                action = 'SELL'
                confidence = 'Medium-High'
            else:
                action = 'STRONG SELL'
                confidence = 'High'
        else:
            # Original logic when ML predicts sideways/up or not available
            if probability >= 70 and rr_ratio >= 2 and not manipulation_warning:
                action = 'STRONG BUY'
                confidence = 'High'
            elif probability >= 60 and rr_ratio >= 1.5 and not manipulation_warning:
                action = 'BUY'
                confidence = 'Medium-High'
            elif probability >= 50 and rr_ratio >= 1:
                action = 'HOLD/ACCUMULATE'
                confidence = 'Medium'
            elif probability >= 40:
                action = 'HOLD'
                confidence = 'Low-Medium'
            elif probability >= 30:
                action = 'SELL'
                confidence = 'Medium-High'
            else:
                action = 'STRONG SELL'
                confidence = 'High'
        
        # Generate reasoning
        reasoning = []
        
        # Add ML prediction insight first if available
        if ml_predictions and 'trend_analysis' in ml_predictions:
            trend = ml_predictions['trend_analysis']
            ml_trend = trend.get('overall_trend', 'N/A')
            ml_change = trend.get('avg_predicted_change', 0)
            
            if ml_change < -2:
                reasoning.append(f"⚠️ ML predicts {ml_trend} trend ({ml_change:+.1f}% avg)")
            elif ml_change > 2:
                reasoning.append(f"✓ ML predicts {ml_trend} trend ({ml_change:+.1f}% avg)")
            else:
                reasoning.append(f"→ ML predicts {ml_trend} trend ({ml_change:+.1f}% avg)")
        
        if technical_analysis.get('technical_score', 0) >= 70:
            reasoning.append("✓ Strong technical indicators")
        if fundamental_analysis.get('overall_score', 0) >= 70:
            reasoning.append("✓ Solid fundamentals")
        if sentiment_analysis.get('overall_sentiment') == 'positive':
            reasoning.append("✓ Positive market sentiment")
        if rr_ratio >= 2:
            reasoning.append("✓ Favorable risk-reward ratio")
        
        # Add broker analysis insights
        if broker_analysis:
            broker_score = broker_analysis['overall_broker_score']['overall_score']
            if broker_score >= 80:
                reasoning.append("✓ Excellent broker activity and liquidity")
            elif broker_score >= 60:
                reasoning.append("✓ Good broker activity")
            
            smart_money = broker_analysis['smart_money_flow']['smart_money_signal']
            if 'BULLISH' in smart_money:
                reasoning.append("✓ Institutional buyers accumulating")
            elif 'BEARISH' in smart_money:
                reasoning.append("⚠️ Institutional sellers distributing")
            
            liquidity = broker_analysis['liquidity_metrics']['liquidity_score']
            if liquidity >= 80:
                reasoning.append("✓ High liquidity - safe for trading")
            elif liquidity < 40:
                reasoning.append("⚠️ Low liquidity - trade with caution")
            
            if manipulation_warning:
                reasoning.append("⚠️ WARNING: Moderate manipulation risk detected")
        
        if not reasoning:
            reasoning.append("⚠️ Mixed signals across indicators")
        
        return {
            'action': action,
            'confidence': confidence,
            'reasoning': reasoning
        }
    
    def _calculate_position_size(self, probability: float, risk_reward: Dict, 
                                 broker_analysis: Dict = None) -> Dict:
        """Calculate recommended position size"""
        
        # Base position size from config
        max_position = self.risk_config['max_position_size']
        
        # Adjust based on probability and risk-reward
        probability_factor = probability / 100
        rr_factor = min(risk_reward.get('ratio', 1) / 3, 1)
        
        recommended_size = max_position * probability_factor * rr_factor
        
        # Adjust for liquidity if broker analysis available
        if broker_analysis:
            liquidity_score = broker_analysis['liquidity_metrics']['liquidity_score']
            
            # Reduce position size for low liquidity
            if liquidity_score < 40:
                recommended_size *= 0.5  # Halve position for low liquidity
                reasoning = f"Based on {probability:.1f}% probability, {risk_reward.get('ratio', 0):.2f} R:R ratio, reduced due to low liquidity"
            elif liquidity_score < 60:
                recommended_size *= 0.75  # Reduce by 25%
                reasoning = f"Based on {probability:.1f}% probability, {risk_reward.get('ratio', 0):.2f} R:R ratio, slightly reduced for moderate liquidity"
            else:
                reasoning = f"Based on {probability:.1f}% probability, {risk_reward.get('ratio', 0):.2f} R:R ratio, good liquidity"
        else:
            reasoning = f"Based on {probability:.1f}% probability and {risk_reward.get('ratio', 0):.2f} R:R ratio"
        
        return {
            'recommended_percent': round(recommended_size * 100, 2),
            'max_percent': round(max_position * 100, 2),
            'reasoning': reasoning
        }
    
    def _identify_entry_exit_points(self, current_price: float,
                                    technical_analysis: Dict,
                                    risk_reward: Dict) -> Dict:
        """Identify entry and exit points"""
        
        sr = technical_analysis.get('support_resistance', {})
        indicators = technical_analysis.get('indicators', {})
        
        # Entry points
        entry_aggressive = current_price
        entry_conservative = sr.get('nearest_support', current_price * 0.98)
        
        # Exit points (take profit)
        tp1 = risk_reward.get('nearest_resistance', current_price * 1.05)
        tp2 = current_price * (1 + self.risk_config['take_profit'])
        
        # Stop loss
        sl = current_price * (1 - self.risk_config['stop_loss'])
        nearest_support = sr.get('nearest_support', sl)
        stop_loss = max(sl, nearest_support * 0.98)
        
        return {
            'entry': {
                'aggressive': round(entry_aggressive, 2),
                'conservative': round(entry_conservative, 2)
            },
            'exit': {
                'target_1': round(tp1, 2),
                'target_2': round(tp2, 2)
            },
            'stop_loss': round(stop_loss, 2),
            'take_profit': round(tp2, 2)
        }
    
    def _generate_key_insights(self, technical_analysis: Dict,
                              fundamental_analysis: Dict,
                              sentiment_analysis: Dict,
                              price_data: pd.DataFrame) -> List[str]:
        """Generate key insights from all analyses"""
        insights = []
        
        # Technical insights
        tech_score = technical_analysis.get('technical_score', 0)
        if tech_score >= 70:
            insights.append(f"✅ Strong technical setup (Score: {tech_score}/100)")
        elif tech_score <= 30:
            insights.append(f"⚠️ Weak technical indicators (Score: {tech_score}/100)")
        
        # Trend insight
        trend = technical_analysis.get('signals', {}).get('trend', 'Unknown')
        insights.append(f"📊 Current trend: {trend}")
        
        # Fundamental insights
        fund_score = fundamental_analysis.get('overall_score', 0)
        if fund_score >= 70:
            insights.append(f"✅ Strong fundamentals (Score: {fund_score}/100)")
        
        strengths = fundamental_analysis.get('strengths', [])
        if strengths and strengths[0] != 'No significant strengths identified':
            insights.append(f"💪 {strengths[0]}")
        
        # Sentiment insight
        sentiment_label = sentiment_analysis.get('sentiment_label', 'NEUTRAL')
        article_count = sentiment_analysis.get('total_articles', 0)
        insights.append(f"📰 Market sentiment: {sentiment_label} ({article_count} articles)")
        
        # Momentum insight
        price_change = ((price_data['close'].iloc[-1] - price_data['close'].iloc[-6]) / 
                       price_data['close'].iloc[-6] * 100) if len(price_data) > 5 else 0
        
        direction = "up" if price_change > 0 else "down"
        insights.append(f"📈 5-day momentum: {direction} {abs(price_change):.2f}%")
        
        return insights
    
    def _identify_warnings(self, technical_analysis: Dict,
                          fundamental_analysis: Dict,
                          sentiment_analysis: Dict) -> List[str]:
        """Identify potential warnings and risks"""
        warnings = []
        
        # Technical warnings
        rsi = technical_analysis.get('indicators', {}).get('rsi', 50)
        if rsi > 70:
            warnings.append("⚠️ RSI indicates overbought conditions")
        elif rsi < 30:
            warnings.append("⚠️ RSI indicates oversold conditions")
        
        # Fundamental warnings
        weaknesses = fundamental_analysis.get('weaknesses', [])
        if weaknesses and weaknesses[0] != 'No significant weaknesses identified':
            warnings.append(f"⚠️ {weaknesses[0]}")
        
        # Sentiment warnings
        if sentiment_analysis.get('sentiment_trend') == 'declining':
            warnings.append("⚠️ Market sentiment is deteriorating")
        
        # Pattern warnings
        patterns = technical_analysis.get('patterns', [])
        for pattern in patterns:
            if pattern['type'] == 'Bearish':
                warnings.append(f"⚠️ Bearish pattern detected: {pattern['pattern']}")
        
        if not warnings:
            warnings.append("✅ No major warnings identified")
        
        return warnings
    
    def _get_confidence_level(self, probability: float) -> str:
        """Get confidence level description"""
        if probability >= 75:
            return 'Very High'
        elif probability >= 60:
            return 'High'
        elif probability >= 40:
            return 'Medium'
        elif probability >= 25:
            return 'Low'
        else:
            return 'Very Low'
    
    def batch_analysis(self, symbols: List[str], time_horizon: str = 'short', 
                      use_cache: bool = True, save_to_csv: bool = False) -> pd.DataFrame:
        """
        Analyze multiple stocks and rank by profitability probability
        
        Args:
            symbols: List of stock symbols
            time_horizon: Time horizon for analysis
            use_cache: Use cached data for faster analysis
            save_to_csv: Save results to CSV file
        """
        results = []
        
        print(f"\n{'='*60}")
        print(f"BATCH ANALYSIS: {len(symbols)} stocks")
        print(f"{'='*60}\n")
        
        for i, symbol in enumerate(symbols, 1):
            try:
                print(f"[{i}/{len(symbols)}] {symbol}...")
                analysis = self.calculate_profitability_probability(symbol, time_horizon, 
                                                                    include_broker_analysis=False,
                                                                    use_cache=use_cache)
                
                if 'error' not in analysis:
                    # Extract ML prediction if available
                    ml_pred = None
                    if analysis.get('ml_predictions') and 'horizons' in analysis['ml_predictions']:
                        week1 = analysis['ml_predictions']['horizons'].get('1_week', {})
                        ml_pred = week1.get('price_change_pct', 0)
                    
                    results.append({
                        'Symbol': analysis['symbol'],
                        'Price': analysis['current_price'],
                        'Probability': analysis['profitability_probability'],
                        'Confidence': analysis['confidence_level'],
                        'Recommendation': analysis['recommendation']['action'],
                        'R:R Ratio': analysis['risk_reward_ratio']['ratio'],
                        'Technical': analysis['scores']['technical'],
                        'Fundamental': analysis['scores']['fundamental'],
                        'Sentiment': analysis['scores']['sentiment'],
                        'ML_1Week%': ml_pred if ml_pred else 0
                    })
            except Exception as e:
                print(f"  ✗ Error analyzing {symbol}: {e}")
                continue
        
        df = pd.DataFrame(results)
        df = df.sort_values('Probability', ascending=False)
        
        if save_to_csv:
            filename = f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False)
            print(f"\n✓ Results saved to: {filename}")
        
        return df
    
    def analyze_all_nepse_stocks(self, time_horizon: str = 'short', 
                                save_results: bool = True) -> pd.DataFrame:
        """
        Analyze all stocks in NEPSE database
        
        Args:
            time_horizon: Time horizon for analysis
            save_results: Save results to CSV
        """
        # Get all unique symbols with price data
        conn = sqlite3.connect(config.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT symbol FROM price_history")
        symbols = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        print(f"\n🔍 Found {len(symbols)} stocks in database")
        
        return self.batch_analysis(symbols, time_horizon, use_cache=True, save_to_csv=save_results)


if __name__ == "__main__":
    # Test the trading insights engine
    engine = TradingInsightsEngine()
    
    # Analyze a single stock
    symbol = "NABIL"
    print(f"\n{'='*60}")
    print(f"TRADING INSIGHTS FOR {symbol}")
    print(f"{'='*60}")
    
    result = engine.calculate_profitability_probability(symbol, time_horizon='short')
    
    if 'error' not in result:
        print(f"\n💰 Current Price: NPR {result['current_price']:.2f}")
        print(f"📊 Profitability Probability: {result['profitability_probability']:.2f}%")
        print(f"🎯 Confidence Level: {result['confidence_level']}")
        print(f"📈 Recommendation: {result['recommendation']['action']}")
        print(f"💡 Risk-Reward Ratio: {result['risk_reward_ratio']['ratio']:.2f}")
        
        print(f"\n📍 Entry Points:")
        print(f"   Aggressive: NPR {result['entry_points']['aggressive']:.2f}")
        print(f"   Conservative: NPR {result['entry_points']['conservative']:.2f}")
        
        print(f"\n🎯 Exit Points:")
        print(f"   Target 1: NPR {result['exit_points']['target_1']:.2f}")
        print(f"   Target 2: NPR {result['exit_points']['target_2']:.2f}")
        print(f"   Stop Loss: NPR {result['stop_loss']:.2f}")
        
        print(f"\n💡 Key Insights:")
        for insight in result['key_insights']:
            print(f"   {insight}")
        
        print(f"\n⚠️ Warnings:")
        for warning in result['warnings']:
            print(f"   {warning}")
    else:
        print(f"\n❌ Error: {result['error']}")
