"""
Export to website with RL signals integration
Enhanced version that includes RL trading signals
"""

import sys
import time
from sharesansar_price_scraper import ShareSansarPriceScraper
from sharesansar_news_scraper import ShareSansarNewsScraper
from website_generator import WebsiteGenerator
from rl_signal_integrator import RLSignalIntegrator
from datetime import datetime


def scrape_and_analyze_with_rl(symbols):
    """Scrape data and perform analysis with RL signals for given symbols"""
    
    price_scraper = ShareSansarPriceScraper(headless=True)
    news_scraper = ShareSansarNewsScraper(headless=True)
    generator = WebsiteGenerator()
    rl_integrator = RLSignalIntegrator()
    
    for symbol in symbols:
        print(f"\n{'='*70}")
        print(f"Processing {symbol}")
        print('='*70)
        
        try:
            # Scrape price history
            print(f"\n📈 Scraping price history for {symbol}...")
            price_data = price_scraper.scrape_price_history(symbol)
            
            if price_data is None or len(price_data) < 10:
                print(f"⚠️ Insufficient price data for {symbol}, skipping...")
                continue
            
            print(f"  ✓ Loaded {len(price_data)} days of price data")
            
            # Scrape news
            print(f"\n📰 Scraping news for {symbol}...")
            news_articles = news_scraper.scrape_company_news(symbol, max_articles=10)
            
            if news_articles:
                print(f"  ✓ Loaded {len(news_articles)} news articles")
                avg_sentiment = sum(a['sentiment_score'] for a in news_articles) / len(news_articles)
            else:
                print(f"  ⚠️ No news articles found")
                avg_sentiment = 0
            
            # Calculate simple metrics
            latest_price = price_data.iloc[-1]['close']
            avg_price_30d = price_data.tail(30)['close'].mean()
            price_change_30d = ((latest_price - avg_price_30d) / avg_price_30d) * 100
            
            avg_volume = price_data['volume'].mean()
            latest_volume = price_data.iloc[-1]['volume']
            volume_ratio = latest_volume / avg_volume if avg_volume > 0 else 1
            
            # Simple scoring
            technical_score = 50 + (price_change_30d * 2)
            technical_score = max(0, min(100, technical_score))
            
            sentiment_score = 50 + (avg_sentiment * 100)
            sentiment_score = max(0, min(100, sentiment_score))
            
            fundamental_score = 60  # Placeholder
            
            profitability_score = (technical_score * 0.4 + sentiment_score * 0.3 + fundamental_score * 0.3)
            
            # Determine sentiment
            if avg_sentiment > 0.1:
                news_sentiment = 'POSITIVE'
            elif avg_sentiment < -0.1:
                news_sentiment = 'NEGATIVE'
            else:
                news_sentiment = 'NEUTRAL'
            
            # Get RL signal (NEW!)
            print(f"\n🤖 Getting RL trading signal for {symbol}...")
            rl_signal = rl_integrator.format_for_website(symbol)
            
            if rl_signal['available']:
                print(f"  ✓ RL Signal: {rl_signal['action']}")
                print(f"  • Confidence: {rl_signal['confidence_pct']}")
                print(f"  • Reason: {rl_signal['reason']}")
                
                # Use RL signal as primary recommendation
                recommendation = rl_signal['action_raw']  # BUY, SELL, HOLD
                recommendation_strength = rl_signal['strength']  # STRONG, MODERATE, WEAK
                recommendation_confidence = rl_signal['confidence']
                recommendation_source = 'RL Agent'
            else:
                print(f"  ⚠️ RL signal not available, using trend-based signal")
                # Fallback to simple trend
                if price_change_30d > 5:
                    recommendation = 'BUY'
                    recommendation_strength = 'MODERATE'
                elif price_change_30d < -5:
                    recommendation = 'SELL'
                    recommendation_strength = 'MODERATE'
                else:
                    recommendation = 'HOLD'
                    recommendation_strength = 'WEAK'
                recommendation_confidence = 0.5
                recommendation_source = 'Trend Analysis'
            
            # Create analysis result
            analysis_result = {
                'profitability_score': profitability_score,
                'technical_score': technical_score,
                'fundamental_score': fundamental_score,
                'news_sentiment': news_sentiment,
                'latest_price': latest_price,
                'price_change_30d': price_change_30d,
                'volume': latest_volume,
                'volume_ratio': volume_ratio,
                'news_count': len(news_articles),
                
                # RL Trading Signal (PRIMARY)
                'recommendation': recommendation,
                'recommendation_strength': recommendation_strength,
                'recommendation_confidence': recommendation_confidence,
                'recommendation_source': recommendation_source,
                'rl_signal': rl_signal,  # Full RL signal data
                
                # ML predictions (for reference)
                'ml_predictions': {
                    'overall_trend': 'BULLISH' if price_change_30d > 0 else 'BEARISH',
                    'avg_confidence': 0.65
                }
            }
            
            # Add to website generator
            generator.add_analysis(symbol, analysis_result)
            
            print(f"\n✓ Analysis complete for {symbol}")
            print(f"  • Profitability Score: {profitability_score:.1f}%")
            print(f"  • Technical Score: {technical_score:.1f}%")
            print(f"  • News Sentiment: {news_sentiment}")
            print(f"  • Recommendation: {recommendation} ({recommendation_strength})")
            
            time.sleep(2)  # Rate limiting
            
        except Exception as e:
            print(f"❌ Error processing {symbol}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Close scrapers
    news_scraper.close_driver()
    
    # Show top RL opportunities
    print(f"\n{'='*70}")
    print("TOP RL OPPORTUNITIES")
    print('='*70)
    
    buy_opportunities = rl_integrator.get_top_opportunities(
        min_confidence=0.7, 
        min_sharpe=1.2, 
        action='BUY'
    )
    
    if buy_opportunities:
        print("\n🔥 Top BUY signals:")
        for i, opp in enumerate(buy_opportunities[:5], 1):
            print(f"{i}. {opp['symbol']}: {opp['confidence']:.0%} confidence | "
                  f"{opp['test_return']:.1f}% return | Sharpe: {opp['sharpe_ratio']:.2f}")
    else:
        print("  No strong BUY opportunities found")
    
    # Generate website
    print(f"\n{'='*70}")
    print("Generating website...")
    print('='*70)
    generator.generate_website()
    
    return generator


if __name__ == '__main__':
    # Get symbols from command line or use defaults
    if len(sys.argv) > 1:
        symbols = sys.argv[1:]
    else:
        # Use 3 popular banks as default
        symbols = ['NABIL', 'NICA', 'SCB']
    
    print(f"\n🚀 Starting analysis with RL signals for: {', '.join(symbols)}")
    print(f"📝 Make sure you've run RL training first:")
    print(f"   python3 rl_trading_agent.py {' '.join(symbols)}")
    print()
    
    scrape_and_analyze_with_rl(symbols)
    
    print(f"\n✅ Complete! Website ready in 'docs/' directory")
    print(f"   Open docs/index.html in your browser to view")
    print(f"   BUY/SELL signals are now powered by RL agent! 🤖")
