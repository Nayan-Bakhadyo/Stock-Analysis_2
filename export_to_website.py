"""
Simple analysis and export to website
Scrapes data for specified companies and exports to static website
"""

import sys
import time
from sharesansar_price_scraper import ShareSansarPriceScraper
from sharesansar_news_scraper import ShareSansarNewsScraper
from nepsealpha_scraper import NepseAlphaScraper
from website_generator import WebsiteGenerator
from datetime import datetime
import pandas as pd


def scrape_and_analyze(symbols):
    """Scrape data and perform simple analysis for given symbols"""
    
    price_scraper = ShareSansarPriceScraper(headless=True)
    news_scraper = ShareSansarNewsScraper(headless=True)
    generator = WebsiteGenerator()
    
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
            technical_score = 50 + (price_change_30d * 2)  # Base 50, adjust by price change
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
            
            # Simple ML predictions (placeholder - based on recent trend)
            predictions = {
                '1_week': {
                    'change': price_change_30d * 0.25,  # 1/4 of monthly trend
                    'confidence': 0.70
                },
                '2_week': {
                    'change': price_change_30d * 0.5,  # 1/2 of monthly trend
                    'confidence': 0.65
                },
                '4_week': {
                    'change': price_change_30d,
                    'confidence': 0.60
                }
            }
            
            ml_trend = 'BULLISH' if price_change_30d > 0 else 'BEARISH'
            avg_confidence = sum(p['confidence'] for p in predictions.values()) / len(predictions)
            
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
                'ml_predictions': {
                    'overall_trend': ml_trend,
                    'avg_confidence': avg_confidence,
                    'predictions': predictions
                }
            }
            
            # Add to website generator
            generator.add_analysis(symbol, analysis_result)
            
            print(f"\n✓ Analysis complete for {symbol}")
            print(f"  • Profitability Score: {profitability_score:.1f}%")
            print(f"  • Technical Score: {technical_score:.1f}%")
            print(f"  • News Sentiment: {news_sentiment}")
            print(f"  • ML Trend: {ml_trend}")
            
            time.sleep(2)  # Rate limiting
            
        except Exception as e:
            print(f"❌ Error processing {symbol}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Close scrapers
    news_scraper.close_driver()
    
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
    
    print(f"\n🚀 Starting analysis for: {', '.join(symbols)}")
    scrape_and_analyze(symbols)
    print(f"\n✅ Complete! Website ready in 'docs/' directory")
    print(f"   Open docs/index.html in your browser to view")
