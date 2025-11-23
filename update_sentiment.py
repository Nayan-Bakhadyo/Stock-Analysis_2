"""
Daily Sentiment Analysis Update
- Incremental sync of news articles
- Sentiment analysis on new articles
- Updates sentiment scores
"""
import json
import os
from datetime import datetime
from sync_manager import SyncManager
from sharesansar_news_scraper import ShareSansarNewsScraper
from stock_tracker import StockTracker


def analyze_sentiment(news_data):
    """
    Simple sentiment analysis on news articles
    Returns: score between -1 (negative) to 1 (positive)
    """
    if not news_data:
        return 0.0
    
    # Keywords for sentiment analysis
    positive_keywords = [
        'profit', 'growth', 'dividend', 'expansion', 'increase', 'gain', 'success',
        'strong', 'positive', 'improve', 'bonus', 'achievement', 'record', 'high',
        'surge', 'rally', 'bullish', 'optimistic', 'upgrade'
    ]
    
    negative_keywords = [
        'loss', 'decline', 'decrease', 'fall', 'drop', 'negative', 'concern',
        'weak', 'problem', 'issue', 'down', 'low', 'bearish', 'pessimistic',
        'downgrade', 'risk', 'debt', 'crisis', 'warning'
    ]
    
    total_score = 0
    article_count = 0
    
    for article in news_data:
        title = article.get('title', '').lower()
        description = article.get('description', '').lower()
        combined_text = f"{title} {description}"
        
        # Count positive and negative keywords
        pos_count = sum(1 for word in positive_keywords if word in combined_text)
        neg_count = sum(1 for word in negative_keywords if word in combined_text)
        
        # Calculate article sentiment (-1 to 1)
        if pos_count > 0 or neg_count > 0:
            article_sentiment = (pos_count - neg_count) / (pos_count + neg_count)
        else:
            article_sentiment = 0
        
        total_score += article_sentiment
        article_count += 1
    
    # Average sentiment
    return total_score / article_count if article_count > 0 else 0.0


def update_sentiment_analysis(symbols=None, max_articles=10):
    """
    Update sentiment analysis for all analyzed stocks
    - Incremental sync of news articles
    - Sentiment analysis on new articles
    
    Args:
        symbols: List of symbols to update (None = update all from analysis_results.json)
        max_articles: Maximum articles to fetch per stock
    """
    print("\n" + "="*70)
    print("SENTIMENT ANALYSIS UPDATE (Incremental)")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Get symbols to update
    if symbols is None:
        try:
            with open('analysis_results.json', 'r') as f:
                existing_results = json.load(f)
            
            # Filter out failed stocks
            symbols = [r['symbol'] for r in existing_results if not r.get('error')]
            
            print(f"Found {len(symbols)} stocks to update")
            print("="*70 + "\n")
            
        except FileNotFoundError:
            print("❌ No analysis_results.json found. Provide symbols manually.")
            return
    
    # Initialize
    sync_manager = SyncManager()
    tracker = StockTracker()
    
    # Track statistics
    success_count = 0
    error_count = 0
    total_articles_added = 0
    sentiment_results = {}
    
    # Update each stock
    for i, symbol in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] Updating sentiment: {symbol}")
        
        try:
            # Sync news (incremental)
            result = sync_manager.sync_news(symbol, max_articles=max_articles)
            
            print(f"  ✅ Status: {result['status']}")
            print(f"  ✅ Articles added: {result.get('articles_added', 0)}")
            print(f"  ✅ Total articles: {result.get('total_articles', 0)}")
            
            # Get all cached news for sentiment analysis
            cached_news = sync_manager.get_cached_news(symbol, days=180, limit=50)
            
            # Analyze sentiment
            sentiment_score = analyze_sentiment(cached_news)
            
            # Classify sentiment
            if sentiment_score > 0.2:
                sentiment_label = "Positive"
            elif sentiment_score < -0.2:
                sentiment_label = "Negative"
            else:
                sentiment_label = "Neutral"
            
            print(f"  📊 Sentiment: {sentiment_label} (score: {sentiment_score:.2f})")
            
            sentiment_results[symbol] = {
                'sentiment_score': sentiment_score,
                'sentiment_label': sentiment_label,
                'total_articles': len(cached_news),
                'timestamp': datetime.now().isoformat()
            }
            
            total_articles_added += result.get('articles_added', 0)
            success_count += 1
            
            # Mark as processed
            tracker.mark_processed(symbol, status='success')
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)[:80]}...")
            tracker.mark_processed(symbol, status='failed', error=str(e))
            error_count += 1
    
    # Save sentiment results
    output_file = 'sentiment_results.json'
    with open(output_file, 'w') as f:
        json.dump(sentiment_results, f, indent=2)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SENTIMENT ANALYSIS UPDATE COMPLETE")
    print(f"{'='*70}")
    print(f"Total stocks processed: {len(symbols)}")
    print(f"Successful: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Total new articles added: {total_articles_added}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}\n")
    
    # Print tracker summary
    tracker.print_summary()


if __name__ == '__main__':
    import sys
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        symbols = [s.strip().upper() for s in sys.argv[1:]]
        update_sentiment_analysis(symbols)
    else:
        # Update all stocks from analysis_results.json
        update_sentiment_analysis()
