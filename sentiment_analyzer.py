"""Sentiment analysis module for Nepali stock news"""
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import List, Dict
import pandas as pd
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import config


class NewsScraper:
    """Scrape news from various Nepali stock market sources"""
    
    def __init__(self):
        self.sources = config.NEWS_SOURCES
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def scrape_sharesansar_news(self, symbol: str, days: int = 7) -> List[Dict]:
        """Scrape news from ShareSansar"""
        news_items = []
        
        try:
            # Search for company-specific news
            url = f"{self.sources['sharesansar']}/company/{symbol}"
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find news articles - adjust selectors based on actual website structure
            articles = soup.find_all('div', class_='news-item')  # Placeholder selector
            
            for article in articles[:20]:  # Limit to recent 20 articles
                try:
                    title_elem = article.find('h3') or article.find('a')
                    date_elem = article.find('span', class_='date')
                    
                    if title_elem:
                        news_items.append({
                            'source': 'sharesansar',
                            'symbol': symbol,
                            'title': title_elem.get_text(strip=True),
                            'url': title_elem.get('href', ''),
                            'date': self._parse_date(date_elem.get_text() if date_elem else ''),
                            'scraped_at': datetime.now().isoformat()
                        })
                except Exception as e:
                    continue
            
        except Exception as e:
            print(f"Error scraping ShareSansar news: {e}")
        
        return news_items
    
    def scrape_merolagani_news(self, symbol: str, days: int = 7) -> List[Dict]:
        """Scrape news from MeroLagani"""
        news_items = []
        
        try:
            url = f"{self.sources['merolagani']}/company/{symbol}"
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find news articles - adjust selectors based on actual website structure
            articles = soup.find_all('div', class_='news-item')  # Placeholder selector
            
            for article in articles[:20]:
                try:
                    title_elem = article.find('a')
                    date_elem = article.find('span', class_='date')
                    
                    if title_elem:
                        news_items.append({
                            'source': 'merolagani',
                            'symbol': symbol,
                            'title': title_elem.get_text(strip=True),
                            'url': title_elem.get('href', ''),
                            'date': self._parse_date(date_elem.get_text() if date_elem else ''),
                            'scraped_at': datetime.now().isoformat()
                        })
                except Exception as e:
                    continue
            
        except Exception as e:
            print(f"Error scraping MeroLagani news: {e}")
        
        return news_items
    
    def scrape_general_market_news(self, days: int = 7) -> List[Dict]:
        """Scrape general market news"""
        news_items = []
        
        # Scrape from multiple sources
        sources_to_scrape = [
            (self.sources['sharesansar'], 'sharesansar'),
            (self.sources['merolagani'], 'merolagani'),
        ]
        
        for source_url, source_name in sources_to_scrape:
            try:
                response = self.session.get(f"{source_url}/news", timeout=30)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                articles = soup.find_all('article')[:15]  # Get latest 15 articles
                
                for article in articles:
                    try:
                        title_elem = article.find('h2') or article.find('h3')
                        link_elem = article.find('a')
                        
                        if title_elem and link_elem:
                            news_items.append({
                                'source': source_name,
                                'symbol': 'MARKET',
                                'title': title_elem.get_text(strip=True),
                                'url': link_elem.get('href', ''),
                                'date': datetime.now().isoformat(),
                                'scraped_at': datetime.now().isoformat()
                            })
                    except Exception as e:
                        continue
                        
            except Exception as e:
                print(f"Error scraping {source_name}: {e}")
        
        return news_items
    
    def get_all_news(self, symbol: str, days: int = 7) -> List[Dict]:
        """Get all news for a symbol from all sources"""
        all_news = []
        
        # Scrape from different sources
        all_news.extend(self.scrape_sharesansar_news(symbol, days))
        all_news.extend(self.scrape_merolagani_news(symbol, days))
        
        # Sort by date
        all_news.sort(key=lambda x: x.get('date', ''), reverse=True)
        
        return all_news
    
    def _parse_date(self, date_str: str) -> str:
        """Parse date string to ISO format"""
        try:
            # Handle various date formats
            # This is a simple implementation - adjust based on actual formats
            if 'ago' in date_str.lower():
                return datetime.now().isoformat()
            
            return datetime.now().isoformat()
        except Exception:
            return datetime.now().isoformat()


class SentimentAnalyzer:
    """Analyze sentiment of news articles"""
    
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        
    def analyze_text(self, text: str) -> Dict:
        """Analyze sentiment of a single text"""
        if not text:
            return {'score': 0, 'sentiment': 'neutral', 'confidence': 0}
        
        # VADER sentiment
        vader_scores = self.vader.polarity_scores(text)
        
        # TextBlob sentiment
        blob = TextBlob(text)
        textblob_score = blob.sentiment.polarity
        
        # Combined score (average)
        combined_score = (vader_scores['compound'] + textblob_score) / 2
        
        # Determine sentiment
        if combined_score >= 0.05:
            sentiment = 'positive'
        elif combined_score <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'score': combined_score,
            'sentiment': sentiment,
            'confidence': abs(combined_score),
            'vader_compound': vader_scores['compound'],
            'textblob_polarity': textblob_score,
            'vader_breakdown': {
                'positive': vader_scores['pos'],
                'negative': vader_scores['neg'],
                'neutral': vader_scores['neu']
            }
        }
    
    def analyze_news_list(self, news_items: List[Dict]) -> pd.DataFrame:
        """Analyze sentiment for a list of news items"""
        results = []
        
        for item in news_items:
            text = item.get('title', '')
            sentiment_result = self.analyze_text(text)
            
            results.append({
                'source': item.get('source', ''),
                'symbol': item.get('symbol', ''),
                'title': text,
                'date': item.get('date', ''),
                'sentiment': sentiment_result['sentiment'],
                'score': sentiment_result['score'],
                'confidence': sentiment_result['confidence'],
                'url': item.get('url', '')
            })
        
        return pd.DataFrame(results)
    
    def get_aggregate_sentiment(self, symbol: str, days: int = 7) -> Dict:
        """Get aggregate sentiment for a symbol"""
        scraper = NewsScraper()
        news_items = scraper.get_all_news(symbol, days)
        
        if not news_items:
            return {
                'symbol': symbol,
                'overall_sentiment': 'neutral',
                'average_score': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'total_articles': 0,
                'confidence': 0
            }
        
        sentiment_df = self.analyze_news_list(news_items)
        
        # Calculate aggregate metrics
        avg_score = sentiment_df['score'].mean()
        positive_count = len(sentiment_df[sentiment_df['sentiment'] == 'positive'])
        negative_count = len(sentiment_df[sentiment_df['sentiment'] == 'negative'])
        neutral_count = len(sentiment_df[sentiment_df['sentiment'] == 'neutral'])
        
        # Determine overall sentiment
        if avg_score >= 0.05:
            overall_sentiment = 'positive'
        elif avg_score <= -0.05:
            overall_sentiment = 'negative'
        else:
            overall_sentiment = 'neutral'
        
        return {
            'symbol': symbol,
            'overall_sentiment': overall_sentiment,
            'average_score': float(avg_score),
            'positive_count': int(positive_count),
            'negative_count': int(negative_count),
            'neutral_count': int(neutral_count),
            'total_articles': len(news_items),
            'confidence': float(sentiment_df['confidence'].mean()),
            'sentiment_trend': self._calculate_sentiment_trend(sentiment_df),
            'recent_headlines': news_items[:5]  # Top 5 recent headlines
        }
    
    def _calculate_sentiment_trend(self, sentiment_df: pd.DataFrame) -> str:
        """Calculate if sentiment is improving or declining"""
        if len(sentiment_df) < 2:
            return 'stable'
        
        # Sort by date and compare first half vs second half
        sentiment_df = sentiment_df.sort_values('date')
        mid_point = len(sentiment_df) // 2
        
        first_half_avg = sentiment_df.iloc[:mid_point]['score'].mean()
        second_half_avg = sentiment_df.iloc[mid_point:]['score'].mean()
        
        diff = second_half_avg - first_half_avg
        
        if diff > 0.1:
            return 'improving'
        elif diff < -0.1:
            return 'declining'
        else:
            return 'stable'


if __name__ == "__main__":
    # Test sentiment analysis
    analyzer = SentimentAnalyzer()
    
    # Test with a sample stock
    symbol = "NABIL"
    print(f"Analyzing sentiment for {symbol}...")
    
    result = analyzer.get_aggregate_sentiment(symbol, days=7)
    
    print(f"\nOverall Sentiment: {result['overall_sentiment']}")
    print(f"Average Score: {result['average_score']:.3f}")
    print(f"Positive: {result['positive_count']}, Negative: {result['negative_count']}, Neutral: {result['neutral_count']}")
    print(f"Sentiment Trend: {result['sentiment_trend']}")
