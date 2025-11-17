"""
ShareSansar News Scraper using Selenium
Scrapes company-specific news articles with date-based sentiment weighting
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime, timedelta
import time
import pandas as pd
import re
from typing import List, Dict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob


class ShareSansarNewsScraper:
    """Scrape and analyze news from ShareSansar"""
    
    def __init__(self, headless=True):
        self.headless = headless
        self.driver = None
        self.base_url = "https://www.sharesansar.com"
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
    def init_driver(self):
        """Initialize Chrome WebDriver"""
        options = Options()
        if self.headless:
            options.add_argument('--headless=new')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('--window-size=1920,1080')
        
        self.driver = webdriver.Chrome(options=options)
        print("✓ Chrome driver initialized")
    
    def close_driver(self):
        """Close the browser"""
        if self.driver:
            self.driver.quit()
            print("✓ Browser closed")
    
    def scrape_company_news(self, symbol: str, max_articles: int = 20) -> List[Dict]:
        """
        Scrape news articles for a specific company
        
        Args:
            symbol: Stock symbol (e.g., 'IGI', 'NABIL')
            max_articles: Maximum number of articles to scrape
            
        Returns:
            List of news articles with title, date, URL, and sentiment
        """
        if not self.driver:
            self.init_driver()
        
        articles = []
        
        try:
            # Navigate to company page
            url = f"{self.base_url}/company/{symbol.lower()}"
            print(f"\n→ Loading {url}...")
            self.driver.get(url)
            time.sleep(2)
            
            # Click News tab (the tab within the page, not navbar)
            try:
                news_tab = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "a[href='#cnews'][data-toggle='tab']"))
                )
                news_tab.click()
                print("✓ Clicked News tab")
                time.sleep(3)  # Wait for content to load
            except Exception as e:
                print(f"⚠️ Could not find News tab: {e}")
                return articles
            
            # Find all news article links after clicking News tab
            # Get all links that point to newsdetail pages
            all_links = self.driver.find_elements(By.TAG_NAME, "a")
            
            newsdetail_links = []
            for link in all_links:
                href = link.get_attribute("href")
                title = link.text.strip()
                if href and '/newsdetail/' in href and title:
                    newsdetail_links.append((title, href))
            
            print(f"→ Found {len(newsdetail_links)} news articles")
            
            # Process each news article
            for title, href in newsdetail_links[:max_articles]:
                try:
                    print(f"  → Processing: {title[:60]}...")
                    
                    # Extract date from URL (format: yyyy-mm-dd at end)
                    date_match = re.search(r'(\d{4})-(\d{2})-(\d{2})$', href)
                    article_date = None
                    
                    if date_match:
                        try:
                            article_date = datetime.strptime(
                                f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}", 
                                "%Y-%m-%d"
                            )
                        except:
                            article_date = datetime.now()
                    else:
                        article_date = datetime.now()
                    
                    # Fetch article content for better sentiment analysis
                    article_content = self._fetch_article_content(href)
                    
                    # Analyze sentiment using both title and content
                    full_text = f"{title}. {article_content}"
                    sentiment_result = self._analyze_sentiment(full_text)
                    
                    # Calculate age-based weight (newer = higher weight)
                    days_old = (datetime.now() - article_date).days
                    recency_weight = self._calculate_recency_weight(days_old)
                    
                    articles.append({
                        'symbol': symbol.upper(),
                        'title': title,
                        'content': article_content,
                        'url': href,
                        'date': article_date,
                        'days_old': days_old,
                        'recency_weight': recency_weight,
                        'sentiment': sentiment_result['sentiment'],
                        'sentiment_score': sentiment_result['score'],
                        'vader_score': sentiment_result['vader_compound'],
                        'textblob_score': sentiment_result['textblob_polarity'],
                        'weighted_score': sentiment_result['score'] * recency_weight,
                        'scraped_at': datetime.now()
                    })
                    
                    print(f"    ✓ Sentiment: {sentiment_result['sentiment'].upper()} ({sentiment_result['score']:.3f})")
                    
                except Exception as e:
                    print(f"    ✗ Error processing article: {e}")
                    continue
            
            print(f"✓ Scraped {len(articles)} news articles for {symbol}")
            
        except Exception as e:
            print(f"✗ Error scraping news: {e}")
        
        return articles
    
    def _fetch_article_content(self, url: str) -> str:
        """
        Fetch full article content from URL
        
        Args:
            url: Article URL
            
        Returns:
            Article text content
        """
        try:
            # Open article in new tab to avoid losing current page state
            self.driver.execute_script(f"window.open('{url}', '_blank');")
            
            # Switch to new tab
            self.driver.switch_to.window(self.driver.window_handles[-1])
            time.sleep(1.5)
            
            # Extract article content
            # Look for common article content containers
            content_text = ""
            
            # Try multiple selectors for article content
            selectors = [
                'div.article-content',
                'div.news-content', 
                'div.content',
                'article',
                'div[class*="detail"]',
                'div[class*="body"]'
            ]
            
            for selector in selectors:
                try:
                    content_elem = self.driver.find_element(By.CSS_SELECTOR, selector)
                    paragraphs = content_elem.find_elements(By.TAG_NAME, 'p')
                    content_text = ' '.join([p.text.strip() for p in paragraphs if p.text.strip()])
                    if content_text and len(content_text) > 100:
                        break
                except:
                    continue
            
            # If no content found with selectors, get all paragraph text
            if not content_text:
                try:
                    paragraphs = self.driver.find_elements(By.TAG_NAME, 'p')
                    # Filter out navigation/menu items (too short)
                    content_paragraphs = [p.text.strip() for p in paragraphs if len(p.text.strip()) > 50]
                    content_text = ' '.join(content_paragraphs[:10])  # First 10 substantial paragraphs
                except:
                    content_text = ""
            
            # Close tab and switch back
            self.driver.close()
            self.driver.switch_to.window(self.driver.window_handles[0])
            
            # Limit content length for sentiment analysis
            return content_text[:2000] if content_text else ""
            
        except Exception as e:
            # If anything fails, close any extra windows and return empty
            try:
                if len(self.driver.window_handles) > 1:
                    self.driver.close()
                    self.driver.switch_to.window(self.driver.window_handles[0])
            except:
                pass
            return ""
    
    def _analyze_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment of text using VADER and TextBlob
        
        Returns:
            Dict with sentiment classification and scores
        """
        if not text:
            return {
                'sentiment': 'neutral',
                'score': 0,
                'vader_compound': 0,
                'textblob_polarity': 0
            }
        
        # VADER sentiment
        vader_scores = self.sentiment_analyzer.polarity_scores(text)
        
        # TextBlob sentiment
        blob = TextBlob(text)
        textblob_score = blob.sentiment.polarity
        
        # Combined score (average)
        combined_score = (vader_scores['compound'] + textblob_score) / 2
        
        # Classify sentiment
        if combined_score >= 0.05:
            sentiment = 'positive'
        elif combined_score <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'score': combined_score,
            'vader_compound': vader_scores['compound'],
            'textblob_polarity': textblob_score
        }
    
    def _calculate_recency_weight(self, days_old: int) -> float:
        """
        Calculate weight based on article age
        
        Recent articles have more weight than older ones:
        - 0-7 days: 1.0 (full weight)
        - 8-30 days: 0.7
        - 31-90 days: 0.4
        - 91-180 days: 0.2
        - 180+ days: 0.1
        
        Args:
            days_old: Number of days since article was published
            
        Returns:
            Weight multiplier (0.1 to 1.0)
        """
        if days_old <= 7:
            return 1.0
        elif days_old <= 30:
            return 0.7
        elif days_old <= 90:
            return 0.4
        elif days_old <= 180:
            return 0.2
        else:
            return 0.1
    
    def get_aggregate_sentiment(self, symbol: str, max_articles: int = 20, days_filter: int = None) -> Dict:
        """
        Get aggregate sentiment analysis for a company
        
        Args:
            symbol: Stock symbol
            max_articles: Maximum articles to analyze
            days_filter: Only include articles from last N days (None = all)
            
        Returns:
            Aggregate sentiment metrics
        """
        articles = self.scrape_company_news(symbol, max_articles)
        
        if not articles:
            return {
                'symbol': symbol,
                'overall_sentiment': 'neutral',
                'average_score': 0,
                'weighted_average_score': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'total_articles': 0,
                'confidence': 0,
                'recent_headlines': []
            }
        
        # Filter by date if specified
        if days_filter:
            cutoff_date = datetime.now() - timedelta(days=days_filter)
            articles = [a for a in articles if a['date'] >= cutoff_date]
        
        if not articles:
            return {
                'symbol': symbol,
                'overall_sentiment': 'neutral',
                'average_score': 0,
                'weighted_average_score': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'total_articles': 0,
                'confidence': 0,
                'recent_headlines': []
            }
        
        # Calculate metrics
        df = pd.DataFrame(articles)
        
        avg_score = df['sentiment_score'].mean()
        weighted_avg_score = df['weighted_score'].sum() / df['recency_weight'].sum()
        
        positive_count = len(df[df['sentiment'] == 'positive'])
        negative_count = len(df[df['sentiment'] == 'negative'])
        neutral_count = len(df[df['sentiment'] == 'neutral'])
        
        # Determine overall sentiment based on weighted average
        if weighted_avg_score >= 0.05:
            overall_sentiment = 'positive'
        elif weighted_avg_score <= -0.05:
            overall_sentiment = 'negative'
        else:
            overall_sentiment = 'neutral'
        
        # Confidence based on number of articles and score consistency
        score_std = df['sentiment_score'].std()
        confidence = min(len(articles) / 20, 1.0) * (1 - min(score_std, 1.0))
        
        return {
            'symbol': symbol,
            'overall_sentiment': overall_sentiment,
            'average_score': float(avg_score),
            'weighted_average_score': float(weighted_avg_score),
            'positive_count': int(positive_count),
            'negative_count': int(negative_count),
            'neutral_count': int(neutral_count),
            'total_articles': len(articles),
            'confidence': float(confidence),
            'sentiment_distribution': {
                'positive': f"{(positive_count/len(articles)*100):.1f}%",
                'negative': f"{(negative_count/len(articles)*100):.1f}%",
                'neutral': f"{(neutral_count/len(articles)*100):.1f}%"
            },
            'recent_headlines': [
                {
                    'title': a['title'],
                    'content_preview': a['content'][:200] + '...' if len(a['content']) > 200 else a['content'],
                    'date': a['date'].strftime('%Y-%m-%d'),
                    'sentiment': a['sentiment'],
                    'score': a['sentiment_score']
                }
                for a in articles[:5]
            ],
            'articles_by_recency': {
                'last_7_days': len([a for a in articles if a['days_old'] <= 7]),
                'last_30_days': len([a for a in articles if a['days_old'] <= 30]),
                'last_90_days': len([a for a in articles if a['days_old'] <= 90])
            }
        }


if __name__ == "__main__":
    # Test the scraper
    scraper = ShareSansarNewsScraper(headless=True)
    
    try:
        print("=" * 70)
        print("TESTING SHARESANSAR NEWS SCRAPER")
        print("=" * 70)
        
        result = scraper.get_aggregate_sentiment('IGI', max_articles=20)
        
        print(f"\n📊 SENTIMENT ANALYSIS FOR {result['symbol']}")
        print(f"Overall Sentiment: {result['overall_sentiment'].upper()}")
        print(f"Weighted Average Score: {result['weighted_average_score']:.3f}")
        print(f"Raw Average Score: {result['average_score']:.3f}")
        print(f"\nArticle Breakdown:")
        print(f"  Total: {result['total_articles']}")
        print(f"  Positive: {result['positive_count']}")
        print(f"  Negative: {result['negative_count']}")
        print(f"  Neutral: {result['neutral_count']}")
        print(f"\nBy Recency:")
        print(f"  Last 7 days: {result['articles_by_recency']['last_7_days']}")
        print(f"  Last 30 days: {result['articles_by_recency']['last_30_days']}")
        print(f"  Last 90 days: {result['articles_by_recency']['last_90_days']}")
        
        print("\n📰 RECENT HEADLINES:")
        for i, article in enumerate(result['recent_headlines'][:5], 1):
            sentiment_emoji = {'positive': '✅', 'negative': '❌', 'neutral': '⚪'}
            print(f"\n{i}. {article['title'][:80]}...")
            print(f"   Date: {article['date']} | {sentiment_emoji[article['sentiment']]} {article['sentiment'].upper()} ({article['score']:.3f})")
            if article['content_preview']:
                print(f"   Preview: {article['content_preview'][:150]}...")
        
    finally:
        scraper.close_driver()
