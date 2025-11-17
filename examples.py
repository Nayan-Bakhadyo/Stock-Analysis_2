"""
Example scripts demonstrating different use cases
"""

# Example 1: Basic stock analysis
def example_basic_analysis():
    """Simple stock analysis example"""
    from trading_insights import TradingInsightsEngine
    
    engine = TradingInsightsEngine()
    
    # Analyze NABIL bank
    result = engine.calculate_profitability_probability("NABIL", "short")
    
    print(f"Stock: {result['symbol']}")
    print(f"Probability: {result['profitability_probability']:.2f}%")
    print(f"Recommendation: {result['recommendation']['action']}")
    
    return result


# Example 2: Compare multiple stocks
def example_compare_stocks():
    """Compare multiple banking stocks"""
    from trading_insights import TradingInsightsEngine
    
    engine = TradingInsightsEngine()
    
    banking_stocks = ['NABIL', 'NICA', 'GBIME', 'SCB', 'EBL']
    
    comparison = engine.batch_analysis(banking_stocks, 'short')
    
    print("\nTop 3 Banks by Probability:")
    print(comparison.head(3))
    
    return comparison


# Example 3: Detailed technical analysis
def example_technical_analysis():
    """Detailed technical analysis example"""
    from technical_analyzer import TechnicalAnalyzer
    from data_fetcher import NepseDataFetcher
    
    fetcher = NepseDataFetcher()
    analyzer = TechnicalAnalyzer()
    
    # Get price data
    symbol = "NABIL"
    price_data = fetcher.get_stock_price_history(symbol, days=180)
    
    # Perform technical analysis
    result = analyzer.comprehensive_analysis(price_data)
    
    print(f"\nTechnical Analysis for {symbol}:")
    print(f"RSI: {result['indicators']['rsi']:.2f}")
    print(f"MACD Signal: {result['signals']['macd_signal']}")
    print(f"Trend: {result['signals']['trend']}")
    print(f"Technical Score: {result['technical_score']}/100")
    
    return result


# Example 4: Sentiment analysis
def example_sentiment_analysis():
    """Sentiment analysis example"""
    from sentiment_analyzer import SentimentAnalyzer
    
    analyzer = SentimentAnalyzer()
    
    symbol = "NABIL"
    sentiment = analyzer.get_aggregate_sentiment(symbol, days=7)
    
    print(f"\nSentiment Analysis for {symbol}:")
    print(f"Overall Sentiment: {sentiment['overall_sentiment']}")
    print(f"Average Score: {sentiment['average_score']:.3f}")
    print(f"Positive Articles: {sentiment['positive_count']}")
    print(f"Negative Articles: {sentiment['negative_count']}")
    print(f"Trend: {sentiment['sentiment_trend']}")
    
    return sentiment


# Example 5: Custom fundamental analysis
def example_fundamental_analysis():
    """Fundamental analysis with custom data"""
    from fundamental_analyzer import FundamentalAnalyzer
    
    analyzer = FundamentalAnalyzer()
    
    # Custom stock data
    stock_data = {
        'symbol': 'NABIL',
        'current_price': 1050,
        'eps': 52.5,
        'previous_eps': 48.0,
        'book_value_per_share': 420,
        'annual_dividend': 31.5,
        'net_income': 5250000,
        'shareholders_equity': 26000000,
        'total_debt': 9500000,
        'current_assets': 52000000,
        'current_liabilities': 31000000,
        'market_cap': 10500000000
    }
    
    result = analyzer.comprehensive_analysis(stock_data)
    
    print(f"\nFundamental Analysis for {stock_data['symbol']}:")
    print(f"Overall Rating: {result['overall_rating']}")
    print(f"Overall Score: {result['overall_score']}/100")
    print(f"P/E Ratio: {result['ratios']['pe_ratio']['value']:.2f}")
    print(f"P/B Ratio: {result['ratios']['pb_ratio']['value']:.2f}")
    print(f"\nStrengths:")
    for strength in result['strengths']:
        print(f"  • {strength}")
    
    return result


# Example 6: Portfolio analysis
def example_portfolio_analysis():
    """Analyze a portfolio of stocks"""
    from trading_insights import TradingInsightsEngine
    
    engine = TradingInsightsEngine()
    
    # Your portfolio
    portfolio = {
        'NABIL': 50,    # 50 shares
        'NICA': 100,    # 100 shares
        'GBIME': 75,    # 75 shares
    }
    
    print("\nPortfolio Analysis:")
    print("=" * 60)
    
    total_value = 0
    
    for symbol, shares in portfolio.items():
        result = engine.calculate_profitability_probability(symbol, 'medium')
        
        if 'error' not in result:
            value = shares * result['current_price']
            total_value += value
            
            print(f"\n{symbol}:")
            print(f"  Shares: {shares}")
            print(f"  Current Price: NPR {result['current_price']:.2f}")
            print(f"  Value: NPR {value:,.2f}")
            print(f"  Probability: {result['profitability_probability']:.2f}%")
            print(f"  Recommendation: {result['recommendation']['action']}")
    
    print(f"\nTotal Portfolio Value: NPR {total_value:,.2f}")
    
    return total_value


# Example 7: Export analysis to file
def example_export_analysis():
    """Export analysis results to JSON"""
    from trading_insights import TradingInsightsEngine
    import json
    from datetime import datetime
    
    engine = TradingInsightsEngine()
    
    symbol = "NABIL"
    result = engine.calculate_profitability_probability(symbol, 'short')
    
    # Export to file
    filename = f"{symbol}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(filename, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✅ Analysis exported to {filename}")
    
    return filename


# Example 8: Monitor top gainers
def example_monitor_gainers():
    """Monitor and analyze top gaining stocks"""
    from data_fetcher import NepseDataFetcher
    from trading_insights import TradingInsightsEngine
    
    fetcher = NepseDataFetcher()
    engine = TradingInsightsEngine()
    
    # Get top gainers
    gainers = fetcher.get_top_gainers(limit=5)
    
    print("\nTop 5 Gainers Analysis:")
    print("=" * 60)
    
    if not gainers.empty and 'symbol' in gainers.columns:
        for idx, row in gainers.iterrows():
            symbol = row['symbol']
            
            try:
                result = engine.calculate_profitability_probability(symbol, 'short')
                
                if 'error' not in result:
                    print(f"\n{symbol}:")
                    print(f"  Probability: {result['profitability_probability']:.2f}%")
                    print(f"  Recommendation: {result['recommendation']['action']}")
                    print(f"  R:R Ratio: {result['risk_reward_ratio']['ratio']:.2f}")
            except Exception as e:
                print(f"\n{symbol}: Error - {str(e)}")


if __name__ == "__main__":
    print("NEPSE Stock Analysis - Example Scripts")
    print("=" * 60)
    
    # Run examples
    print("\n1. Running basic analysis...")
    # example_basic_analysis()
    
    print("\n2. Running stock comparison...")
    # example_compare_stocks()
    
    print("\n3. Running technical analysis...")
    # example_technical_analysis()
    
    print("\nUncomment the example you want to run in the __main__ section")
