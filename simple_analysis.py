"""
Simple analysis script for multiple stocks using TradingInsightsEngine
Generates comprehensive analysis matching main.py output format
"""

import sys
import json
import os
from datetime import datetime
from trading_insights import TradingInsightsEngine
from data_fetcher import NepseDataFetcher
from stock_tracker import StockTracker


def analyze_stock(symbol: str, time_horizon: str = 'short', tracker: StockTracker = None) -> dict:
    """Perform complete analysis on a single stock using TradingInsightsEngine"""
    
    print(f"\n{'='*70}")
    print(f"ANALYZING: {symbol}")
    print('='*70)
    
    try:
        # Use TradingInsightsEngine for comprehensive analysis
        insights_engine = TradingInsightsEngine()
        
        # Get comprehensive analysis (includes all data: technical, fundamental, sentiment, ML)
        result = insights_engine.calculate_profitability_probability(
            symbol=symbol,
            time_horizon=time_horizon,
            include_broker_analysis=False,  # Skip broker analysis for speed
            use_cache=True  # Use sync manager for intelligent caching
        )
        
        # Check for errors
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'error': result['error']
            }
        
        # Get price summary from database
        data_fetcher = NepseDataFetcher()
        price_data = data_fetcher.get_stock_price_history(symbol, days=None)
        
        if not price_data.empty:
            price_summary = {
                'total_days': len(price_data),
                'latest_price': float(price_data['close'].iloc[-1]),
                'latest_date': str(price_data['date'].iloc[-1]),
                'highest_price': float(price_data['close'].max()),
                'lowest_price': float(price_data['close'].min()),
                'avg_volume': float(price_data['volume'].mean())
            }
        else:
            price_summary = None
        
        # Get news summary from sentiment details
        news_summary = None
        if result.get('sentiment_details'):
            sentiment = result['sentiment_details']
            news_summary = {
                'total_articles': sentiment.get('articles_analyzed', 0),
                'avg_sentiment': sentiment.get('overall_sentiment', 0),
                'sentiment_label': sentiment.get('sentiment_label', 'NEUTRAL'),
                'articles': []
            }
        
        # Structure the complete result
        complete_result = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            
            # Price data summary
            'price_data': price_summary,
            
            # News and sentiment
            'news': news_summary,
            
            # Fundamental analysis
            'fundamentals': result.get('fundamental_analysis', {}),
            
            # Technical analysis (comprehensive)
            'technical_analysis': result.get('technical_analysis', {}),
            
            # Candlestick patterns
            'candlestick_patterns': result.get('technical_analysis', {}).get('candlestick_patterns', []),
            
            # Trading insights (profitability probability, recommendations, etc.)
            'trading_insights': {
                'profitability_probability': result.get('profitability_probability', 0),
                'confidence_level': result.get('confidence_level', 'Unknown'),
                'time_horizon': result.get('time_horizon', time_horizon),
                'recommendation': result.get('recommendation', {}),
                'risk_reward_ratio': result.get('risk_reward_ratio', {}),
                'position_size': result.get('position_size', {}),
                'entry_points': result.get('entry_points', {}),
                'exit_points': result.get('exit_points', {}),
                'stop_loss': result.get('stop_loss'),
                'take_profit': result.get('take_profit'),
                'current_price': result.get('current_price'),
                'potential_profit_pct': result.get('potential_profit_pct', 0),
                'potential_loss_pct': result.get('potential_loss_pct', 0)
            },
            
            # Scores
            'scores': result.get('scores', {}),
            
            # Key insights and warnings
            'key_insights': result.get('key_insights', []),
            'warnings': result.get('warnings', []),
            
            # ML predictions
            'ml_predictions': result.get('ml_predictions'),
            
            # Metadata
            'analysis_timestamp': result.get('analysis_timestamp'),
            'data_points': result.get('data_points', 0),
            
            'error': None
        }
        
        print(f"\n✅ Analysis complete for {symbol}")
        
        # Track successful analysis
        if tracker:
            tracker.mark_processed(symbol, status='success')
        
        return complete_result
        
    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ Analysis failed for {symbol}: {error_msg}")
        import traceback
        traceback.print_exc()
        
        # Track failed analysis
        if tracker:
            tracker.mark_processed(symbol, status='failed', error=error_msg)
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'error': error_msg
        }


def main():
    """Main function to analyze multiple stocks"""
    
    # Initialize tracker
    tracker = StockTracker()
    
    # Default stocks to analyze
    symbols = ['NABIL', 'NICA', 'SCB']
    
    # Override with command line arguments if provided
    if len(sys.argv) > 1:
        symbols = sys.argv[1:]
    
    print("\n" + "="*70)
    print("STOCK ANALYSIS FOR WEBSITE GENERATION")
    print("="*70)
    print(f"Analyzing {len(symbols)} stocks: {', '.join(symbols)}")
    print("="*70)
    
    # Load existing results if file exists
    output_file = 'analysis_results.json'
    existing_results = {}
    
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                existing_data = json.load(f)
                # Create dict with symbol as key for easy lookup/update
                existing_results = {r['symbol']: r for r in existing_data}
                print(f"ℹ️ Loaded {len(existing_results)} existing analyses")
        except:
            pass
    
    # Analyze new stocks and save after each one
    for symbol in symbols:
        result = analyze_stock(symbol, time_horizon='short', tracker=tracker)
        
        # Update or add to existing results
        existing_results[symbol] = result
        
        # Save immediately after each stock completes
        all_results = list(existing_results.values())
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"  💾 Saved progress ({len(existing_results)} stocks total)")
    
    print(f"\n{'='*70}")
    print(f"✅ ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {output_file}")
    print(f"Total stocks analyzed: {len(all_results)}")
    print(f"Successful: {sum(1 for r in all_results if not r.get('error'))}")
    print(f"Failed: {sum(1 for r in all_results if r.get('error'))}")
    print(f"\nNext step: Run 'python generate_website.py' to create the website")
    print("="*70)
    
    # Print tracker summary
    tracker.print_summary()
    
    # Print tracker summary
    tracker.print_summary()


if __name__ == '__main__':
    main()
