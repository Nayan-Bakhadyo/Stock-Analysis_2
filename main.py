"""Main application for NEPSE Stock Analysis"""
import argparse
import sys
from datetime import datetime
from colorama import init, Fore, Style
from tabulate import tabulate
import json
import sqlite3
import pandas as pd
import config

from trading_insights import TradingInsightsEngine
from data_fetcher import NepseDataFetcher
from sentiment_analyzer import SentimentAnalyzer
from fundamental_analyzer import FundamentalAnalyzer
from technical_analyzer import TechnicalAnalyzer
from broker_analyzer import BrokerAnalyzer
from real_data_adapter import RealDataAdapter
from pathlib import Path

# Initialize colorama
init(autoreset=True)


class NepseStockAnalyzer:
    """Main application class"""
    
    def __init__(self):
        self.insights_engine = TradingInsightsEngine()
        self.data_fetcher = NepseDataFetcher()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.fundamental_analyzer = FundamentalAnalyzer()
        self.technical_analyzer = TechnicalAnalyzer()
        self.broker_analyzer = BrokerAnalyzer()
    
    def print_header(self, text: str):
        """Print formatted header"""
        print(f"\n{Fore.CYAN}{'='*70}")
        print(f"{Fore.CYAN}{text.center(70)}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}\n")
    
    def print_section(self, text: str):
        """Print section header"""
        print(f"\n{Fore.YELLOW}{text}")
        print(f"{Fore.YELLOW}{'-'*len(text)}{Style.RESET_ALL}")
    
    def analyze_stock(self, symbol: str, time_horizon: str = 'short', 
                     detailed: bool = False):
        """Analyze a single stock"""
        
        self.print_header(f"NEPSE STOCK ANALYSIS: {symbol}")
        
        print(f"{Fore.GREEN}⏳ Analyzing {symbol}... Please wait...{Style.RESET_ALL}\n")
        
        try:
            # Get comprehensive analysis
            result = self.insights_engine.calculate_profitability_probability(
                symbol, 
                time_horizon
            )
            
            if 'error' in result:
                print(f"{Fore.RED}❌ Error: {result['error']}{Style.RESET_ALL}")
                return
            
            # Display results
            self._display_analysis_results(result, detailed)
            
        except Exception as e:
            print(f"{Fore.RED}❌ Analysis failed: {str(e)}{Style.RESET_ALL}")
    
    def _display_analysis_results(self, result: dict, detailed: bool = False):
        """Display analysis results in a formatted way"""
        
        # Basic Information
        self.print_section("📊 BASIC INFORMATION")
        basic_info = [
            ["Symbol", result['symbol']],
            ["Current Price", f"NPR {result['current_price']:.2f}"],
            ["Analysis Date", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["Time Horizon", result['time_horizon'].upper()]
        ]
        print(tabulate(basic_info, tablefmt="grid"))
        
        # Profitability Analysis
        self.print_section("💰 PROFITABILITY ANALYSIS")
        prob = result['profitability_probability']
        
        # Color code based on probability
        if prob >= 70:
            prob_color = Fore.GREEN
        elif prob >= 50:
            prob_color = Fore.YELLOW
        else:
            prob_color = Fore.RED
        
        print(f"\n{prob_color}Profitability Probability: {prob:.2f}%{Style.RESET_ALL}")
        print(f"Confidence Level: {result['confidence_level']}")
        
        # Progress bar
        bar_length = 50
        filled = int(bar_length * prob / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"\n[{prob_color}{bar}{Style.RESET_ALL}] {prob:.1f}%\n")
        
        # Recommendation
        self.print_section("📈 TRADING RECOMMENDATION")
        rec = result['recommendation']
        
        action = rec['action']
        if 'BUY' in action:
            action_color = Fore.GREEN
        elif 'SELL' in action:
            action_color = Fore.RED
        else:
            action_color = Fore.YELLOW
        
        print(f"{action_color}Action: {action}{Style.RESET_ALL}")
        print(f"Confidence: {rec['confidence']}")
        print(f"\nReasoning:")
        for reason in rec['reasoning']:
            print(f"  • {reason}")
        
        # Risk-Reward Analysis
        self.print_section("⚖️ RISK-REWARD ANALYSIS")
        rr = result['risk_reward_ratio']
        
        risk_reward_data = [
            ["Risk-Reward Ratio", f"{rr['ratio']:.2f}"],
            ["Potential Profit", f"{rr['potential_profit_percent']:.2f}%"],
            ["Potential Loss", f"{rr['potential_loss_percent']:.2f}%"],
            ["Nearest Support", f"NPR {rr['nearest_support']:.2f}"],
            ["Nearest Resistance", f"NPR {rr['nearest_resistance']:.2f}"]
        ]
        print(tabulate(risk_reward_data, tablefmt="grid"))
        
        # Entry and Exit Points
        self.print_section("🎯 ENTRY & EXIT STRATEGY")
        
        entry_exit_data = [
            ["Entry (Aggressive)", f"NPR {result['entry_points']['aggressive']:.2f}"],
            ["Entry (Conservative)", f"NPR {result['entry_points']['conservative']:.2f}"],
            ["", ""],
            ["Target 1", f"NPR {result['exit_points']['target_1']:.2f}"],
            ["Target 2", f"NPR {result['exit_points']['target_2']:.2f}"],
            ["", ""],
            ["Stop Loss", f"{Fore.RED}NPR {result['stop_loss']:.2f}{Style.RESET_ALL}"],
            ["Take Profit", f"{Fore.GREEN}NPR {result['take_profit']:.2f}{Style.RESET_ALL}"]
        ]
        print(tabulate(entry_exit_data, tablefmt="grid"))
        
        # Position Size
        self.print_section("💼 POSITION SIZING")
        pos_size = result['position_size']
        print(f"Recommended Position: {pos_size['recommended_percent']:.2f}% of portfolio")
        print(f"Maximum Position: {pos_size['max_percent']:.2f}% of portfolio")
        print(f"Reasoning: {pos_size['reasoning']}")
        
        # Scores Breakdown
        self.print_section("📊 ANALYSIS SCORES")
        scores = result['scores']
        
        scores_data = [
            ["Technical Analysis", f"{scores['technical']:.2f}/100", self._get_score_bar(scores['technical'])],
            ["Fundamental Analysis", f"{scores['fundamental']:.2f}/100", self._get_score_bar(scores['fundamental'])],
            ["Sentiment Analysis", f"{scores['sentiment']:.2f}/100", self._get_score_bar(scores['sentiment'])],
            ["Momentum Analysis", f"{scores['momentum']:.2f}/100", self._get_score_bar(scores['momentum'])]
        ]
        print(tabulate(scores_data, headers=["Category", "Score", ""], tablefmt="grid"))
        
        # Key Insights
        self.print_section("💡 KEY INSIGHTS")
        for insight in result['key_insights']:
            print(f"  {insight}")
        
        # ML Price Predictions
        if result.get('ml_predictions') and 'horizons' in result['ml_predictions']:
            self.print_section("🔮 ML PRICE PREDICTIONS")
            ml_preds = result['ml_predictions']
            
            print(f"\nCurrent Price: NPR {ml_preds['current_price']:.2f}")
            
            # Display predictions table
            pred_data = []
            for horizon_key, horizon in ml_preds['horizons'].items():
                trend_symbol = "📈" if horizon['trend'] == 'UP' else "📉" if horizon['trend'] == 'DOWN' else "➡️"
                pred_data.append([
                    f"{horizon['weeks_ahead']} Week",
                    horizon['target_date'],
                    f"NPR {horizon['predicted_price']:.2f}",
                    f"{horizon['price_change_pct']:+.2f}%",
                    f"{trend_symbol} {horizon['trend']}",
                    f"{horizon['confidence_score']:.1f}%"
                ])
            
            print(tabulate(pred_data, 
                          headers=['Horizon', 'Target Date', 'Predicted Price', 'Change', 'Trend', 'Confidence'],
                          tablefmt='grid'))
            
            # Trend analysis summary
            if 'trend_analysis' in ml_preds:
                trend = ml_preds['trend_analysis']
                print(f"\n{'='*60}")
                print(f"Overall ML Trend: {Fore.GREEN if trend['overall_trend'] == 'BULLISH' else Fore.RED if trend['overall_trend'] == 'BEARISH' else Fore.YELLOW}{trend['overall_trend']}{Style.RESET_ALL}")
                print(f"Avg Predicted Change: {trend['avg_predicted_change']:+.2f}%")
                print(f"Avg Confidence: {trend['avg_confidence']:.1f}%")
                print(f"Best Outlook: {trend['best_horizon']['period']} ({trend['best_horizon']['change_pct']:+.2f}% → NPR {trend['best_horizon']['predicted_price']:.2f})")
                print(f"{'='*60}")
        
        # Warnings
        self.print_section("⚠️ WARNINGS & RISKS")
        for warning in result['warnings']:
            print(f"  {warning}")
        
        # Detailed analysis (if requested)
        if detailed:
            self._display_detailed_analysis(result)
    
    def _get_score_bar(self, score: float, width: int = 20) -> str:
        """Generate a visual score bar"""
        filled = int(width * score / 100)
        
        if score >= 70:
            color = Fore.GREEN
        elif score >= 40:
            color = Fore.YELLOW
        else:
            color = Fore.RED
        
        bar = '█' * filled + '░' * (width - filled)
        return f"{color}{bar}{Style.RESET_ALL}"
    
    def _display_detailed_analysis(self, result: dict):
        """Display detailed technical and fundamental metrics"""
        
        self.print_section("🔍 DETAILED ANALYSIS")
        print("\n[This would show detailed technical indicators, fundamentals, etc.]")
        print("Use --export option to save full report to file.")
    
    def compare_stocks(self, symbols: list, time_horizon: str = 'short'):
        """Compare multiple stocks"""
        
        self.print_header("NEPSE STOCK COMPARISON")
        
        print(f"{Fore.GREEN}⏳ Analyzing {len(symbols)} stocks... Please wait...{Style.RESET_ALL}\n")
        
        try:
            df = self.insights_engine.batch_analysis(symbols, time_horizon)
            
            if df.empty:
                print(f"{Fore.RED}❌ No data available for comparison{Style.RESET_ALL}")
                return
            
            print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
            
            # Top picks
            self.print_section("🏆 TOP PICKS")
            top_3 = df.head(3)
            
            for idx, row in top_3.iterrows():
                print(f"\n{idx + 1}. {Fore.GREEN}{row['Symbol']}{Style.RESET_ALL}")
                print(f"   Price: NPR {row['Price']:.2f}")
                print(f"   Probability: {row['Probability']:.2f}%")
                print(f"   Recommendation: {row['Recommendation']}")
                
        except Exception as e:
            print(f"{Fore.RED}❌ Comparison failed: {str(e)}{Style.RESET_ALL}")
    
    def market_overview(self):
        """Display market overview"""
        
        self.print_header("NEPSE MARKET OVERVIEW")
        
        try:
            print(f"{Fore.GREEN}📊 Fetching market data...{Style.RESET_ALL}\n")
            
            # Market summary
            summary = self.data_fetcher.get_market_summary()
            
            if summary:
                print("Market Summary:")
                for key, value in summary.items():
                    print(f"  {key}: {value}")
            
            # Top gainers
            self.print_section("📈 TOP GAINERS")
            gainers = self.data_fetcher.get_top_gainers(limit=5)
            if not gainers.empty:
                print(tabulate(gainers, headers='keys', tablefmt='grid', showindex=False))
            
            # Top losers
            self.print_section("📉 TOP LOSERS")
            losers = self.data_fetcher.get_top_losers(limit=5)
            if not losers.empty:
                print(tabulate(losers, headers='keys', tablefmt='grid', showindex=False))
                
        except Exception as e:
            print(f"{Fore.RED}❌ Failed to fetch market overview: {str(e)}{Style.RESET_ALL}")
    
    def export_analysis(self, symbol: str, filename: str = None):
        """Export analysis to JSON file"""
        
        if filename is None:
            filename = f"{symbol}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            result = self.insights_engine.calculate_profitability_probability(symbol)
            
            if 'error' in result:
                print(f"{Fore.RED}❌ Error: {result['error']}{Style.RESET_ALL}")
                return
            
            with open(filename, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"{Fore.GREEN}✅ Analysis exported to {filename}{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"{Fore.RED}❌ Export failed: {str(e)}{Style.RESET_ALL}")
    
    def analyze_broker_activity(self, symbol: str = None, days: int = 30):
        """Analyze broker trading patterns and detect manipulation"""
        
        if symbol:
            self.print_header(f"BROKER ANALYSIS: {symbol}")
        else:
            self.print_header("BROKER ANALYSIS: MARKET WIDE")
        
        print(f"{Fore.GREEN}⏳ Fetching floorsheet data... Please wait...{Style.RESET_ALL}\n")
        
        try:
            # Get floorsheet data
            floorsheet = self.data_fetcher.get_floorsheet_data(symbol, days)
            
            if floorsheet.empty:
                print(f"{Fore.RED}❌ No floorsheet data available{Style.RESET_ALL}")
                return
            
            # Analyze broker activity
            if symbol:
                analysis = self.broker_analyzer.comprehensive_broker_analysis(
                    floorsheet, symbol, days
                )
            else:
                # Analyze all stocks
                print(f"{Fore.YELLOW}Analyzing all stocks in floorsheet...{Style.RESET_ALL}\n")
                symbols = floorsheet['symbol'].unique()
                print(f"Found {len(symbols)} stocks with recent activity\n")
                
                # Analyze each stock
                results = []
                for sym in symbols[:10]:  # Limit to top 10 for display
                    try:
                        analysis = self.broker_analyzer.comprehensive_broker_analysis(
                            floorsheet, sym, days
                        )
                        results.append(analysis)
                    except:
                        continue
                
                # Display summary
                self._display_broker_summary(results)
                return
            
            # Display detailed broker analysis
            self._display_broker_analysis(analysis)
            
        except Exception as e:
            print(f"{Fore.RED}❌ Broker analysis failed: {str(e)}{Style.RESET_ALL}")
    
    def _display_broker_analysis(self, analysis: dict):
        """Display detailed broker analysis results"""
        
        if 'error' in analysis:
            print(f"{Fore.RED}❌ {analysis['error']}{Style.RESET_ALL}")
            return
        
        # Overall Score
        self.print_section("🏆 OVERALL BROKER SCORE")
        score = analysis['overall_broker_score']
        
        score_val = score['overall_score']
        if score_val >= 80:
            score_color = Fore.GREEN
        elif score_val >= 60:
            score_color = Fore.YELLOW
        else:
            score_color = Fore.RED
        
        print(f"\n{score_color}Overall Score: {score_val:.2f}/100{Style.RESET_ALL}")
        print(f"Grade: {score['grade']}")
        print(f"\nRecommendation: {analysis['recommendation']}\n")
        
        # Component scores
        print("Component Breakdown:")
        comp_data = [
            ["Concentration (Lower is Better)", f"{score['component_scores']['concentration']:.2f}/100"],
            ["Smart Money Activity", f"{score['component_scores']['smart_money']:.2f}/100"],
            ["Buy/Sell Pressure", f"{score['component_scores']['pressure']:.2f}/100"],
            ["Liquidity", f"{score['component_scores']['liquidity']:.2f}/100"],
            ["Manipulation Risk (Inverted)", f"{score['component_scores']['manipulation_risk']:.2f}/100"]
        ]
        print(tabulate(comp_data, tablefmt="grid"))
        
        # Manipulation Risk
        self.print_section("🚨 MANIPULATION RISK ANALYSIS")
        manip = analysis['manipulation_risk']
        
        risk_score = manip['risk_score']
        if risk_score >= 70:
            risk_color = Fore.RED
        elif risk_score >= 50:
            risk_color = Fore.YELLOW
        else:
            risk_color = Fore.GREEN
        
        print(f"\n{risk_color}Risk Level: {manip['risk_level']}{Style.RESET_ALL}")
        print(f"Risk Score: {risk_score}/100")
        print(f"Recommendation: {manip['recommendation']}\n")
        
        if manip['risk_flags']:
            print(f"{Fore.RED}⚠️ Risk Flags:{Style.RESET_ALL}")
            for flag in manip['risk_flags']:
                print(f"  • {flag}")
        else:
            print(f"{Fore.GREEN}✓ No manipulation risk flags detected{Style.RESET_ALL}")
        
        # Broker Concentration
        self.print_section("📊 BROKER CONCENTRATION")
        conc = analysis['broker_concentration']
        
        print(f"Unique Buyers: {conc['total_unique_buyers']}")
        print(f"Unique Sellers: {conc['total_unique_sellers']}")
        print(f"\nBuyer Concentration: {conc['buyer_concentration_level']}")
        print(f"Seller Concentration: {conc['seller_concentration_level']}\n")
        
        print("Top 5 Buyers:")
        top_buyers_data = [
            [b['broker'], f"{b['volume']:,}", f"{b['percentage']:.2f}%"]
            for b in conc['top_buyers']
        ]
        print(tabulate(top_buyers_data, headers=['Broker', 'Volume', '% of Total'], tablefmt='grid'))
        
        print("\nTop 5 Sellers:")
        top_sellers_data = [
            [s['broker'], f"{s['volume']:,}", f"{s['percentage']:.2f}%"]
            for s in conc['top_sellers']
        ]
        print(tabulate(top_sellers_data, headers=['Broker', 'Volume', '% of Total'], tablefmt='grid'))
        
        # Smart Money Flow
        self.print_section("💼 SMART MONEY FLOW")
        smart = analysis['smart_money_flow']
        
        signal = smart['smart_money_signal']
        if 'BULLISH' in signal:
            signal_color = Fore.GREEN
        elif 'BEARISH' in signal:
            signal_color = Fore.RED
        else:
            signal_color = Fore.YELLOW
        
        print(f"\n{signal_color}Signal: {signal}{Style.RESET_ALL}")
        print(f"Net Institutional Flow: {smart['net_institutional_flow']:,} shares")
        print(f"Accumulation Score: {smart['accumulation_score']}/100\n")
        
        print("Large Volume Brokers (Top 10):")
        smart_data = [
            [b['broker'], f"{b['total_bought']:,}", f"{b['total_sold']:,}", 
             f"{b['net_position']:,}", b['activity_type'], b['signal']]
            for b in smart['large_volume_brokers']
        ]
        print(tabulate(smart_data, headers=['Broker', 'Bought', 'Sold', 'Net', 'Activity', 'Signal'], 
                      tablefmt='grid'))
        
        # Buy/Sell Pressure
        self.print_section("⚖️ BUY/SELL PRESSURE")
        pressure = analysis['buy_sell_pressure']
        
        pressure_type = pressure['pressure_type']
        if 'BUYING' in pressure_type:
            pressure_color = Fore.GREEN
        elif 'SELLING' in pressure_type:
            pressure_color = Fore.RED
        else:
            pressure_color = Fore.YELLOW
        
        print(f"\n{pressure_color}{pressure_type}{Style.RESET_ALL}")
        print(f"Imbalance Ratio: {pressure['buy_sell_imbalance_ratio']:.2f}")
        print(f"VWAP: NPR {pressure['vwap']:.2f}")
        print(f"Price Change: {pressure['price_change_percent']:.2f}%\n")
        
        pressure_data = [
            ["Total Transactions", f"{pressure['total_transactions']:,}"],
            ["Total Volume", f"{pressure['total_volume']:,}"],
            ["Aggressive Buy Volume", f"{pressure['aggressive_buy_volume']:,}"],
            ["Aggressive Sell Volume", f"{pressure['aggressive_sell_volume']:,}"]
        ]
        print(tabulate(pressure_data, tablefmt='grid'))
        
        # Liquidity Analysis
        self.print_section("💧 LIQUIDITY ANALYSIS")
        liquidity = analysis['liquidity_metrics']
        
        print(f"\nLiquidity: {liquidity['liquidity_classification']}")
        print(f"Trade Safety: {liquidity['trade_safety']}")
        print(f"Liquidity Score: {liquidity['liquidity_score']}/100\n")
        
        liq_data = [
            ["Total Volume", f"{liquidity['total_volume']:,}"],
            ["Total Value", f"NPR {liquidity['total_value']:,.2f}"],
            ["Unique Brokers", f"{liquidity['unique_brokers']}"],
            ["Transactions/Day", f"{liquidity['transactions_per_day']:.2f}"],
            ["Avg Transaction Size", f"{liquidity['avg_transaction_size']:,.2f}"],
            ["Price Volatility", f"{liquidity['price_volatility_percent']:.2f}%"]
        ]
        print(tabulate(liq_data, tablefmt='grid'))
    
    def _display_broker_summary(self, results: list):
        """Display summary of multiple stocks' broker analysis"""
        
        self.print_section("📊 BROKER ANALYSIS SUMMARY")
        
        summary_data = []
        for res in results:
            if 'error' in res:
                continue
            
            score = res['overall_broker_score']['overall_score']
            manip_risk = res['manipulation_risk']['risk_score']
            
            summary_data.append([
                res['symbol'],
                f"{score:.1f}",
                res['overall_broker_score']['grade'],
                f"{manip_risk:.1f}",
                res['smart_money_flow']['smart_money_signal'][:20]
            ])
        
        print(tabulate(summary_data, 
                      headers=['Symbol', 'Score', 'Grade', 'Manip Risk', 'Smart Money'],
                      tablefmt='grid'))


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description='NEPSE Stock Analysis - Comprehensive stock analysis tool for Nepal Stock Exchange',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single stock
  python main.py analyze NABIL
  
  # Analyze with medium-term horizon
  python main.py analyze NABIL --horizon medium
  
  # Compare multiple stocks
  python main.py compare NABIL NICA GBIME
  
  # Market overview
  python main.py market
  
  # Broker analysis for a stock
  python main.py broker NABIL --days 30
  
  # Market-wide broker analysis
  python main.py broker --days 7
  
  # Export analysis to file
  python main.py analyze NABIL --export
  
  # Import real data from CSV
  python main.py import my_data.csv price
  python main.py import floorsheet.csv floorsheet
  
  # Import from Excel with specific sheet
  python main.py import nepse_data.xlsx price --sheet "Price History"
  
  # Show guide for getting real NEPSE data
  python main.py sources
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a single stock')
    analyze_parser.add_argument('symbol', help='Stock symbol (e.g., NABIL)')
    analyze_parser.add_argument('--horizon', choices=['short', 'medium', 'long'], 
                               default='short', help='Time horizon for analysis')
    analyze_parser.add_argument('--detailed', action='store_true', 
                               help='Show detailed analysis')
    analyze_parser.add_argument('--export', action='store_true', 
                               help='Export analysis to JSON file')
    analyze_parser.add_argument('--visualize', action='store_true',
                               help='Generate visual diagram of the analysis pipeline')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare multiple stocks')
    compare_parser.add_argument('symbols', nargs='+', help='Stock symbols to compare')
    compare_parser.add_argument('--horizon', choices=['short', 'medium', 'long'], 
                               default='short', help='Time horizon for analysis')
    
    # Market overview command
    market_parser = subparsers.add_parser('market', help='Show market overview')
    
    # Broker analysis command
    broker_parser = subparsers.add_parser('broker', help='Analyze broker trading patterns')
    broker_parser.add_argument('symbol', nargs='?', help='Stock symbol (optional - analyzes all if omitted)')
    broker_parser.add_argument('--days', type=int, default=30, 
                              help='Number of days to analyze (default: 30)')
    
    # Visualize process command (NEW)
    viz_parser = subparsers.add_parser('visualize', help='Show analysis pipeline visualization')
    viz_parser.add_argument('--symbol', default='IGI', help='Symbol for context (default: IGI)')
    viz_parser.add_argument('--save', help='Save to file instead of displaying')
    
    # Sync command (NEW)
    sync_parser = subparsers.add_parser('sync', help='Sync data for stocks (incremental updates)')
    sync_parser.add_argument('symbols', nargs='*', help='Stock symbols to sync (omit for all)')
    sync_parser.add_argument('--price', action='store_true', help='Sync price history')
    sync_parser.add_argument('--news', action='store_true', help='Sync news articles')
    sync_parser.add_argument('--all', action='store_true', help='Sync all NEPSE stocks (328 companies)')
    
    # Analyze-all command (NEW)
    analyze_all_parser = subparsers.add_parser('analyze-all', help='Analyze all NEPSE stocks')
    analyze_all_parser.add_argument('--output', default='nepse_analysis_results.csv', 
                                   help='Output CSV file (default: nepse_analysis_results.csv)')
    analyze_all_parser.add_argument('--limit', type=int, help='Limit number of stocks to analyze')
    analyze_all_parser.add_argument('--min-data', type=int, default=100, 
                                   help='Minimum days of price data required (default: 100)')
    
    # Import data command (NEW)
    import_parser = subparsers.add_parser('import', help='Import real NEPSE data from CSV/Excel files')
    import_parser.add_argument('file', help='Path to CSV or Excel file')
    import_parser.add_argument('type', choices=['price', 'floorsheet', 'companies'], 
                              help='Type of data to import')
    import_parser.add_argument('--sheet', help='Excel sheet name (for .xlsx/.xls files)')
    
    # Data sources guide command (NEW)
    subparsers.add_parser('sources', help='Show guide for obtaining real NEPSE data')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Create analyzer instance
    analyzer = NepseStockAnalyzer()
    
    # Execute command
    if args.command == 'analyze':
        result = analyzer.analyze_stock(
            args.symbol.upper(), 
            args.horizon, 
            args.detailed
        )
        
        if args.export:
            analyzer.export_analysis(args.symbol.upper())
        
        if args.visualize:
            # Generate visualization
            try:
                from analysis_visualizer import AnalysisVisualizer
                print(f"\n{Fore.CYAN}📊 Generating analysis visualization...{Style.RESET_ALL}")
                visualizer = AnalysisVisualizer()
                # Get the full analysis result again with all details
                full_result = analyzer.insights_engine.calculate_profitability_probability(
                    args.symbol.upper(), 
                    args.horizon
                )
                filename = visualizer.visualize_complete_analysis(full_result, args.symbol.upper())
                print(f"{Fore.GREEN}✓ Visualization saved: {filename}{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}✗ Visualization failed: {e}{Style.RESET_ALL}")
                import traceback
                traceback.print_exc()
            
    elif args.command == 'compare':
        symbols = [s.upper() for s in args.symbols]
        analyzer.compare_stocks(symbols, args.horizon)
    
    elif args.command == 'market':
        analyzer.market_overview()
    
    elif args.command == 'broker':
        if args.symbol:
            analyzer.analyze_broker_activity(args.symbol.upper(), args.days)
        else:
            analyzer.analyze_broker_activity(None, args.days)
    
    elif args.command == 'sync':
        # Sync data
        from sync_manager import SyncManager
        from symbol_scraper import ShareSansarSymbolScraper
        
        sync_manager = SyncManager()
        
        # Get symbols to sync
        if args.all:
            # Get all NEPSE stocks
            print(f"{Fore.CYAN}🔄 Fetching all NEPSE stock symbols...{Style.RESET_ALL}")
            scraper = ShareSansarSymbolScraper(headless=True)
            try:
                symbols = scraper.get_all_symbols(save_to_db=True)
                scraper.close_driver()
                print(f"{Fore.GREEN}✓ Found {len(symbols)} stocks{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}✗ Error fetching symbols: {e}{Style.RESET_ALL}")
                return
        elif args.symbols:
            symbols = [s.upper() for s in args.symbols]
        else:
            print(f"{Fore.RED}✗ Please specify symbols or use --all{Style.RESET_ALL}")
            return
        
        # Determine what to sync
        sync_price = args.price or (not args.price and not args.news)  # Default to both if neither specified
        sync_news = args.news or (not args.price and not args.news)
        
        # Bulk sync
        results = sync_manager.bulk_sync(symbols, sync_price=sync_price, sync_news=sync_news)
        
        print(f"\n{Fore.GREEN}✓ Sync complete!{Style.RESET_ALL}")
    
    elif args.command == 'analyze-all':
        # Analyze all NEPSE stocks
        import sqlite3
        from datetime import datetime
        
        print(f"\n{Fore.CYAN}📊 Analyzing all NEPSE stocks...{Style.RESET_ALL}\n")
        
        # Get all symbols from database
        conn = sqlite3.connect(config.DB_PATH)
        symbols_df = pd.read_sql("SELECT DISTINCT symbol FROM companies ORDER BY symbol", conn)
        all_symbols = symbols_df['symbol'].tolist()
        conn.close()
        
        if args.limit:
            all_symbols = all_symbols[:args.limit]
        
        print(f"Found {len(all_symbols)} stocks to analyze")
        
        results = []
        successful = 0
        failed = 0
        
        for i, symbol in enumerate(all_symbols, 1):
            try:
                print(f"\n[{i}/{len(all_symbols)}] Analyzing {symbol}...")
                
                # Quick check: does it have enough data?
                conn = sqlite3.connect(config.DB_PATH)
                count = conn.execute(
                    "SELECT COUNT(*) FROM price_history WHERE UPPER(symbol) = UPPER(?)", 
                    (symbol,)
                ).fetchone()[0]
                conn.close()
                
                if count < args.min_data:
                    print(f"  ⚠️ Skipping {symbol}: only {count} days of data (need {args.min_data})")
                    failed += 1
                    continue
                
                # Run analysis (use cache to avoid re-scraping)
                result = analyzer.insights_engine.calculate_profitability_probability(
                    symbol, 
                    time_horizon='short',
                    include_broker_analysis=False,  # Skip broker analysis for speed
                    use_cache=True
                )
                
                if 'error' not in result:
                    results.append({
                        'Symbol': symbol,
                        'Probability': result['profitability_probability'],
                        'Recommendation': result['recommendation']['action'],
                        'Technical_Score': result['scores']['technical'],
                        'Fundamental_Score': result['scores']['fundamental'],
                        'Sentiment_Score': result['scores']['sentiment'],
                        'Current_Price': result['current_price'],
                        'Risk_Reward_Ratio': result['risk_reward_ratio'].get('ratio', 0),
                        'Analysis_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                    successful += 1
                    print(f"  ✓ {result['profitability_probability']:.1f}% probability | {result['recommendation']['action']}")
                else:
                    failed += 1
                    print(f"  ✗ Analysis failed: {result.get('error', 'Unknown error')}")
            
            except Exception as e:
                failed += 1
                print(f"  ✗ Error: {e}")
                continue
        
        # Save results
        if results:
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('Probability', ascending=False)
            results_df.to_csv(args.output, index=False)
            
            print(f"\n{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
            print(f"{Fore.GREEN}ANALYSIS COMPLETE{Style.RESET_ALL}")
            print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
            print(f"✓ Successful: {successful}/{len(all_symbols)}")
            print(f"✗ Failed/Skipped: {failed}")
            print(f"📄 Results saved to: {args.output}")
            
            # Show top 10
            print(f"\n{Fore.CYAN}🏆 TOP 10 STOCKS BY PROBABILITY:{Style.RESET_ALL}")
            print(tabulate(results_df.head(10), headers='keys', tablefmt='grid', showindex=False))
        else:
            print(f"\n{Fore.RED}✗ No successful analyses{Style.RESET_ALL}")
    
    elif args.command == 'visualize':
        # Show process flow visualization
        try:
            from analysis_visualizer import AnalysisVisualizer
            import matplotlib.pyplot as plt
            print(f"\n{Fore.CYAN}📊 Generating analysis pipeline visualization...{Style.RESET_ALL}")
            visualizer = AnalysisVisualizer()
            fig = visualizer.visualize_process_flow(args.symbol.upper())
            
            if args.save:
                fig.savefig(args.save, dpi=300, bbox_inches='tight')
                print(f"{Fore.GREEN}✓ Visualization saved: {args.save}{Style.RESET_ALL}")
            else:
                print(f"{Fore.GREEN}✓ Displaying visualization...{Style.RESET_ALL}")
                plt.show()
        except Exception as e:
            print(f"{Fore.RED}✗ Visualization failed: {e}{Style.RESET_ALL}")
            import traceback
            traceback.print_exc()
    
    elif args.command == 'import':
        # Import real data
        adapter = RealDataAdapter()
        file_path = args.file
        
        if not Path(file_path).exists():
            print(f"{Fore.RED}✗ File not found: {file_path}{Style.RESET_ALL}")
            return
        
        # Map command type to database type
        type_map = {
            'price': 'price_history',
            'floorsheet': 'floorsheet',
            'companies': 'companies'
        }
        data_type = type_map[args.type]
        
        # Import based on file extension
        if file_path.endswith('.csv'):
            adapter.import_from_csv(file_path, data_type)
        elif file_path.endswith(('.xlsx', '.xls')):
            sheet = args.sheet or 0
            adapter.import_from_excel(file_path, data_type, sheet)
        else:
            print(f"{Fore.RED}✗ Unsupported file format. Use .csv or .xlsx{Style.RESET_ALL}")
    
    elif args.command == 'sources':
        # Show data sources guide
        adapter = RealDataAdapter()
        adapter.get_data_sources_guide()


if __name__ == "__main__":
    main()
