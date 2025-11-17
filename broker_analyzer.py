"""Broker Analysis Module for NEPSE Stock Trading
Analyzes broker activity to detect manipulation, track smart money, and assess liquidity
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import config


class BrokerAnalyzer:
    """Analyze broker trading patterns to detect manipulation and smart money"""
    
    def __init__(self):
        self.config = config.BROKER_ANALYSIS_CONFIG
    
    def comprehensive_broker_analysis(self, 
                                     floorsheet_data: pd.DataFrame,
                                     symbol: str,
                                     days: int = 30) -> Dict:
        """
        Perform complete broker analysis
        
        Args:
            floorsheet_data: DataFrame with columns [date, broker_no, symbol, quantity, rate, amount, buyer_broker, seller_broker]
            symbol: Stock symbol
            days: Analysis period
        
        Returns:
            Dict with all broker analysis metrics
        """
        # Filter for specific symbol
        stock_data = floorsheet_data[floorsheet_data['symbol'] == symbol].copy()
        
        if stock_data.empty:
            return {'error': f'No floorsheet data found for {symbol}'}
        
        # Run all analyses
        concentration = self.analyze_broker_concentration(stock_data)
        smart_money = self.track_smart_money(stock_data)
        pressure = self.calculate_buy_sell_pressure(stock_data)
        liquidity = self.analyze_liquidity(stock_data)
        manipulation_risk = self.detect_manipulation_risk(stock_data, concentration)
        
        # Aggregate scores
        overall_score = self._calculate_overall_broker_score(
            concentration, smart_money, pressure, liquidity, manipulation_risk
        )
        
        return {
            'symbol': symbol,
            'analysis_date': datetime.now().isoformat(),
            'data_period_days': days,
            'total_transactions': len(stock_data),
            'broker_concentration': concentration,
            'smart_money_flow': smart_money,
            'buy_sell_pressure': pressure,
            'liquidity_metrics': liquidity,
            'manipulation_risk': manipulation_risk,
            'overall_broker_score': overall_score,
            'recommendation': self._generate_broker_recommendation(overall_score, manipulation_risk)
        }
    
    def analyze_broker_concentration(self, stock_data: pd.DataFrame) -> Dict:
        """
        Analyze how concentrated trading is among brokers
        
        High concentration = Few brokers dominating = Manipulation risk
        Low concentration = Many brokers active = Healthy market
        """
        # Aggregate by buyer broker
        buyer_volume = stock_data.groupby('buyer_broker')['quantity'].sum().sort_values(ascending=False)
        seller_volume = stock_data.groupby('seller_broker')['quantity'].sum().sort_values(ascending=False)
        
        total_volume = stock_data['quantity'].sum()
        
        # Calculate concentration metrics
        top_1_buyer_pct = (buyer_volume.iloc[0] / total_volume * 100) if len(buyer_volume) > 0 else 0
        top_3_buyer_pct = (buyer_volume.head(3).sum() / total_volume * 100) if len(buyer_volume) >= 3 else 0
        top_5_buyer_pct = (buyer_volume.head(5).sum() / total_volume * 100) if len(buyer_volume) >= 5 else 0
        
        top_1_seller_pct = (seller_volume.iloc[0] / total_volume * 100) if len(seller_volume) > 0 else 0
        top_3_seller_pct = (seller_volume.head(3).sum() / total_volume * 100) if len(seller_volume) >= 3 else 0
        top_5_seller_pct = (seller_volume.head(5).sum() / total_volume * 100) if len(seller_volume) >= 5 else 0
        
        # Herfindahl-Hirschman Index (HHI) - measures market concentration
        # HHI = sum of squared market shares (0-10000)
        # <1500: Competitive, 1500-2500: Moderate concentration, >2500: High concentration
        buyer_market_shares = (buyer_volume / total_volume * 100) ** 2
        seller_market_shares = (seller_volume / total_volume * 100) ** 2
        
        hhi_buyers = buyer_market_shares.sum()
        hhi_sellers = seller_market_shares.sum()
        
        # Determine concentration level
        def get_concentration_level(hhi, top_3_pct):
            if hhi > 2500 or top_3_pct > 70:
                return 'CRITICAL - Very High Concentration'
            elif hhi > 1500 or top_3_pct > 50:
                return 'WARNING - High Concentration'
            elif hhi > 800 or top_3_pct > 35:
                return 'MODERATE - Some Concentration'
            else:
                return 'HEALTHY - Well Distributed'
        
        return {
            'total_unique_buyers': len(buyer_volume),
            'total_unique_sellers': len(seller_volume),
            'top_buyers': [
                {'broker': idx, 'volume': int(vol), 'percentage': round(vol/total_volume*100, 2)}
                for idx, vol in buyer_volume.head(5).items()
            ],
            'top_sellers': [
                {'broker': idx, 'volume': int(vol), 'percentage': round(vol/total_volume*100, 2)}
                for idx, vol in seller_volume.head(5).items()
            ],
            'concentration_metrics': {
                'top_1_buyer_percentage': round(top_1_buyer_pct, 2),
                'top_3_buyers_percentage': round(top_3_buyer_pct, 2),
                'top_5_buyers_percentage': round(top_5_buyer_pct, 2),
                'top_1_seller_percentage': round(top_1_seller_pct, 2),
                'top_3_sellers_percentage': round(top_3_seller_pct, 2),
                'top_5_sellers_percentage': round(top_5_seller_pct, 2),
                'hhi_buyers': round(hhi_buyers, 2),
                'hhi_sellers': round(hhi_sellers, 2),
            },
            'buyer_concentration_level': get_concentration_level(hhi_buyers, top_3_buyer_pct),
            'seller_concentration_level': get_concentration_level(hhi_sellers, top_3_seller_pct),
            'concentration_risk_score': self._calculate_concentration_risk(hhi_buyers, hhi_sellers, top_3_buyer_pct)
        }
    
    def track_smart_money(self, stock_data: pd.DataFrame) -> Dict:
        """
        Track institutional/smart money flows
        
        Identifies brokers with:
        - Large consistent volumes
        - Sustained accumulation/distribution patterns
        - High average transaction sizes
        """
        # Identify large-volume brokers (top 20% by volume)
        buyer_volume = stock_data.groupby('buyer_broker')['quantity'].sum()
        seller_volume = stock_data.groupby('seller_broker')['quantity'].sum()
        
        volume_threshold = buyer_volume.quantile(0.80)  # Top 20%
        
        large_buyers = buyer_volume[buyer_volume >= volume_threshold].index.tolist()
        large_sellers = seller_volume[seller_volume >= volume_threshold].index.tolist()
        
        # Calculate net position for each large broker
        broker_flows = []
        
        all_large_brokers = set(large_buyers + large_sellers)
        
        for broker in all_large_brokers:
            bought = stock_data[stock_data['buyer_broker'] == broker]['quantity'].sum()
            sold = stock_data[stock_data['seller_broker'] == broker]['quantity'].sum()
            net_position = bought - sold
            
            # Calculate average transaction size
            buy_transactions = stock_data[stock_data['buyer_broker'] == broker]
            sell_transactions = stock_data[stock_data['seller_broker'] == broker]
            
            avg_buy_size = buy_transactions['quantity'].mean() if len(buy_transactions) > 0 else 0
            avg_sell_size = sell_transactions['quantity'].mean() if len(sell_transactions) > 0 else 0
            
            # Determine activity type
            if net_position > 0 and bought > sold * 1.2:
                activity = 'ACCUMULATING'
                signal = 'BULLISH'
            elif net_position < 0 and sold > bought * 1.2:
                activity = 'DISTRIBUTING'
                signal = 'BEARISH'
            else:
                activity = 'NEUTRAL'
                signal = 'NEUTRAL'
            
            broker_flows.append({
                'broker': broker,
                'total_bought': int(bought),
                'total_sold': int(sold),
                'net_position': int(net_position),
                'avg_buy_size': round(avg_buy_size, 2),
                'avg_sell_size': round(avg_sell_size, 2),
                'activity_type': activity,
                'signal': signal
            })
        
        # Sort by absolute net position
        broker_flows = sorted(broker_flows, key=lambda x: abs(x['net_position']), reverse=True)
        
        # Calculate overall smart money flow
        total_accumulation = sum(b['net_position'] for b in broker_flows if b['net_position'] > 0)
        total_distribution = abs(sum(b['net_position'] for b in broker_flows if b['net_position'] < 0))
        
        net_institutional_flow = total_accumulation - total_distribution
        
        # Determine overall smart money signal
        if net_institutional_flow > 0 and total_accumulation > total_distribution * 1.3:
            smart_money_signal = 'STRONG BULLISH - Institutions Accumulating'
        elif net_institutional_flow > 0:
            smart_money_signal = 'BULLISH - Net Institutional Buying'
        elif net_institutional_flow < 0 and total_distribution > total_accumulation * 1.3:
            smart_money_signal = 'STRONG BEARISH - Institutions Distributing'
        elif net_institutional_flow < 0:
            smart_money_signal = 'BEARISH - Net Institutional Selling'
        else:
            smart_money_signal = 'NEUTRAL - Balanced Institutional Activity'
        
        return {
            'large_volume_brokers': broker_flows[:10],  # Top 10
            'total_accumulation': int(total_accumulation),
            'total_distribution': int(total_distribution),
            'net_institutional_flow': int(net_institutional_flow),
            'smart_money_signal': smart_money_signal,
            'accumulation_score': self._calculate_accumulation_score(net_institutional_flow, total_accumulation, total_distribution)
        }
    
    def calculate_buy_sell_pressure(self, stock_data: pd.DataFrame) -> Dict:
        """
        Calculate buy/sell pressure and imbalance ratios
        
        Strong buy pressure > 1.2 = Bullish momentum
        Strong sell pressure < 0.8 = Bearish momentum
        """
        total_buy_volume = stock_data['quantity'].sum()
        total_sell_volume = stock_data['quantity'].sum()  # Equal in floorsheet, but weighted differently
        
        # Calculate VWAP (Volume Weighted Average Price)
        stock_data['value'] = stock_data['quantity'] * stock_data['rate']
        vwap = stock_data['value'].sum() / stock_data['quantity'].sum()
        
        # Transactions above VWAP (aggressive buying) vs below VWAP (aggressive selling)
        above_vwap = stock_data[stock_data['rate'] > vwap]
        below_vwap = stock_data[stock_data['rate'] < vwap]
        
        aggressive_buy_volume = above_vwap['quantity'].sum()
        aggressive_sell_volume = below_vwap['quantity'].sum()
        
        # Buy/Sell imbalance ratio
        if aggressive_sell_volume > 0:
            imbalance_ratio = aggressive_buy_volume / aggressive_sell_volume
        else:
            imbalance_ratio = 10.0 if aggressive_buy_volume > 0 else 1.0
        
        # Pressure classification
        if imbalance_ratio > 1.5:
            pressure_type = 'STRONG BUYING PRESSURE'
            pressure_signal = 'BULLISH'
        elif imbalance_ratio > 1.2:
            pressure_type = 'MODERATE BUYING PRESSURE'
            pressure_signal = 'BULLISH'
        elif imbalance_ratio < 0.67:
            pressure_type = 'STRONG SELLING PRESSURE'
            pressure_signal = 'BEARISH'
        elif imbalance_ratio < 0.83:
            pressure_type = 'MODERATE SELLING PRESSURE'
            pressure_signal = 'BEARISH'
        else:
            pressure_type = 'BALANCED PRESSURE'
            pressure_signal = 'NEUTRAL'
        
        # Calculate price momentum from transactions
        stock_data_sorted = stock_data.sort_values('date')
        if len(stock_data_sorted) > 1:
            first_price = stock_data_sorted.iloc[0]['rate']
            last_price = stock_data_sorted.iloc[-1]['rate']
            price_change_pct = ((last_price - first_price) / first_price * 100) if first_price > 0 else 0
        else:
            price_change_pct = 0
        
        return {
            'total_transactions': len(stock_data),
            'total_volume': int(total_buy_volume),
            'vwap': round(vwap, 2),
            'aggressive_buy_volume': int(aggressive_buy_volume),
            'aggressive_sell_volume': int(aggressive_sell_volume),
            'buy_sell_imbalance_ratio': round(imbalance_ratio, 2),
            'pressure_type': pressure_type,
            'pressure_signal': pressure_signal,
            'price_change_percent': round(price_change_pct, 2),
            'pressure_score': self._calculate_pressure_score(imbalance_ratio, price_change_pct)
        }
    
    def analyze_liquidity(self, stock_data: pd.DataFrame) -> Dict:
        """
        Analyze liquidity and market depth
        
        High liquidity = Safe for large trades
        Low liquidity = High slippage risk, manipulation risk
        """
        # Basic liquidity metrics
        total_volume = stock_data['quantity'].sum()
        total_value = (stock_data['quantity'] * stock_data['rate']).sum()
        unique_brokers = len(set(stock_data['buyer_broker'].unique().tolist() + 
                                 stock_data['seller_broker'].unique().tolist()))
        
        # Transaction frequency (transactions per day)
        if 'date' in stock_data.columns:
            date_range = (stock_data['date'].max() - stock_data['date'].min()).days
            if date_range == 0:
                date_range = 1
            transactions_per_day = len(stock_data) / date_range
        else:
            transactions_per_day = len(stock_data)
        
        # Average transaction size
        avg_transaction_size = stock_data['quantity'].mean()
        median_transaction_size = stock_data['quantity'].median()
        
        # Price spread analysis
        price_std = stock_data['rate'].std()
        avg_price = stock_data['rate'].mean()
        volatility_pct = (price_std / avg_price * 100) if avg_price > 0 else 0
        
        # Liquidity classification
        liquidity_score = 0
        
        # Volume-based scoring
        if total_volume > 50000:
            liquidity_score += 30
        elif total_volume > 20000:
            liquidity_score += 20
        elif total_volume > 5000:
            liquidity_score += 10
        
        # Broker diversity scoring
        if unique_brokers > 30:
            liquidity_score += 30
        elif unique_brokers > 15:
            liquidity_score += 20
        elif unique_brokers > 5:
            liquidity_score += 10
        
        # Transaction frequency scoring
        if transactions_per_day > 100:
            liquidity_score += 25
        elif transactions_per_day > 50:
            liquidity_score += 15
        elif transactions_per_day > 10:
            liquidity_score += 5
        
        # Volatility scoring (lower volatility = better liquidity)
        if volatility_pct < 2:
            liquidity_score += 15
        elif volatility_pct < 5:
            liquidity_score += 10
        elif volatility_pct < 10:
            liquidity_score += 5
        
        # Classify liquidity
        if liquidity_score >= 80:
            liquidity_class = 'EXCELLENT - Highly Liquid'
            trade_safety = 'SAFE for large positions'
        elif liquidity_score >= 60:
            liquidity_class = 'GOOD - Liquid'
            trade_safety = 'SAFE for medium positions'
        elif liquidity_score >= 40:
            liquidity_class = 'MODERATE - Fairly Liquid'
            trade_safety = 'CAUTION for large positions'
        elif liquidity_score >= 20:
            liquidity_class = 'LOW - Thinly Traded'
            trade_safety = 'HIGH RISK - Small positions only'
        else:
            liquidity_class = 'VERY LOW - Illiquid'
            trade_safety = 'VERY HIGH RISK - Avoid or micro positions'
        
        return {
            'total_volume': int(total_volume),
            'total_value': round(total_value, 2),
            'unique_brokers': unique_brokers,
            'transactions_per_day': round(transactions_per_day, 2),
            'avg_transaction_size': round(avg_transaction_size, 2),
            'median_transaction_size': round(median_transaction_size, 2),
            'price_volatility_percent': round(volatility_pct, 2),
            'liquidity_score': liquidity_score,
            'liquidity_classification': liquidity_class,
            'trade_safety': trade_safety
        }
    
    def detect_manipulation_risk(self, stock_data: pd.DataFrame, concentration_data: Dict) -> Dict:
        """
        Detect potential market manipulation patterns
        
        Red flags:
        - Single broker > 60% of volume
        - Top 3 brokers > 80% of volume
        - Coordinated buying followed by selling
        - Unusual price movements with low volume
        """
        risk_flags = []
        risk_score = 0
        
        # Check concentration metrics
        top_1_buyer = concentration_data['concentration_metrics']['top_1_buyer_percentage']
        top_3_buyers = concentration_data['concentration_metrics']['top_3_buyers_percentage']
        hhi_buyers = concentration_data['concentration_metrics']['hhi_buyers']
        
        # Flag 1: Single broker dominance
        if top_1_buyer > 60:
            risk_flags.append('CRITICAL: Single broker controls >60% of buying volume')
            risk_score += 40
        elif top_1_buyer > 45:
            risk_flags.append('WARNING: Single broker controls >45% of buying volume')
            risk_score += 25
        
        # Flag 2: Top 3 broker dominance
        if top_3_buyers > 80:
            risk_flags.append('CRITICAL: Top 3 brokers control >80% of volume')
            risk_score += 35
        elif top_3_buyers > 70:
            risk_flags.append('WARNING: Top 3 brokers control >70% of volume')
            risk_score += 20
        
        # Flag 3: Very high HHI
        if hhi_buyers > 3000:
            risk_flags.append('CRITICAL: Extremely high market concentration (HHI > 3000)')
            risk_score += 30
        elif hhi_buyers > 2500:
            risk_flags.append('WARNING: High market concentration (HHI > 2500)')
            risk_score += 15
        
        # Flag 4: Coordinated pump-and-dump pattern
        # Check if same brokers are both top buyers and sellers
        top_buyers = [b['broker'] for b in concentration_data['top_buyers'][:3]]
        top_sellers = [b['broker'] for b in concentration_data['top_sellers'][:3]]
        common_brokers = set(top_buyers) & set(top_sellers)
        
        if len(common_brokers) >= 2:
            risk_flags.append('WARNING: Same brokers appear in both top buyers and sellers')
            risk_score += 20
        
        # Flag 5: Low liquidity with high concentration
        unique_brokers = concentration_data['total_unique_buyers']
        if unique_brokers < 5 and top_1_buyer > 40:
            risk_flags.append('CRITICAL: Very few brokers with high concentration')
            risk_score += 25
        
        # Determine overall risk level
        if risk_score >= 70:
            risk_level = 'CRITICAL - Very High Manipulation Risk'
            recommendation = 'AVOID - Do not trade this stock'
        elif risk_score >= 50:
            risk_level = 'HIGH - Significant Manipulation Risk'
            recommendation = 'HIGH RISK - Trade with extreme caution'
        elif risk_score >= 30:
            risk_level = 'MODERATE - Some Manipulation Risk'
            recommendation = 'CAUTION - Small positions only'
        elif risk_score >= 15:
            risk_level = 'LOW - Minimal Manipulation Risk'
            recommendation = 'ACCEPTABLE - Monitor closely'
        else:
            risk_level = 'VERY LOW - Healthy Trading Pattern'
            recommendation = 'SAFE - Normal market activity'
        
        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'risk_flags': risk_flags,
            'recommendation': recommendation,
            'coordinated_brokers': list(common_brokers) if common_brokers else []
        }
    
    def _calculate_concentration_risk(self, hhi_buyers: float, hhi_sellers: float, top_3_pct: float) -> int:
        """Calculate concentration risk score (0-100)"""
        risk = 0
        
        # HHI component
        if hhi_buyers > 3000:
            risk += 40
        elif hhi_buyers > 2500:
            risk += 30
        elif hhi_buyers > 1500:
            risk += 20
        elif hhi_buyers > 800:
            risk += 10
        
        # Top 3 concentration
        if top_3_pct > 80:
            risk += 35
        elif top_3_pct > 70:
            risk += 25
        elif top_3_pct > 50:
            risk += 15
        elif top_3_pct > 35:
            risk += 5
        
        return min(risk, 100)
    
    def _calculate_accumulation_score(self, net_flow: int, accumulation: int, distribution: int) -> int:
        """Calculate smart money accumulation score (-100 to +100)"""
        if accumulation + distribution == 0:
            return 0
        
        # Ratio-based scoring
        total = accumulation + distribution
        net_ratio = net_flow / total if total > 0 else 0
        
        score = int(net_ratio * 100)
        return max(-100, min(100, score))
    
    def _calculate_pressure_score(self, imbalance_ratio: float, price_change: float) -> int:
        """Calculate buy/sell pressure score (-100 to +100)"""
        # Ratio component (-50 to +50)
        if imbalance_ratio > 1:
            ratio_score = min(50, (imbalance_ratio - 1) * 50)
        else:
            ratio_score = max(-50, (imbalance_ratio - 1) * 50)
        
        # Price momentum component (-50 to +50)
        price_score = max(-50, min(50, price_change * 5))
        
        total_score = ratio_score + price_score
        return int(max(-100, min(100, total_score)))
    
    def _calculate_overall_broker_score(self, concentration: Dict, smart_money: Dict, 
                                       pressure: Dict, liquidity: Dict, 
                                       manipulation: Dict) -> Dict:
        """Calculate overall broker analysis score"""
        # Weights
        weights = config.BROKER_SIGNAL_WEIGHTS
        
        # Individual scores (normalized to 0-100)
        concentration_score = 100 - concentration['concentration_risk_score']  # Invert (lower concentration = better)
        accumulation_score = (smart_money['accumulation_score'] + 100) / 2  # Convert -100/+100 to 0-100
        pressure_score = (pressure['pressure_score'] + 100) / 2
        liquidity_score = liquidity['liquidity_score']
        manipulation_score = 100 - manipulation['risk_score']  # Invert
        
        # Weighted average
        overall_score = (
            concentration_score * weights['concentration'] +
            accumulation_score * weights['smart_money'] +
            pressure_score * weights['pressure'] +
            liquidity_score * weights['liquidity'] +
            manipulation_score * weights['manipulation_risk']
        )
        
        return {
            'overall_score': round(overall_score, 2),
            'component_scores': {
                'concentration': round(concentration_score, 2),
                'smart_money': round(accumulation_score, 2),
                'pressure': round(pressure_score, 2),
                'liquidity': round(liquidity_score, 2),
                'manipulation_risk': round(manipulation_score, 2)
            },
            'grade': self._score_to_grade(overall_score)
        }
    
    def _score_to_grade(self, score: float) -> str:
        """Convert score to letter grade"""
        if score >= 90:
            return 'A+ (Excellent)'
        elif score >= 85:
            return 'A (Very Good)'
        elif score >= 80:
            return 'A- (Good)'
        elif score >= 75:
            return 'B+ (Above Average)'
        elif score >= 70:
            return 'B (Average)'
        elif score >= 65:
            return 'B- (Below Average)'
        elif score >= 60:
            return 'C+ (Marginal)'
        elif score >= 50:
            return 'C (Poor)'
        else:
            return 'D/F (Very Poor - Avoid)'
    
    def _generate_broker_recommendation(self, overall_score: Dict, manipulation_risk: Dict) -> str:
        """Generate trading recommendation based on broker analysis"""
        score = overall_score['overall_score']
        risk_level = manipulation_risk['risk_score']
        
        # Critical risk override
        if risk_level >= 70:
            return '🚫 AVOID - Critical manipulation risk detected'
        
        if score >= 80 and risk_level < 30:
            return '✅ STRONG BUY - Healthy broker activity, good liquidity'
        elif score >= 70 and risk_level < 40:
            return '✅ BUY - Positive broker signals, acceptable risk'
        elif score >= 60 and risk_level < 50:
            return '⚠️ HOLD - Mixed signals, monitor closely'
        elif score >= 50:
            return '⚠️ CAUTION - Weak broker activity, consider waiting'
        else:
            return '❌ SELL/AVOID - Poor broker metrics, high risk'


if __name__ == "__main__":
    # Example usage with sample data
    print("Broker Analysis Module - Testing")
    print("=" * 60)
    
    # Sample floorsheet data structure
    sample_data = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=100),
        'symbol': ['NABIL'] * 100,
        'buyer_broker': np.random.choice(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8'], 100),
        'seller_broker': np.random.choice(['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8'], 100),
        'quantity': np.random.randint(100, 5000, 100),
        'rate': np.random.uniform(800, 850, 100)
    })
    
    analyzer = BrokerAnalyzer()
    results = analyzer.comprehensive_broker_analysis(sample_data, 'NABIL')
    
    print(f"\nOverall Score: {results['overall_broker_score']['overall_score']}")
    print(f"Grade: {results['overall_broker_score']['grade']}")
    print(f"Recommendation: {results['recommendation']}")
    print(f"\nManipulation Risk: {results['manipulation_risk']['risk_level']}")
    print(f"Smart Money Signal: {results['smart_money_flow']['smart_money_signal']}")
    print(f"Liquidity: {results['liquidity_metrics']['liquidity_classification']}")
