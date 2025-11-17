"""Fundamental analysis module for NEPSE stocks"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime
import config


class FundamentalAnalyzer:
    """Perform fundamental analysis on stocks"""
    
    def __init__(self):
        self.thresholds = config.FUNDAMENTAL_THRESHOLDS
    
    def calculate_pe_ratio(self, current_price: float, eps: float) -> Optional[float]:
        """Calculate Price-to-Earnings ratio"""
        if eps == 0 or eps is None:
            return None
        return current_price / eps
    
    def calculate_pb_ratio(self, current_price: float, book_value_per_share: float) -> Optional[float]:
        """Calculate Price-to-Book ratio"""
        if book_value_per_share == 0 or book_value_per_share is None:
            return None
        return current_price / book_value_per_share
    
    def calculate_dividend_yield(self, annual_dividend: float, current_price: float) -> Optional[float]:
        """Calculate dividend yield percentage"""
        if current_price == 0 or current_price is None or annual_dividend is None:
            return None
        return (annual_dividend / current_price) * 100
    
    def calculate_eps_growth(self, current_eps: float, previous_eps: float) -> Optional[float]:
        """Calculate EPS growth rate"""
        if previous_eps == 0 or previous_eps is None:
            return None
        return ((current_eps - previous_eps) / abs(previous_eps)) * 100
    
    def calculate_roe(self, net_income: float, shareholders_equity: float) -> Optional[float]:
        """Calculate Return on Equity"""
        if shareholders_equity == 0 or shareholders_equity is None:
            return None
        return (net_income / shareholders_equity) * 100
    
    def calculate_debt_to_equity(self, total_debt: float, shareholders_equity: float) -> Optional[float]:
        """Calculate Debt-to-Equity ratio"""
        if shareholders_equity == 0 or shareholders_equity is None:
            return None
        return total_debt / shareholders_equity
    
    def calculate_current_ratio(self, current_assets: float, current_liabilities: float) -> Optional[float]:
        """Calculate current ratio (liquidity)"""
        if current_liabilities == 0 or current_liabilities is None:
            return None
        return current_assets / current_liabilities
    
    def evaluate_pe_ratio(self, pe_ratio: Optional[float]) -> Dict:
        """Evaluate PE ratio and provide interpretation"""
        if pe_ratio is None:
            return {'rating': 'N/A', 'interpretation': 'Insufficient data', 'score': 0}
        
        thresholds = self.thresholds['pe_ratio']
        
        if pe_ratio < thresholds['undervalued']:
            rating = 'Undervalued'
            interpretation = 'Stock may be undervalued - potential buy opportunity'
            score = 1.0
        elif pe_ratio < thresholds['fair']:
            rating = 'Fair Value'
            interpretation = 'Stock is fairly valued'
            score = 0.7
        elif pe_ratio < thresholds['overvalued']:
            rating = 'Moderately Overvalued'
            interpretation = 'Stock may be slightly overvalued - exercise caution'
            score = 0.4
        else:
            rating = 'Overvalued'
            interpretation = 'Stock appears overvalued - high risk'
            score = 0.1
        
        return {
            'value': pe_ratio,
            'rating': rating,
            'interpretation': interpretation,
            'score': score
        }
    
    def evaluate_pb_ratio(self, pb_ratio: Optional[float]) -> Dict:
        """Evaluate PB ratio and provide interpretation"""
        if pb_ratio is None:
            return {'rating': 'N/A', 'interpretation': 'Insufficient data', 'score': 0}
        
        thresholds = self.thresholds['pb_ratio']
        
        if pb_ratio < thresholds['undervalued']:
            rating = 'Undervalued'
            interpretation = 'Trading below book value - potentially undervalued'
            score = 1.0
        elif pb_ratio < thresholds['fair']:
            rating = 'Fair Value'
            interpretation = 'Trading at reasonable value'
            score = 0.7
        elif pb_ratio < thresholds['overvalued']:
            rating = 'Moderately Overvalued'
            interpretation = 'Slightly above book value'
            score = 0.4
        else:
            rating = 'Overvalued'
            interpretation = 'Trading significantly above book value'
            score = 0.1
        
        return {
            'value': pb_ratio,
            'rating': rating,
            'interpretation': interpretation,
            'score': score
        }
    
    def evaluate_dividend_yield(self, dividend_yield: Optional[float]) -> Dict:
        """Evaluate dividend yield"""
        if dividend_yield is None:
            return {'rating': 'N/A', 'interpretation': 'No dividend data', 'score': 0}
        
        thresholds = self.thresholds['dividend_yield']
        
        if dividend_yield >= thresholds['excellent']:
            rating = 'Excellent'
            interpretation = 'High dividend yield - attractive for income investors'
            score = 1.0
        elif dividend_yield >= thresholds['good']:
            rating = 'Good'
            interpretation = 'Decent dividend yield'
            score = 0.7
        elif dividend_yield >= thresholds['poor']:
            rating = 'Fair'
            interpretation = 'Low dividend yield'
            score = 0.4
        else:
            rating = 'No Dividend'
            interpretation = 'Company does not pay dividends'
            score = 0.1
        
        return {
            'value': dividend_yield,
            'rating': rating,
            'interpretation': interpretation,
            'score': score
        }
    
    def evaluate_eps_growth(self, eps_growth: Optional[float]) -> Dict:
        """Evaluate EPS growth rate"""
        if eps_growth is None:
            return {'rating': 'N/A', 'interpretation': 'Insufficient data', 'score': 0}
        
        thresholds = self.thresholds['eps_growth']
        
        if eps_growth >= thresholds['excellent']:
            rating = 'Excellent'
            interpretation = 'Strong earnings growth'
            score = 1.0
        elif eps_growth >= thresholds['good']:
            rating = 'Good'
            interpretation = 'Positive earnings growth'
            score = 0.7
        elif eps_growth >= thresholds['poor']:
            rating = 'Fair'
            interpretation = 'Modest earnings growth'
            score = 0.4
        else:
            rating = 'Declining'
            interpretation = 'Negative earnings growth - concerning'
            score = 0.1
        
        return {
            'value': eps_growth,
            'rating': rating,
            'interpretation': interpretation,
            'score': score
        }
    
    def comprehensive_analysis(self, stock_data: Dict) -> Dict:
        """Perform comprehensive fundamental analysis"""
        
        # Extract data
        current_price = stock_data.get('current_price', 0)
        eps = stock_data.get('eps', 0)
        previous_eps = stock_data.get('previous_eps', 0)
        book_value = stock_data.get('book_value_per_share', 0)
        annual_dividend = stock_data.get('annual_dividend', 0)
        net_income = stock_data.get('net_income', 0)
        shareholders_equity = stock_data.get('shareholders_equity', 0)
        total_debt = stock_data.get('total_debt', 0)
        current_assets = stock_data.get('current_assets', 0)
        current_liabilities = stock_data.get('current_liabilities', 0)
        market_cap = stock_data.get('market_cap', 0)
        
        # Calculate ratios
        pe_ratio = self.calculate_pe_ratio(current_price, eps)
        pb_ratio = self.calculate_pb_ratio(current_price, book_value)
        dividend_yield = self.calculate_dividend_yield(annual_dividend, current_price)
        eps_growth = self.calculate_eps_growth(eps, previous_eps)
        roe = self.calculate_roe(net_income, shareholders_equity)
        debt_to_equity = self.calculate_debt_to_equity(total_debt, shareholders_equity)
        current_ratio = self.calculate_current_ratio(current_assets, current_liabilities)
        
        # Evaluate ratios
        pe_eval = self.evaluate_pe_ratio(pe_ratio)
        pb_eval = self.evaluate_pb_ratio(pb_ratio)
        dividend_eval = self.evaluate_dividend_yield(dividend_yield)
        eps_growth_eval = self.evaluate_eps_growth(eps_growth)
        
        # Calculate overall fundamental score (0-100)
        scores = [
            pe_eval['score'],
            pb_eval['score'],
            dividend_eval['score'],
            eps_growth_eval['score']
        ]
        
        # Filter out zero scores (missing data) for average
        valid_scores = [s for s in scores if s > 0]
        overall_score = (sum(valid_scores) / len(valid_scores) * 100) if valid_scores else 0
        
        # Determine overall rating
        if overall_score >= 75:
            overall_rating = 'Strong Buy'
        elif overall_score >= 60:
            overall_rating = 'Buy'
        elif overall_score >= 40:
            overall_rating = 'Hold'
        elif overall_score >= 25:
            overall_rating = 'Sell'
        else:
            overall_rating = 'Strong Sell'
        
        return {
            'symbol': stock_data.get('symbol', 'N/A'),
            'current_price': current_price,
            'market_cap': market_cap,
            'ratios': {
                'pe_ratio': pe_eval,
                'pb_ratio': pb_eval,
                'dividend_yield': dividend_eval,
                'eps_growth': eps_growth_eval,
                'roe': roe,
                'debt_to_equity': debt_to_equity,
                'current_ratio': current_ratio
            },
            'overall_score': round(overall_score, 2),
            'overall_rating': overall_rating,
            'strengths': self._identify_strengths(pe_eval, pb_eval, dividend_eval, eps_growth_eval),
            'weaknesses': self._identify_weaknesses(pe_eval, pb_eval, dividend_eval, eps_growth_eval),
            'analysis_date': datetime.now().isoformat()
        }
    
    def _identify_strengths(self, pe_eval: Dict, pb_eval: Dict, 
                           dividend_eval: Dict, eps_growth_eval: Dict) -> list:
        """Identify fundamental strengths"""
        strengths = []
        
        if pe_eval['score'] >= 0.7:
            strengths.append(f"Attractive P/E ratio ({pe_eval['value']:.2f})")
        if pb_eval['score'] >= 0.7:
            strengths.append(f"Good P/B ratio ({pb_eval['value']:.2f})")
        if dividend_eval['score'] >= 0.7:
            strengths.append(f"Strong dividend yield ({dividend_eval['value']:.2f}%)")
        if eps_growth_eval['score'] >= 0.7:
            strengths.append(f"Positive EPS growth ({eps_growth_eval['value']:.2f}%)")
        
        return strengths if strengths else ['No significant strengths identified']
    
    def _identify_weaknesses(self, pe_eval: Dict, pb_eval: Dict, 
                            dividend_eval: Dict, eps_growth_eval: Dict) -> list:
        """Identify fundamental weaknesses"""
        weaknesses = []
        
        if pe_eval['score'] <= 0.4 and pe_eval['score'] > 0:
            weaknesses.append(f"High P/E ratio ({pe_eval['value']:.2f})")
        if pb_eval['score'] <= 0.4 and pb_eval['score'] > 0:
            weaknesses.append(f"High P/B ratio ({pb_eval['value']:.2f})")
        if dividend_eval['score'] <= 0.4:
            weaknesses.append(f"Low dividend yield ({dividend_eval.get('value', 0):.2f}%)")
        if eps_growth_eval['score'] <= 0.4 and eps_growth_eval['score'] > 0:
            weaknesses.append(f"Weak EPS growth ({eps_growth_eval['value']:.2f}%)")
        
        return weaknesses if weaknesses else ['No significant weaknesses identified']
    
    def compare_stocks(self, stocks_data: list) -> pd.DataFrame:
        """Compare fundamental metrics across multiple stocks"""
        comparisons = []
        
        for stock_data in stocks_data:
            analysis = self.comprehensive_analysis(stock_data)
            
            comparisons.append({
                'Symbol': analysis['symbol'],
                'Price': analysis['current_price'],
                'P/E Ratio': analysis['ratios']['pe_ratio']['value'],
                'P/B Ratio': analysis['ratios']['pb_ratio']['value'],
                'Dividend Yield': analysis['ratios']['dividend_yield']['value'],
                'EPS Growth': analysis['ratios']['eps_growth']['value'],
                'Overall Score': analysis['overall_score'],
                'Rating': analysis['overall_rating']
            })
        
        df = pd.DataFrame(comparisons)
        df = df.sort_values('Overall Score', ascending=False)
        
        return df


if __name__ == "__main__":
    # Test fundamental analysis
    analyzer = FundamentalAnalyzer()
    
    # Sample stock data
    sample_data = {
        'symbol': 'NABIL',
        'current_price': 1000,
        'eps': 50,
        'previous_eps': 45,
        'book_value_per_share': 400,
        'annual_dividend': 30,
        'net_income': 5000000,
        'shareholders_equity': 25000000,
        'total_debt': 10000000,
        'current_assets': 50000000,
        'current_liabilities': 30000000,
        'market_cap': 10000000000
    }
    
    result = analyzer.comprehensive_analysis(sample_data)
    
    print(f"Fundamental Analysis for {result['symbol']}")
    print(f"Overall Rating: {result['overall_rating']}")
    print(f"Overall Score: {result['overall_score']}/100")
    print(f"\nP/E Ratio: {result['ratios']['pe_ratio']['value']:.2f} - {result['ratios']['pe_ratio']['rating']}")
    print(f"Strengths: {', '.join(result['strengths'])}")
    print(f"Weaknesses: {', '.join(result['weaknesses'])}")
