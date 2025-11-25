"""
RL Signal Integrator for Website
Reads RL trading signals and formats them for website display
"""

import json
from pathlib import Path
from typing import Dict, Optional


class RLSignalIntegrator:
    """Integrate RL trading signals into website data"""
    
    def __init__(self, rl_results_dir='rl_results'):
        self.rl_results_dir = Path(rl_results_dir)
        self.all_signals_path = self.rl_results_dir / 'rl_trading_signals_all.json'
    
    def get_signal_for_symbol(self, symbol: str) -> Optional[Dict]:
        """
        Get RL signal for a specific symbol
        
        Args:
            symbol: Stock symbol (e.g., 'HRL', 'NABIL')
            
        Returns:
            Dictionary with signal data or None if not available
        """
        symbol = symbol.upper()
        
        # Try individual file first
        individual_file = self.rl_results_dir / f"{symbol}_rl_signals.json"
        if individual_file.exists():
            with open(individual_file, 'r') as f:
                return json.load(f)
        
        # Fall back to combined file
        if self.all_signals_path.exists():
            with open(self.all_signals_path, 'r') as f:
                all_signals = json.load(f)
                return all_signals.get('signals', {}).get(symbol)
        
        return None
    
    def get_all_signals(self) -> Dict:
        """
        Get all RL signals from combined file
        
        Returns:
            Dictionary with all signals
        """
        if self.all_signals_path.exists():
            with open(self.all_signals_path, 'r') as f:
                return json.load(f)
        return {'signals': {}}
    
    def format_for_website(self, symbol: str) -> Dict:
        """
        Format RL signal for website display
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with formatted signal data for website
        """
        signal_data = self.get_signal_for_symbol(symbol)
        
        if not signal_data:
            return {
                'available': False,
                'action': 'HOLD',
                'confidence': 0,
                'reason': 'RL signal not available'
            }
        
        current_signal = signal_data.get('current_signal', {})
        performance = signal_data.get('performance_metrics', {})
        
        # Determine recommendation strength
        confidence = current_signal.get('confidence', 0)
        uncertainty = current_signal.get('ensemble_uncertainty', 1)
        action = current_signal.get('action', 'HOLD')
        
        # Calculate strength
        if confidence > 0.75 and uncertainty < 0.15:
            strength = 'STRONG'
        elif confidence > 0.6 and uncertainty < 0.25:
            strength = 'MODERATE'
        else:
            strength = 'WEAK'
        
        # Format action
        if action == 'BUY':
            action_text = f"{strength} BUY"
            action_color = 'green'
        elif action == 'SELL':
            action_text = f"{strength} SELL"
            action_color = 'red'
        else:
            action_text = "HOLD"
            action_color = 'gray'
        
        # Get reason
        reasons = []
        if confidence > 0.75:
            reasons.append(f"High confidence ({confidence:.0%})")
        if uncertainty < 0.15:
            reasons.append(f"Low uncertainty ({uncertainty:.2f})")
        if performance.get('test_return', 0) > 10:
            reasons.append(f"Strong test returns ({performance['test_return']:.1f}%)")
        if performance.get('sharpe_ratio', 0) > 1.5:
            reasons.append(f"Good risk-adjusted return (Sharpe: {performance['sharpe_ratio']:.2f})")
        
        reason = " • ".join(reasons) if reasons else "Standard trading signal"
        
        return {
            'available': True,
            'action': action_text,
            'action_raw': action,
            'confidence': confidence,
            'confidence_pct': f"{confidence:.1%}",
            'uncertainty': uncertainty,
            'strength': strength,
            'color': action_color,
            'reason': reason,
            'performance': {
                'test_return': performance.get('test_return', 0),
                'sharpe_ratio': performance.get('sharpe_ratio', 0),
                'win_rate': performance.get('win_rate', 0),
                'total_trades': performance.get('total_trades', 0),
                'max_drawdown': performance.get('max_drawdown', 0)
            },
            'probabilities': current_signal.get('action_probabilities', {}),
            'last_updated': signal_data.get('last_updated', 'N/A')
        }
    
    def enrich_website_data(self, website_data: list) -> list:
        """
        Add RL signals to existing website data
        
        Args:
            website_data: List of stock analysis dictionaries
            
        Returns:
            Enriched website data with RL signals
        """
        for stock in website_data:
            symbol = stock.get('symbol')
            if symbol:
                rl_signal = self.format_for_website(symbol)
                stock['rl_signal'] = rl_signal
        
        return website_data
    
    def get_top_opportunities(self, min_confidence=0.75, min_sharpe=1.5, action='BUY') -> list:
        """
        Get top trading opportunities based on RL signals
        
        Args:
            min_confidence: Minimum confidence threshold
            min_sharpe: Minimum Sharpe ratio
            action: Action to filter ('BUY', 'SELL', or None for all)
            
        Returns:
            List of opportunities sorted by confidence
        """
        all_signals = self.get_all_signals()
        opportunities = []
        
        for symbol, data in all_signals.get('signals', {}).items():
            signal = data.get('current_signal', {})
            performance = data.get('performance_metrics', {})
            
            # Apply filters
            if signal.get('confidence', 0) < min_confidence:
                continue
            if performance.get('sharpe_ratio', 0) < min_sharpe:
                continue
            if action and signal.get('action') != action:
                continue
            
            opportunities.append({
                'symbol': symbol,
                'action': signal.get('action'),
                'confidence': signal.get('confidence', 0),
                'uncertainty': signal.get('ensemble_uncertainty', 1),
                'test_return': performance.get('test_return', 0),
                'sharpe_ratio': performance.get('sharpe_ratio', 0),
                'win_rate': performance.get('win_rate', 0)
            })
        
        # Sort by confidence
        opportunities.sort(key=lambda x: x['confidence'], reverse=True)
        return opportunities


# Example usage
if __name__ == '__main__':
    integrator = RLSignalIntegrator()
    
    # Get signal for one stock
    print("\n=== HRL Signal ===")
    hrl_signal = integrator.format_for_website('HRL')
    print(json.dumps(hrl_signal, indent=2))
    
    # Get top opportunities
    print("\n=== Top BUY Opportunities ===")
    opportunities = integrator.get_top_opportunities(min_confidence=0.7, action='BUY')
    for opp in opportunities[:5]:
        print(f"{opp['symbol']}: {opp['action']} ({opp['confidence']:.1%} confidence, "
              f"{opp['test_return']:.1f}% return, Sharpe: {opp['sharpe_ratio']:.2f})")
