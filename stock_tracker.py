"""
Track which stocks have been analyzed and when
"""

import json
import os
from datetime import datetime


class StockTracker:
    def __init__(self, tracker_file='stock_tracker.json'):
        self.tracker_file = tracker_file
        self.data = self._load_tracker()
    
    def _load_tracker(self):
        """Load existing tracker data"""
        if os.path.exists(self.tracker_file):
            with open(self.tracker_file, 'r') as f:
                return json.load(f)
        return {
            'last_updated': None,
            'total_processed': 0,
            'stocks': {}
        }
    
    def _save_tracker(self):
        """Save tracker data to file"""
        with open(self.tracker_file, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def mark_processed(self, symbol: str, status: str = 'success', error: str = None):
        """
        Mark a stock as processed
        
        Args:
            symbol: Stock symbol
            status: 'success' or 'failed'
            error: Error message if failed
        """
        timestamp = datetime.now().isoformat()
        
        if symbol not in self.data['stocks']:
            self.data['total_processed'] += 1
        
        self.data['stocks'][symbol] = {
            'last_analyzed': timestamp,
            'status': status,
            'error': error,
            'analysis_count': self.data['stocks'].get(symbol, {}).get('analysis_count', 0) + 1
        }
        
        self.data['last_updated'] = timestamp
        self._save_tracker()
        
        print(f"  ✓ Tracked: {symbol} ({status})")
    
    def is_processed(self, symbol: str, days_old: int = 7) -> bool:
        """
        Check if stock was processed recently
        
        Args:
            symbol: Stock symbol
            days_old: Consider processed if analyzed within this many days
            
        Returns:
            True if processed recently
        """
        if symbol not in self.data['stocks']:
            return False
        
        last_analyzed = self.data['stocks'][symbol].get('last_analyzed')
        if not last_analyzed:
            return False
        
        from datetime import timedelta
        last_date = datetime.fromisoformat(last_analyzed)
        age = datetime.now() - last_date
        
        return age.days < days_old
    
    def get_unprocessed(self, all_symbols: list, days_old: int = 7) -> list:
        """
        Get list of symbols that haven't been processed recently
        
        Args:
            all_symbols: List of all available symbols
            days_old: Consider unprocessed if not analyzed within this many days
            
        Returns:
            List of unprocessed symbols
        """
        return [s for s in all_symbols if not self.is_processed(s, days_old)]
    
    def get_status(self, symbol: str = None):
        """
        Get processing status
        
        Args:
            symbol: Specific symbol (optional). If None, returns all
            
        Returns:
            Status dictionary
        """
        if symbol:
            return self.data['stocks'].get(symbol, {'status': 'not_processed'})
        return self.data
    
    def print_summary(self):
        """Print tracking summary"""
        print(f"\n{'='*70}")
        print(f"STOCK ANALYSIS TRACKER")
        print(f"{'='*70}")
        print(f"Total stocks processed: {self.data['total_processed']}")
        print(f"Last updated: {self.data['last_updated']}")
        print(f"\nRecently processed stocks:")
        
        # Sort by last analyzed date
        sorted_stocks = sorted(
            self.data['stocks'].items(),
            key=lambda x: x[1].get('last_analyzed', ''),
            reverse=True
        )
        
        for symbol, info in sorted_stocks[:10]:  # Show last 10
            status_icon = '✓' if info['status'] == 'success' else '✗'
            print(f"  {status_icon} {symbol:10} - {info['last_analyzed'][:19]} ({info['analysis_count']} times)")
            if info.get('error'):
                print(f"     Error: {info['error']}")
        
        print(f"{'='*70}\n")


if __name__ == '__main__':
    # Test the tracker
    tracker = StockTracker()
    tracker.print_summary()
