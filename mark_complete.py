#!/usr/bin/env python3
"""
Mark a stock as completed in the optimization tracker.

Usage:
    python mark_complete.py SYMBOL
    python mark_complete.py SYMBOL1 SYMBOL2 SYMBOL3
    python mark_complete.py --status  # Show current status
    python mark_complete.py --reorder # Reorder queue based on priority list
"""

import json
import sys
from pathlib import Path

TRACKER_FILE = Path("optimization_tracker.json")
PRIORITY_FILE = Path("priority_stocks/priority_list.txt")


def load_tracker():
    if TRACKER_FILE.exists():
        with open(TRACKER_FILE, 'r') as f:
            return json.load(f)
    return None


def save_tracker(tracker):
    with open(TRACKER_FILE, 'w') as f:
        json.dump(tracker, f, indent=2)


def get_stock_results(symbol):
    """Load results from optimization_results_{symbol}.json"""
    results_file = Path(f"optimization_results_{symbol}.json")
    if not results_file.exists():
        return None
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    best = data.get('best_results', {})
    return {
        'total_trials': data.get('n_trials', 0),
        'best_trial': data.get('best_trial', best.get('trial', 0)),
        'avg_direction_acc': best.get('avg_direction_acc', 0),
        'avg_mape': best.get('avg_mape', 0),
    }


def mark_complete(symbols):
    """Mark one or more stocks as completed"""
    tracker = load_tracker()
    if not tracker:
        print("❌ Tracker file not found")
        return
    
    for symbol in symbols:
        symbol = symbol.upper()
        
        # Get results
        results = get_stock_results(symbol)
        
        # Move from in_progress/pending to completed
        if tracker.get('in_progress') == symbol:
            tracker['in_progress'] = ''
        if symbol in tracker.get('pending', []):
            tracker['pending'].remove(symbol)
        if symbol not in tracker.get('completed', []):
            tracker['completed'].append(symbol)
        
        # Add results
        if results:
            tracker['results'][symbol] = {
                'best_trial': results['best_trial'],
                'avg_direction_acc': results['avg_direction_acc'],
                'avg_mape': results['avg_mape'],
                'status': 'completed'
            }
            print(f"✅ {symbol}: Trial {results['best_trial']}, Dir Acc {results['avg_direction_acc']:.1f}%, MAPE {results['avg_mape']:.2f}%")
        else:
            tracker['results'][symbol] = {'status': 'completed_no_data'}
            print(f"✅ {symbol}: Marked complete (no results file found)")
    
    save_tracker(tracker)
    print(f"\n📊 Total completed: {len(tracker['completed'])} stocks")


def show_status():
    """Show current optimization status"""
    tracker = load_tracker()
    if not tracker:
        print("❌ Tracker file not found")
        return
    
    completed = tracker.get('completed', [])
    pending = tracker.get('pending', [])
    in_progress = tracker.get('in_progress', '')
    results = tracker.get('results', {})
    
    print(f"\n{'='*60}")
    print(f"📊 OPTIMIZATION STATUS")
    print(f"{'='*60}")
    print(f"✅ Completed: {len(completed)}")
    print(f"🔄 In Progress: {in_progress or 'None'}")
    print(f"⏳ Pending: {len(pending)}")
    print(f"{'='*60}")
    
    if completed:
        print(f"\n✅ Completed stocks:")
        for s in completed:
            r = results.get(s, {})
            acc = r.get('avg_direction_acc', 0)
            mape = r.get('avg_mape', 0)
            trial = r.get('best_trial', '?')
            print(f"   {s}: Dir Acc {acc:.1f}%, MAPE {mape:.2f}% (Trial {trial})")
    
    if pending:
        print(f"\n⏳ Next 10 pending: {pending[:10]}")


def reorder_queue():
    """Reorder pending queue based on priority list"""
    tracker = load_tracker()
    if not tracker:
        print("❌ Tracker file not found")
        return
    
    if not PRIORITY_FILE.exists():
        print("❌ Priority file not found")
        return
    
    with open(PRIORITY_FILE, 'r') as f:
        priority = [line.strip() for line in f if line.strip()]
    
    completed = set(tracker.get('completed', []))
    pending = tracker.get('pending', [])
    
    # Find priority stocks that are still pending
    priority_pending = [s for s in priority if s in pending]
    non_priority_pending = [s for s in pending if s not in priority]
    
    # Reorder
    new_pending = priority_pending + non_priority_pending
    tracker['pending'] = new_pending
    
    save_tracker(tracker)
    
    print(f"✅ Reordered queue")
    print(f"Priority stocks in queue: {priority_pending}")
    print(f"Next 5: {new_pending[:5]}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    if sys.argv[1] == '--status':
        show_status()
    elif sys.argv[1] == '--reorder':
        reorder_queue()
    else:
        symbols = [s.upper() for s in sys.argv[1:]]
        mark_complete(symbols)
