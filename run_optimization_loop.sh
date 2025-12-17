#!/bin/bash
#
# Run optimization one stock at a time with full process restart between stocks
# This prevents memory leaks by completely restarting Python after each stock
#
# Usage:
#   ./run_optimization_loop.sh
#   ./run_optimization_loop.sh --max 10   # Run max 10 stocks then stop

cd /Users/nayanbakhadyo/Desktop/Stock_Analysis
source Stock_Prediction/bin/activate

MAX_STOCKS=${1:-1000}  # Default: run up to 1000 stocks
COOLDOWN_SECONDS=60    # 1 minute GPU rest between stocks
STOCKS_COMPLETED=0

echo "========================================"
echo "🚀 OPTIMIZATION LOOP STARTED"
echo "   Max stocks: $MAX_STOCKS"
echo "   GPU cooldown: ${COOLDOWN_SECONDS}s between stocks"
echo "========================================"

while [ $STOCKS_COMPLETED -lt $MAX_STOCKS ]; do
    # Get next pending stock
    NEXT_STOCK=$(python3 -c "
import json
with open('optimization_tracker.json', 'r') as f:
    tracker = json.load(f)
pending = tracker.get('pending', [])
in_progress = tracker.get('in_progress', '')
if in_progress:
    print(in_progress)
elif pending:
    print(pending[0])
else:
    print('DONE')
")

    if [ "$NEXT_STOCK" == "DONE" ]; then
        echo "✅ All stocks completed!"
        break
    fi

    echo ""
    echo "========================================"
    echo "📈 Stock $((STOCKS_COMPLETED + 1)): $NEXT_STOCK"
    echo "   Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"

    # Run optimization for single stock (this is a fresh Python process)
    python3 run_full_optimization.py --stock "$NEXT_STOCK"
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ $NEXT_STOCK completed successfully"
        STOCKS_COMPLETED=$((STOCKS_COMPLETED + 1))
    else
        echo "❌ $NEXT_STOCK failed with exit code $EXIT_CODE"
        # Mark as failed and continue
        python3 -c "
import json
with open('optimization_tracker.json', 'r') as f:
    tracker = json.load(f)
symbol = '$NEXT_STOCK'
if tracker.get('in_progress') == symbol:
    tracker['in_progress'] = ''
if symbol in tracker.get('pending', []):
    tracker['pending'].remove(symbol)
if symbol not in tracker.get('failed', []):
    if 'failed' not in tracker:
        tracker['failed'] = []
    tracker['failed'].append(symbol)
with open('optimization_tracker.json', 'w') as f:
    json.dump(tracker, f, indent=2)
print(f'Marked {symbol} as failed')
"
    fi

    # GPU cooldown - let the GPU rest
    echo ""
    echo "😴 GPU cooldown: ${COOLDOWN_SECONDS} seconds..."
    echo "   Memory before: $(python3 -c 'import torch; print(f\"MPS cache cleared\") if torch.backends.mps.is_available() and torch.mps.empty_cache() is None else print(\"N/A\")')"
    
    # Force garbage collection
    python3 -c "
import gc
import torch
gc.collect()
gc.collect()
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
print('Memory cleared')
"
    
    sleep $COOLDOWN_SECONDS
    
    echo "   Resumed: $(date '+%Y-%m-%d %H:%M:%S')"
done

echo ""
echo "========================================"
echo "🏁 OPTIMIZATION LOOP FINISHED"
echo "   Stocks completed this session: $STOCKS_COMPLETED"
echo "   Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
