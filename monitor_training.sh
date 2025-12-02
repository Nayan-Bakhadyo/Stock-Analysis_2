#!/bin/bash
# Monitor xLSTM training progress

echo "Monitoring xLSTM training progress..."
echo "Press Ctrl+C to stop monitoring"
echo ""

while true; do
    clear
    echo "========================================"
    echo "xLSTM Training Monitor - $(date '+%H:%M:%S')"
    echo "========================================"
    echo ""
    
    # Show last 30 lines of log
    if [ -f "comparison_final.log" ]; then
        tail -30 comparison_final.log
    else
        echo "Log file not found: comparison_final.log"
    fi
    
    echo ""
    echo "========================================"
    echo "Refreshing every 10 seconds..."
    echo "========================================"
    
    sleep 10
done
