#!/bin/bash
# Full Stock Optimization Script
# ================================
# Run Bayesian optimization on all stocks with sector information.
# Uses caffeinate to prevent Mac from sleeping.
#
# Usage:
#   ./run_optimization.sh              # Run all unoptimized stocks
#   ./run_optimization.sh --status     # Show progress
#   ./run_optimization.sh --stock IGI  # Optimize single stock
#   ./run_optimization.sh --list       # List all stocks
#   ./run_optimization.sh --reset      # Reset and start fresh

cd /Users/nayanbakhadyo/Desktop/Stock_Analysis

# Create logs directory
mkdir -p logs

# Activate virtual environment
source Stock_Prediction/bin/activate

# Log start
echo ""
echo "========================================"
echo "Full Optimization Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo "Logs: logs/optimization.log"
echo ""

# Run with caffeinate to prevent sleep
caffeinate -ims python run_full_optimization.py "$@" 2>&1 | tee -a logs/optimization.log

echo ""
echo "========================================"
echo "Optimization Ended: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
