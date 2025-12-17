#!/bin/bash
# Daily Morning Update Script for Stock Analysis
# Run at 4 AM Central Time (Oklahoma) daily via cron
#
# Cron entry (add with: crontab -e):
# 0 4 * * * /Users/nayanbakhadyo/Desktop/Stock_Analysis/run_daily_update.sh
#
# Note: Uses caffeinate to prevent Mac from sleeping during training
# Logs are saved to logs/daily_YYYY-MM-DD.log (one file per day)

# Set working directory
cd /Users/nayanbakhadyo/Desktop/Stock_Analysis

# Create logs directory if it doesn't exist
mkdir -p logs

# Create log file with today's date
LOG_FILE="logs/daily_$(date '+%Y-%m-%d').log"

# Redirect all output to dated log file
exec >> "$LOG_FILE" 2>&1

# Log start time
echo ""
echo "========================================"
echo "Daily Update Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# Activate virtual environment
source Stock_Prediction/bin/activate

# Run the daily morning update with caffeinate to prevent sleep
# -i: prevent idle sleep, -m: prevent disk sleep, -s: prevent system sleep
caffeinate -ims python daily_morning_update.py

# Log end time
echo ""
echo "========================================"
echo "Daily Update Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# Deactivate virtual environment
deactivate
