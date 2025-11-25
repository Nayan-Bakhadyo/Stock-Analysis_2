#!/usr/bin/env python3
"""
Test subprocess-based grid search with memory tracking.
Should show stable memory across configs.
"""

import psutil
import os

def get_memory():
    """Get current process memory in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

print(f"📊 Initial memory: {get_memory():.2f} MB\n")

# Import and run grid search
from stock_predictor import StockPredictor

predictor = StockPredictor(symbol='IGI')

print(f"📊 After import: {get_memory():.2f} MB\n")

# Run grid search with only 2 configs
results = predictor.grid_search(
    architectures=['bidirectional'],
    lookback_periods=[60],  # Only 1 lookback
    verbose=True
)

print(f"\n📊 After grid search: {get_memory():.2f} MB")
print(f"\n✓ Grid search completed!")
print(f"Best config: {results[0] if results[0] else 'None found'}")
