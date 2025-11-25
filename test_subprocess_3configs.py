#!/usr/bin/env python3
"""
Test subprocess-based grid search with 3 configs to verify stable memory.
"""

import psutil
import os
import time

def get_memory():
    """Get current process memory in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

print(f"📊 Initial memory: {get_memory():.2f} MB\n")

# Import and run grid search
from stock_predictor import StockPredictor

predictor = StockPredictor(symbol='IGI')

print(f"📊 After import: {get_memory():.2f} MB\n")

# Override to generate only 3 configs for testing
original_generate = predictor._generate_configs

def generate_only_3(config_grid, architecture, lookback):
    all_configs = original_generate(config_grid, architecture, lookback)
    return all_configs[:3]  # Only return first 3 configs

predictor._generate_configs = generate_only_3

# Run grid search with only 3 configs
print("Running grid search with 3 configs via subprocess isolation...\n")
start_time = time.time()

results = predictor.grid_search(
    architectures=['bidirectional'],
    lookback_periods=[60],  # Only 1 lookback
    verbose=True
)

elapsed = time.time() - start_time

print(f"\n📊 After grid search: {get_memory():.2f} MB")
print(f"⏱️  Time elapsed: {elapsed:.1f} seconds")
print(f"\n✓ Grid search completed with subprocess isolation!")
print(f"Best config: {results[0] if results and results[0] else 'None found'}")
print(f"\n🎯 Memory should be stable (not growing) since each config ran in its own subprocess")
