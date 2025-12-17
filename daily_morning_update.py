#!/usr/bin/env python3
"""
Daily Morning Update Script
============================
Run this at 4 AM daily to:
1. Update price data, fundamental data, sentiment data (IN PARALLEL)
2. Train xLSTM models on priority stocks (with Bayesian optimization)
3. Run inference and save predictions
4. Generate website

Uses caffeinate to prevent Mac from sleeping during long training.

Usage:
    python daily_morning_update.py              # Full update + train + website
    python daily_morning_update.py --data-only  # Only update data (no training)
    python daily_morning_update.py --predict-only  # Only run inference (no training)
    python daily_morning_update.py --website-only  # Only generate website
    python daily_morning_update.py --skip-data  # Skip data update, only train + website
    
Progress tracking:
    - Training logs: logs/daily_update.log
    - Per-stock optimization: optimization_results_{SYMBOL}.json
    - Real-time: tail -f logs/daily_update.log
"""

import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


# Lock for print statements
print_lock = threading.Lock()


def prevent_sleep():
    """Start caffeinate to prevent Mac from sleeping"""
    try:
        # Start caffeinate in background, it will be killed when parent process exits
        process = subprocess.Popen(
            ['caffeinate', '-ims'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print("☕ Caffeinate started - Mac will stay awake during training")
        return process
    except Exception as e:
        print(f"⚠️  Could not start caffeinate: {e}")
        return None


def run_script(script_name, description, args=None):
    """Run a Python script and return success status"""
    with print_lock:
        print(f"\n🚀 Starting: {description} ({script_name})")
    
    cmd = [sys.executable, script_name]
    if args:
        cmd.extend(args)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        with print_lock:
            print(f"✅ {description} - Complete")
        return script_name, True, None
    except subprocess.CalledProcessError as e:
        with print_lock:
            print(f"❌ {description} - Failed (exit code: {e.returncode})")
            if e.stderr:
                print(f"   Error: {e.stderr[:200]}")
        return script_name, False, e.stderr
    except FileNotFoundError:
        with print_lock:
            print(f"⚠️  Script not found: {script_name}")
        return script_name, None, "Script not found"


def run_scripts_parallel(scripts):
    """Run multiple scripts in parallel"""
    results = {}
    
    with ThreadPoolExecutor(max_workers=len(scripts)) as executor:
        futures = {}
        for script_name, description in scripts:
            if os.path.exists(script_name):
                future = executor.submit(run_script, script_name, description)
                futures[future] = script_name
            else:
                print(f"⚠️  {script_name} not found, skipping...")
                results[script_name] = None
        
        for future in as_completed(futures):
            script_name, success, error = future.result()
            results[script_name] = success
    
    return results


def run_script_sequential(script_name, description, args=None):
    """Run a Python script sequentially (with visible output)"""
    print(f"\n{'='*70}")
    print(f"⏳ {description}")
    print(f"   Script: {script_name}")
    print(f"{'='*70}\n")
    
    cmd = [sys.executable, script_name]
    if args:
        cmd.extend(args)
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✅ {description} - Complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} - Failed (exit code: {e.returncode})")
        return False
    except FileNotFoundError:
        print(f"\n⚠️  Script not found: {script_name}")
        return None


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Daily Morning Update')
    parser.add_argument('--data-only', action='store_true', 
                       help='Only update data (skip training)')
    parser.add_argument('--predict-only', action='store_true',
                       help='Only run inference (no training)')
    parser.add_argument('--website-only', action='store_true',
                       help='Only generate website')
    parser.add_argument('--skip-data', action='store_true',
                       help='Skip data updates')
    parser.add_argument('--mode', choices=['train', 'optimize', 'inference'], default='inference',
                       help='Mode: inference (default, uses saved config), train (quick retrain), optimize (bayesian)')
    parser.add_argument('--models', type=int, default=5,
                       help='Number of ensemble models')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs')
    
    args = parser.parse_args()
    
    start_time = datetime.now()
    
    print(f"\n{'#'*70}")
    print(f"#  DAILY MORNING UPDATE")
    print(f"#  Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*68}#")
    
    # Prevent Mac from sleeping during training
    caffeinate_process = prevent_sleep()
    
    results = {}
    
    # Website only mode
    if args.website_only:
        results['website'] = run_script_sequential('generate_website.py', 'Generating Website')
        _print_summary(results, start_time)
        return
    
    # Predict only mode
    if args.predict_only:
        results['predict'] = run_script_sequential('train_and_predict.py', 
                                        'Running Inference',
                                        ['--mode', 'predict'])
        results['website'] = run_script_sequential('generate_website.py', 'Generating Website')
        _print_summary(results, start_time)
        return
    
    # Step 1: Update Data IN PARALLEL (unless skipped)
    if not args.skip_data:
        print(f"\n{'#'*70}")
        print(f"#  STEP 1: UPDATE DATA (PARALLEL)")
        print(f"{'#'*70}")
        
        # Define data update scripts to run in parallel
        data_scripts = [
            ('update_price_data.py', 'Updating Price Data'),
            ('update_fundamental_data.py', 'Updating Fundamental Data'),
            ('update_sentiment.py', 'Updating Sentiment Data'),
            ('update_sector_data.py', 'Updating Sector Data'),
            ('update_index_data.py', 'Updating NEPSE & Sector Indices'),
        ]
        
        print(f"\n🔄 Running {len(data_scripts)} data update scripts in parallel...")
        parallel_results = run_scripts_parallel(data_scripts)
        
        # Map results
        results['price_data'] = parallel_results.get('update_price_data.py')
        results['fundamental'] = parallel_results.get('update_fundamental_data.py')
        results['sentiment'] = parallel_results.get('update_sentiment.py')
        results['sector_data'] = parallel_results.get('update_sector_data.py')
        results['index_data'] = parallel_results.get('update_index_data.py')
        
        print(f"\n✅ Parallel data updates complete")
    
    # Data only mode - stop here
    if args.data_only:
        _print_summary(results, start_time)
        return
    
    # Step 2: Train and Predict
    print(f"\n{'#'*70}")
    print(f"#  STEP 2: TRAIN & PREDICT (xLSTM Multi-Horizon)")
    print(f"{'#'*70}")
    
    train_args = [
        '--mode', args.mode,
        '--models', str(args.models),
        '--epochs', str(args.epochs)
    ]
    results['train_predict'] = run_script_sequential('train_and_predict.py',
                                          f'Training & Predicting ({args.mode} mode)',
                                          train_args)
    
    # Step 3: Generate Website
    print(f"\n{'#'*70}")
    print(f"#  STEP 3: GENERATE WEBSITE")
    print(f"{'#'*70}")
    
    results['website'] = run_script_sequential('generate_website.py', 'Generating Website')
    
    _print_summary(results, start_time)


def _print_summary(results, start_time):
    """Print summary of all operations"""
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n{'#'*70}")
    print(f"#  DAILY UPDATE SUMMARY")
    print(f"{'#'*70}")
    print(f"  Started:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Duration: {duration}")
    print(f"\n  Results:")
    
    for task, success in results.items():
        if success is None:
            status = "⏭️  Skipped"
        elif success:
            status = "✅ Success"
        else:
            status = "❌ Failed"
        print(f"    {task}: {status}")
    
    print(f"{'#'*70}\n")
    
    # Check if website was generated
    if results.get('website'):
        print(f"🌐 Website ready at: stock_website/index.html")
    
    print(f"\n📊 Progress tracking files:")
    print(f"   - Training log: logs/daily_update.log")
    print(f"   - Per-stock optimization: optimization_results_{{SYMBOL}}.json")
    print(f"   - Real-time monitoring: tail -f logs/daily_update.log")


if __name__ == '__main__':
    main()
