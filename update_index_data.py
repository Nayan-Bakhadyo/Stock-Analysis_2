#!/usr/bin/env python3
"""
Update Index Data (Incremental)
================================
Updates NEPSE main index and all sector sub-indices price history.

This script:
1. Checks last update date for each index in database
2. Only fetches new data since last update (incremental)
3. Saves to index_history table in database

These indices are used by xLSTM model as market features.
"""

import sys
import argparse
import sqlite3
from datetime import datetime, timedelta
from sharesansar_index_scraper import ShareSansarIndexScraper


def get_latest_index_date(db_path: str, index_code: str) -> datetime:
    """Get the latest date we have for an index in the database"""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute('''
            SELECT MAX(date) FROM index_history WHERE index_code = ?
        ''', (index_code,))
        result = cursor.fetchone()
        if result and result[0]:
            return datetime.strptime(result[0], '%Y-%m-%d')
        return None
    finally:
        conn.close()


def update_index_data(days=None, indices=None, force_full=False):
    """
    Update index data for NEPSE and sector indices (INCREMENTAL)
    
    Args:
        days: Override days to fetch (default: auto-detect based on last update)
        indices: List of index codes to update (default: all)
        force_full: Force full historical fetch instead of incremental
    """
    
    print(f"\n{'#'*70}")
    print(f"#  UPDATE INDEX DATA - INCREMENTAL")
    print(f"#  NEPSE + Sector Sub-Indices")
    print(f"#  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")
    
    scraper = ShareSansarIndexScraper(headless=True)
    db_path = 'data/nepse_stocks.db'
    
    # Show available indices
    print("\n📊 Available Indices:")
    for code, name in scraper.INDICES.items():
        print(f"   {code:<20} → {name}")
    
    # Determine which indices to scrape
    if indices:
        target_indices = {k: v for k, v in scraper.INDICES.items() if k in indices}
    else:
        target_indices = scraper.INDICES
    
    print(f"\n🎯 Indices to update: {list(target_indices.keys())}")
    
    success_count = 0
    error_count = 0
    total_rows = 0
    skipped_count = 0
    
    try:
        scraper.setup_driver()
        
        for index_code, index_name in target_indices.items():
            print(f"\n{'='*70}")
            print(f"📈 {index_code} ({index_name})")
            print(f"{'='*70}")
            
            # Check last update date
            latest_date = get_latest_index_date(db_path, index_code)
            
            if latest_date and not force_full:
                days_old = (datetime.now() - latest_date).days
                print(f"   ℹ️  Latest data: {latest_date.strftime('%Y-%m-%d')} ({days_old} days old)")
                
                if days_old == 0:
                    print(f"   ✓ Already up-to-date (today's data exists)")
                    skipped_count += 1
                    continue
                elif days_old <= 7:
                    fetch_days = days_old + 3  # Fetch a few extra days for safety
                    print(f"   → Incremental update: fetching {fetch_days} days")
                else:
                    fetch_days = min(days_old + 5, 365)  # Cap at 1 year
                    print(f"   → Data is stale, fetching {fetch_days} days")
            else:
                if force_full:
                    fetch_days = days if days else 365  # Default 1 year for full
                    print(f"   → Forced full sync: fetching {fetch_days} days")
                else:
                    fetch_days = days if days else 60  # Default 60 days for new
                    print(f"   → No existing data, fetching {fetch_days} days")
            
            try:
                df = scraper.scrape_index_history(index_code, days=fetch_days)
                
                if not df.empty:
                    # Filter only new records if we have existing data
                    if latest_date and not force_full:
                        df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d') if isinstance(x, str) else x)
                        original_len = len(df)
                        df = df[df['date'] > latest_date]
                        df['date'] = df['date'].apply(lambda x: x.strftime('%Y-%m-%d') if hasattr(x, 'strftime') else x)
                        print(f"   📊 Filtered: {original_len} → {len(df)} new records")
                    
                    if not df.empty:
                        scraper.save_to_database(df)
                        print(f"   ✅ Saved {len(df)} new rows")
                        success_count += 1
                        total_rows += len(df)
                    else:
                        print(f"   ℹ️  No new records to add")
                        skipped_count += 1
                else:
                    print(f"   ⚠️ No data returned from scraper")
                    error_count += 1
                    
            except Exception as e:
                print(f"   ❌ Error - {e}")
                error_count += 1
            
            # Rate limiting
            import time
            time.sleep(2)
    
    finally:
        scraper.close_driver()
    
    # Summary
    print(f"\n{'='*70}")
    print(f"✅ INDEX UPDATE COMPLETE")
    print(f"{'='*70}")
    print(f"   Updated: {success_count}")
    print(f"   Skipped (up-to-date): {skipped_count}")
    print(f"   Errors: {error_count}")
    print(f"   Total new rows: {total_rows}")
    print(f"{'='*70}\n")
    
    return success_count > 0 or skipped_count > 0


def update_priority_indices(days=None, force_full=False):
    """
    Update only the most important indices for xLSTM model (INCREMENTAL):
    - NEPSE (main index)
    - All sector sub-indices used in model
    """
    priority_indices = [
        'NEPSE',        # Main market index
        'BANKING',      # Commercial banks
        'DEVBANK',      # Development banks
        'FINANCE',      # Finance companies
        'MICROFINANCE', # Microfinance
        'LIFEINSURANCE',    # Life insurance
        'NONLIFEINSURANCE', # Non-life insurance
        'HYDROPOWER',   # Hydropower
        'MANUFACTURING', # Manufacturing
        'HOTELS',       # Hotels & tourism
        'INVESTMENT',   # Investment
        'TRADING',      # Trading
        'MUTUAL',       # Mutual funds
        'OTHERS',       # Others
    ]
    
    return update_index_data(days=days, indices=priority_indices, force_full=force_full)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Update NEPSE and Sector Index Data (Incremental)')
    parser.add_argument('--days', type=int, default=None,
                       help='Override days of history to fetch (default: auto-detect)')
    parser.add_argument('--indices', nargs='*',
                       help='Specific indices to update (default: all)')
    parser.add_argument('--priority', action='store_true',
                       help='Update only priority indices for xLSTM')
    parser.add_argument('--force-full', action='store_true',
                       help='Force full historical fetch instead of incremental')
    
    args = parser.parse_args()
    
    if args.priority:
        success = update_priority_indices(days=args.days, force_full=args.force_full)
    else:
        success = update_index_data(days=args.days, indices=args.indices, force_full=args.force_full)
    
    sys.exit(0 if success else 1)
