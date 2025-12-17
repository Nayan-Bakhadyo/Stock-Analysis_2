#!/usr/bin/env python3
"""
Update Sector Data
==================
Scrapes sector information for all NEPSE stocks and saves to database.

This script:
1. Scrapes sector information from ShareSansar
2. Saves to stock_sectors table in database
3. Updates companies table with sector info

Run this periodically to keep sector data updated.
"""

import sys
from datetime import datetime
from sector_mapper import SectorMapper


def update_sector_data():
    """Main function to update sector data"""
    
    print(f"\n{'#'*70}")
    print(f"#  UPDATE SECTOR DATA")
    print(f"#  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")
    
    mapper = SectorMapper()
    
    # Step 1: Scrape sectors
    print("\n📡 Step 1: Scraping sector data from ShareSansar...")
    df = mapper.scrape_sectors()
    
    if df.empty:
        print("❌ Failed to scrape sector data")
        return False
    
    # Step 2: Save to database
    print("\n💾 Step 2: Saving to database...")
    mapper.save_to_database(df)
    
    # Step 3: Print summary
    print(f"\n{'='*70}")
    print(f"✅ SECTOR UPDATE COMPLETE")
    print(f"{'='*70}")
    print(f"  Total stocks with sector info: {len(df)}")
    print(f"  Sectors covered: {df['sector'].nunique()}")
    
    # Show any stocks without sector index
    no_index = df[df['sector_index_code'].isna()]
    if not no_index.empty:
        print(f"\n  ⚠ Stocks without sector index ({len(no_index)}):")
        for _, row in no_index.iterrows():
            print(f"    - {row['symbol']}: {row['sector']}")
    
    print(f"{'='*70}\n")
    
    return True


if __name__ == '__main__':
    success = update_sector_data()
    sys.exit(0 if success else 1)
