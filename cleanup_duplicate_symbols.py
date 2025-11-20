"""
Cleanup script to remove duplicate stock entries with comma suffixes
Fixes symbols like "BANDIPUR," -> "BANDIPUR"
"""
import json
from datetime import datetime


def cleanup_symbols():
    """Remove duplicate entries with comma suffixes and clean up symbols"""
    print("\n" + "="*70)
    print("CLEANING UP DUPLICATE SYMBOLS")
    print("="*70)
    
    # Load analysis results
    try:
        with open('analysis_results.json', 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print("❌ analysis_results.json not found")
        return
    
    print(f"Total entries: {len(results)}")
    
    # Track symbols
    cleaned_results = {}
    duplicates_removed = []
    symbols_cleaned = []
    
    for entry in results:
        original_symbol = entry.get('symbol', '')
        
        # Clean the symbol (remove trailing comma and whitespace)
        cleaned_symbol = original_symbol.strip().rstrip(',').strip()
        
        # Track if symbol was modified
        if original_symbol != cleaned_symbol:
            symbols_cleaned.append(f"{original_symbol} -> {cleaned_symbol}")
            entry['symbol'] = cleaned_symbol
        
        # Check for duplicates (keep the one without error, or the newest)
        if cleaned_symbol in cleaned_results:
            existing = cleaned_results[cleaned_symbol]
            current = entry
            
            # Prioritize entries without errors
            if existing.get('error') and not current.get('error'):
                # Replace with current (no error)
                duplicates_removed.append(f"{cleaned_symbol} (kept newer, error-free)")
                cleaned_results[cleaned_symbol] = current
            elif not existing.get('error') and current.get('error'):
                # Keep existing (no error)
                duplicates_removed.append(f"{cleaned_symbol} (kept existing, error-free)")
            else:
                # Both have errors or both are good - keep newer
                existing_time = datetime.fromisoformat(existing.get('timestamp', '2000-01-01'))
                current_time = datetime.fromisoformat(current.get('timestamp', '2000-01-01'))
                
                if current_time > existing_time:
                    duplicates_removed.append(f"{cleaned_symbol} (kept newer)")
                    cleaned_results[cleaned_symbol] = current
                else:
                    duplicates_removed.append(f"{cleaned_symbol} (kept existing)")
        else:
            cleaned_results[cleaned_symbol] = entry
    
    # Summary
    print(f"\n📊 Cleanup Summary:")
    print(f"  Original entries: {len(results)}")
    print(f"  Cleaned entries: {len(cleaned_results)}")
    print(f"  Duplicates removed: {len(duplicates_removed)}")
    print(f"  Symbols cleaned: {len(symbols_cleaned)}")
    
    if symbols_cleaned:
        print(f"\n🧹 Symbols cleaned:")
        for change in symbols_cleaned[:20]:
            print(f"  • {change}")
        if len(symbols_cleaned) > 20:
            print(f"  ... and {len(symbols_cleaned) - 20} more")
    
    if duplicates_removed:
        print(f"\n🗑️ Duplicates removed:")
        for dup in duplicates_removed[:20]:
            print(f"  • {dup}")
        if len(duplicates_removed) > 20:
            print(f"  ... and {len(duplicates_removed) - 20} more")
    
    # Save cleaned results
    print(f"\n💾 Saving cleaned results...")
    with open('analysis_results.json', 'w') as f:
        json.dump(list(cleaned_results.values()), f, indent=2)
    
    print(f"✅ Cleanup complete!")
    print("="*70 + "\n")


if __name__ == '__main__':
    cleanup_symbols()
