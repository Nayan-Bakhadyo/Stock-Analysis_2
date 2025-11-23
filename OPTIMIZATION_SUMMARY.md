# Stock Predictor Optimization Summary

## Changes Made

### 1. Reduced Configuration Space
**Before**: 288 configurations per 3 lookback periods = ~288 configs
**After**: 16 configurations per 2 lookback periods = 32 configs

**Time Savings**: ~90% faster (32 vs 288 configs)

### Optimized Parameters (Based on SPC Results):
- **Layers**: 2 only (3 layers didn't improve results)
- **Units**: 64, 128 (keeping both for stock variability)
- **Dropout**: 0.2, 0.3 (removed 0.1 - less effective)
- **Learning Rate**: 0.001 only (removed 0.0005 - slower convergence)
- **Batch Size**: 64 only (good balance of speed/accuracy)
- **Patience**: 10, 15 (keep both for early stopping flexibility)
- **Lookback**: 120, 180 days (removed 60 - too short; removed 365, 500 - diminishing returns)

### 2. True Future Predictions
**Before**: Predictions were for test set (past 7 days with known actuals)
**After**: Two-phase approach:
1. Validate on test set (get MAPE metrics)
2. Retrain on ALL data and predict TRUE FUTURE (7 days beyond last known date)

### 3. Metrics Reporting
- Grid search MAPE (best validation performance) is reported
- Final predictions use model trained on 100% of data (not held-out test set)

## Usage

### Quick Analysis (Default - Optimized)
```bash
# Uses lookback [120, 180] by default
python3 stock_predictor.py ADBL
```

### Multiple Stocks
```bash
python3 stock_predictor.py HRL SPC IGI AHPC ADBL
```

### Custom Lookback (if needed)
```bash
python3 stock_predictor.py ADBL --lookback 60 120 180 365
```

## Expected Runtime
- **Per stock**: ~5-10 minutes (down from ~30-60 minutes)
- **5 stocks**: ~25-50 minutes total

## Performance Expectations (Based on SPC)
- **MAPE**: 0.5-2% for good fits
- **MAE**: 2-10 Rs depending on stock price
- Predictions are iterative 7-day forecasts beyond last known price
