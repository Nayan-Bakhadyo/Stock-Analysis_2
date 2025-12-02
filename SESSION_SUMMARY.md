# Stock Analysis - Session Summary (Dec 1, 2025)

## Project Status

### What We Built Today:

1. **xLSTM Stock Forecaster** (`xlstm_stock_forecaster.py`)
   - 3 model variations: Single, Ensemble, Market-Aware
   - 14-22 features (technical + market indicators)
   - Multi-horizon predictions: 1, 3, 5, 10, 15, 21 days
   - DirectionalLoss for better direction prediction (α=0.3)

2. **Index Scraper** (`sharesansar_index_scraper.py`)
   - Scrapes NEPSE and sector indices from ShareSansar
   - Currently scraped: NEPSE (6514 rows), SENSITIVE (2200), FLOAT (3345), BANKING (3357)

3. **Sector Mapper** (`sector_mapper.py`)
   - Maps 184 stocks to 14 sectors
   - Links sectors to index codes (e.g., Commercial Bank → BANKING)

4. **Batch Predictions** (`run_all_predictions.py`)
   - Run predictions for all stocks
   - Resume support, sector filtering

5. **Backtest Script** (`backtest_model.py`)
   - Compare Standard vs Market-Aware models
   - Calculates MAPE and Direction Accuracy

---

## Backtest Results (NABIL, 30 days)

| Horizon | Standard MAPE | Market MAPE | Direction Acc |
|---------|---------------|-------------|---------------|
| 1-day   | 8.18%         | 9.93%       | 58.6%         |
| 3-day   | 22.11%        | 21.07%      | 48.1%         |
| 5-day   | 22.89%        | 20.48%      | 48.0%         |
| 10-day  | 21.13%        | 22.46%      | 45.0%         |

**Assessment:** 
- 1-day predictions are usable (~8% MAPE)
- Multi-day predictions need improvement (~20% MAPE)
- Direction accuracy (~50%) needs significant improvement

---

## Commands for Mac Studio

### Setup Environment:
```bash
conda create -n Stock_Prediction python=3.11 -y
conda activate Stock_Prediction
pip install torch torchvision torchaudio
pip install -r requirements_mac_studio.txt
```

### Verify MPS (Metal GPU):
```bash
python3 -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
```

### Run Predictions:
```bash
# Quick test (fast config)
python3 xlstm_stock_forecaster.py NABIL --market --fast

# Full accuracy
python3 xlstm_stock_forecaster.py NABIL --market --n-models 5 --epochs 100

# Run all stocks
python3 run_all_predictions.py

# Backtest comparison
python3 backtest_model.py NABIL --compare --days 30
```

### Scrape remaining indices:
```bash
python3 sharesansar_index_scraper.py --all
```

---

## Next Steps for Mac Studio Optimization

1. **Increase DirectionalLoss alpha**: Change from 0.3 to 0.5-0.7
   - File: `xlstm_stock_forecaster.py`, line ~280
   - `self.direction_weight = 0.5  # was 0.3`

2. **Use larger batch sizes**: `--batch-size 64` or `128`

3. **More training epochs**: `--epochs 150`

4. **Larger model**: `--hidden-size 512 --num-blocks 7`

5. **Create direction-focused model**: Binary classification (up/down) instead of price prediction

---

## Files to Transfer:
- Entire `Stock_Analysis/` folder
- Especially: `data/nepse_stocks.db` (all scraped data)

## Expected Speedup on Mac Studio M1 Max:
- Training: 3-4x faster
- Full backtest: ~20 min (vs 60 min on MacBook)
