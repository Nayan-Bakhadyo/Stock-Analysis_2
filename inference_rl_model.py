"""
RL Model Inference - Generate trading signals from trained RL model
"""
import sys
sys.path.insert(0, '/Users/Nayan/Documents/Business/Stock_Analysis')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import sqlite3
import json
from datetime import datetime
import config

# Register custom layer
@tf.keras.saving.register_keras_serializable()
class MeanSubtraction(tf.keras.layers.Layer):
    """Custom layer to subtract mean from advantage values for Dueling DQN"""
    def call(self, inputs):
        advantage = inputs
        mean_advantage = tf.reduce_mean(advantage, axis=1, keepdims=True)
        return advantage - mean_advantage
    
    def compute_output_shape(self, input_shape):
        return input_shape

# Get symbol from command line argument
if len(sys.argv) < 2:
    print("\nUsage: python3 inference_rl_model.py SYMBOL")
    print("Example: python3 inference_rl_model.py IGI")
    sys.exit(1)

symbol = sys.argv[1].upper()

# Detect which model file to load based on symbol
model_dir = 'rl_models'
available_models = {}

if os.path.exists(model_dir):
    for filename in os.listdir(model_dir):
        if filename.endswith('_rl_model.keras'):
            # Extract symbols from filename (e.g., "SPC_IGI_AHPC_rl_model.keras" -> ["SPC", "IGI", "AHPC"])
            model_name = filename.replace('_rl_model.keras', '')
            symbols = model_name.split('_')
            for sym in symbols:
                if sym.isupper() and len(sym) <= 5:  # Valid stock symbol
                    available_models[sym] = os.path.join(model_dir, filename)

if symbol not in available_models:
    print(f"\n❌ ERROR: No trained RL model found for {symbol}!")
    if available_models:
        print(f"\n✓ Available trained models:")
        model_files = set(available_models.values())
        for model_file in model_files:
            model_symbols = [s for s, f in available_models.items() if f == model_file]
            print(f"  • {', '.join(model_symbols)}: {model_file}")
    print(f"\nTo get predictions for {symbol}, you need to either:")
    print(f"  1. Train RL model: python3 train_rl_model.py {symbol}")
    print(f"  2. Use ML predictions: python3 predict_single_config.py {symbol}")
    sys.exit(1)

model_path = available_models[symbol]
print(f"Loading model from {model_path}...")
model = tf.keras.models.load_model(model_path, custom_objects={'MeanSubtraction': MeanSubtraction})

# Quick validation check - predict with zeros to detect NaN model
try:
    test_input = np.zeros((1, model.input_shape[1]))
    test_output = model.predict(test_input, verbose=0)[0]
    if not np.isfinite(test_output).all():
        raise ValueError("Model produces NaN/inf outputs")
except Exception as e:
    # Model is corrupted, try to load checkpoint
    checkpoint_path = model_path.replace('_rl_model.keras', '_rl_best.keras')
    if os.path.exists(checkpoint_path):
        print(f"\n⚠️ Main model appears corrupted, loading checkpoint instead...")
        print(f"   Checkpoint: {checkpoint_path}")
        model = tf.keras.models.load_model(checkpoint_path, custom_objects={'MeanSubtraction': MeanSubtraction})
        model_path = checkpoint_path
        print(f"✓ Checkpoint loaded successfully")
    else:
        print(f"\n❌ ERROR: Model is corrupted and no checkpoint found!")
        print(f"  Recommendation: Retrain the model with:")
        print(f"    conda run -n Stock_Prediction python3 train_rl_model.py {symbol}")
        sys.exit(1)

print(f"✓ Model loaded successfully")
print(f"  Input shape: {model.input_shape}")
print(f"  Output shape: {model.output_shape}")

print(f"\n✓ {symbol} is in the trained model")
print(f"Loading {symbol} data from database...")


conn = sqlite3.connect(config.DB_PATH)
query = f"""
    SELECT date, open, high, low, close, volume
    FROM price_history 
    WHERE symbol = '{symbol}'
    ORDER BY date DESC
    LIMIT 100
"""
df = pd.read_sql_query(query, conn)
conn.close()

# Reverse to chronological order for feature calculation
df = df.iloc[::-1].reset_index(drop=True)

print(f"✓ Loaded {len(df)} days of data")
print(f"  Latest: {df['date'].iloc[-1]} - Price: {df['close'].iloc[-1]:.2f}")

# ============================================================================
# PREPARE FEATURES - EXACT SAME AS TRAINING (train_rl_model.py)
# ============================================================================

data = df.copy()

# Price-based features - using PREVIOUS close to avoid lookahead
data['returns'] = data['close'].pct_change().shift(1)
data['log_returns'] = np.log(data['close'] / data['close'].shift(1)).shift(1)

# Moving averages (normalized) - using PAST data only
for window in [5, 10, 20, 50]:
    sma = data['close'].shift(1).rolling(window=window, min_periods=window).mean()
    data[f'sma_{window}'] = sma / data['close'].shift(1)

# RSI - using PAST data only
close_shifted = data['close'].shift(1)
delta = close_shifted.diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
rs = gain / (loss + 1e-10)
rsi = 100 - (100 / (1 + rs))
data['rsi_norm'] = (rsi - 50) / 50

# MACD - using PAST data only
close_shifted = data['close'].shift(1)
ema12 = close_shifted.ewm(span=12, adjust=False).mean()
ema26 = close_shifted.ewm(span=26, adjust=False).mean()
data['macd'] = (ema12 - ema26) / (close_shifted + 1e-10)

# Bollinger Bands - using PAST data only
close_shifted = data['close'].shift(1)
bb_middle = close_shifted.rolling(window=20, min_periods=20).mean()
bb_std = close_shifted.rolling(window=20, min_periods=20).std()
data['bb_position'] = (close_shifted - bb_middle) / (bb_std + 1e-10)

# Volatility - using PAST returns only
data['volatility'] = data['returns'].rolling(20, min_periods=20).std()

# Volume - using PAST data only
volume_shifted = data['volume'].shift(1)
data['volume_ratio'] = volume_shifted / (volume_shifted.rolling(20, min_periods=20).mean() + 1e-10)

# Momentum - using PAST data only
close_shifted = data['close'].shift(1)
data['momentum_5'] = (close_shifted - close_shifted.shift(5)) / (close_shifted.shift(5) + 1e-10)
data['momentum_10'] = (close_shifted - close_shifted.shift(10)) / (close_shifted.shift(10) + 1e-10)

# Drop NaN
data = data.dropna()

if len(data) == 0:
    print(f"\n❌ ERROR: Not enough historical data for {symbol}")
    print(f"   Need at least 60 days of price history for feature calculation")
    sys.exit(1)

# Feature columns - SAME ORDER as training
feature_columns = ['returns', 'log_returns', 'sma_5', 'sma_10', 'sma_20', 'sma_50',
                   'rsi_norm', 'macd', 'bb_position', 'volatility', 
                   'volume_ratio', 'momentum_5', 'momentum_10']

# Get most recent state (market features)
market_features = data[feature_columns].iloc[-1].values.astype(np.float32)

# Check for NaN/inf in features BEFORE feeding to model
if not np.isfinite(market_features).all():
    print(f"\n❌ ERROR: Invalid feature values detected!")
    print(f"\nFeature values:")
    for i, col in enumerate(feature_columns):
        value = market_features[i]
        status = "✓" if np.isfinite(value) else "❌"
        print(f"  {status} {col:15s}: {value}")
    print(f"\n⚠️ Check your data quality for {symbol}")
    print(f"   Some features contain NaN or infinity values")
    sys.exit(1)

# Portfolio features (for inference, assume we're evaluating from neutral position)
current_price = data['close'].iloc[-1]
initial_balance = 100000  # Same as training
portfolio_features = np.array([
    1.0,  # Normalized balance (full cash, no position)
    0.0,  # No current position
    1.0,  # Net worth = initial balance
    0.0,  # Position ratio = 0
], dtype=np.float32)

# Determine expected input size from model
expected_features = model.input_shape[1]
market_and_portfolio = len(market_features) + len(portfolio_features)
num_embedding_features = expected_features - market_and_portfolio

if num_embedding_features < 0:
    print(f"\n❌ ERROR: Model expects {expected_features} features but we have {market_and_portfolio} market+portfolio features!")
    sys.exit(1)

# Stock embedding (adjust based on model size)
if num_embedding_features == 0:
    # Single-stock model (no embedding)
    stock_embedding = np.array([], dtype=np.float32)
    num_stocks = 0
elif num_embedding_features == 1:
    # Single-stock model with scalar embedding
    stock_embedding = np.array([1.0], dtype=np.float32)
    num_stocks = 1
else:
    # Multi-stock model - determine stock ID from filename
    model_filename = os.path.basename(model_path).replace('_rl_model.keras', '')
    trained_symbols = [s for s in model_filename.split('_') if s.isupper() and len(s) <= 5]
    
    stock_id_map = {sym: idx for idx, sym in enumerate(trained_symbols)}
    num_stocks = len(trained_symbols)
    
    stock_embedding = np.zeros(num_stocks, dtype=np.float32)
    if symbol in stock_id_map:
        stock_embedding[stock_id_map[symbol]] = 1.0

# Combine features - EXACT SAME as training
state = np.concatenate([market_features, portfolio_features, stock_embedding]).astype(np.float32)
state = state.reshape(1, -1)

print(f"\nState vector shape: {state.shape}")
print(f"State features: {len(market_features)} market + 4 portfolio + {num_stocks} embedding = {state.shape[1]} total")
print(f"Market features sample: {market_features[:5]}")
print(f"Expected by model: {expected_features} features")

# Get prediction
print("\nGetting model prediction...")
q_values = model.predict(state, verbose=0)[0]

# Check for NaN/inf values
if not np.isfinite(q_values).all():
    print(f"\n❌ ERROR: Model still produced invalid Q-values (NaN or inf)!")
    print(f"  Q-Values: {q_values}")
    print(f"  Model used: {model_path}")
    print(f"\n⚠️ Both the main model and checkpoint appear corrupted.")
    print(f"  This can happen if training was unstable throughout.")
    print(f"  Recommendation: Retrain with the improved stability fixes:")
    print(f"    conda run -n Stock_Prediction python3 train_rl_model.py {symbol}")
    sys.exit(1)

print(f"\nQ-Values:")
print(f"  HOLD: {q_values[0]:.4f}")
print(f"  BUY:  {q_values[1]:.4f}")
print(f"  SELL: {q_values[2]:.4f}")

# Convert to probabilities (softmax)
exp_q = np.exp(q_values - np.max(q_values))
probs = exp_q / exp_q.sum()

print(f"\nAction Probabilities:")
print(f"  HOLD: {probs[0]:.2%}")
print(f"  BUY:  {probs[1]:.2%}")
print(f"  SELL: {probs[2]:.2%}")

action_names = ['HOLD', 'BUY', 'SELL']
best_action = action_names[np.argmax(q_values)]
confidence = probs[np.argmax(q_values)]

# Calculate expected return estimates based on Q-values
# Q-values represent expected cumulative discounted returns
# Higher Q-value = better expected outcome
q_range = q_values.max() - q_values.min()
if q_range > 0:
    # Normalize Q-values to percentage scale (0-100%)
    # Positive Q-values suggest profitable action
    expected_return_buy = (q_values[1] - q_values.min()) / q_range * 100
    expected_return_hold = (q_values[0] - q_values.min()) / q_range * 100
    
    # Estimate holding period based on action confidence
    # Higher confidence = stronger signal = potentially shorter optimal holding period
    # Lower confidence = weaker signal = may need longer holding period
    if best_action == 'BUY':
        # Strong BUY signals (>70% confidence) suggest 3-7 day holds
        # Moderate BUY signals (50-70%) suggest 7-14 day holds
        # Weak BUY signals (<50%) suggest 14-30 day holds
        if confidence > 0.70:
            holding_days = "3-7 days (strong signal)"
        elif confidence > 0.50:
            holding_days = "7-14 days (moderate signal)"
        else:
            holding_days = "14-30 days (weak signal)"
    elif best_action == 'HOLD':
        holding_days = "Continue monitoring (neutral position)"
    else:  # SELL
        holding_days = "Exit position immediately"
else:
    expected_return_buy = 0
    expected_return_hold = 0
    holding_days = "Insufficient signal strength"

# Calculate risk indicator based on Q-value spread
risk_indicator = "LOW" if q_range < 0.5 else ("MEDIUM" if q_range < 1.0 else "HIGH")

# Calculate estimated return based on Q-values
# Q-values represent expected cumulative discounted future returns
# The difference between actions indicates expected opportunity cost

if best_action == 'BUY':
    # BUY signal: Expected gain from buying vs holding
    # Higher Q(BUY) - Q(HOLD) = stronger expected upside
    expected_gain = q_values[1] - q_values[0]  # BUY vs HOLD
    
    # Scale to realistic percentage (based on training environment)
    # During training, typical episode returns ranged from -20% to +30%
    # Q-values are cumulative discounted, so we normalize
    estimated_return = np.clip(expected_gain * 0.01, 0, 15)  # Cap at 15%
    
elif best_action == 'SELL':
    # SELL signal: Avoid expected loss by selling
    # Higher Q(SELL) - Q(HOLD) = stronger expected downside if we don't sell
    expected_loss = q_values[0] - q_values[2]  # HOLD vs SELL (potential loss avoided)
    
    # Negative return (protecting from loss)
    estimated_return = -np.clip(expected_loss * 0.01, 0, 10)  # Cap at -10%
    
else:  # HOLD
    # HOLD signal: Best to do nothing
    # Compare HOLD vs best alternative action
    alternative_value = max(q_values[1], q_values[2])  # BUY or SELL
    expected_move = alternative_value - q_values[0]  # How much worse is the alternative?
    
    # If HOLD is much better than alternatives, stock is fairly valued (small expected return)
    # If HOLD is only slightly better, there's some uncertainty (moderate expected return)
    if q_range < 1.0:
        # Very small Q-range = very uncertain = assume minimal movement
        estimated_return = 0.0
    elif expected_move < -100:
        # HOLD is much better than alternatives = strongly expect minimal movement
        estimated_return = np.clip(abs(expected_move) * 0.001, 0, 2)
    else:
        # HOLD slightly better = moderate uncertainty = small expected return
        estimated_return = np.clip(abs(expected_move) * 0.005, 0, 5)

# Calculate average holding period from confidence and Q-value strength
# Higher confidence + higher Q-range = stronger signal = more actionable
if best_action == 'BUY':
    if confidence > 0.70:
        avg_holding_days = 5  # Strong BUY: 3-7 days
    elif confidence > 0.50:
        avg_holding_days = 10  # Moderate BUY: 7-14 days
    else:
        avg_holding_days = 21  # Weak BUY: 14-30 days
        
elif best_action == 'SELL':
    avg_holding_days = 0  # Exit immediately
    
else:  # HOLD
    # For HOLD, holding period depends on conviction strength
    # High conviction HOLD (certain it won't move much) = longer hold is fine
    # Low conviction HOLD (uncertain) = re-evaluate soon
    
    if confidence > 0.90:
        # Very strong HOLD (>90% confidence) = very stable, long-term hold
        avg_holding_days = int(60 + (confidence - 0.9) * 1000)  # 60-160 days
    elif confidence > 0.70:
        # Strong HOLD (70-90%) = stable, medium-term hold
        avg_holding_days = int(30 + (confidence - 0.7) * 150)  # 30-60 days
    elif confidence > 0.55:
        # Moderate HOLD (55-70%) = somewhat stable, re-evaluate regularly
        avg_holding_days = int(14 + (confidence - 0.55) * 100)  # 14-30 days
    else:
        # Weak HOLD (50-55%) = very uncertain, re-evaluate soon
        avg_holding_days = int(7 + (confidence - 0.5) * 140)  # 7-14 days

print(f"\nESTIMATED RETURN: {estimated_return:+.2f}%")
print(f"AVERAGE HOLDING: {avg_holding_days} days")
print(f"  (Q-value range: {q_range:.4f})")
print(f"{'='*60}")

# Save results to rl_results/
results_dir = Path("rl_results")
results_dir.mkdir(exist_ok=True)

# Prepare detailed results (use latest data - last row after chronological sort)
current_price = float(df['close'].iloc[-1])
latest_date = df['date'].iloc[-1]

# Extract model info
model_filename = os.path.basename(model_path).replace('_rl_model.keras', '')
trained_symbols_list = [s for s in model_filename.split('_') if s.isupper() and len(s) <= 5]

results = {
    'symbol': symbol,
    'timestamp': datetime.now().isoformat(),
    'model': os.path.basename(model_path),
    'trained_symbols': trained_symbols_list,
    'current_signal': {
        'recommendation': best_action,
        'date': latest_date,
        'price': current_price,
        'confidence': float(confidence),
        'action_probabilities': {
            'HOLD': float(probs[0]),
            'BUY': float(probs[1]),
            'SELL': float(probs[2])
        },
        'q_values': {
            'HOLD': float(q_values[0]),
            'BUY': float(q_values[1]),
            'SELL': float(q_values[2])
        },
        'holding_period': holding_days,
        'signal_strength': {
            'buy': float(expected_return_buy),
            'hold': float(expected_return_hold)
        },
        'signal_quality': risk_indicator,
        'q_value_range': float(q_range),
        'estimated_return_pct': float(estimated_return),
        'average_holding_days': int(avg_holding_days)
    },
    'diagnostics': {
        'algorithm': 'Double DQN + Dueling + Prioritized Replay',
        'state_features': state.shape[1],
        'multi_stock_training': True
    }
}

# Save individual stock results
output_file = results_dir / f"{symbol}_rl_signals.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved: {output_file}")

# Update consolidated results
consolidated_file = results_dir / "rl_trading_signals_all.json"

if consolidated_file.exists():
    with open(consolidated_file, 'r') as f:
        all_results = json.load(f)
    # Ensure 'signals' key exists in loaded file
    if 'signals' not in all_results:
        all_results['signals'] = {}
else:
    all_results = {
        'last_updated': None,
        'multi_stock_training': True,
        'training_pool': TRAINED_SYMBOLS,
        'signals': {}
    }

# Update signals in the format expected by generate_website.py
all_results['signals'][symbol] = {
    'recommendation': best_action,
    'confidence': float(confidence),
    'date': latest_date,
    'price': current_price,
    'action_probabilities': {
        'HOLD': float(probs[0]),
        'BUY': float(probs[1]),
        'SELL': float(probs[2])
    },
    'holding_period': holding_days,
    'signal_strength': {
        'buy': float(expected_return_buy),
        'hold': float(expected_return_hold)
    },
    'signal_quality': risk_indicator,
    'timestamp': datetime.now().isoformat(),
    'q_values': {
        'HOLD': float(q_values[0]),
        'BUY': float(q_values[1]),
        'SELL': float(q_values[2])
    },
    'q_value_range': float(q_range),
    'estimated_return_pct': float(estimated_return),
    'average_holding_days': int(avg_holding_days)
}

all_results['last_updated'] = datetime.now().isoformat()

with open(consolidated_file, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"✓ Consolidated results updated: {consolidated_file}")
print(f"\n💡 To see results in website, run: python3 generate_website.py")

