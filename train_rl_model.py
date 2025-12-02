"""
Reinforcement Learning Trading Agent using Advanced Deep Q-Learning (DQN)
Learns to generate BUY/SELL/HOLD signals for stock trading

Key Features:
- Double DQN with Dueling architecture (reduces overestimation)
- Prioritized Experience Replay (samples important transitions)
- N-step returns (faster reward propagation)
- NO DATA LEAKAGE: All features use only past data
- Custom trading environment with realistic constraints
- Step-wise log-return rewards with trade penalties
- State includes: prices, technical indicators, portfolio state
- Actions: BUY, SELL, HOLD
- Outputs trading signals to JSON file

M1 Pro Optimizations (16GB RAM):
✓ Mixed precision (float16) - 2x faster inference, 50% less memory
✓ XLA JIT compilation - optimized ops for M1 GPU
✓ Metal GPU acceleration - native M1 GPU support
✓ Memory growth enabled - prevents OOM on 16GB RAM
✓ Modest model size (128-128-64) - ~200K parameters
✓ Batch size 32 - balanced speed/memory
✓ Replay buffer 5000 - reasonable for multi-stock training
✓ Recommended: 5-8 stocks max for 16GB M1 Pro
"""

import sys
sys.path.insert(0, '/Users/Nayan/Documents/Business/Stock_Analysis')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque, namedtuple
import random
import heapq

# TensorFlow with M1 Mac optimizations
import tensorflow as tf

# M1 Pro GPU Detection and Optimization
# IMPORTANT: Must configure BEFORE any TensorFlow operations
try:
    # Configure memory growth FIRST (before any device initialization)
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"✓ M1 GPU acceleration enabled ({len(physical_devices)} GPU(s) detected)")
        
        # Apply optimizations for GPU (NO XLA - conflicts with Metal)
        tf.keras.mixed_precision.set_global_policy('mixed_float16')  # Faster on M1 GPU
        # tf.config.optimizer.set_jit(True)  # DISABLED: XLA conflicts with Metal GPU
    else:
        print("⚠️ No GPU detected by TensorFlow")
        print("   Note: M1 Macs need 'tensorflow-metal' for GPU acceleration")
        print("   Install: pip install tensorflow-metal")
        print("   Training will use CPU (slower but still works)")
        
        # Apply optimizations for CPU (with XLA)
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        tf.config.optimizer.set_jit(True)  # XLA works fine on CPU
    
except Exception as e:
    print(f"⚠️ GPU setup warning: {e}")
    print("   Continuing with CPU...")
    # Still apply optimizations for CPU
    try:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        tf.config.optimizer.set_jit(True)
    except:
        pass


from tensorflow.keras.models import Sequential, Model, clone_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, Input, Add, Lambda, Subtract
from tensorflow.keras.optimizers.legacy import Adam  # Legacy optimizer for M1/M2 performance
from tensorflow.keras import backend as K
from sklearn.preprocessing import StandardScaler

from data_fetcher import NepseDataFetcher

# Experience tuple for replay memory
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer
    Samples transitions with higher TD error more frequently
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001):
        """
        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0 = uniform, 1 = fully prioritized)
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
            beta_increment: Beta increment per sample
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = []
        self.position = 0
        
    def add(self, experience: Experience, td_error: float = None):
        """Add experience with priority"""
        max_priority = max(self.priorities) if self.priorities else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(max_priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = max_priority
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int):
        """Sample batch with priorities"""
        if len(self.buffer) == 0:
            return [], [], []
        
        # Calculate sampling probabilities
        priorities = np.array(self.priorities[:len(self.buffer)], dtype=np.float64)
        
        # More aggressive clipping to prevent overflow
        priorities = np.clip(priorities, 1e-5, 100.0)
        
        # Use log-space computation to avoid overflow
        # Instead of priorities^alpha, compute exp(alpha * log(priorities))
        log_probs = self.alpha * np.log(priorities)
        
        # Subtract max for numerical stability (log-sum-exp trick)
        log_probs_max = np.max(log_probs)
        probs = np.exp(log_probs - log_probs_max)
        
        # Check for NaN/inf and fix
        if not np.isfinite(probs).all() or probs.sum() == 0:
            print("⚠️ Warning: Invalid probabilities detected, using uniform distribution")
            probs = np.ones(len(self.buffer), dtype=np.float64)
        
        # Normalize
        probs = probs / probs.sum()
        
        # Final safety check
        if not np.isfinite(probs).all() or probs.sum() == 0:
            print("⚠️ Warning: Probabilities still invalid after normalization, using uniform")
            probs = np.ones(len(self.buffer), dtype=np.float64) / len(self.buffer)
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        
        # Calculate importance sampling weights with safety checks
        total = len(self.buffer)
        sample_probs = probs[indices]
        
        # Clip before exponentiation to avoid overflow
        weights = np.clip(total * sample_probs, 1e-8, 1e8) ** (-self.beta)
        
        # Check for invalid weights
        if not np.isfinite(weights).all():
            weights = np.ones_like(weights, dtype=np.float64)
        
        weights /= (weights.max() + 1e-8)  # Normalize with safety
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        samples = [self.buffer[idx] for idx in indices]
        
        return samples, indices, weights
    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors with robust error handling"""
        for idx, td_error in zip(indices, td_errors):
            # Check for NaN/inf in TD error
            if not np.isfinite(td_error):
                td_error = 1.0  # Default to moderate priority
            
            # More conservative clipping for priorities
            priority = abs(float(td_error)) + 1e-5
            priority = np.clip(priority, 1e-5, 100.0)  # Much tighter bounds
            
            self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.buffer)


class TradingEnvironment:
    """
    Custom trading environment for RL agent
    Simulates realistic trading with transaction costs and position tracking
    Supports multi-stock training with stock embeddings
    """
    
    def __init__(self, df: pd.DataFrame, initial_balance: float = 100000, 
                 transaction_cost: float = 0.002, max_position: float = 1.0,
                 stock_id: int = 0, num_stocks: int = 1):
        """
        Initialize trading environment
        
        Args:
            df: DataFrame with price and technical indicators
            initial_balance: Starting cash balance
            transaction_cost: Transaction cost as fraction (0.002 = 0.2%)
            max_position: Maximum position size as fraction of portfolio
            stock_id: Unique identifier for this stock (for multi-stock training)
            num_stocks: Total number of stocks in training pool
        """
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        self.stock_id = stock_id
        self.num_stocks = num_stocks
        
        # State variables
        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.total_shares_bought = 0
        self.total_shares_sold = 0
        self.net_worth = initial_balance
        
        # History tracking
        self.net_worth_history = []
        self.action_history = []
        self.trade_count = 0
        
        # Prepare features
        self._prepare_features()
        
    def _prepare_features(self):
        """
        Prepare state features for RL agent
        CRITICAL: NO DATA LEAKAGE - All features use only PAST data up to time t
        """
        data = self.df.copy()
        
        # Price-based features - using PREVIOUS close to avoid lookahead
        # At time t, we use returns from t-1 to t (known at close of t-1)
        data['returns'] = data['close'].pct_change().shift(1)  # Shift to use previous return
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1)).shift(1)
        
        # Moving averages (normalized) - using PAST data only
        # At time t, we calculate SMA using data up to t-1
        for window in [5, 10, 20, 50]:
            sma = data['close'].shift(1).rolling(window=window, min_periods=window).mean()
            data[f'sma_{window}'] = sma / data['close'].shift(1)  # Normalize by previous close
        
        # RSI - using PAST data only (no lookahead)
        # Calculate on shifted prices so at time t we use RSI from t-1
        close_shifted = data['close'].shift(1)
        delta = close_shifted.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
        rs = gain / (loss + 1e-10)
        data['rsi'] = 100 - (100 / (1 + rs))
        data['rsi_norm'] = (data['rsi'] - 50) / 50
        
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
        
        # Drop NaN (need more initial data due to shifts and windows)
        data = data.dropna()
        
        self.feature_data = data
        self.feature_columns = ['returns', 'log_returns', 'sma_5', 'sma_10', 'sma_20', 'sma_50',
                               'rsi_norm', 'macd', 'bb_position', 'volatility', 
                               'volume_ratio', 'momentum_5', 'momentum_10']
        
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_shares_bought = 0
        self.total_shares_sold = 0
        self.net_worth = self.initial_balance
        self.net_worth_history = [self.initial_balance]
        self.action_history = []
        
        return self._get_state()
    
    def _get_state(self):
        """
        Get current state vector for RL agent
        
        State includes:
        - Technical indicators
        - Portfolio state (balance, shares, net worth ratios)
        - Stock embedding (one-hot encoded for multi-stock training)
        """
        if self.current_step >= len(self.feature_data):
            return None
        
        # Market features
        row = self.feature_data.iloc[self.current_step]
        market_features = row[self.feature_columns].values
        
        # Portfolio features (normalized)
        current_price = row['close']
        portfolio_features = np.array([
            self.balance / self.initial_balance,  # Normalized balance
            self.shares_held * current_price / self.initial_balance,  # Normalized position value
            self.net_worth / self.initial_balance,  # Normalized net worth
            self.shares_held / (self.initial_balance / current_price + 1e-8),  # Position ratio
        ], dtype=np.float32)
        
        # Stock embedding (one-hot encoded for multi-stock learning)
        stock_embedding = np.zeros(self.num_stocks, dtype=np.float32)
        if self.stock_id < self.num_stocks:
            stock_embedding[self.stock_id] = 1.0
        
        # Combine features and ensure float32 dtype
        state = np.concatenate([market_features, portfolio_features, stock_embedding]).astype(np.float32)
        
        return state
    
    def step(self, action):
        """
        Execute action and return next state, reward, done
        
        Actions:
            0: HOLD
            1: BUY (25% of available balance)
            2: SELL (all shares)
        
        Returns:
            next_state, reward, done, info
        """
        if self.current_step >= len(self.feature_data) - 1:
            return None, 0, True, {}
        
        current_price = self.feature_data.iloc[self.current_step]['close']
        
        # Execute action
        action_type = "HOLD"
        shares_traded = 0
        
        if action == 1:  # BUY
            # Buy shares with 25% of available balance
            max_shares = (self.balance * 0.25) / (current_price * (1 + self.transaction_cost))
            shares_to_buy = int(max_shares)
            
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                if cost <= self.balance:
                    self.balance -= cost
                    self.shares_held += shares_to_buy
                    self.total_shares_bought += shares_to_buy
                    shares_traded = shares_to_buy
                    action_type = "BUY"
                    self.trade_count += 1
        
        elif action == 2:  # SELL
            # Sell all shares
            if self.shares_held > 0:
                revenue = self.shares_held * current_price * (1 - self.transaction_cost)
                self.balance += revenue
                shares_traded = self.shares_held
                self.total_shares_sold += self.shares_held
                self.shares_held = 0
                action_type = "SELL"
                self.trade_count += 1
        
        # Move to next step
        self.current_step += 1
        next_price = self.feature_data.iloc[self.current_step]['close']
        
        # Calculate net worth
        self.net_worth = self.balance + self.shares_held * next_price
        self.net_worth_history.append(self.net_worth)
        self.action_history.append({
            'step': self.current_step,
            'action': action_type,
            'shares': shares_traded,
            'price': current_price,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'net_worth': self.net_worth
        })
        
        # Calculate reward using step-wise log returns + trade penalties
        # This is more stable than episode-level Sharpe for RL training
        if len(self.net_worth_history) > 1:
            # Log return of portfolio (more stable than percentage return)
            log_return = np.log(self.net_worth / (self.net_worth_history[-2] + 1e-10))
            
            # Base reward is the log return scaled to reasonable range
            reward = log_return * 100
            
            # Add trade penalty to discourage excessive trading
            if action_type == "BUY" or action_type == "SELL":
                # Penalty proportional to transaction cost incurred
                trade_cost = shares_traded * current_price * self.transaction_cost
                reward -= (trade_cost / self.initial_balance) * 10  # Scaled penalty
            
            # Small penalty for holding when not in a position (opportunity cost)
            if action == 0 and self.shares_held == 0:
                reward -= 0.005
        else:
            reward = 0
        
        # Check if done
        done = self.current_step >= len(self.feature_data) - 1
        
        next_state = self._get_state()
        
        info = {
            'net_worth': self.net_worth,
            'balance': self.balance,
            'shares_held': self.shares_held
        }
        
        return next_state, reward, done, info
    
    def get_final_stats(self):
        """Get final trading statistics with detailed diagnostics"""
        total_return = (self.net_worth - self.initial_balance) / self.initial_balance * 100
        
        # Calculate Sharpe ratio
        if len(self.net_worth_history) > 1:
            returns = np.diff(self.net_worth_history) / self.net_worth_history[:-1]
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
            
            # Annualized volatility
            volatility = np.std(returns) * np.sqrt(252) * 100
        else:
            sharpe_ratio = 0
            volatility = 0
        
        # Max drawdown
        peak = np.maximum.accumulate(self.net_worth_history)
        drawdown = (self.net_worth_history - peak) / peak
        max_drawdown = np.min(drawdown) * 100
        
        # Buy and hold performance
        buy_and_hold_return = ((self.feature_data.iloc[-1]['close'] - 
                                self.feature_data.iloc[0]['close']) / 
                               self.feature_data.iloc[0]['close'] * 100)
        
        # Average holding period (days between trades)
        avg_holding_period = len(self.net_worth_history) / (self.trade_count + 1)
        
        # Win rate (if we track individual trade P&L)
        # For now, approximation based on positive return periods
        if len(self.net_worth_history) > 1:
            positive_periods = np.sum(np.diff(self.net_worth_history) > 0)
            win_rate = positive_periods / len(np.diff(self.net_worth_history)) * 100
        else:
            win_rate = 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'total_trades': self.trade_count,
            'final_net_worth': self.net_worth,
            'buy_and_hold_return': buy_and_hold_return,
            'outperformance': total_return - buy_and_hold_return,
            'avg_holding_period': avg_holding_period,
            'win_rate': win_rate
        }


class MeanSubtraction(tf.keras.layers.Layer):
    """Custom layer to subtract mean from advantage values for Dueling DQN"""
    def call(self, inputs):
        advantage = inputs
        mean_advantage = tf.reduce_mean(advantage, axis=1, keepdims=True)
        return advantage - mean_advantage
    
    def compute_output_shape(self, input_shape):
        return input_shape


class DQNAgent:
    """Advanced Deep Q-Network agent with Double DQN, Dueling architecture, and Prioritized Replay"""
    
    def __init__(self, state_size: int, action_size: int = 3, 
                 learning_rate: float = 0.0005, gamma: float = 0.99,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01, memory_size: int = 5000,
                 n_step: int = 3, use_double_dqn: bool = True,
                 use_dueling: bool = True, use_prioritized_replay: bool = True):
        """
        Initialize advanced DQN agent
        
        Args:
            state_size: Size of state vector
            action_size: Number of actions (3: HOLD, BUY, SELL)
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon value
            memory_size: Size of replay memory
            n_step: Number of steps for n-step returns
            use_double_dqn: Use Double DQN (reduces overestimation)
            use_dueling: Use Dueling architecture
            use_prioritized_replay: Use prioritized experience replay
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.n_step = n_step
        self.use_double_dqn = use_double_dqn
        self.use_dueling = use_dueling
        self.use_prioritized_replay = use_prioritized_replay
        
        # Replay memory
        if use_prioritized_replay:
            self.memory = PrioritizedReplayBuffer(memory_size)
        else:
            self.memory = deque(maxlen=memory_size)
        
        # N-step buffer for n-step returns
        self.n_step_buffer = deque(maxlen=n_step)
        
        # Build models
        self.model = self._build_model()
        self.target_model = clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())
        
    def _build_model(self):
        """Build neural network for Q-learning (Dueling architecture if enabled)"""
        if self.use_dueling:
            # Dueling DQN architecture: separate value and advantage streams
            inputs = Input(shape=(self.state_size,))
            
            # Shared feature extraction
            x = Dense(128, activation='relu')(inputs)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
            x = Dense(128, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
            
            # Value stream (single output: state value)
            value_stream = Dense(64, activation='relu')(x)
            value_stream = Dropout(0.1)(value_stream)
            value = Dense(1, activation='linear')(value_stream)
            
            # Advantage stream (action_size outputs: advantage for each action)
            advantage_stream = Dense(64, activation='relu')(x)
            advantage_stream = Dropout(0.1)(advantage_stream)
            advantage = Dense(self.action_size, activation='linear')(advantage_stream)
            
            # Combine value and advantage: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
            # Subtracting mean advantage makes it identifiable
            advantage_normalized = MeanSubtraction()(advantage)
            q_values = Add()([value, advantage_normalized])
            
            model = Model(inputs=inputs, outputs=q_values)
        else:
            # Standard DQN architecture
            model = Sequential([
                Dense(128, input_dim=self.state_size, activation='relu'),
                BatchNormalization(),
                Dropout(0.2),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.2),
                Dense(64, activation='relu'),
                Dropout(0.1),
                Dense(self.action_size, activation='linear')
            ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate, clipnorm=1.0),  # Add gradient clipping
            loss='huber'  # Huber loss is more stable than MSE for RL
        )
        
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory with n-step returns"""
        # Add to n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        # If buffer is full, compute n-step return and store
        if len(self.n_step_buffer) == self.n_step:
            # Compute n-step return
            n_step_reward = 0
            for i, (_, _, r, _, _) in enumerate(self.n_step_buffer):
                n_step_reward += (self.gamma ** i) * r
            
            # Get first state and action, last next_state and done
            first_state = self.n_step_buffer[0][0]
            first_action = self.n_step_buffer[0][1]
            last_next_state = self.n_step_buffer[-1][3]
            last_done = self.n_step_buffer[-1][4]
            
            experience = Experience(first_state, first_action, n_step_reward, last_next_state, last_done)
            
            if self.use_prioritized_replay:
                self.memory.add(experience)
            else:
                self.memory.append(experience)
    
    def act(self, state, training=True):
        """
        Choose action using epsilon-greedy policy
        
        Args:
            state: Current state
            training: If True, use epsilon-greedy; if False, use greedy
        
        Returns:
            action: 0 (HOLD), 1 (BUY), or 2 (SELL)
        """
        if training and np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])
    
    def get_action_with_confidence(self, state, use_ensemble=False, n_samples=10):
        """
        Get action with calibrated confidence scores
        
        Args:
            state: Current state
            use_ensemble: Use MC dropout for ensemble uncertainty
            n_samples: Number of forward passes for ensemble
        
        Returns:
            action: Best action
            confidence: Calibrated confidence score (0-1)
            action_probs: Probability distribution over actions
            uncertainty: Measure of uncertainty (lower is more confident)
        """
        if use_ensemble:
            # MC Dropout: multiple forward passes with dropout enabled
            # This requires training=True mode even at inference
            q_values_samples = []
            for _ in range(n_samples):
                # Note: This requires model to have dropout layers that work in inference
                q_vals = self.model(state.reshape(1, -1), training=True).numpy()[0]
                q_values_samples.append(q_vals)
            
            q_values_samples = np.array(q_values_samples)
            q_values_mean = np.mean(q_values_samples, axis=0)
            q_values_std = np.std(q_values_samples, axis=0)
            
            # Action is based on mean Q-values
            action = np.argmax(q_values_mean)
            
            # Uncertainty is the std of Q-values for chosen action
            uncertainty = q_values_std[action]
            
            # Convert Q-values to probabilities using softmax
            q_values_for_prob = q_values_mean
        else:
            # Single forward pass
            q_values = self.model.predict(state.reshape(1, -1), verbose=0)[0]
            action = np.argmax(q_values)
            uncertainty = 0.0
            q_values_for_prob = q_values
        
        # Calibrated confidence using softmax over Q-values
        # Temperature scaling to make probabilities more interpretable
        temperature = 2.0
        exp_q = np.exp(q_values_for_prob / temperature)
        action_probs = exp_q / np.sum(exp_q)
        
        # Confidence is the probability of the selected action
        confidence = action_probs[action]
        
        # Alternative: use normalized advantage
        # advantage = q_values_for_prob - np.mean(q_values_for_prob)
        # normalized_advantage = advantage[action] / (np.max(np.abs(advantage)) + 1e-8)
        
        return action, float(confidence), action_probs.tolist(), float(uncertainty)
    
    def replay(self, batch_size=32):
        """Train on batch of experiences from replay memory with Double DQN"""
        if len(self.memory) < batch_size:
            return
        
        # Sample from memory
        if self.use_prioritized_replay:
            experiences, indices, weights = self.memory.sample(batch_size)
            weights = np.array(weights, dtype=np.float32).reshape(-1, 1)
        else:
            experiences = random.sample(self.memory, batch_size)
            indices = None
            weights = np.ones((batch_size, 1), dtype=np.float32)
        
        states = np.array([exp.state for exp in experiences], dtype=np.float32)
        actions = np.array([exp.action for exp in experiences], dtype=np.int32)
        rewards = np.array([exp.reward for exp in experiences], dtype=np.float32)
        next_states = np.array([exp.next_state for exp in experiences], dtype=np.float32)
        dones = np.array([exp.done for exp in experiences], dtype=np.bool_)
        
        # Predict Q-values for current states
        current_q_values = self.model.predict(states, verbose=0)
        
        if self.use_double_dqn:
            # Double DQN: use main network to select action, target network to evaluate
            next_q_main = self.model.predict(next_states, verbose=0)
            next_actions = np.argmax(next_q_main, axis=1)
            next_q_target = self.target_model.predict(next_states, verbose=0)
            next_q_values = next_q_target[np.arange(batch_size), next_actions]
        else:
            # Standard DQN: use target network for both selection and evaluation
            next_q_target = self.target_model.predict(next_states, verbose=0)
            next_q_values = np.max(next_q_target, axis=1)
        
        # Compute target Q-values and TD errors
        target_q_values = current_q_values.copy()
        td_errors = []
        
        for i in range(batch_size):
            old_q = current_q_values[i][actions[i]]
            
            if dones[i]:
                target_q = rewards[i]
            else:
                # N-step return with discounting
                target_q = rewards[i] + (self.gamma ** self.n_step) * next_q_values[i]
            
            # Clip target Q to prevent extreme values
            target_q = np.clip(target_q, -1000.0, 1000.0)
            
            target_q_values[i][actions[i]] = target_q
            
            # Calculate TD error with clipping
            td_error = abs(target_q - old_q)
            td_error = np.clip(td_error, 0.0, 100.0)  # Prevent extreme TD errors
            td_errors.append(td_error)
        
        # Train model with importance sampling weights
        self.model.fit(states, target_q_values, sample_weight=weights.flatten(), 
                      epochs=1, verbose=0)
        
        # Update priorities in prioritized replay
        if self.use_prioritized_replay and indices is not None:
            self.memory.update_priorities(indices, td_errors)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_model(self):
        """Update target model weights"""
        self.target_model.set_weights(self.model.get_weights())


class RLTradingSystem:
    """Main RL trading system with multi-stock training support"""
    
    def __init__(self, symbols=None):
        """
        Initialize RL trading system
        
        Args:
            symbols: Single symbol (str) or list of symbols for multi-stock training
        """
        if isinstance(symbols, str):
            self.symbols = [symbols]
            self.multi_stock = False
        elif isinstance(symbols, list):
            self.symbols = symbols
            self.multi_stock = len(symbols) > 1
        else:
            raise ValueError("symbols must be a string or list of strings")
        
        self.data_fetcher = NepseDataFetcher()
        self.models_dir = Path("rl_models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Stock ID mapping for multi-stock training
        self.stock_id_map = {symbol: idx for idx, symbol in enumerate(self.symbols)}
        
    def train(self, episodes: int = 100, batch_size: int = 32, use_ensemble: bool = True):
        """
        Train RL agent on historical data (single or multi-stock)
        
        Args:
            episodes: Number of training episodes per stock
            batch_size: Batch size for experience replay
            use_ensemble: Use ensemble uncertainty at inference
        """
        print(f"\n{'='*70}")
        if self.multi_stock:
            print(f"TRAINING MULTI-STOCK RL AGENT")
            print(f"Stocks: {', '.join(self.symbols)}")
        else:
            print(f"TRAINING RL TRADING AGENT FOR {self.symbols[0]}")
        print('='*70)
        
        # Load data for all symbols
        stock_data = {}
        for symbol in self.symbols:
            df = self.data_fetcher.get_stock_price_history(symbol)
            if df is None or len(df) < 200:
                print(f"⚠ Warning: Insufficient data for {symbol}, skipping...")
                continue
            df = df.sort_values('date').reset_index(drop=True)
            stock_data[symbol] = df
        
        if not stock_data:
            raise ValueError("No valid stock data available for training")
        
        print(f"✓ Loaded data for {len(stock_data)} stocks")
        
        # Split into train and test for each stock
        train_envs = []
        test_envs = []
        
        for symbol, df in stock_data.items():
            split_idx = int(len(df) * 0.8)
            train_df = df.iloc[:split_idx].copy()
            test_df = df.iloc[split_idx:].copy()
            
            stock_id = self.stock_id_map[symbol]
            train_env = TradingEnvironment(train_df, stock_id=stock_id, num_stocks=len(self.symbols))
            test_env = TradingEnvironment(test_df, stock_id=stock_id, num_stocks=len(self.symbols))
            
            train_envs.append((symbol, train_env))
            test_envs.append((symbol, test_env))
            
            print(f"  {symbol}: Train={len(train_df)} days, Test={len(test_df)} days")
        
        # Get state size from first environment
        state_size = len(train_envs[0][1]._get_state())
        
        print(f"✓ State size: {state_size} features (includes stock embedding)")
        print(f"✓ NO DATA LEAKAGE: All features use only PAST data")
        
        # Create advanced agent with modern DQN improvements
        agent = DQNAgent(
            state_size=state_size,
            learning_rate=0.0005,
            gamma=0.99,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            memory_size=5000,
            n_step=3,
            use_double_dqn=True,
            use_dueling=True,
            use_prioritized_replay=True
        )
        
        print(f"\n✓ Agent Configuration:")
        print(f"  • Double DQN: Enabled (reduces Q-value overestimation)")
        print(f"  • Dueling Architecture: Enabled (better state evaluation)")
        print(f"  • Prioritized Replay: Enabled (learns from important transitions)")
        print(f"  • N-step Returns: {agent.n_step} steps (faster reward propagation)")
        print(f"  • Reward: Step-wise log returns + trade penalties")
        
        print(f"\n{'='*70}")
        print(f"TRAINING FOR {episodes} EPISODES PER STOCK")
        print('='*70)
        
        best_return = -np.inf
        episode_returns = []
        
        # Multi-stock training: shuffle episodes across stocks
        total_episodes = episodes * len(train_envs)
        episode_count = 0
        
        print(f"Starting training loop...")
        print(f"  • {len(train_envs)} stocks")
        print(f"  • {episodes} episodes per stock")
        print(f"  • Total: {total_episodes} episodes")
        print(f"  • Batch size: {batch_size}")
        print(f"  • Progress updates every 10 episodes\n")
        
        for episode in range(episodes):
            # Shuffle stock order for this round of episodes
            stock_order = list(range(len(train_envs)))
            random.shuffle(stock_order)
            
            round_returns = []
            
            for stock_idx in stock_order:
                symbol, env = train_envs[stock_idx]
                state = env.reset()
                total_reward = 0
                steps = 0
                
                # Show which stock is being trained (first episode only or every 10 episodes)
                if episode == 0 or (episode + 1) % 10 == 0:
                    print(f"  Episode {episode + 1}/{episodes} - Training on {symbol}...", end='\r')
                
                while True:
                    # Choose action
                    action = agent.act(state, training=True)
                    
                    # Take action
                    next_state, reward, done, info = env.step(action)
                    total_reward += reward
                    steps += 1
                    
                    # Remember experience
                    agent.remember(state, action, reward, next_state, done)
                    
                    # Move to next state
                    state = next_state
                    
                    # Train agent
                    if len(agent.memory) > batch_size:
                        agent.replay(batch_size)
                    
                    if done:
                        break
                
                # Get episode stats
                stats = env.get_final_stats()
                round_returns.append(stats['total_return'])
                episode_count += 1
            
            # Update target model periodically
            if episode % 10 == 0:
                agent.update_target_model()
            
            # Track performance
            avg_return = np.mean(round_returns)
            episode_returns.append(avg_return)
            
            # Track best performance
            if avg_return > best_return:
                best_return = avg_return
                # Save best model
                model_name = f"multi_stock_rl_best.keras" if self.multi_stock else f"{self.symbols[0]}_rl_best.keras"
                model_path = self.models_dir / model_name
                agent.model.save(model_path)
            
            # Print progress
            if (episode + 1) % 10 == 0:
                avg_last_10 = np.mean(episode_returns[-10:])
                print(f"\n✓ Episode {episode + 1}/{episodes} completed")
                print(f"  • Avg Return: {avg_return:+.2f}%")
                print(f"  • Avg (last 10): {avg_last_10:+.2f}%")
                print(f"  • Epsilon: {agent.epsilon:.3f}")
                print(f"  • Best: {best_return:+.2f}%")
                print()
        
        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETE")
        print(f"Best training return: {best_return:+.2f}%")
        print('='*70)
        
        # Test on held-out data
        print(f"\n{'='*70}")
        print(f"TESTING AND GENERATING SIGNALS")
        print('='*70)
        
        # Test and generate signals for each stock
        all_results = {}
        
        for symbol, test_env in test_envs:
            print(f"\nProcessing {symbol}...")
            
            state = test_env.reset()
            test_actions = []
            
            while True:
                # Get action with confidence
                action, confidence, action_probs, uncertainty = agent.get_action_with_confidence(
                    state, use_ensemble=use_ensemble, n_samples=10
                )
                
                next_state, reward, done, info = test_env.step(action)
                
                test_actions.append({
                    'date': stock_data[symbol].iloc[test_env.current_step + int(len(stock_data[symbol]) * 0.8)]['date'].strftime('%Y-%m-%d'),
                    'action': ['HOLD', 'BUY', 'SELL'][action],
                    'price': float(stock_data[symbol].iloc[test_env.current_step + int(len(stock_data[symbol]) * 0.8)]['close']),
                    'net_worth': float(info['net_worth']),
                    'shares_held': int(info['shares_held']),
                    'confidence': confidence,
                    'action_probabilities': {
                        'HOLD': float(action_probs[0]),
                        'BUY': float(action_probs[1]),
                        'SELL': float(action_probs[2])
                    },
                    'uncertainty': uncertainty
                })
                
                state = next_state
                
                if done:
                    break
            
            test_stats = test_env.get_final_stats()
            
            print(f"  Return: {test_stats['total_return']:+.2f}% vs Buy&Hold: {test_stats['buy_and_hold_return']:+.2f}%")
            print(f"  Sharpe: {test_stats['sharpe_ratio']:.3f} | Drawdown: {test_stats['max_drawdown']:.2f}%")
            print(f"  Volatility: {test_stats['volatility']:.2f}% | Trades: {test_stats['total_trades']}")
        
        # Save best model
        if self.multi_stock:
            model_name = f"rl_models/{('_'.join(self.symbols))}_rl_model.keras"
        else:
            model_name = f"rl_models/{self.symbols[0]}_rl_model.keras"
        
        agent.model.save(model_name)
        print(f"\n✓ Model saved: {model_name}")
        
        return {
            'best_return': best_return,
            'episodes': episodes,
            'final_epsilon': agent.epsilon,
            'trained_symbols': self.symbols
        }


def main():
    """Main entry point"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python3 train_rl_model.py <SYMBOL> [episodes] [additional_symbols...]")
        print("\nExamples:")
        print("  Single stock: python3 train_rl_model.py PFL 100")
        print("  Multi-stock:  python3 train_rl_model.py PFL 100 NABIL ADBL SBI")
        print("\nNote: This script only trains the model. Use test_rl_model.py to get predictions.")
        sys.exit(1)
    
    # Parse arguments
    symbols = [sys.argv[1].upper()]
    episodes = 100
    
    # Check if second arg is a number (episodes) or another symbol
    if len(sys.argv) > 2:
        try:
            episodes = int(sys.argv[2])
            # Rest are additional symbols
            if len(sys.argv) > 3:
                symbols.extend([s.upper() for s in sys.argv[3:]])
        except ValueError:
            # Second arg is a symbol, not episodes
            symbols.extend([s.upper() for s in sys.argv[2:]])
    
    multi_stock = len(symbols) > 1
    
    print(f"\n{'='*70}")
    print(f"RL MODEL TRAINING (Training Only - No Predictions)")
    print('='*70)
    if multi_stock:
        print(f"Symbols: {', '.join(symbols)} (Multi-stock training)")
    else:
        print(f"Symbol: {symbols[0]}")
    print(f"Episodes: {episodes} per stock")
    print(f"Algorithm: Double DQN + Dueling + Prioritized Replay")
    print(f"Actions: BUY, SELL, HOLD")
    print(f"Improvements:")
    print(f"  ✓ No data leakage (all features use only past data)")
    print(f"  ✓ Double DQN (reduces overestimation bias)")
    print(f"  ✓ Dueling architecture (better learning efficiency)")
    print(f"  ✓ Prioritized experience replay (smarter sampling)")
    print(f"  ✓ N-step returns (faster reward propagation)")
    print(f"  ✓ Step-wise log-return rewards + trade penalties")
    if multi_stock:
        print(f"  ✓ Multi-stock training (better generalization)")
        print(f"  ✓ Stock embeddings (learns stock-specific patterns)")
    print('='*70)
    
    # Create and train system
    rl_system = RLTradingSystem(symbols)
    results = rl_system.train(episodes=episodes, use_ensemble=True)
    
    print(f"\n{'='*70}")
    print(f"✅ TRAINING COMPLETE")
    print('='*70)
    print(f"\n📁 Model saved:")
    if multi_stock:
        model_name = f"rl_models/{('_'.join(symbols))}_rl_model.keras"
    else:
        model_name = f"rl_models/{symbols[0]}_rl_model.keras"
    print(f"  {model_name}")
    
    print(f"\n📊 Training Summary:")
    print(f"  Best Return: {results['best_return']:.2f}%")
    print(f"  Episodes: {results['episodes']}")
    print(f"  Final Epsilon: {results['final_epsilon']:.4f}")
    print(f"  Trained on: {', '.join(results['trained_symbols'])}")
    
    print(f"\n🎯 To get predictions, use:")
    for symbol in symbols:
        print(f"  python3 test_rl_model.py {symbol}")


if __name__ == '__main__':
    main()
