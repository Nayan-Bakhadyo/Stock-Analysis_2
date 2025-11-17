"""Technical analysis module for chart analysis and indicators"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime
import config


class TechnicalAnalyzer:
    """Perform technical analysis using various indicators"""
    
    def __init__(self):
        self.params = config.TECHNICAL_INDICATORS
    
    def calculate_sma(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return data.rolling(window=period).mean()
    
    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    def calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(self, data: pd.Series, fast: int = 12, slow: int = 26, 
                       signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        ema_fast = self.calculate_ema(data, fast)
        ema_slow = self.calculate_ema(data, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def calculate_bollinger_bands(self, data: pd.Series, period: int = 20, 
                                  std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = self.calculate_sma(data, period)
        std = data.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, sma, lower_band
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                      period: int = 14) -> pd.Series:
        """Calculate Average True Range (volatility indicator)"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                            period: int = 14) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=3).mean()
        
        return k_percent, d_percent
    
    def calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume"""
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def identify_support_resistance(self, data: pd.Series, window: int = 20) -> Dict:
        """Identify support and resistance levels"""
        # Find local minima (support) and maxima (resistance)
        local_min = data[(data.shift(1) > data) & (data.shift(-1) > data)]
        local_max = data[(data.shift(1) < data) & (data.shift(-1) < data)]
        
        # Get recent levels
        support_levels = sorted(local_min.tail(window).tolist())
        resistance_levels = sorted(local_max.tail(window).tolist(), reverse=True)
        
        return {
            'support': support_levels[:3] if support_levels else [],
            'resistance': resistance_levels[:3] if resistance_levels else [],
            'current_price': data.iloc[-1]
        }
    
    def detect_chart_patterns(self, data: pd.DataFrame) -> list:
        """Detect common chart patterns"""
        patterns = []
        
        if len(data) < 20:
            return patterns
        
        close = data['close']
        
        # Head and Shoulders
        if self._is_head_and_shoulders(close):
            patterns.append({
                'pattern': 'Head and Shoulders',
                'type': 'Bearish',
                'confidence': 0.7
            })
        
        # Double Top/Bottom
        if self._is_double_top(close):
            patterns.append({
                'pattern': 'Double Top',
                'type': 'Bearish',
                'confidence': 0.65
            })
        
        if self._is_double_bottom(close):
            patterns.append({
                'pattern': 'Double Bottom',
                'type': 'Bullish',
                'confidence': 0.65
            })
        
        # Triangle patterns
        if self._is_ascending_triangle(data):
            patterns.append({
                'pattern': 'Ascending Triangle',
                'type': 'Bullish',
                'confidence': 0.6
            })
        
        return patterns
    
    def _is_head_and_shoulders(self, close: pd.Series) -> bool:
        """Detect head and shoulders pattern (simplified)"""
        if len(close) < 30:
            return False
        
        recent = close.tail(30)
        peaks = recent[(recent.shift(1) < recent) & (recent.shift(-1) < recent)]
        
        if len(peaks) >= 3:
            peak_values = peaks.tail(3).values
            # Head should be higher than shoulders
            if peak_values[1] > peak_values[0] and peak_values[1] > peak_values[2]:
                return True
        
        return False
    
    def _is_double_top(self, close: pd.Series) -> bool:
        """Detect double top pattern (simplified)"""
        if len(close) < 20:
            return False
        
        recent = close.tail(20)
        peaks = recent[(recent.shift(1) < recent) & (recent.shift(-1) < recent)]
        
        if len(peaks) >= 2:
            last_two_peaks = peaks.tail(2).values
            # Two peaks should be roughly at same level
            if abs(last_two_peaks[0] - last_two_peaks[1]) / last_two_peaks[0] < 0.02:
                return True
        
        return False
    
    def _is_double_bottom(self, close: pd.Series) -> bool:
        """Detect double bottom pattern (simplified)"""
        if len(close) < 20:
            return False
        
        recent = close.tail(20)
        troughs = recent[(recent.shift(1) > recent) & (recent.shift(-1) > recent)]
        
        if len(troughs) >= 2:
            last_two_troughs = troughs.tail(2).values
            # Two troughs should be roughly at same level
            if abs(last_two_troughs[0] - last_two_troughs[1]) / last_two_troughs[0] < 0.02:
                return True
        
        return False
    
    def detect_candlestick_patterns(self, data: pd.DataFrame, lookback: int = 5) -> list:
        """
        Detect candlestick patterns in recent price data
        
        Args:
            data: DataFrame with OHLC data
            lookback: Number of recent candles to analyze
            
        Returns:
            List of detected candlestick patterns with type and confidence
        """
        patterns = []
        
        if len(data) < 3:
            return patterns
        
        # Analyze recent candles
        recent_data = data.tail(lookback)
        
        for i in range(len(recent_data)):
            candle = recent_data.iloc[i]
            
            # Get previous candle if available
            prev_candle = recent_data.iloc[i-1] if i > 0 else None
            next_candle = recent_data.iloc[i+1] if i < len(recent_data) - 1 else None
            
            # Calculate candle properties
            body = abs(candle['close'] - candle['open'])
            range_high_low = candle['high'] - candle['low']
            upper_shadow = candle['high'] - max(candle['open'], candle['close'])
            lower_shadow = min(candle['open'], candle['close']) - candle['low']
            
            is_bullish = candle['close'] > candle['open']
            is_bearish = candle['close'] < candle['open']
            
            # Skip if no range
            if range_high_low == 0:
                continue
            
            body_ratio = body / range_high_low
            
            # 1. Doji - Small body with long shadows (indecision)
            if body_ratio < 0.1:
                patterns.append({
                    'pattern': 'Doji',
                    'type': 'Neutral',
                    'confidence': 0.7,
                    'position': i,
                    'description': 'Indecision - potential reversal'
                })
            
            # 2. Hammer - Small body at top, long lower shadow (bullish reversal)
            if (body_ratio < 0.3 and 
                lower_shadow > body * 2 and 
                upper_shadow < body * 0.3 and
                i == len(recent_data) - 1):  # Only recent
                patterns.append({
                    'pattern': 'Hammer',
                    'type': 'Bullish',
                    'confidence': 0.75,
                    'position': i,
                    'description': 'Bullish reversal signal'
                })
            
            # 3. Hanging Man - Same as hammer but at top of uptrend (bearish)
            if (body_ratio < 0.3 and 
                lower_shadow > body * 2 and 
                upper_shadow < body * 0.3 and
                prev_candle is not None and
                candle['close'] > prev_candle['close'] and
                i == len(recent_data) - 1):
                patterns.append({
                    'pattern': 'Hanging Man',
                    'type': 'Bearish',
                    'confidence': 0.7,
                    'position': i,
                    'description': 'Bearish reversal after uptrend'
                })
            
            # 4. Inverted Hammer - Small body at bottom, long upper shadow
            if (body_ratio < 0.3 and 
                upper_shadow > body * 2 and 
                lower_shadow < body * 0.3 and
                i == len(recent_data) - 1):
                patterns.append({
                    'pattern': 'Inverted Hammer',
                    'type': 'Bullish',
                    'confidence': 0.65,
                    'position': i,
                    'description': 'Potential bullish reversal'
                })
            
            # 5. Marubozu - Very large body, almost no shadows (strong trend)
            if body_ratio > 0.9:
                pattern_type = 'Bullish' if is_bullish else 'Bearish'
                patterns.append({
                    'pattern': 'Marubozu',
                    'type': pattern_type,
                    'confidence': 0.8,
                    'position': i,
                    'description': f'Strong {pattern_type.lower()} momentum'
                })
            
            # 6. Spinning Top - Small body, long shadows on both sides (indecision)
            if (0.1 < body_ratio < 0.3 and 
                upper_shadow > body and 
                lower_shadow > body):
                patterns.append({
                    'pattern': 'Spinning Top',
                    'type': 'Neutral',
                    'confidence': 0.65,
                    'position': i,
                    'description': 'Indecision in market'
                })
            
            # 7. Bullish Engulfing - Current bullish candle engulfs previous bearish
            if (prev_candle is not None and 
                is_bullish and 
                prev_candle['close'] < prev_candle['open'] and
                candle['open'] < prev_candle['close'] and
                candle['close'] > prev_candle['open'] and
                i == len(recent_data) - 1):
                patterns.append({
                    'pattern': 'Bullish Engulfing',
                    'type': 'Bullish',
                    'confidence': 0.8,
                    'position': i,
                    'description': 'Strong bullish reversal'
                })
            
            # 8. Bearish Engulfing - Current bearish candle engulfs previous bullish
            if (prev_candle is not None and 
                is_bearish and 
                prev_candle['close'] > prev_candle['open'] and
                candle['open'] > prev_candle['close'] and
                candle['close'] < prev_candle['open'] and
                i == len(recent_data) - 1):
                patterns.append({
                    'pattern': 'Bearish Engulfing',
                    'type': 'Bearish',
                    'confidence': 0.8,
                    'position': i,
                    'description': 'Strong bearish reversal'
                })
        
        # 9. Morning Star - 3-candle bullish reversal pattern
        if len(recent_data) >= 3:
            last_3 = recent_data.tail(3)
            c1, c2, c3 = last_3.iloc[0], last_3.iloc[1], last_3.iloc[2]
            
            # First candle bearish, second small body, third bullish
            if (c1['close'] < c1['open'] and
                abs(c2['close'] - c2['open']) < abs(c1['close'] - c1['open']) * 0.3 and
                c3['close'] > c3['open'] and
                c3['close'] > (c1['open'] + c1['close']) / 2):
                patterns.append({
                    'pattern': 'Morning Star',
                    'type': 'Bullish',
                    'confidence': 0.85,
                    'position': len(recent_data) - 1,
                    'description': '3-candle bullish reversal'
                })
        
        # 10. Evening Star - 3-candle bearish reversal pattern
        if len(recent_data) >= 3:
            last_3 = recent_data.tail(3)
            c1, c2, c3 = last_3.iloc[0], last_3.iloc[1], last_3.iloc[2]
            
            # First candle bullish, second small body, third bearish
            if (c1['close'] > c1['open'] and
                abs(c2['close'] - c2['open']) < abs(c1['close'] - c1['open']) * 0.3 and
                c3['close'] < c3['open'] and
                c3['close'] < (c1['open'] + c1['close']) / 2):
                patterns.append({
                    'pattern': 'Evening Star',
                    'type': 'Bearish',
                    'confidence': 0.85,
                    'position': len(recent_data) - 1,
                    'description': '3-candle bearish reversal'
                })
        
        return patterns
    
    def _is_ascending_triangle(self, data: pd.DataFrame) -> bool:
        """Detect ascending triangle pattern (simplified)"""
        if len(data) < 20:
            return False
        
        recent = data.tail(20)
        highs = recent['high']
        lows = recent['low']
        
        # Check if highs are relatively flat and lows are rising
        high_std = highs.std() / highs.mean()
        low_trend = (lows.iloc[-1] - lows.iloc[0]) / lows.iloc[0]
        
        if high_std < 0.02 and low_trend > 0.05:
            return True
        
        return False
    
    def comprehensive_analysis(self, data: pd.DataFrame) -> Dict:
        """Perform comprehensive technical analysis"""
        
        if len(data) < 50:
            return {
                'error': 'Insufficient data for technical analysis',
                'min_required': 50,
                'available': len(data)
            }
        
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']
        
        # Calculate all indicators
        sma_short = self.calculate_sma(close, self.params['short_ma'])
        sma_medium = self.calculate_sma(close, self.params['medium_ma'])
        sma_long = self.calculate_sma(close, self.params['long_ma'])
        
        rsi = self.calculate_rsi(close, self.params['rsi_period'])
        
        macd_line, signal_line, histogram = self.calculate_macd(
            close, 
            self.params['macd_fast'],
            self.params['macd_slow'],
            self.params['macd_signal']
        )
        
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(
            close,
            self.params['bb_period'],
            self.params['bb_std']
        )
        
        atr = self.calculate_atr(high, low, close)
        
        stoch_k, stoch_d = self.calculate_stochastic(high, low, close)
        
        obv = self.calculate_obv(close, volume)
        
        # Get latest values
        current_price = close.iloc[-1]
        current_rsi = rsi.iloc[-1]
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        current_stoch_k = stoch_k.iloc[-1]
        
        # Analyze signals
        signals = self._analyze_signals(
            current_price, sma_short.iloc[-1], sma_medium.iloc[-1], sma_long.iloc[-1],
            current_rsi, current_macd, current_signal, current_stoch_k,
            bb_upper.iloc[-1], bb_lower.iloc[-1]
        )
        
        # Detect patterns
        patterns = self.detect_chart_patterns(data)
        candlestick_patterns = self.detect_candlestick_patterns(data, lookback=10)
        
        # Support and resistance
        sr_levels = self.identify_support_resistance(close)
        
        # Calculate momentum
        momentum = self._calculate_momentum(close, volume, obv)
        
        # Overall technical score
        technical_score = self._calculate_technical_score(signals, patterns, momentum)
        
        return {
            'symbol': data.get('symbol', ['N/A'])[0] if 'symbol' in data.columns else 'N/A',
            'current_price': float(current_price),
            'indicators': {
                'sma_short': float(sma_short.iloc[-1]),
                'sma_medium': float(sma_medium.iloc[-1]),
                'sma_long': float(sma_long.iloc[-1]),
                'rsi': float(current_rsi),
                'macd': float(current_macd),
                'macd_signal': float(current_signal),
                'macd_histogram': float(histogram.iloc[-1]),
                'stochastic_k': float(current_stoch_k),
                'stochastic_d': float(stoch_d.iloc[-1]),
                'bb_upper': float(bb_upper.iloc[-1]),
                'bb_middle': float(bb_middle.iloc[-1]),
                'bb_lower': float(bb_lower.iloc[-1]),
                'atr': float(atr.iloc[-1])
            },
            'signals': signals,
            'patterns': patterns,
            'candlestick_patterns': candlestick_patterns,
            'support_resistance': sr_levels,
            'momentum': momentum,
            'technical_score': technical_score,
            'recommendation': self._get_recommendation(technical_score),
            'analysis_date': datetime.now().isoformat()
        }
    
    def _analyze_signals(self, price: float, sma_short: float, sma_medium: float, 
                        sma_long: float, rsi: float, macd: float, signal: float, 
                        stoch_k: float, bb_upper: float, bb_lower: float) -> Dict:
        """Analyze technical signals"""
        signals = {}
        
        # Moving Average signals
        if price > sma_short > sma_medium > sma_long:
            signals['trend'] = 'Strong Uptrend'
            signals['trend_score'] = 1.0
        elif price > sma_short > sma_medium:
            signals['trend'] = 'Uptrend'
            signals['trend_score'] = 0.7
        elif price < sma_short < sma_medium < sma_long:
            signals['trend'] = 'Strong Downtrend'
            signals['trend_score'] = 0.0
        elif price < sma_short < sma_medium:
            signals['trend'] = 'Downtrend'
            signals['trend_score'] = 0.3
        else:
            signals['trend'] = 'Sideways'
            signals['trend_score'] = 0.5
        
        # RSI signals
        if rsi > 70:
            signals['rsi_signal'] = 'Overbought'
            signals['rsi_score'] = 0.2
        elif rsi > 60:
            signals['rsi_signal'] = 'Bullish'
            signals['rsi_score'] = 0.7
        elif rsi < 30:
            signals['rsi_signal'] = 'Oversold'
            signals['rsi_score'] = 0.8
        elif rsi < 40:
            signals['rsi_signal'] = 'Bearish'
            signals['rsi_score'] = 0.3
        else:
            signals['rsi_signal'] = 'Neutral'
            signals['rsi_score'] = 0.5
        
        # MACD signals
        if macd > signal and macd > 0:
            signals['macd_signal'] = 'Strong Buy'
            signals['macd_score'] = 1.0
        elif macd > signal:
            signals['macd_signal'] = 'Buy'
            signals['macd_score'] = 0.7
        elif macd < signal and macd < 0:
            signals['macd_signal'] = 'Strong Sell'
            signals['macd_score'] = 0.0
        elif macd < signal:
            signals['macd_signal'] = 'Sell'
            signals['macd_score'] = 0.3
        else:
            signals['macd_signal'] = 'Neutral'
            signals['macd_score'] = 0.5
        
        # Stochastic signals
        if stoch_k > 80:
            signals['stochastic_signal'] = 'Overbought'
            signals['stochastic_score'] = 0.2
        elif stoch_k < 20:
            signals['stochastic_signal'] = 'Oversold'
            signals['stochastic_score'] = 0.8
        else:
            signals['stochastic_signal'] = 'Neutral'
            signals['stochastic_score'] = 0.5
        
        # Bollinger Bands signals
        if price > bb_upper:
            signals['bb_signal'] = 'Above Upper Band - Overbought'
            signals['bb_score'] = 0.2
        elif price < bb_lower:
            signals['bb_signal'] = 'Below Lower Band - Oversold'
            signals['bb_score'] = 0.8
        else:
            signals['bb_signal'] = 'Within Bands'
            signals['bb_score'] = 0.5
        
        return signals
    
    def _calculate_momentum(self, close: pd.Series, volume: pd.Series, obv: pd.Series) -> Dict:
        """Calculate momentum indicators"""
        # Price momentum
        price_change_5d = ((close.iloc[-1] - close.iloc[-6]) / close.iloc[-6] * 100) if len(close) > 5 else 0
        price_change_20d = ((close.iloc[-1] - close.iloc[-21]) / close.iloc[-21] * 100) if len(close) > 20 else 0
        
        # Volume trend
        avg_volume_recent = volume.tail(5).mean()
        avg_volume_overall = volume.mean()
        volume_ratio = avg_volume_recent / avg_volume_overall if avg_volume_overall > 0 else 1
        
        # OBV trend
        obv_slope = (obv.iloc[-1] - obv.iloc[-6]) if len(obv) > 5 else 0
        
        return {
            'price_change_5d': float(price_change_5d),
            'price_change_20d': float(price_change_20d),
            'volume_ratio': float(volume_ratio),
            'obv_trend': 'Rising' if obv_slope > 0 else 'Falling',
            'momentum_strength': 'Strong' if abs(price_change_5d) > 5 else 'Moderate' if abs(price_change_5d) > 2 else 'Weak'
        }
    
    def _calculate_technical_score(self, signals: Dict, patterns: list, momentum: Dict) -> float:
        """Calculate overall technical score (0-100)"""
        # Weight different signals
        scores = [
            signals.get('trend_score', 0.5) * 0.25,
            signals.get('rsi_score', 0.5) * 0.20,
            signals.get('macd_score', 0.5) * 0.20,
            signals.get('stochastic_score', 0.5) * 0.15,
            signals.get('bb_score', 0.5) * 0.10,
        ]
        
        # Add pattern bonus/penalty
        pattern_score = 0
        for pattern in patterns:
            if pattern['type'] == 'Bullish':
                pattern_score += pattern['confidence'] * 0.1
            else:
                pattern_score -= pattern['confidence'] * 0.1
        
        total_score = sum(scores) + pattern_score
        total_score = max(0, min(1, total_score))  # Clamp between 0 and 1
        
        return round(total_score * 100, 2)
    
    def _get_recommendation(self, score: float) -> str:
        """Get trading recommendation based on technical score"""
        if score >= 75:
            return 'Strong Buy'
        elif score >= 60:
            return 'Buy'
        elif score >= 40:
            return 'Hold'
        elif score >= 25:
            return 'Sell'
        else:
            return 'Strong Sell'


if __name__ == "__main__":
    # Test with sample data
    analyzer = TechnicalAnalyzer()
    
    # Generate sample price data
    dates = pd.date_range(start='2024-01-01', end='2024-11-15', freq='D')
    np.random.seed(42)
    
    prices = 1000 + np.cumsum(np.random.randn(len(dates)) * 10)
    
    sample_data = pd.DataFrame({
        'date': dates,
        'open': prices + np.random.randn(len(dates)) * 5,
        'high': prices + np.abs(np.random.randn(len(dates)) * 10),
        'low': prices - np.abs(np.random.randn(len(dates)) * 10),
        'close': prices,
        'volume': np.random.randint(100000, 1000000, len(dates)),
        'symbol': 'TEST'
    })
    
    result = analyzer.comprehensive_analysis(sample_data)
    
    print(f"Technical Analysis for {result['symbol']}")
    print(f"Current Price: {result['current_price']:.2f}")
    print(f"Technical Score: {result['technical_score']}/100")
    print(f"Recommendation: {result['recommendation']}")
    print(f"\nTrend: {result['signals']['trend']}")
    print(f"RSI Signal: {result['signals']['rsi_signal']}")
    print(f"MACD Signal: {result['signals']['macd_signal']}")
