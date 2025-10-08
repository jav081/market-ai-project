"""
Chart Reading AI Module
Analyzes historical price charts (candles, volume, RSI, patterns).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volume import VolumeWeightedAveragePrice


class ChartAnalyzer:
    """Analyzes price charts and identifies patterns and indicators."""
    
    def __init__(self):
        self.patterns = []
    
    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            prices: Series of closing prices
            window: Period for RSI calculation
            
        Returns:
            RSI values
        """
        rsi_indicator = RSIIndicator(close=prices, window=window)
        return rsi_indicator.rsi()
    
    def calculate_macd(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            prices: Series of closing prices
            
        Returns:
            Dictionary with MACD line, signal line, and histogram
        """
        macd = MACD(close=prices)
        return {
            'macd_line': macd.macd(),
            'signal_line': macd.macd_signal(),
            'histogram': macd.macd_diff()
        }
    
    def calculate_moving_averages(self, prices: pd.Series, 
                                  windows: List[int] = [20, 50, 200]) -> Dict[str, pd.Series]:
        """
        Calculate Simple Moving Averages.
        
        Args:
            prices: Series of closing prices
            windows: List of periods for moving averages
            
        Returns:
            Dictionary of moving averages
        """
        mas = {}
        for window in windows:
            sma = SMAIndicator(close=prices, window=window)
            mas[f'SMA_{window}'] = sma.sma_indicator()
        return mas
    
    def identify_candlestick_pattern(self, open_price: float, high: float, 
                                     low: float, close: float) -> str:
        """
        Identify candlestick patterns.
        
        Args:
            open_price: Opening price
            high: High price
            low: Low price
            close: Closing price
            
        Returns:
            Pattern name
        """
        body = abs(close - open_price)
        range_val = high - low
        
        if range_val == 0:
            return 'DOJI'
        
        body_ratio = body / range_val
        upper_shadow = high - max(open_price, close)
        lower_shadow = min(open_price, close) - low
        
        # Doji pattern
        if body_ratio < 0.1:
            return 'DOJI'
        
        # Hammer pattern (bullish)
        if (close > open_price and 
            lower_shadow > 2 * body and 
            upper_shadow < body * 0.3):
            return 'HAMMER'
        
        # Shooting Star pattern (bearish)
        if (close < open_price and 
            upper_shadow > 2 * body and 
            lower_shadow < body * 0.3):
            return 'SHOOTING_STAR'
        
        # Bullish Engulfing
        if close > open_price and body_ratio > 0.6:
            return 'BULLISH_CANDLE'
        
        # Bearish Engulfing
        if close < open_price and body_ratio > 0.6:
            return 'BEARISH_CANDLE'
        
        return 'NEUTRAL'
    
    def detect_support_resistance(self, prices: pd.Series, 
                                  window: int = 20) -> Dict[str, List[float]]:
        """
        Detect support and resistance levels.
        
        Args:
            prices: Series of prices
            window: Window for local extrema detection
            
        Returns:
            Dictionary with support and resistance levels
        """
        support_levels = []
        resistance_levels = []
        
        for i in range(window, len(prices) - window):
            # Check for local minimum (support)
            if prices.iloc[i] == min(prices.iloc[i-window:i+window]):
                support_levels.append(prices.iloc[i])
            
            # Check for local maximum (resistance)
            if prices.iloc[i] == max(prices.iloc[i-window:i+window]):
                resistance_levels.append(prices.iloc[i])
        
        return {
            'support': sorted(set(support_levels)),
            'resistance': sorted(set(resistance_levels), reverse=True)
        }
    
    def analyze_trend(self, prices: pd.Series, window: int = 20) -> str:
        """
        Analyze price trend.
        
        Args:
            prices: Series of closing prices
            window: Window for trend analysis
            
        Returns:
            Trend direction (UPTREND, DOWNTREND, SIDEWAYS)
        """
        if len(prices) < window:
            return 'INSUFFICIENT_DATA'
        
        recent_prices = prices.tail(window)
        
        # Calculate linear regression slope
        x = np.arange(len(recent_prices))
        y = recent_prices.values
        
        # Simple linear regression
        slope = np.polyfit(x, y, 1)[0]
        
        # Normalize slope by average price
        avg_price = y.mean()
        normalized_slope = slope / avg_price if avg_price != 0 else 0
        
        if normalized_slope > 0.01:
            return 'UPTREND'
        elif normalized_slope < -0.01:
            return 'DOWNTREND'
        else:
            return 'SIDEWAYS'
    
    def analyze_volume(self, volume: pd.Series, window: int = 20) -> Dict[str, any]:
        """
        Analyze volume patterns.
        
        Args:
            volume: Series of volume data
            window: Window for volume analysis
            
        Returns:
            Volume analysis results
        """
        if len(volume) < window:
            return {'status': 'INSUFFICIENT_DATA'}
        
        recent_volume = volume.tail(window)
        avg_volume = recent_volume.mean()
        current_volume = volume.iloc[-1]
        
        volume_ratio = current_volume / avg_volume if avg_volume != 0 else 1
        
        if volume_ratio > 1.5:
            status = 'HIGH_VOLUME'
        elif volume_ratio < 0.5:
            status = 'LOW_VOLUME'
        else:
            status = 'NORMAL_VOLUME'
        
        return {
            'status': status,
            'current_volume': current_volume,
            'average_volume': avg_volume,
            'volume_ratio': round(volume_ratio, 2)
        }
    
    def comprehensive_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Perform comprehensive chart analysis.
        
        Args:
            df: DataFrame with columns: open, high, low, close, volume
            
        Returns:
            Complete analysis results
        """
        if len(df) < 50:
            return {'error': 'Insufficient data for analysis (minimum 50 data points)'}
        
        results = {}
        
        # Technical Indicators
        results['rsi'] = {
            'current': round(self.calculate_rsi(df['close']).iloc[-1], 2),
            'status': self._interpret_rsi(self.calculate_rsi(df['close']).iloc[-1])
        }
        
        # MACD
        macd_data = self.calculate_macd(df['close'])
        results['macd'] = {
            'macd_line': round(macd_data['macd_line'].iloc[-1], 2),
            'signal_line': round(macd_data['signal_line'].iloc[-1], 2),
            'histogram': round(macd_data['histogram'].iloc[-1], 2),
            'status': 'BULLISH' if macd_data['histogram'].iloc[-1] > 0 else 'BEARISH'
        }
        
        # Moving Averages
        mas = self.calculate_moving_averages(df['close'])
        current_price = df['close'].iloc[-1]
        results['moving_averages'] = {
            'SMA_20': round(mas['SMA_20'].iloc[-1], 2),
            'SMA_50': round(mas['SMA_50'].iloc[-1], 2),
            'SMA_200': round(mas['SMA_200'].iloc[-1], 2),
            'price_vs_sma20': 'ABOVE' if current_price > mas['SMA_20'].iloc[-1] else 'BELOW'
        }
        
        # Candlestick Pattern
        last_candle = df.iloc[-1]
        results['candlestick_pattern'] = self.identify_candlestick_pattern(
            last_candle['open'], last_candle['high'], 
            last_candle['low'], last_candle['close']
        )
        
        # Trend Analysis
        results['trend'] = self.analyze_trend(df['close'])
        
        # Volume Analysis
        results['volume'] = self.analyze_volume(df['volume'])
        
        # Support and Resistance
        sr_levels = self.detect_support_resistance(df['close'])
        results['support_resistance'] = {
            'nearest_support': sr_levels['support'][-1] if sr_levels['support'] else None,
            'nearest_resistance': sr_levels['resistance'][0] if sr_levels['resistance'] else None
        }
        
        # Overall Signal
        results['overall_signal'] = self._generate_signal(results)
        
        return results
    
    def _interpret_rsi(self, rsi_value: float) -> str:
        """Interpret RSI value."""
        if rsi_value > 70:
            return 'OVERBOUGHT'
        elif rsi_value < 30:
            return 'OVERSOLD'
        else:
            return 'NEUTRAL'
    
    def _generate_signal(self, analysis: Dict) -> str:
        """Generate overall trading signal from analysis."""
        bullish_signals = 0
        bearish_signals = 0
        
        # RSI
        if analysis['rsi']['status'] == 'OVERSOLD':
            bullish_signals += 1
        elif analysis['rsi']['status'] == 'OVERBOUGHT':
            bearish_signals += 1
        
        # MACD
        if analysis['macd']['status'] == 'BULLISH':
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        # Trend
        if analysis['trend'] == 'UPTREND':
            bullish_signals += 1
        elif analysis['trend'] == 'DOWNTREND':
            bearish_signals += 1
        
        # Moving Average
        if analysis['moving_averages']['price_vs_sma20'] == 'ABOVE':
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        if bullish_signals > bearish_signals:
            return 'BUY'
        elif bearish_signals > bullish_signals:
            return 'SELL'
        else:
            return 'HOLD'


if __name__ == "__main__":
    # Example usage with sample data
    analyzer = ChartAnalyzer()
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # Generate synthetic price data
    close_prices = 100 + np.cumsum(np.random.randn(100) * 2)
    open_prices = close_prices + np.random.randn(100) * 0.5
    high_prices = np.maximum(open_prices, close_prices) + np.abs(np.random.randn(100) * 1)
    low_prices = np.minimum(open_prices, close_prices) - np.abs(np.random.randn(100) * 1)
    volume = np.random.randint(1000000, 5000000, 100)
    
    df = pd.DataFrame({
        'date': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    })
    
    # Perform analysis
    analysis = analyzer.comprehensive_analysis(df)
    
    print("Chart Analysis Results:")
    print("=" * 60)
    print(f"\nRSI: {analysis['rsi']['current']} ({analysis['rsi']['status']})")
    print(f"\nMACD Status: {analysis['macd']['status']}")
    print(f"MACD Line: {analysis['macd']['macd_line']}")
    print(f"Signal Line: {analysis['macd']['signal_line']}")
    print(f"\nTrend: {analysis['trend']}")
    print(f"\nCandlestick Pattern: {analysis['candlestick_pattern']}")
    print(f"\nVolume Status: {analysis['volume']['status']}")
    print(f"\nOverall Signal: {analysis['overall_signal']}")
