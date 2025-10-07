"""
Market AI - Main Application
Integrates sentiment analysis, chart reading, and historical learning.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sentiment_analyzer import SentimentAnalyzer
from chart_analyzer import ChartAnalyzer
from historical_learner import HistoricalLearner


class MarketAI:
    """Main Market AI system integrating all analysis modules."""
    
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.chart_analyzer = ChartAnalyzer()
        self.historical_learner = HistoricalLearner()
    
    def analyze_market(self, 
                       price_data: pd.DataFrame,
                       sentiment_texts: Optional[List[str]] = None) -> Dict:
        """
        Perform comprehensive market analysis.
        
        Args:
            price_data: DataFrame with OHLCV data
            sentiment_texts: Optional list of texts for sentiment analysis
            
        Returns:
            Complete market analysis results
        """
        results = {
            'chart_analysis': None,
            'sentiment_analysis': None,
            'historical_analysis': None,
            'ml_prediction': None,
            'overall_recommendation': None
        }
        
        # Chart Analysis
        if len(price_data) >= 50:
            results['chart_analysis'] = self.chart_analyzer.comprehensive_analysis(price_data)
        
        # Sentiment Analysis
        if sentiment_texts and len(sentiment_texts) > 0:
            results['sentiment_analysis'] = self.sentiment_analyzer.aggregate_sentiment(sentiment_texts)
        
        # Historical Analysis
        if len(price_data) >= 50:
            results['historical_analysis'] = self.historical_learner.analyze_historical_patterns(price_data)
        
        # Generate overall recommendation
        results['overall_recommendation'] = self._generate_recommendation(results)
        
        return results
    
    def train_and_predict(self, historical_data: pd.DataFrame, 
                         recent_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Train on historical data and make predictions.
        
        Args:
            historical_data: Historical OHLCV data for training
            recent_data: Recent data for prediction (uses historical_data if not provided)
            
        Returns:
            Training metrics and predictions
        """
        # Train model
        training_metrics = self.historical_learner.train(historical_data)
        
        # Make prediction
        prediction_data = recent_data if recent_data is not None else historical_data
        prediction = self.historical_learner.predict(prediction_data)
        
        return {
            'training_metrics': training_metrics,
            'prediction': prediction
        }
    
    def get_market_summary(self, 
                          price_data: pd.DataFrame,
                          sentiment_texts: Optional[List[str]] = None) -> str:
        """
        Generate a comprehensive market summary report.
        
        Args:
            price_data: DataFrame with OHLCV data
            sentiment_texts: Optional list of texts for sentiment analysis
            
        Returns:
            Formatted text summary
        """
        analysis = self.analyze_market(price_data, sentiment_texts)
        
        summary = "=" * 70 + "\n"
        summary += "MARKET AI - COMPREHENSIVE ANALYSIS REPORT\n"
        summary += "=" * 70 + "\n\n"
        
        # Chart Analysis Section
        if analysis['chart_analysis']:
            chart = analysis['chart_analysis']
            summary += "TECHNICAL ANALYSIS\n"
            summary += "-" * 70 + "\n"
            summary += f"Current Trend: {chart.get('trend', 'N/A')}\n"
            summary += f"Overall Signal: {chart.get('overall_signal', 'N/A')}\n"
            summary += f"RSI: {chart.get('rsi', {}).get('current', 'N/A')} ({chart.get('rsi', {}).get('status', 'N/A')})\n"
            summary += f"MACD: {chart.get('macd', {}).get('status', 'N/A')}\n"
            summary += f"Candlestick Pattern: {chart.get('candlestick_pattern', 'N/A')}\n"
            summary += f"Volume: {chart.get('volume', {}).get('status', 'N/A')}\n"
            summary += "\n"
        
        # Sentiment Analysis Section
        if analysis['sentiment_analysis']:
            sentiment = analysis['sentiment_analysis']
            summary += "SENTIMENT ANALYSIS\n"
            summary += "-" * 70 + "\n"
            summary += f"Overall Sentiment: {sentiment.get('overall_sentiment', 'N/A')}\n"
            summary += f"Average Score: {sentiment.get('average_score', 'N/A')}\n"
            summary += f"Bullish Ratio: {sentiment.get('bullish_ratio', 0) * 100:.1f}%\n"
            summary += f"Bearish Ratio: {sentiment.get('bearish_ratio', 0) * 100:.1f}%\n"
            summary += f"Neutral Ratio: {sentiment.get('neutral_ratio', 0) * 100:.1f}%\n"
            summary += f"Total Texts Analyzed: {sentiment.get('total_texts', 0)}\n"
            summary += "\n"
        
        # Historical Analysis Section
        if analysis['historical_analysis']:
            hist = analysis['historical_analysis']
            summary += "HISTORICAL PERFORMANCE\n"
            summary += "-" * 70 + "\n"
            summary += f"Total Return: {hist.get('total_return_pct', 'N/A')}%\n"
            summary += f"Win Rate: {hist.get('win_rate', 0) * 100:.1f}%\n"
            summary += f"Sharpe Ratio: {hist.get('sharpe_ratio', 'N/A')}\n"
            summary += f"Max Drawdown: {hist.get('max_drawdown', 'N/A')}%\n"
            summary += f"Risk-Reward Ratio: {hist.get('risk_reward_ratio', 'N/A')}\n"
            summary += "\n"
        
        # Overall Recommendation
        summary += "OVERALL RECOMMENDATION\n"
        summary += "-" * 70 + "\n"
        summary += f"{analysis['overall_recommendation']}\n"
        summary += "\n" + "=" * 70 + "\n"
        
        return summary
    
    def _generate_recommendation(self, analysis: Dict) -> str:
        """Generate overall trading recommendation."""
        signals = []
        
        # Technical signal
        if analysis['chart_analysis']:
            tech_signal = analysis['chart_analysis'].get('overall_signal', 'HOLD')
            signals.append(('Technical', tech_signal))
        
        # Sentiment signal
        if analysis['sentiment_analysis']:
            sentiment = analysis['sentiment_analysis'].get('overall_sentiment', 'NEUTRAL')
            if sentiment == 'BULLISH':
                signals.append(('Sentiment', 'BUY'))
            elif sentiment == 'BEARISH':
                signals.append(('Sentiment', 'SELL'))
            else:
                signals.append(('Sentiment', 'HOLD'))
        
        # Count signals
        buy_count = sum(1 for _, signal in signals if signal == 'BUY')
        sell_count = sum(1 for _, signal in signals if signal == 'SELL')
        
        # Generate recommendation
        if buy_count > sell_count:
            recommendation = "BULLISH - Consider buying or holding long positions"
        elif sell_count > buy_count:
            recommendation = "BEARISH - Consider selling or holding short positions"
        else:
            recommendation = "NEUTRAL - Wait for clearer signals before taking action"
        
        # Add details
        details = "\nSignal Breakdown:\n"
        for source, signal in signals:
            details += f"  - {source}: {signal}\n"
        
        return recommendation + details


def main():
    """Example usage of the Market AI system."""
    
    # Initialize Market AI
    market_ai = MarketAI()
    
    # Create sample price data
    print("Generating sample market data...\n")
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # Generate synthetic price data
    close_prices = 100 + np.cumsum(np.random.randn(100) * 2)
    open_prices = close_prices + np.random.randn(100) * 0.5
    high_prices = np.maximum(open_prices, close_prices) + np.abs(np.random.randn(100) * 1)
    low_prices = np.minimum(open_prices, close_prices) - np.abs(np.random.randn(100) * 1)
    volume = np.random.randint(1000000, 5000000, 100)
    
    price_data = pd.DataFrame({
        'date': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    })
    
    # Sample sentiment texts
    sentiment_texts = [
        "Stock is breaking out! Very bullish signal here.",
        "Major resistance ahead, expecting pullback.",
        "Strong buying pressure, momentum is building.",
        "Bears are taking control, sell signal confirmed.",
        "Market looks neutral, waiting for direction.",
        "This is going to the moon! 🚀",
        "Correction incoming, time to take profits."
    ]
    
    # Perform comprehensive analysis
    print("Analyzing market data...\n")
    summary = market_ai.get_market_summary(price_data, sentiment_texts)
    print(summary)
    
    # Train and predict
    print("\nTraining ML models and making predictions...\n")
    
    # Generate more data for training
    train_dates = pd.date_range(start='2022-01-01', periods=300, freq='D')
    train_close = 80 + np.cumsum(np.random.randn(300) * 2)
    train_open = train_close + np.random.randn(300) * 0.5
    train_high = np.maximum(train_open, train_close) + np.abs(np.random.randn(300) * 1)
    train_low = np.minimum(train_open, train_close) - np.abs(np.random.randn(300) * 1)
    train_volume = np.random.randint(1000000, 5000000, 300)
    
    training_data = pd.DataFrame({
        'date': train_dates,
        'open': train_open,
        'high': train_high,
        'low': train_low,
        'close': train_close,
        'volume': train_volume
    })
    
    ml_results = market_ai.train_and_predict(training_data, price_data)
    
    print("ML Training Results:")
    print("-" * 70)
    print(f"Direction Accuracy: {ml_results['training_metrics'].get('direction_accuracy', 'N/A')}")
    print(f"Price Prediction R²: {ml_results['training_metrics'].get('price_prediction_r2', 'N/A')}")
    
    print("\nML Predictions:")
    print("-" * 70)
    pred = ml_results['prediction']
    print(f"Predicted Direction: {pred.get('predicted_direction', 'N/A')}")
    print(f"Confidence: {pred.get('confidence', 'N/A')}")
    print(f"Predicted Return: {pred.get('predicted_return', 'N/A')}%")
    print(f"Current Price: ${pred.get('current_price', 'N/A')}")
    print(f"Predicted Price: ${pred.get('predicted_price', 'N/A')}")
    
    print("\n" + "=" * 70)
    print("Market AI Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
