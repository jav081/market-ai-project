#!/usr/bin/env python3
"""
Quick Demo Script - Market AI System
Run this to see a quick demonstration of all features
"""

import pandas as pd
import numpy as np
from main import MarketAI

def demo():
    print("\n" + "=" * 80)
    print(" " * 25 + "MARKET AI - QUICK DEMO")
    print("=" * 80 + "\n")
    
    # Initialize the AI
    print("🤖 Initializing Market AI System...")
    market_ai = MarketAI()
    print("✓ System ready!\n")
    
    # Sample price data
    print("📊 Generating sample market data...")
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
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
    print(f"✓ Generated {len(price_data)} days of market data\n")
    
    # Sample sentiment data
    print("💭 Collecting market sentiment from multiple sources...")
    sentiment_texts = [
        "📰 NEWS: Company reports strong earnings, beating analyst expectations",
        "🐦 TWITTER: This stock is going to the moon! 🚀 Major breakout pattern!",
        "📱 REDDIT: Technical analysis shows clear bullish divergence",
        "💬 COMMENT: I'm very optimistic about this company's future prospects",
        "📰 NEWS: Market analysts upgrade rating to strong buy",
        "🐦 TWITTER: Bears are getting destroyed, this rally is unstoppable",
        "💬 COMMENT: Some resistance expected at current levels"
    ]
    print(f"✓ Collected {len(sentiment_texts)} sentiment data points\n")
    
    # Run comprehensive analysis
    print("🔍 Running comprehensive market analysis...")
    print("-" * 80)
    summary = market_ai.get_market_summary(price_data, sentiment_texts)
    print(summary)
    
    # Train ML model and predict
    print("\n🧠 Training machine learning models...")
    print("-" * 80)
    
    # Generate training data
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
    
    print(f"✓ Model trained with {ml_results['training_metrics']['training_samples']} samples")
    print(f"✓ Direction Accuracy: {ml_results['training_metrics']['direction_accuracy']}")
    print(f"✓ Price Prediction R²: {ml_results['training_metrics']['price_prediction_r2']}\n")
    
    print("📈 ML Predictions:")
    pred = ml_results['prediction']
    print(f"  → Direction: {pred['predicted_direction']} (Confidence: {pred['confidence']})")
    print(f"  → Expected Return: {pred['predicted_return']}%")
    print(f"  → Current Price: ${pred['current_price']:.2f}")
    print(f"  → Target Price: ${pred['predicted_price']:.2f}")
    
    print("\n" + "=" * 80)
    print(" " * 25 + "Demo completed successfully! ✨")
    print("=" * 80 + "\n")
    
    print("📚 Next Steps:")
    print("  • Read the README.md for detailed documentation")
    print("  • Run 'python examples.py' for more examples")
    print("  • Import modules in your own scripts")
    print("  • Use with real market data via yfinance or other data sources\n")

if __name__ == "__main__":
    demo()
