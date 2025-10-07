"""
Example Usage Scripts for Market AI
Demonstrates how to use each module independently and together.
"""

import pandas as pd
import numpy as np
from sentiment_analyzer import SentimentAnalyzer
from chart_analyzer import ChartAnalyzer
from historical_learner import HistoricalLearner
from main import MarketAI


def example_sentiment_analysis():
    """Example: Sentiment Analysis"""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: MARKET SENTIMENT ANALYSIS")
    print("=" * 70 + "\n")
    
    analyzer = SentimentAnalyzer()
    
    # Example texts from different sources
    news_articles = [
        "Stock market rallies on positive economic data, analysts expect continued growth",
        "Tech sector faces headwinds as interest rates rise, correction likely"
    ]
    
    social_posts = [
        "This stock is going to moon! 🚀 Bullish breakout confirmed!",
        "Major dump incoming, bears taking control",
        "Strong buy signal on the daily chart"
    ]
    
    comments = [
        "I'm very optimistic about this company's future",
        "Sell everything, market crash is coming",
        "Holding my position, waiting for more clarity"
    ]
    
    # Analyze each source
    print("News Sentiment:")
    news_sentiment = analyzer.aggregate_sentiment(news_articles)
    print(f"  Overall: {news_sentiment['overall_sentiment']}")
    print(f"  Score: {news_sentiment['average_score']}\n")
    
    print("Social Media Sentiment:")
    social_sentiment = analyzer.aggregate_sentiment(social_posts)
    print(f"  Overall: {social_sentiment['overall_sentiment']}")
    print(f"  Bullish Ratio: {social_sentiment['bullish_ratio'] * 100:.1f}%\n")
    
    print("Comments Sentiment:")
    comment_sentiment = analyzer.aggregate_sentiment(comments)
    print(f"  Overall: {comment_sentiment['overall_sentiment']}")
    print(f"  Bearish Ratio: {comment_sentiment['bearish_ratio'] * 100:.1f}%\n")


def example_chart_analysis():
    """Example: Chart Analysis"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: CHART READING AI")
    print("=" * 70 + "\n")
    
    analyzer = ChartAnalyzer()
    
    # Create sample chart data
    dates = pd.date_range(start='2023-06-01', periods=60, freq='D')
    np.random.seed(123)
    
    close_prices = 150 + np.cumsum(np.random.randn(60) * 3)
    open_prices = close_prices + np.random.randn(60) * 1
    high_prices = np.maximum(open_prices, close_prices) + np.abs(np.random.randn(60) * 2)
    low_prices = np.minimum(open_prices, close_prices) - np.abs(np.random.randn(60) * 2)
    volume = np.random.randint(500000, 3000000, 60)
    
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
    
    print("Technical Indicators:")
    print(f"  RSI: {analysis['rsi']['current']} ({analysis['rsi']['status']})")
    print(f"  MACD Status: {analysis['macd']['status']}")
    print(f"  Trend: {analysis['trend']}")
    print(f"  Volume: {analysis['volume']['status']}\n")
    
    print("Chart Patterns:")
    print(f"  Candlestick Pattern: {analysis['candlestick_pattern']}")
    print(f"  Support Level: ${analysis['support_resistance']['nearest_support']}")
    print(f"  Resistance Level: ${analysis['support_resistance']['nearest_resistance']}\n")
    
    print(f"Trading Signal: {analysis['overall_signal']}")


def example_historical_learning():
    """Example: Historical Learning"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: HISTORICAL DATA LEARNING")
    print("=" * 70 + "\n")
    
    learner = HistoricalLearner()
    
    # Create historical data
    dates = pd.date_range(start='2021-01-01', periods=500, freq='D')
    np.random.seed(456)
    
    trend = np.linspace(100, 180, 500)
    noise = np.random.randn(500) * 8
    close_prices = trend + noise
    
    open_prices = close_prices + np.random.randn(500) * 2
    high_prices = np.maximum(open_prices, close_prices) + np.abs(np.random.randn(500) * 3)
    low_prices = np.minimum(open_prices, close_prices) - np.abs(np.random.randn(500) * 3)
    volume = np.random.randint(1000000, 8000000, 500)
    
    df = pd.DataFrame({
        'date': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    })
    
    # Train model
    print("Training ML Models...")
    metrics = learner.train(df)
    print(f"  Direction Accuracy: {metrics['direction_accuracy']}")
    print(f"  Price Prediction R²: {metrics['price_prediction_r2']}")
    print(f"  Features Used: {metrics['features_used']}\n")
    
    # Make predictions
    print("Making Predictions...")
    prediction = learner.predict(df)
    print(f"  Predicted Direction: {prediction['predicted_direction']}")
    print(f"  Confidence: {prediction['confidence']}")
    print(f"  Expected Return: {prediction['predicted_return']}%")
    print(f"  Target Price: ${prediction['predicted_price']}\n")
    
    # Analyze patterns
    print("Historical Pattern Analysis:")
    patterns = learner.analyze_historical_patterns(df)
    print(f"  Total Return: {patterns['total_return_pct']}%")
    print(f"  Win Rate: {patterns['win_rate'] * 100:.1f}%")
    print(f"  Sharpe Ratio: {patterns['sharpe_ratio']}")
    print(f"  Max Drawdown: {patterns['max_drawdown']}%")


def example_integrated_analysis():
    """Example: Integrated Market AI"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: INTEGRATED MARKET AI ANALYSIS")
    print("=" * 70 + "\n")
    
    market_ai = MarketAI()
    
    # Prepare data
    dates = pd.date_range(start='2023-03-01', periods=100, freq='D')
    np.random.seed(789)
    
    close_prices = 200 + np.cumsum(np.random.randn(100) * 4)
    open_prices = close_prices + np.random.randn(100) * 1.5
    high_prices = np.maximum(open_prices, close_prices) + np.abs(np.random.randn(100) * 2)
    low_prices = np.minimum(open_prices, close_prices) - np.abs(np.random.randn(100) * 2)
    volume = np.random.randint(2000000, 6000000, 100)
    
    price_data = pd.DataFrame({
        'date': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    })
    
    # Sentiment data
    sentiment_texts = [
        "Market showing strong bullish momentum",
        "Expecting resistance at current levels",
        "Great buying opportunity at this price",
        "Technical indicators suggest uptrend continuation",
        "Some profit-taking expected soon"
    ]
    
    # Get comprehensive analysis
    summary = market_ai.get_market_summary(price_data, sentiment_texts)
    print(summary)


def example_real_world_workflow():
    """Example: Real-world workflow"""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: REAL-WORLD TRADING WORKFLOW")
    print("=" * 70 + "\n")
    
    print("Step 1: Collect market sentiment from multiple sources")
    print("-" * 70)
    
    sentiment_analyzer = SentimentAnalyzer()
    
    all_texts = [
        # News
        "Company reports strong earnings, beating expectations",
        # Twitter/X
        "This is the best setup I've seen all year! Going long 🚀",
        # Reddit
        "Technical analysis shows clear breakout pattern",
        # Comments
        "I'm bullish on this stock for the long term"
    ]
    
    sentiment = sentiment_analyzer.aggregate_sentiment(all_texts)
    print(f"Market Sentiment: {sentiment['overall_sentiment']}")
    print(f"Confidence: {sentiment['average_score']}\n")
    
    print("Step 2: Analyze price charts and technical indicators")
    print("-" * 70)
    
    # Sample price data
    dates = pd.date_range(start='2023-01-01', periods=90, freq='D')
    close_prices = 100 + np.cumsum(np.random.randn(90) * 2)
    
    df = pd.DataFrame({
        'date': dates,
        'open': close_prices + np.random.randn(90) * 0.5,
        'high': close_prices + np.abs(np.random.randn(90) * 1),
        'low': close_prices - np.abs(np.random.randn(90) * 1),
        'close': close_prices,
        'volume': np.random.randint(1000000, 5000000, 90)
    })
    
    chart_analyzer = ChartAnalyzer()
    chart_analysis = chart_analyzer.comprehensive_analysis(df)
    print(f"Technical Signal: {chart_analysis['overall_signal']}")
    print(f"Trend: {chart_analysis['trend']}")
    print(f"RSI: {chart_analysis['rsi']['current']} ({chart_analysis['rsi']['status']})\n")
    
    print("Step 3: Learn from historical data and predict")
    print("-" * 70)
    
    learner = HistoricalLearner()
    
    # More historical data for training
    hist_dates = pd.date_range(start='2022-01-01', periods=365, freq='D')
    hist_close = 80 + np.cumsum(np.random.randn(365) * 2)
    
    hist_df = pd.DataFrame({
        'date': hist_dates,
        'open': hist_close + np.random.randn(365) * 0.5,
        'high': hist_close + np.abs(np.random.randn(365) * 1),
        'low': hist_close - np.abs(np.random.randn(365) * 1),
        'close': hist_close,
        'volume': np.random.randint(1000000, 5000000, 365)
    })
    
    learner.train(hist_df)
    prediction = learner.predict(df)
    print(f"ML Prediction: {prediction['predicted_direction']}")
    print(f"Confidence: {prediction['confidence']}")
    print(f"Expected Move: {prediction['predicted_return']}%\n")
    
    print("Step 4: Make trading decision")
    print("-" * 70)
    
    # Combine all signals
    signals = {
        'sentiment': sentiment['overall_sentiment'],
        'technical': chart_analysis['overall_signal'],
        'ml_prediction': prediction['predicted_direction']
    }
    
    print("Signal Summary:")
    for source, signal in signals.items():
        print(f"  {source.capitalize()}: {signal}")
    
    # Final decision
    bullish_count = sum(1 for s in signals.values() if s in ['BULLISH', 'BUY', 'UP'])
    bearish_count = sum(1 for s in signals.values() if s in ['BEARISH', 'SELL', 'DOWN'])
    
    if bullish_count > bearish_count:
        print("\n✓ TRADING DECISION: BUY/LONG")
    elif bearish_count > bullish_count:
        print("\n✓ TRADING DECISION: SELL/SHORT")
    else:
        print("\n✓ TRADING DECISION: HOLD/WAIT")


def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("MARKET AI - EXAMPLE USAGE DEMONSTRATIONS")
    print("=" * 70)
    
    example_sentiment_analysis()
    example_chart_analysis()
    example_historical_learning()
    example_integrated_analysis()
    example_real_world_workflow()
    
    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
