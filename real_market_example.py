"""
Real Market Data Integration Example
Demonstrates how to use Market AI with real stock data from Yahoo Finance
"""

import yfinance as yf
import pandas as pd
from main import MarketAI
from datetime import datetime, timedelta


def get_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    Fetch real stock data from Yahoo Finance.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'TSLA', 'MSFT')
        period: Time period ('1mo', '3mo', '6mo', '1y', '2y', '5y')
    
    Returns:
        DataFrame with OHLCV data
    """
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    
    # Rename columns to match our format
    data = data.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })
    
    # Reset index to have date as column
    data = data.reset_index()
    data = data.rename(columns={'Date': 'date'})
    
    return data[['date', 'open', 'high', 'low', 'close', 'volume']]


def analyze_stock(ticker: str, sentiment_texts: list = None):
    """
    Perform comprehensive analysis on a real stock.
    
    Args:
        ticker: Stock ticker symbol
        sentiment_texts: Optional list of sentiment texts about the stock
    """
    print(f"\n{'=' * 80}")
    print(f" Analysis for {ticker}")
    print(f"{'=' * 80}\n")
    
    # Fetch real market data
    print(f"📡 Fetching market data for {ticker}...")
    try:
        stock_data = get_stock_data(ticker, period="1y")
        print(f"✓ Retrieved {len(stock_data)} days of historical data")
        print(f"   Date Range: {stock_data['date'].min().strftime('%Y-%m-%d')} to {stock_data['date'].max().strftime('%Y-%m-%d')}")
        print(f"   Current Price: ${stock_data['close'].iloc[-1]:.2f}")
        print(f"   52-Week High: ${stock_data['high'].max():.2f}")
        print(f"   52-Week Low: ${stock_data['low'].min():.2f}\n")
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return
    
    # Initialize Market AI
    market_ai = MarketAI()
    
    # Get comprehensive analysis
    print(f"🔍 Running AI analysis...")
    summary = market_ai.get_market_summary(stock_data, sentiment_texts)
    print(summary)
    
    # Train ML model on historical data and make prediction
    print("\n🧠 Training ML model on historical data...")
    
    # Use first 80% for training, last 20% for prediction
    split_idx = int(len(stock_data) * 0.8)
    train_data = stock_data[:split_idx]
    recent_data = stock_data[split_idx:]
    
    try:
        ml_results = market_ai.train_and_predict(train_data, stock_data)
        
        print(f"✓ Training completed")
        print(f"   Direction Accuracy: {ml_results['training_metrics']['direction_accuracy']}")
        print(f"   Price R² Score: {ml_results['training_metrics']['price_prediction_r2']}\n")
        
        pred = ml_results['prediction']
        print(f"📊 Price Prediction:")
        print(f"   Current Price: ${pred['current_price']:.2f}")
        print(f"   Predicted Price: ${pred['predicted_price']:.2f}")
        print(f"   Expected Return: {pred['predicted_return']:.2f}%")
        print(f"   Direction: {pred['predicted_direction']} (Confidence: {pred['confidence']:.1%})")
        
    except Exception as e:
        print(f"⚠️  ML prediction unavailable: {e}")
    
    print(f"\n{'=' * 80}\n")


def main():
    """Run analysis on multiple stocks."""
    
    print("\n" + "=" * 80)
    print(" " * 20 + "REAL MARKET DATA ANALYSIS")
    print("=" * 80)
    
    # Example 1: Tech stock with sentiment
    print("\n📱 EXAMPLE 1: Apple Inc. (AAPL)")
    print("-" * 80)
    
    aapl_sentiment = [
        "Apple's new product lineup exceeds expectations, strong demand",
        "iPhone sales continue to show robust growth in key markets",
        "Services revenue reaches all-time high, analysts bullish",
        "Some concerns about supply chain, but overall outlook positive"
    ]
    
    analyze_stock('AAPL', aapl_sentiment)
    
    # Example 2: EV stock
    print("\n⚡ EXAMPLE 2: Tesla Inc. (TSLA)")
    print("-" * 80)
    
    tsla_sentiment = [
        "Tesla deliveries beat estimates, production ramping up",
        "Energy storage business showing massive growth potential",
        "Competition intensifying in EV market",
        "Autonomous driving progress impressive, regulatory approval pending"
    ]
    
    analyze_stock('TSLA', tsla_sentiment)
    
    # Example 3: Index ETF (no sentiment)
    print("\n📈 EXAMPLE 3: S&P 500 ETF (SPY)")
    print("-" * 80)
    
    analyze_stock('SPY')
    
    print("\n" + "=" * 80)
    print("✨ Analysis complete!")
    print("=" * 80 + "\n")
    
    print("💡 Tips:")
    print("  • Use this script as a template for your own analysis")
    print("  • Add real-time sentiment from Twitter API, Reddit API, or news APIs")
    print("  • Integrate with trading platforms for automated trading")
    print("  • Backtest strategies using historical predictions")
    print("  • Combine multiple timeframes for better accuracy\n")


if __name__ == "__main__":
    # Note: This script requires internet connection to fetch data from Yahoo Finance
    import sys
    
    # Check if running with specific ticker
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
        print(f"\nAnalyzing {ticker}...")
        analyze_stock(ticker)
    else:
        # Run full demo
        main()
