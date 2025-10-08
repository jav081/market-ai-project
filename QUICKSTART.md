# Quick Reference Guide

## Installation
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

## Quick Commands

### Run Demos
```bash
python demo.py              # Quick demo with sample data
python main.py              # Main application demo
python examples.py          # All feature examples
```

### Analyze Real Stocks
```bash
python real_market_example.py AAPL    # Apple
python real_market_example.py TSLA    # Tesla
python real_market_example.py SPY     # S&P 500 ETF
python real_market_example.py [TICKER] # Any ticker
```

## Code Snippets

### 1. Sentiment Analysis Only
```python
from sentiment_analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer()

# Single text
result = analyzer.analyze_sentiment("Stock going to moon! 🚀")
print(result['classification'])  # BULLISH

# Multiple texts
texts = ["Great earnings!", "Market crash coming", "Neutral day"]
summary = analyzer.aggregate_sentiment(texts)
print(summary['overall_sentiment'])
print(summary['bullish_ratio'])
```

### 2. Chart Analysis Only
```python
from chart_analyzer import ChartAnalyzer
import pandas as pd

analyzer = ChartAnalyzer()

# Your OHLCV data
df = pd.DataFrame({
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...]
})

analysis = analyzer.comprehensive_analysis(df)
print(f"Signal: {analysis['overall_signal']}")
print(f"RSI: {analysis['rsi']['current']}")
print(f"Trend: {analysis['trend']}")
```

### 3. ML Predictions Only
```python
from historical_learner import HistoricalLearner
import pandas as pd

learner = HistoricalLearner()

# Historical data (min 100 rows)
historical_df = pd.DataFrame({...})

# Train
metrics = learner.train(historical_df)
print(f"Accuracy: {metrics['direction_accuracy']}")

# Predict
prediction = learner.predict(historical_df)
print(f"Direction: {prediction['predicted_direction']}")
print(f"Target: ${prediction['predicted_price']}")

# Analyze patterns
summary = learner.generate_summary(historical_df)
print(summary)
```

### 4. Complete Analysis
```python
from main import MarketAI
import pandas as pd

market_ai = MarketAI()

# Price data
price_data = pd.DataFrame({...})

# Sentiment data (optional)
sentiment_texts = [
    "Bullish news about the company",
    "Strong technical setup",
    # ...
]

# Get full analysis
summary = market_ai.get_market_summary(price_data, sentiment_texts)
print(summary)

# Or get detailed results
analysis = market_ai.analyze_market(price_data, sentiment_texts)
print(analysis['chart_analysis'])
print(analysis['sentiment_analysis'])
print(analysis['overall_recommendation'])
```

### 5. Train and Predict
```python
from main import MarketAI

market_ai = MarketAI()

# Train on historical data
results = market_ai.train_and_predict(
    historical_data=historical_df,
    recent_data=recent_df  # optional
)

print(results['training_metrics'])
print(results['prediction'])
```

### 6. Real Market Data
```python
from real_market_example import get_stock_data, analyze_stock

# Get data
data = get_stock_data('AAPL', period='1y')
print(data.head())

# Full analysis
sentiment = ["Strong buy signal", "Bullish momentum"]
analyze_stock('AAPL', sentiment)
```

## Data Format

### Required DataFrame Format
```python
import pandas as pd

df = pd.DataFrame({
    'date': pd.date_range(start='2023-01-01', periods=100),
    'open': [100.0, 101.5, ...],     # Opening price
    'high': [102.0, 103.0, ...],     # High price
    'low': [99.0, 100.5, ...],       # Low price
    'close': [101.0, 102.0, ...],    # Closing price
    'volume': [1000000, 1200000, ...]  # Volume
})
```

### Minimum Data Requirements
- Sentiment Analysis: 1+ texts
- Chart Analysis: 50+ data points (days)
- ML Training: 100+ data points
- ML Prediction: Requires trained model

## Output Interpretation

### Sentiment Scores
- **Bullish**: Positive market sentiment (score > 0.1)
- **Bearish**: Negative market sentiment (score < -0.1)
- **Neutral**: No clear direction (-0.1 to 0.1)

### Technical Signals
- **RSI > 70**: Overbought (potential sell)
- **RSI < 30**: Oversold (potential buy)
- **MACD > 0**: Bullish momentum
- **MACD < 0**: Bearish momentum
- **Uptrend**: Price moving up
- **Downtrend**: Price moving down

### Trading Signals
- **BUY**: Majority bullish indicators
- **SELL**: Majority bearish indicators
- **HOLD**: Mixed or neutral signals

### ML Predictions
- **Direction**: UP or DOWN
- **Confidence**: 0.0 to 1.0 (higher is better)
- **Return**: Expected % change
- **Target Price**: Predicted next price

## Common Use Cases

### Day Trading
```python
# Get latest data and sentiment
data = get_stock_data('AAPL', period='1mo')
sentiment = fetch_social_sentiment('AAPL')  # your function

# Quick analysis
market_ai = MarketAI()
analysis = market_ai.analyze_market(data, sentiment)

if analysis['overall_recommendation'].startswith('BULLISH'):
    print("Consider LONG position")
elif analysis['overall_recommendation'].startswith('BEARISH'):
    print("Consider SHORT position")
```

### Swing Trading
```python
# Get historical data
data = get_stock_data('TSLA', period='6mo')

# Train model
learner = HistoricalLearner()
learner.train(data)

# Predict
prediction = learner.predict(data)
if prediction['predicted_direction'] == 'UP':
    print(f"Target: ${prediction['predicted_price']}")
```

### Risk Assessment
```python
# Analyze historical performance
learner = HistoricalLearner()
patterns = learner.analyze_historical_patterns(historical_data)

print(f"Max Drawdown: {patterns['max_drawdown']}%")
print(f"Sharpe Ratio: {patterns['sharpe_ratio']}")
print(f"Win Rate: {patterns['win_rate']}%")
```

## Troubleshooting

### "Insufficient data"
- Need at least 50 rows for chart analysis
- Need at least 100 rows for ML training
- Use longer time period or more data

### "Model not trained"
- Call `learner.train(data)` before `predict()`
- Or use `market_ai.train_and_predict()`

### Feature name warnings
- Safe to ignore sklearn feature name warnings
- Does not affect functionality

### No internet connection
- Use sample data from examples
- Can't fetch real stock data without internet
- All modules work with local data

## Tips & Best Practices

1. **More data = Better predictions**
   - Use at least 6 months for training
   - More sentiment texts = more accurate sentiment

2. **Combine multiple signals**
   - Don't rely on one indicator
   - Use the integrated MarketAI for best results

3. **Validate predictions**
   - Check confidence scores
   - Compare with your own analysis

4. **Update regularly**
   - Retrain models with new data
   - Keep sentiment data fresh

5. **Use appropriate timeframes**
   - Day trading: 1min-1hour data
   - Swing trading: Daily data
   - Long-term: Weekly/Monthly data

## Next Steps

- Integrate with real-time data APIs
- Add more technical indicators
- Implement backtesting
- Connect to trading platforms
- Build a web dashboard
- Add more ML models
- Implement ensemble predictions

## Support

- Read README.md for detailed documentation
- Check SUMMARY.md for implementation details
- Run examples.py to see all features
- Modify code for your specific needs
