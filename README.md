# Market AI Project

AI-powered market analysis system that combines sentiment analysis, technical chart reading, and machine learning predictions.

## Features

### 1. **Market Sentiment Analysis** 📊
Analyzes news articles, social media posts, and comments to detect bullish/bearish market sentiment.

- Text preprocessing and cleaning
- Sentiment classification (Bullish/Bearish/Neutral)
- Keyword-based sentiment scoring
- Aggregate sentiment analysis from multiple sources
- Support for news, social posts, and comments

### 2. **Chart Reading AI** 📈
Analyzes historical price charts with technical indicators and pattern recognition.

- **Technical Indicators:**
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Moving Averages (SMA 20, 50, 200)
  - Volume analysis
  
- **Pattern Recognition:**
  - Candlestick patterns (Doji, Hammer, Shooting Star, etc.)
  - Support and resistance levels
  - Trend detection (Uptrend/Downtrend/Sideways)
  
- **Trading Signals:**
  - Buy/Sell/Hold recommendations
  - Multi-indicator signal aggregation

### 3. **Historical Data Learning** 🤖
Machine learning models that learn from past market events to make predictions.

- **Feature Engineering:**
  - Price-based features (returns, log returns)
  - Technical indicators (RSI, moving averages)
  - Volatility measures
  - Volume patterns
  - Momentum indicators

- **ML Models:**
  - Direction prediction (Up/Down)
  - Price prediction (expected return)
  - Confidence scoring

- **Historical Analysis:**
  - Win rate calculation
  - Risk-reward ratio
  - Sharpe ratio
  - Maximum drawdown
  - Performance summaries

## Installation

1. Clone the repository:
```bash
git clone https://github.com/jav081/market-ai-project.git
cd market-ai-project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data (first time only):
```python
import nltk
nltk.download('punkt')
```

## Quick Start

### Basic Usage

```python
from main import MarketAI
import pandas as pd

# Initialize Market AI
market_ai = MarketAI()

# Prepare your price data (DataFrame with columns: open, high, low, close, volume)
price_data = pd.DataFrame({
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...]
})

# Prepare sentiment texts (optional)
sentiment_texts = [
    "Stock showing strong momentum!",
    "Bearish signals on the chart",
    # ... more texts
]

# Get comprehensive analysis
summary = market_ai.get_market_summary(price_data, sentiment_texts)
print(summary)
```

### Run Examples

```bash
# Run quick demo
python demo.py

# Run main application demo
python main.py

# Run all feature examples
python examples.py

# Analyze real stocks (requires internet)
python real_market_example.py AAPL
python real_market_example.py TSLA
python real_market_example.py SPY
```

## Module Documentation

### Sentiment Analyzer

```python
from sentiment_analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer()

# Analyze single text
result = analyzer.analyze_sentiment("This stock is going to moon!")
print(result['classification'])  # BULLISH/BEARISH/NEUTRAL

# Analyze multiple texts
texts = ["Bullish breakout!", "Market crash incoming"]
aggregate = analyzer.aggregate_sentiment(texts)
print(aggregate['overall_sentiment'])
```

### Chart Analyzer

```python
from chart_analyzer import ChartAnalyzer

analyzer = ChartAnalyzer()

# Comprehensive analysis
analysis = analyzer.comprehensive_analysis(price_dataframe)
print(analysis['overall_signal'])  # BUY/SELL/HOLD
print(analysis['rsi'])
print(analysis['trend'])
```

### Historical Learner

```python
from historical_learner import HistoricalLearner

learner = HistoricalLearner()

# Train on historical data
metrics = learner.train(historical_dataframe)
print(f"Accuracy: {metrics['direction_accuracy']}")

# Make predictions
prediction = learner.predict(recent_dataframe)
print(f"Direction: {prediction['predicted_direction']}")
print(f"Expected Price: ${prediction['predicted_price']}")

# Analyze patterns
summary = learner.generate_summary(historical_dataframe)
print(summary)
```

## Data Format

All modules expect price data in the following format:

```python
import pandas as pd

df = pd.DataFrame({
    'date': pd.date_range(start='2023-01-01', periods=100),
    'open': [...],    # Opening prices
    'high': [...],    # High prices
    'low': [...],     # Low prices
    'close': [...],   # Closing prices
    'volume': [...]   # Volume
})
```

## Example Output

### Technical Analysis
```
TECHNICAL ANALYSIS
----------------------------------------------------------------------
Current Trend: UPTREND
Overall Signal: BUY
RSI: 45.2 (NEUTRAL)
MACD: BULLISH
Candlestick Pattern: BULLISH_CANDLE
Volume: HIGH_VOLUME
```

### Sentiment Analysis
```
SENTIMENT ANALYSIS
----------------------------------------------------------------------
Overall Sentiment: BULLISH
Average Score: 0.342
Bullish Ratio: 60.0%
Bearish Ratio: 20.0%
Neutral Ratio: 20.0%
Total Texts Analyzed: 10
```

### ML Predictions
```
ML PREDICTIONS
----------------------------------------------------------------------
Predicted Direction: UP
Confidence: 0.78
Predicted Return: 2.3%
Current Price: $150.25
Predicted Price: $153.71
```

## Use Cases

1. **Day Trading**: Real-time sentiment analysis + technical signals
2. **Swing Trading**: Historical pattern learning + chart analysis
3. **Market Research**: Aggregate sentiment from multiple sources
4. **Risk Management**: Historical volatility and drawdown analysis
5. **Strategy Development**: Backtesting with ML predictions

## Real-World Integration

The `real_market_example.py` script demonstrates integration with real market data:

```python
# Fetch and analyze real stock data
python real_market_example.py AAPL

# Or use the module in your code
from real_market_example import get_stock_data, analyze_stock

data = get_stock_data('MSFT', period='1y')
analyze_stock('MSFT', sentiment_texts=[...])
```

You can integrate with:
- **Yahoo Finance** (yfinance) - Historical price data
- **Twitter API** - Real-time social sentiment
- **Reddit API** - Community sentiment analysis
- **News APIs** - News sentiment tracking
- **Trading Platforms** - Automated trading execution

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- textblob
- nltk
- transformers
- torch
- ta (Technical Analysis library)
- matplotlib
- yfinance

## Project Structure

```
market-ai-project/
├── main.py                    # Main application & integration
├── sentiment_analyzer.py      # Sentiment analysis module
├── chart_analyzer.py          # Chart reading module
├── historical_learner.py      # ML prediction module
├── examples.py                # Usage examples
├── demo.py                    # Quick demo script
├── real_market_example.py     # Real market data integration
├── requirements.txt           # Dependencies
├── .gitignore                # Git ignore rules
└── README.md                 # Documentation
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Disclaimer

This tool is for educational and research purposes only. It should not be considered as financial advice. Always do your own research and consult with financial professionals before making investment decisions.
