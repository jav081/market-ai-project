"""
Historical Data Learning Module
Learns from past market events to make predictions or summaries.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import pickle
import os


class HistoricalLearner:
    """Learns from historical market data to make predictions."""
    
    def __init__(self):
        self.price_predictor = None
        self.direction_classifier = None
        self.scaler = MinMaxScaler()
        self.feature_columns = []
        self.is_trained = False
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features from historical data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with engineered features
        """
        features = df.copy()
        
        # Price-based features
        features['returns'] = features['close'].pct_change()
        features['log_returns'] = np.log(features['close'] / features['close'].shift(1))
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            features[f'sma_{window}'] = features['close'].rolling(window=window).mean()
            features[f'price_to_sma_{window}'] = features['close'] / features[f'sma_{window}']
        
        # Volatility
        features['volatility_5'] = features['returns'].rolling(window=5).std()
        features['volatility_20'] = features['returns'].rolling(window=20).std()
        
        # Volume features
        features['volume_ma_20'] = features['volume'].rolling(window=20).mean()
        features['volume_ratio'] = features['volume'] / features['volume_ma_20']
        
        # Price range
        features['high_low_ratio'] = features['high'] / features['low']
        features['close_open_ratio'] = features['close'] / features['open']
        
        # Momentum indicators
        features['rsi'] = self._calculate_rsi(features['close'])
        features['momentum'] = features['close'] - features['close'].shift(10)
        
        # Trend features
        features['trend_5'] = features['close'].rolling(window=5).apply(
            lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1, raw=False
        )
        features['trend_20'] = features['close'].rolling(window=20).apply(
            lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1, raw=False
        )
        
        # Target variables for prediction
        features['next_day_return'] = features['returns'].shift(-1)
        features['next_day_direction'] = (features['next_day_return'] > 0).astype(int)
        features['price_change_5d'] = features['close'].shift(-5) - features['close']
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Train models on historical data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Training metrics
        """
        # Prepare features
        features_df = self.prepare_features(df)
        
        # Drop NaN values
        features_df = features_df.dropna()
        
        if len(features_df) < 100:
            return {'error': 'Insufficient data for training (minimum 100 samples)'}
        
        # Define feature columns (exclude target and original OHLCV)
        self.feature_columns = [
            col for col in features_df.columns 
            if col not in ['open', 'high', 'low', 'close', 'volume', 
                          'next_day_return', 'next_day_direction', 'price_change_5d', 'date']
        ]
        
        X = features_df[self.feature_columns]
        
        # Target for direction classification
        y_direction = features_df['next_day_direction']
        
        # Target for price prediction (next day return)
        y_price = features_df['next_day_return']
        
        # Split data
        X_train, X_test, y_dir_train, y_dir_test = train_test_split(
            X, y_direction, test_size=0.2, random_state=42
        )
        
        _, _, y_price_train, y_price_test = train_test_split(
            X, y_price, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train direction classifier
        self.direction_classifier = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42
        )
        self.direction_classifier.fit(X_train_scaled, y_dir_train)
        dir_accuracy = self.direction_classifier.score(X_test_scaled, y_dir_test)
        
        # Train price predictor
        self.price_predictor = GradientBoostingRegressor(
            n_estimators=100, max_depth=5, random_state=42
        )
        self.price_predictor.fit(X_train_scaled, y_price_train)
        price_score = self.price_predictor.score(X_test_scaled, y_price_test)
        
        self.is_trained = True
        
        return {
            'direction_accuracy': round(dir_accuracy, 3),
            'price_prediction_r2': round(price_score, 3),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features_used': len(self.feature_columns)
        }
    
    def predict(self, df: pd.DataFrame) -> Dict:
        """
        Make predictions based on recent data.
        
        Args:
            df: DataFrame with recent OHLCV data
            
        Returns:
            Prediction results
        """
        if not self.is_trained:
            return {'error': 'Model not trained. Call train() first.'}
        
        # Prepare features
        features_df = self.prepare_features(df)
        features_df = features_df.dropna()
        
        if len(features_df) == 0:
            return {'error': 'Insufficient data for prediction'}
        
        # Get latest data point
        X_latest = features_df[self.feature_columns].iloc[-1:].values
        X_scaled = self.scaler.transform(X_latest)
        
        # Predict direction
        direction_prob = self.direction_classifier.predict_proba(X_scaled)[0]
        direction = 'UP' if direction_prob[1] > 0.5 else 'DOWN'
        confidence = max(direction_prob)
        
        # Predict return
        predicted_return = self.price_predictor.predict(X_scaled)[0]
        
        # Calculate predicted price
        current_price = df['close'].iloc[-1]
        predicted_price = current_price * (1 + predicted_return)
        
        return {
            'predicted_direction': direction,
            'confidence': round(confidence, 3),
            'predicted_return': round(predicted_return * 100, 2),  # in percentage
            'current_price': round(current_price, 2),
            'predicted_price': round(predicted_price, 2),
            'up_probability': round(direction_prob[1], 3),
            'down_probability': round(direction_prob[0], 3)
        }
    
    def analyze_historical_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Analyze historical patterns and generate insights.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Historical pattern analysis
        """
        features_df = self.prepare_features(df)
        features_df = features_df.dropna()
        
        if len(features_df) < 50:
            return {'error': 'Insufficient data for pattern analysis'}
        
        # Calculate statistics
        returns = features_df['returns']
        
        # Best and worst days
        best_days = returns.nlargest(5)
        worst_days = returns.nsmallest(5)
        
        # Volatility analysis
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        # Trend analysis
        total_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
        
        # Win rate
        win_rate = (returns > 0).sum() / len(returns)
        
        # Average gains and losses
        avg_gain = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = returns[returns < 0].mean() if (returns < 0).any() else 0
        
        # Risk-reward ratio
        risk_reward = abs(avg_gain / avg_loss) if avg_loss != 0 else 0
        
        return {
            'total_return_pct': round(total_return, 2),
            'annualized_volatility': round(volatility * 100, 2),
            'win_rate': round(win_rate, 3),
            'avg_gain_pct': round(avg_gain * 100, 2),
            'avg_loss_pct': round(avg_loss * 100, 2),
            'risk_reward_ratio': round(risk_reward, 2),
            'best_day_return': round(best_days.iloc[0] * 100, 2),
            'worst_day_return': round(worst_days.iloc[0] * 100, 2),
            'max_drawdown': round(self._calculate_max_drawdown(df['close']) * 100, 2),
            'sharpe_ratio': round(self._calculate_sharpe_ratio(returns), 2)
        }
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cummax = prices.cummax()
        drawdown = (prices - cummax) / cummax
        return drawdown.min()
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = returns.mean() * 252 - risk_free_rate
        return excess_returns / (returns.std() * np.sqrt(252))
    
    def generate_summary(self, df: pd.DataFrame) -> str:
        """
        Generate a text summary of historical analysis.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Text summary
        """
        patterns = self.analyze_historical_patterns(df)
        
        if 'error' in patterns:
            return patterns['error']
        
        summary = f"""
Historical Market Analysis Summary
{'=' * 50}

Performance Metrics:
- Total Return: {patterns['total_return_pct']}%
- Annualized Volatility: {patterns['annualized_volatility']}%
- Win Rate: {patterns['win_rate'] * 100:.1f}%
- Sharpe Ratio: {patterns['sharpe_ratio']}

Risk-Reward Analysis:
- Average Gain: {patterns['avg_gain_pct']}%
- Average Loss: {patterns['avg_loss_pct']}%
- Risk-Reward Ratio: {patterns['risk_reward_ratio']}
- Maximum Drawdown: {patterns['max_drawdown']}%

Extreme Events:
- Best Day: {patterns['best_day_return']}%
- Worst Day: {patterns['worst_day_return']}%

Interpretation:
"""
        
        # Add interpretation
        if patterns['sharpe_ratio'] > 1:
            summary += "- Strong risk-adjusted returns (Sharpe > 1)\n"
        elif patterns['sharpe_ratio'] > 0:
            summary += "- Positive risk-adjusted returns\n"
        else:
            summary += "- Negative risk-adjusted returns\n"
        
        if patterns['win_rate'] > 0.55:
            summary += "- High win rate indicates consistent positive moves\n"
        elif patterns['win_rate'] < 0.45:
            summary += "- Low win rate indicates frequent negative moves\n"
        
        if patterns['risk_reward_ratio'] > 1.5:
            summary += "- Favorable risk-reward ratio (gains > losses)\n"
        elif patterns['risk_reward_ratio'] < 0.67:
            summary += "- Unfavorable risk-reward ratio (losses > gains)\n"
        
        return summary.strip()
    
    def save_model(self, filepath: str):
        """Save trained model to file."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        model_data = {
            'direction_classifier': self.direction_classifier,
            'price_predictor': self.price_predictor,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load trained model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.direction_classifier = model_data['direction_classifier']
        self.price_predictor = model_data['price_predictor']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.is_trained = True


if __name__ == "__main__":
    # Example usage
    learner = HistoricalLearner()
    
    # Create sample historical data
    dates = pd.date_range(start='2022-01-01', periods=500, freq='D')
    np.random.seed(42)
    
    # Generate synthetic price data with trend
    trend = np.linspace(100, 150, 500)
    noise = np.random.randn(500) * 5
    close_prices = trend + noise
    
    open_prices = close_prices + np.random.randn(500) * 2
    high_prices = np.maximum(open_prices, close_prices) + np.abs(np.random.randn(500) * 3)
    low_prices = np.minimum(open_prices, close_prices) - np.abs(np.random.randn(500) * 3)
    volume = np.random.randint(1000000, 10000000, 500)
    
    df = pd.DataFrame({
        'date': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    })
    
    # Train model
    print("Training Models...")
    print("-" * 50)
    metrics = learner.train(df)
    print(f"Direction Accuracy: {metrics['direction_accuracy']}")
    print(f"Price Prediction R²: {metrics['price_prediction_r2']}")
    print(f"Features Used: {metrics['features_used']}")
    
    # Make prediction
    print("\nMaking Predictions...")
    print("-" * 50)
    prediction = learner.predict(df)
    print(f"Predicted Direction: {prediction['predicted_direction']}")
    print(f"Confidence: {prediction['confidence']}")
    print(f"Predicted Return: {prediction['predicted_return']}%")
    print(f"Current Price: ${prediction['current_price']}")
    print(f"Predicted Price: ${prediction['predicted_price']}")
    
    # Analyze patterns
    print("\nHistorical Pattern Analysis...")
    print("-" * 50)
    print(learner.generate_summary(df))
