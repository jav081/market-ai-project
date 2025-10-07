"""
Market Sentiment Analysis Module
Reads news, social posts, or comments to detect bullish/bearish mood.
"""

import re
from typing import Dict, List, Union
from textblob import TextBlob
import nltk
from collections import Counter

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class SentimentAnalyzer:
    """Analyzes market sentiment from text data."""
    
    def __init__(self):
        self.bullish_keywords = [
            'bullish', 'moon', 'surge', 'rally', 'breakout', 'uptrend',
            'buy', 'long', 'calls', 'pump', 'rocket', 'gain', 'profit',
            'strong', 'growth', 'positive', 'optimistic', 'bull'
        ]
        
        self.bearish_keywords = [
            'bearish', 'crash', 'dump', 'decline', 'downtrend', 'sell',
            'short', 'puts', 'drop', 'fall', 'loss', 'weak', 'negative',
            'pessimistic', 'bear', 'resistance', 'correction'
        ]
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for sentiment analysis."""
        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def analyze_sentiment(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing sentiment classification and scores
        """
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Get TextBlob sentiment polarity (-1 to 1)
        blob = TextBlob(processed_text)
        polarity = blob.sentiment.polarity
        
        # Count bullish/bearish keywords
        words = processed_text.split()
        bullish_count = sum(1 for word in words if word in self.bullish_keywords)
        bearish_count = sum(1 for word in words if word in self.bearish_keywords)
        
        # Calculate keyword-based sentiment
        keyword_sentiment = bullish_count - bearish_count
        
        # Combine both methods
        combined_score = (polarity + (keyword_sentiment * 0.1)) / 1.1
        
        # Classify sentiment
        if combined_score > 0.1:
            classification = 'BULLISH'
        elif combined_score < -0.1:
            classification = 'BEARISH'
        else:
            classification = 'NEUTRAL'
        
        return {
            'text': text,
            'classification': classification,
            'polarity_score': round(polarity, 3),
            'keyword_score': keyword_sentiment,
            'combined_score': round(combined_score, 3),
            'bullish_keywords_found': bullish_count,
            'bearish_keywords_found': bearish_count
        }
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Union[str, float]]]:
        """
        Analyze sentiment of multiple texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of sentiment analysis results
        """
        return [self.analyze_sentiment(text) for text in texts]
    
    def aggregate_sentiment(self, texts: List[str]) -> Dict[str, Union[str, float, int]]:
        """
        Aggregate sentiment from multiple texts to get overall market mood.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Aggregated sentiment analysis
        """
        if not texts:
            return {
                'overall_sentiment': 'NEUTRAL',
                'average_score': 0.0,
                'bullish_ratio': 0.0,
                'bearish_ratio': 0.0,
                'neutral_ratio': 0.0,
                'total_texts': 0
            }
        
        results = self.analyze_batch(texts)
        
        # Count classifications
        classifications = [r['classification'] for r in results]
        counts = Counter(classifications)
        
        # Calculate ratios
        total = len(results)
        bullish_ratio = counts.get('BULLISH', 0) / total
        bearish_ratio = counts.get('BEARISH', 0) / total
        neutral_ratio = counts.get('NEUTRAL', 0) / total
        
        # Calculate average score
        avg_score = sum(r['combined_score'] for r in results) / total
        
        # Determine overall sentiment
        if avg_score > 0.1:
            overall = 'BULLISH'
        elif avg_score < -0.1:
            overall = 'BEARISH'
        else:
            overall = 'NEUTRAL'
        
        return {
            'overall_sentiment': overall,
            'average_score': round(avg_score, 3),
            'bullish_ratio': round(bullish_ratio, 3),
            'bearish_ratio': round(bearish_ratio, 3),
            'neutral_ratio': round(neutral_ratio, 3),
            'total_texts': total,
            'bullish_count': counts.get('BULLISH', 0),
            'bearish_count': counts.get('BEARISH', 0),
            'neutral_count': counts.get('NEUTRAL', 0)
        }


def analyze_news(news_texts: List[str]) -> Dict:
    """Analyze sentiment from news articles."""
    analyzer = SentimentAnalyzer()
    return analyzer.aggregate_sentiment(news_texts)


def analyze_social_posts(posts: List[str]) -> Dict:
    """Analyze sentiment from social media posts."""
    analyzer = SentimentAnalyzer()
    return analyzer.aggregate_sentiment(posts)


def analyze_comments(comments: List[str]) -> Dict:
    """Analyze sentiment from comments."""
    analyzer = SentimentAnalyzer()
    return analyzer.aggregate_sentiment(comments)


if __name__ == "__main__":
    # Example usage
    analyzer = SentimentAnalyzer()
    
    # Test with sample texts
    sample_texts = [
        "This stock is going to moon! 🚀 Bullish breakout incoming!",
        "Major resistance ahead, expecting a correction soon.",
        "Strong buy signal, the trend is definitely bullish.",
        "Bears are taking control, sell before it crashes.",
        "Market looks neutral today, waiting for direction."
    ]
    
    print("Individual Sentiment Analysis:")
    print("-" * 60)
    for text in sample_texts:
        result = analyzer.analyze_sentiment(text)
        print(f"Text: {result['text']}")
        print(f"Classification: {result['classification']}")
        print(f"Combined Score: {result['combined_score']}")
        print("-" * 60)
    
    print("\nAggregated Sentiment Analysis:")
    print("-" * 60)
    aggregate = analyzer.aggregate_sentiment(sample_texts)
    print(f"Overall Sentiment: {aggregate['overall_sentiment']}")
    print(f"Average Score: {aggregate['average_score']}")
    print(f"Bullish Ratio: {aggregate['bullish_ratio']}")
    print(f"Bearish Ratio: {aggregate['bearish_ratio']}")
    print(f"Neutral Ratio: {aggregate['neutral_ratio']}")
