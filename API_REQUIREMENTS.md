# API Requirements for Non-Hallucinated Data

All agents now validate against live data. Here's what you need to connect:

---

## ✅ Already Working (No Additional APIs Needed)

### 1. TechnicalAnalyst
- **Data Source**: Technical indicators calculated from price data
- **Status**: ✅ **FULLY WORKING**
- **Uses**: Price/volume data from yfinance (already in codebase)
- **No action required** - Gets live RSI, MACD, EMA values from FeatureEngineer

### 2. BreadthHealthAnalyst
- **Data Source**: Market breadth calculated from multiple stocks
- **Status**: ✅ **FULLY WORKING**
- **Uses**: Aggregated data from your watchlist
- **No action required** - Calculates breadth from actual stock data

### 3. TraderAgent
- **Data Source**: Receives validated data from other agents
- **Status**: ✅ **FULLY WORKING**
- **No action required** - Validates trade plans against real prices

### 4. BullBearResearchers
- **Data Source**: Uses analyst outputs
- **Status**: ✅ **FULLY WORKING**
- **No action required** - Validates against real analyst signals

---

## ⚠️ Requires API Integration (Currently Fail Loudly)

### 5. MacroAnalyst - Needs VIX Data

**Required API**: yfinance (FREE, already installed)

**Implementation**:
```python
import yfinance as yf

# Fetch VIX (Volatility Index)
def get_live_vix():
    vix = yf.Ticker("^VIX")
    data = vix.history(period="1d")
    return float(data['Close'].iloc[-1])

# Use in your code
vix_current = get_live_vix()
macro_analyst = MacroAnalyst()
signal = macro_analyst.analyze(macro_data={'vix': vix_current})
```

**Cost**: FREE
**Frequency**: Fetch once per trading session or every hour
**Alternative Sources**:
- CBOE API (official VIX source, requires registration)
- Alpha Vantage (free tier: 5 calls/minute)

---

### 6. SentimentAnalyst - Needs News/Social Sentiment

**Option 1: NewsAPI (EASIEST)**

**Required API**: newsapi.org
**Cost**: FREE tier (100 requests/day) or $449/month for unlimited
**Sign up**: https://newsapi.org/register

**Implementation**:
```python
from newsapi import NewsApiClient
import re

newsapi = NewsApiClient(api_key='YOUR_API_KEY')

def get_news_sentiment(symbol):
    # Fetch recent articles
    articles = newsapi.get_everything(
        q=symbol,
        language='en',
        sort_by='publishedAt',
        page_size=20
    )

    # Simple keyword-based sentiment
    positive_words = ['surge', 'rally', 'gain', 'beat', 'bullish', 'growth',
                      'strong', 'outperform', 'upgrade']
    negative_words = ['fall', 'drop', 'loss', 'miss', 'bearish', 'decline',
                      'weak', 'underperform', 'downgrade']

    sentiment_scores = []
    headlines = []

    for article in articles['articles']:
        text = article['title'].lower() + ' ' + article.get('description', '').lower()
        headlines.append(article['title'])

        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)

        if pos_count + neg_count > 0:
            score = (pos_count - neg_count) / (pos_count + neg_count)
            sentiment_scores.append(score)

    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0

    return {
        'sentiment_score': avg_sentiment,
        'drivers': headlines[:3]
    }

# Use in your code
sentiment_data = get_news_sentiment("AAPL")
sentiment_analyst = SentimentAnalyst()
signal = sentiment_analyst.analyze("AAPL", sentiment_data=sentiment_data)
```

**Frequency**: Once per day per symbol (to stay within free tier)

---

**Option 2: FinBERT (FREE but requires setup)**

**Required**: Hugging Face transformers library (FREE)
**Cost**: FREE (runs locally)
**Compute**: Requires GPU recommended, can run on CPU

**Installation**:
```bash
pip install transformers torch newsapi-python
```

**Implementation**:
```python
from transformers import BertTokenizer, BertForSequenceClassification
from newsapi import NewsApiClient
import torch
import numpy as np

# Load FinBERT model (one-time setup)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
newsapi = NewsApiClient(api_key='YOUR_API_KEY')

def get_finbert_sentiment(symbol):
    # Fetch news
    articles = newsapi.get_everything(q=symbol, language='en', page_size=20)

    sentiment_scores = []
    headlines = []

    for article in articles['articles'][:10]:  # Limit to avoid slowness
        text = article['title']
        headlines.append(text)

        # Tokenize and analyze
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

        # FinBERT outputs: [negative, neutral, positive]
        sentiment = probs[0][2].item() - probs[0][0].item()  # positive - negative
        sentiment_scores.append(sentiment)

    avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0

    return {
        'sentiment_score': float(np.clip(avg_sentiment, -1.0, 1.0)),
        'drivers': headlines[:3],
        'news_sentiment': float(avg_sentiment),
        'num_articles': len(articles['articles'])
    }

# Use in your code
sentiment_data = get_finbert_sentiment("AAPL")
sentiment_analyst = SentimentAnalyst()
signal = sentiment_analyst.analyze("AAPL", sentiment_data=sentiment_data)
```

**Pros**: More accurate than keyword matching, FREE
**Cons**: Slower, requires setup, needs NewsAPI for articles

---

**Option 3: Reddit Sentiment (FREE)**

**Required API**: Reddit API via PRAW library (FREE)
**Sign up**: https://www.reddit.com/prefs/apps

**Implementation**:
```python
import praw
import re

# Set up Reddit API
reddit = praw.Reddit(
    client_id='YOUR_CLIENT_ID',
    client_secret='YOUR_CLIENT_SECRET',
    user_agent='trading_bot'
)

def get_reddit_sentiment(symbol):
    subreddit = reddit.subreddit('wallstreetbets+stocks+investing')

    positive_words = ['calls', 'moon', 'bullish', 'buy', 'long', 'rocket']
    negative_words = ['puts', 'bearish', 'sell', 'short', 'dump', 'crash']

    sentiment_scores = []
    titles = []

    # Search recent posts mentioning symbol
    for post in subreddit.search(symbol, limit=50, time_filter='day'):
        text = (post.title + ' ' + post.selftext).lower()
        titles.append(post.title)

        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)

        if pos_count + neg_count > 0:
            score = (pos_count - neg_count) / (pos_count + neg_count)
            sentiment_scores.append(score)

    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0

    return {
        'sentiment_score': float(np.clip(avg_sentiment, -1.0, 1.0)),
        'drivers': titles[:3],
        'social_sentiment': float(avg_sentiment)
    }

# Use in your code
sentiment_data = get_reddit_sentiment("AAPL")
sentiment_analyst = SentimentAnalyst()
signal = sentiment_analyst.analyze("AAPL", sentiment_data=sentiment_data)
```

**Cost**: FREE
**Frequency**: Once per symbol per day
**Note**: WSB sentiment can be noisy/meme-driven

---

## Recommended Setup for Production

### Minimal Setup (MacroAnalyst only)
```python
import yfinance as yf

# Just add VIX fetching
vix = yf.Ticker("^VIX").history(period="1d")['Close'].iloc[-1]

# Disable SentimentAnalyst by not calling it
analyst_outputs = {
    'technical': technical_analyst.analyze(...),
    'breadth': breadth_analyst.analyze(...),
    'macro': macro_analyst.analyze(macro_data={'vix': vix})
    # Don't include 'sentiment' - system will work without it
}
```

**Cost**: $0/month
**Time to implement**: 5 minutes

---

### Full Setup (All Analysts)
```python
import yfinance as yf
from newsapi import NewsApiClient

# Fetch VIX
vix = yf.Ticker("^VIX").history(period="1d")['Close'].iloc[-1]

# Fetch news sentiment
newsapi = NewsApiClient(api_key='YOUR_KEY')
articles = newsapi.get_everything(q=symbol, language='en', page_size=20)
sentiment_score = calculate_sentiment(articles)  # Your sentiment logic

# Run all analysts
analyst_outputs = {
    'technical': technical_analyst.analyze(symbol, features, market_snapshot),
    'breadth': breadth_analyst.analyze(market_snapshot),
    'macro': macro_analyst.analyze(macro_data={'vix': vix}),
    'sentiment': sentiment_analyst.analyze(symbol, sentiment_data={'sentiment_score': sentiment_score})
}
```

**Cost**: $0-449/month (depending on NewsAPI tier)
**Time to implement**: 1-2 hours

---

## Quick Start Code

Add this to your `main.py` or orchestration code:

```python
import yfinance as yf
from newsapi import NewsApiClient
import os

# Initialize APIs (do this once at startup)
newsapi = NewsApiClient(api_key=os.getenv('NEWS_API_KEY'))  # Optional

def fetch_macro_data():
    """Fetch live macro data (VIX)"""
    try:
        vix = yf.Ticker("^VIX")
        vix_current = float(vix.history(period="1d")['Close'].iloc[-1])
        return {'vix': vix_current}
    except Exception as e:
        logger.error(f"Failed to fetch VIX: {e}")
        return None  # MacroAnalyst will fail loudly

def fetch_sentiment_data(symbol):
    """Fetch live sentiment data (optional)"""
    if not newsapi:
        return None  # SentimentAnalyst will fail loudly

    try:
        articles = newsapi.get_everything(
            q=symbol,
            language='en',
            sort_by='publishedAt',
            page_size=20
        )

        # Simple keyword sentiment
        positive_words = ['surge', 'rally', 'gain', 'beat', 'bullish']
        negative_words = ['fall', 'drop', 'loss', 'miss', 'bearish']

        scores = []
        for article in articles['articles']:
            text = article['title'].lower()
            pos = sum(1 for w in positive_words if w in text)
            neg = sum(1 for w in negative_words if w in text)
            if pos + neg > 0:
                scores.append((pos - neg) / (pos + neg))

        avg_score = sum(scores) / len(scores) if scores else 0.0

        return {
            'sentiment_score': avg_score,
            'drivers': [a['title'] for a in articles['articles'][:3]]
        }
    except Exception as e:
        logger.error(f"Failed to fetch sentiment for {symbol}: {e}")
        return None

# In your trading loop:
def run_analysis(symbol, features, market_snapshot):
    analyst_outputs = {}

    # Technical (always works - uses price data)
    analyst_outputs['technical'] = technical_analyst.analyze(
        symbol, features, market_snapshot
    )

    # Breadth (always works - uses aggregated data)
    analyst_outputs['breadth'] = breadth_analyst.analyze(market_snapshot)

    # Macro (requires VIX)
    macro_data = fetch_macro_data()
    if macro_data:
        try:
            analyst_outputs['macro'] = macro_analyst.analyze(macro_data)
        except ValueError as e:
            logger.warning(f"Macro analyst disabled: {e}")

    # Sentiment (optional - requires NewsAPI)
    sentiment_data = fetch_sentiment_data(symbol)
    if sentiment_data:
        try:
            analyst_outputs['sentiment'] = sentiment_analyst.analyze(
                symbol, sentiment_data
            )
        except ValueError as e:
            logger.warning(f"Sentiment analyst disabled: {e}")

    return analyst_outputs
```

---

## Environment Variables

Add to your `.env` file:

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...  # Already have this

# Optional but recommended for MacroAnalyst
# (None needed - yfinance is free)

# Optional for SentimentAnalyst
NEWS_API_KEY=your_newsapi_key_here  # Get from newsapi.org

# Optional for Reddit sentiment
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
```

---

## Cost Summary

| Data Source | API | Cost | Status |
|-------------|-----|------|--------|
| Stock Prices | yfinance | FREE | ✅ Already working |
| Technical Indicators | Calculated | FREE | ✅ Already working |
| Market Breadth | Calculated | FREE | ✅ Already working |
| VIX (Macro) | yfinance | FREE | ⚠️ Need to add fetch |
| News Sentiment | NewsAPI | FREE (100/day) or $449/mo | ⚠️ Optional |
| News Sentiment | FinBERT | FREE | ⚠️ Optional (needs setup) |
| Social Sentiment | Reddit API | FREE | ⚠️ Optional |

**Minimum to make all agents work**: $0/month (just add VIX fetch)
**Full production setup**: $0-449/month (depending on NewsAPI tier)

---

## Next Steps

1. **Immediate** (5 minutes): Add VIX fetching for MacroAnalyst
   ```bash
   # Already have yfinance, just add to main.py
   vix = yf.Ticker("^VIX").history(period="1d")['Close'].iloc[-1]
   ```

2. **Optional** (1 hour): Add NewsAPI for SentimentAnalyst
   ```bash
   pip install newsapi-python
   # Get free API key from newsapi.org
   ```

3. **Advanced** (2-3 hours): Set up FinBERT for better sentiment
   ```bash
   pip install transformers torch
   # Download model (one-time, ~500MB)
   ```

---

## What Happens If You Don't Connect These?

### Without VIX (MacroAnalyst)
- ❌ MacroAnalyst will raise `ValueError`
- ✅ System can still work - just don't call MacroAnalyst
- ✅ Technical and Breadth analysts provide enough signal

### Without Sentiment (SentimentAnalyst)
- ❌ SentimentAnalyst will raise `ValueError`
- ✅ System can still work - sentiment is optional
- ✅ Most quantitative strategies don't use sentiment

### Recommended Approach
- **Week 1**: Run with just Technical + Breadth analysts (no external APIs needed except yfinance)
- **Week 2**: Add VIX for Macro analyst ($0 cost, 5 min setup)
- **Week 3**: Evaluate if you need sentiment, add NewsAPI if desired

---

**Bottom line**: You can run the system RIGHT NOW with just the existing yfinance integration. MacroAnalyst and SentimentAnalyst are OPTIONAL enhancements that require API keys.
