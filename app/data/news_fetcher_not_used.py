import requests
from typing import List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_fixed
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Replace with your actual Alpha Vantage API key
ALPHA_VANTAGE_API_KEY = "L5Z5ABID5UOQ744Q"  # Replace with your API key

def validate_symbol(symbol: str) -> bool:
    if not symbol:
        raise ValueError("Stock symbol cannot be empty.")
    if not re.match(r"^[A-Za-z]+$", symbol):
        raise ValueError("Stock symbol must contain only letters.")
    return True

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def fetch_stock_news(symbol: str, limit: int = 5) -> List[Dict[str, Any]]:
    try:
        validate_symbol(symbol)
        logger.info(f"Fetching news for symbol {symbol}")
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": symbol.upper(),
            "apikey": ALPHA_VANTAGE_API_KEY,
            "limit": 50
        }
        response = requests.get(url, params=params)
        if response.status_code == 429:
            logger.error("Alpha Vantage rate limit exceeded")
            return {"Error": "Rate limit exceeded. Please try again later."}
        response.raise_for_status()
        data = response.json()
        if "Error Message" in data:
            logger.error(f"Alpha Vantage API error: {data['Error Message']}")
            return {"Error": f"Failed to fetch news: {data['Error Message']}"}
        if "Information" in data and "rate limit" in data["Information"].lower():
            logger.error("Alpha Vantage rate limit exceeded")
            return {"Error": "Rate limit exceeded. Please try again later."}
        if "feed" not in data or not data["feed"]:
            logger.warning(f"No news found for symbol {symbol}")
            return {"Error": f"No news found for symbol {symbol}"}
        news_items = []
        for item in data["feed"][:limit]:
            news_item = {
                "title": item.get("title", "N/A"),
                "published_at": item.get("time_published", "N/A"),
                "url": item.get("url", "N/A"),
                "source": item.get("source", "N/A"),
                "summary": item.get("summary", "N/A"),
                "sentiment_score": item.get("overall_sentiment_score", "N/A"),
                "sentiment_label": item.get("overall_sentiment_label", "N/A")
            }
            news_items.append(news_item)
        logger.info(f"Successfully fetched {len(news_items)} news items for {symbol}")
        return news_items
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching news for {symbol}: {str(e)}")
        return {"Error": f"Failed to fetch news for {symbol}: {str(e)}"}
    except ValueError as e:
        logger.error(f"Invalid symbol {symbol}: {str(e)}")
        return {"Error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error fetching news for {symbol}: {str(e)}")
        return {"Error": f"Unexpected error: {str(e)}"}

# Execute the function
if __name__ == "__main__":
    symbol = "AAPL"  # Example stock symbol for Apple
    news = fetch_stock_news(symbol, limit=5)
    if isinstance(news, dict) and "Error" in news:
        print(news["Error"])
    else:
        for item in news:
            print(f"Title: {item['title']}")
            print(f"Published: {item['published_at']}")
            print(f"Source: {item['source']}")
            print(f"URL: {item['url']}")
            print(f"Summary: {item['summary']}")
            print(f"Sentiment Score: {item['sentiment_score']}")
            print(f"Sentiment Label: {item['sentiment_label']}")
            print("-" * 50)