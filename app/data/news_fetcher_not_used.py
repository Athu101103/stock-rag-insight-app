import requests
import pandas as pd
from datetime import datetime, timedelta
import json
import time
from typing import List, Dict, Optional
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

class StockNewsFetcher:
    """
    A comprehensive stock news fetcher using multiple APIs
    Supports: NewsAPI, Alpha Vantage, Finnhub, and Yahoo Finance
    """
    
    def __init__(self):
        # API Keys - Replace with your actual keys
        self.newsapi_key = "YOUR_NEWSAPI_KEY"  # Get from https://newsapi.org/
        self.alpha_vantage_key = "YOUR_ALPHA_VANTAGE_KEY"  # Get from https://www.alphavantage.co/
        self.finnhub_key = "YOUR_FINNHUB_KEY"  # Get from https://finnhub.io/
        
        # API endpoints
        self.newsapi_url = "https://newsapi.org/v2/everything"
        self.alpha_vantage_url = "https://www.alphavantage.co/query"
        self.finnhub_url = "https://finnhub.io/api/v1/company-news"
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1  # seconds between requests
    
    def _rate_limit(self):
        """Simple rate limiting to avoid hitting API limits"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(self, url: str, params: dict, timeout: int = 10) -> Optional[dict]:
        """Make HTTP request with error handling"""
        try:
            self._rate_limit()
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return None
    
    def fetch_newsapi_news(self, symbol: str, company_name: str = None, days: int = 7) -> List[Dict]:
        """
        Fetch news from NewsAPI
        Free tier: 100 requests/day, 1000 requests/month
        """
        if self.newsapi_key == "YOUR_NEWSAPI_KEY":
            print("⚠️ NewsAPI key not configured. Skipping NewsAPI...")
            return []
        
        from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        # Create search query
        query_terms = [symbol]
        if company_name:
            query_terms.append(f'"{company_name}"')
        
        query = " OR ".join(query_terms)
        
        params = {
            'q': query,
            'from': from_date,
            'sortBy': 'publishedAt',
            'language': 'en',
            'apiKey': self.newsapi_key,
            'pageSize': 50
        }
        
        data = self._make_request(self.newsapi_url, params)
        if not data or data.get('status') != 'ok':
            print(f"NewsAPI error: {data.get('message') if data else 'No response'}")
            return []
        
        articles = []
        for article in data.get('articles', []):
            articles.append({
                'source': 'NewsAPI',
                'title': article.get('title', ''),
                'description': article.get('description', ''),
                'content': article.get('content', ''),
                'url': article.get('url', ''),
                'published_at': article.get('publishedAt', ''),
                'source_name': article.get('source', {}).get('name', ''),
                'relevance_score': self._calculate_relevance(article.get('title', '') + ' ' + article.get('description', ''), symbol, company_name)
            })
        
        return articles
    
    def fetch_alpha_vantage_news(self, symbol: str, limit: int = 50) -> List[Dict]:
        """
        Fetch news from Alpha Vantage
        Free tier: 25 requests/day
        """
        if self.alpha_vantage_key == "YOUR_ALPHA_VANTAGE_KEY":
            print("⚠️ Alpha Vantage key not configured. Skipping Alpha Vantage...")
            return []
        
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': symbol,
            'limit': limit,
            'apikey': self.alpha_vantage_key
        }
        
        data = self._make_request(self.alpha_vantage_url, params)
        if not data or 'feed' not in data:
            print(f"Alpha Vantage error: {data.get('Error Message') if data else 'No response'}")
            return []
        
        articles = []
        for article in data.get('feed', []):
            # Extract sentiment for the specific ticker
            ticker_sentiment = None
            for ticker_data in article.get('ticker_sentiment', []):
                if ticker_data.get('ticker') == symbol:
                    ticker_sentiment = ticker_data
                    break
            
            articles.append({
                'source': 'Alpha Vantage',
                'title': article.get('title', ''),
                'summary': article.get('summary', ''),
                'url': article.get('url', ''),
                'published_at': article.get('time_published', ''),
                'source_name': article.get('source', ''),
                'overall_sentiment': article.get('overall_sentiment_label', ''),
                'overall_sentiment_score': article.get('overall_sentiment_score', 0),
                'ticker_sentiment': ticker_sentiment.get('ticker_sentiment_label') if ticker_sentiment else None,
                'ticker_sentiment_score': ticker_sentiment.get('ticker_sentiment_score') if ticker_sentiment else None,
                'relevance_score': ticker_sentiment.get('relevance_score') if ticker_sentiment else 0
            })
        
        return articles
    
    def fetch_finnhub_news(self, symbol: str, days: int = 7) -> List[Dict]:
        """
        Fetch news from Finnhub
        Free tier: 60 calls/minute
        """
        if self.finnhub_key == "YOUR_FINNHUB_KEY":
            print("⚠️ Finnhub key not configured. Skipping Finnhub...")
            return []
        
        from_date = datetime.now() - timedelta(days=days)
        to_date = datetime.now()
        
        params = {
            'symbol': symbol,
            'from': from_date.strftime('%Y-%m-%d'),
            'to': to_date.strftime('%Y-%m-%d'),
            'token': self.finnhub_key
        }
        
        data = self._make_request(self.finnhub_url, params)
        if not data or isinstance(data, dict) and 'error' in data:
            print(f"Finnhub error: {data.get('error') if data else 'No response'}")
            return []
        
        articles = []
        for article in data:
            articles.append({
                'source': 'Finnhub',
                'title': article.get('headline', ''),
                'summary': article.get('summary', ''),
                'url': article.get('url', ''),
                'published_at': datetime.fromtimestamp(article.get('datetime', 0)).isoformat(),
                'source_name': article.get('source', ''),
                'category': article.get('category', ''),
                'image': article.get('image', ''),
                'relevance_score': 1.0  # Finnhub news is already filtered by symbol
            })
        
        return articles
    
    def fetch_yahoo_finance_news(self, symbol: str) -> List[Dict]:
        """
        Fetch news from Yahoo Finance (free, no API key needed)
        """
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            articles = []
            for article in news:
                articles.append({
                    'source': 'Yahoo Finance',
                    'title': article.get('title', ''),
                    'summary': article.get('summary', ''),
                    'url': article.get('link', ''),
                    'published_at': datetime.fromtimestamp(article.get('providerPublishTime', 0)).isoformat(),
                    'source_name': article.get('publisher', ''),
                    'type': article.get('type', ''),
                    'relevance_score': 1.0  # Yahoo Finance news is already filtered by symbol
                })
            
            return articles
        except Exception as e:
            print(f"Yahoo Finance news error: {e}")
            return []
    
    def _calculate_relevance(self, text: str, symbol: str, company_name: str = None) -> float:
        """Calculate relevance score based on symbol/company mentions"""
        if not text:
            return 0.0
        
        text_lower = text.lower()
        score = 0.0
        
        # Symbol mentions
        if symbol.lower() in text_lower:
            score += 0.5
        
        # Company name mentions
        if company_name and company_name.lower() in text_lower:
            score += 0.3
        
        # Financial keywords
        financial_keywords = ['stock', 'shares', 'trading', 'market', 'price', 'earnings', 
                             'revenue', 'profit', 'loss', 'investment', 'analyst', 'forecast']
        for keyword in financial_keywords:
            if keyword in text_lower:
                score += 0.1
                break
        
        return min(score, 1.0)
    
    def get_company_name(self, symbol: str) -> Optional[str]:
        """Get company name from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info.get('longName') or info.get('shortName')
        except:
            return None
    
    def fetch_all_news(self, symbol: str, days: int = 7, include_sentiment: bool = True) -> Dict:
        """
        Fetch news from all available sources
        """
        print(f"🔍 Fetching news for {symbol}...")
        
        # Get company name for better search results
        company_name = self.get_company_name(symbol)
        if company_name:
            print(f"📊 Company: {company_name}")
        
        all_news = {
            'symbol': symbol,
            'company_name': company_name,
            'fetch_date': datetime.now().isoformat(),
            'sources': {}
        }
        
        # Fetch from all sources
        sources = [
            ('yahoo_finance', lambda: self.fetch_yahoo_finance_news(symbol)),
            ('newsapi', lambda: self.fetch_newsapi_news(symbol, company_name, days)),
            ('alpha_vantage', lambda: self.fetch_alpha_vantage_news(symbol)),
            ('finnhub', lambda: self.fetch_finnhub_news(symbol, days))
        ]
        
        total_articles = 0
        for source_name, fetch_func in sources:
            try:
                print(f"📰 Fetching from {source_name.replace('_', ' ').title()}...")
                articles = fetch_func()
                all_news['sources'][source_name] = {
                    'articles': articles,
                    'count': len(articles),
                    'status': 'success' if articles else 'no_data'
                }
                total_articles += len(articles)
                print(f"   ✅ Found {len(articles)} articles")
            except Exception as e:
                print(f"   ❌ Error: {e}")
                all_news['sources'][source_name] = {
                    'articles': [],
                    'count': 0,
                    'status': 'error',
                    'error': str(e)
                }
        
        # Combine and deduplicate articles
        combined_articles = self._combine_and_deduplicate(all_news['sources'])
        all_news['combined_articles'] = combined_articles
        all_news['total_articles'] = len(combined_articles)
        
        # Generate summary
        if include_sentiment:
            all_news['sentiment_summary'] = self._analyze_sentiment_summary(combined_articles)
        
        print(f"📊 Total unique articles found: {len(combined_articles)}")
        
        return all_news
    
    def _combine_and_deduplicate(self, sources: Dict) -> List[Dict]:
        """Combine articles from all sources and remove duplicates"""
        all_articles = []
        
        for source_name, source_data in sources.items():
            for article in source_data.get('articles', []):
                article['source_api'] = source_name
                all_articles.append(article)
        
        # Simple deduplication based on title similarity
        unique_articles = []
        seen_titles = set()
        
        for article in all_articles:
            title = article.get('title', '').lower().strip()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_articles.append(article)
        
        # Sort by published date (newest first)
        unique_articles.sort(key=lambda x: x.get('published_at', ''), reverse=True)
        
        return unique_articles
    
    def _analyze_sentiment_summary(self, articles: List[Dict]) -> Dict:
        """Analyze overall sentiment from articles"""
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        sentiment_scores = []
        
        for article in articles:
            # From Alpha Vantage sentiment
            if 'overall_sentiment' in article:
                sentiment = article['overall_sentiment'].lower()
                if 'positive' in sentiment:
                    sentiment_counts['positive'] += 1
                elif 'negative' in sentiment:
                    sentiment_counts['negative'] += 1
                else:
                    sentiment_counts['neutral'] += 1
            
            # Collect sentiment scores
            if 'overall_sentiment_score' in article:
                sentiment_scores.append(article['overall_sentiment_score'])
            elif 'ticker_sentiment_score' in article and article['ticker_sentiment_score']:
                sentiment_scores.append(article['ticker_sentiment_score'])
        
        total_sentiment_articles = sum(sentiment_counts.values())
        avg_sentiment_score = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        
        return {
            'sentiment_distribution': sentiment_counts,
            'total_sentiment_articles': total_sentiment_articles,
            'average_sentiment_score': avg_sentiment_score,
            'overall_sentiment': self._determine_overall_sentiment(sentiment_counts, avg_sentiment_score)
        }
    
    def _determine_overall_sentiment(self, counts: Dict, avg_score: float) -> str:
        """Determine overall sentiment based on counts and scores"""
        if avg_score > 0.1:
            return 'Bullish'
        elif avg_score < -0.1:
            return 'Bearish'
        elif counts['positive'] > counts['negative']:
            return 'Slightly Bullish'
        elif counts['negative'] > counts['positive']:
            return 'Slightly Bearish'
        else:
            return 'Neutral'
    
    def save_news_to_json(self, news_data: Dict, filename: str = None):
        """Save news data to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"news_{news_data['symbol']}_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(news_data, f, indent=2, ensure_ascii=False)
            print(f"💾 News data saved to {filename}")
            return filename
        except Exception as e:
            print(f"❌ Error saving to file: {e}")
            return None
    
    def create_news_dataframe(self, news_data: Dict) -> pd.DataFrame:
        """Convert news data to pandas DataFrame for analysis"""
        articles = news_data.get('combined_articles', [])
        
        if not articles:
            return pd.DataFrame()
        
        df_data = []
        for article in articles:
            df_data.append({
                'title': article.get('title', ''),
                'source_api': article.get('source_api', ''),
                'source_name': article.get('source_name', ''),
                'published_at': article.get('published_at', ''),
                'url': article.get('url', ''),
                'summary': article.get('summary', '') or article.get('description', ''),
                'sentiment': article.get('overall_sentiment', ''),
                'sentiment_score': article.get('overall_sentiment_score', 0),
                'relevance_score': article.get('relevance_score', 0)
            })
        
        df = pd.DataFrame(df_data)
        if not df.empty and 'published_at' in df.columns:
            df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
            df = df.sort_values('published_at', ascending=False)
        
        return df
    
    def print_news_summary(self, news_data: Dict, max_articles: int = 10):
        """Print a formatted summary of the news"""
        print("=" * 100)
        print(f"📰 NEWS SUMMARY FOR {news_data['symbol']}")
        if news_data.get('company_name'):
            print(f"🏢 Company: {news_data['company_name']}")
        print("=" * 100)
        
        # Source summary
        print("\n📊 SOURCE SUMMARY:")
        for source, data in news_data['sources'].items():
            status_emoji = "✅" if data['status'] == 'success' else "❌" if data['status'] == 'error' else "⚠️"
            print(f"   {status_emoji} {source.replace('_', ' ').title()}: {data['count']} articles")
        
        print(f"\n📈 Total Unique Articles: {news_data['total_articles']}")
        
        # Sentiment summary
        if 'sentiment_summary' in news_data:
            sentiment_data = news_data['sentiment_summary']
            print(f"\n💭 SENTIMENT ANALYSIS:")
            print(f"   📊 Overall Sentiment: {sentiment_data['overall_sentiment']}")
            if sentiment_data['total_sentiment_articles'] > 0:
                dist = sentiment_data['sentiment_distribution']
                print(f"   📈 Positive: {dist['positive']}, 📉 Negative: {dist['negative']}, ➡️ Neutral: {dist['neutral']}")
        
        # Recent articles
        articles = news_data.get('combined_articles', [])[:max_articles]
        if articles:
            print(f"\n📰 RECENT ARTICLES (Top {len(articles)}):")
            for i, article in enumerate(articles, 1):
                title = article.get('title', 'No title')[:80] + "..." if len(article.get('title', '')) > 80 else article.get('title', 'No title')
                source = article.get('source_name', article.get('source_api', 'Unknown'))
                published = article.get('published_at', '')[:10] if article.get('published_at') else 'Unknown date'
                
                print(f"\n   {i}. {title}")
                print(f"      📅 {published} | 📰 {source}")
                
                if article.get('summary'):
                    summary = article['summary'][:150] + "..." if len(article['summary']) > 150 else article['summary']
                    print(f"      💬 {summary}")
                
                if article.get('overall_sentiment'):
                    sentiment_emoji = {"Positive": "😊", "Negative": "😔", "Neutral": "😐"}.get(article['overall_sentiment'], "")
                    print(f"      💭 Sentiment: {sentiment_emoji} {article['overall_sentiment']}")
        
        print("\n" + "=" * 100)


# Integration function for existing stock analysis
def integrate_news_with_stock_analysis(symbol: str, period: str = "1d"):
    """
    Integrate news fetching with existing stock analysis
    """
    from stock_insights import fetch_and_analyze_stock  # Import from your existing module
    
    print(f"🔍 Comprehensive Analysis for {symbol}")
    print("=" * 80)
    
    # Get stock analysis
    print("📊 Fetching stock technical analysis...")
    stock_analysis = fetch_and_analyze_stock(symbol, period)
    
    # Get news analysis
    print("\n📰 Fetching news analysis...")
    news_fetcher = StockNewsFetcher()
    news_analysis = news_fetcher.fetch_all_news(symbol, days=7)
    
    # Combined analysis
    combined_analysis = {
        'symbol': symbol,
        'analysis_date': datetime.now().isoformat(),
        'stock_analysis': stock_analysis,
        'news_analysis': news_analysis,
        'integrated_insights': generate_integrated_insights(stock_analysis, news_analysis)
    }
    
    return combined_analysis

def generate_integrated_insights(stock_data: Dict, news_data: Dict) -> Dict:
    """
    Generate insights by combining stock technical analysis with news sentiment
    """
    insights = {
        'correlation_analysis': [],
        'risk_factors': [],
        'opportunity_factors': [],
        'overall_recommendation': 'Hold'  # Default
    }
    
    # Get stock trend
    stock_trend = stock_data.get('Trend', 'Unknown')
    price_change = stock_data.get('Price Change %', 0)
    
    # Get news sentiment
    news_sentiment = 'Neutral'
    if 'sentiment_summary' in news_data:
        news_sentiment = news_data['sentiment_summary'].get('overall_sentiment', 'Neutral')
    
    # Correlation analysis
    if 'Bullish' in news_sentiment and ('Uptrend' in stock_trend or price_change > 0):
        insights['correlation_analysis'].append("✅ Positive news sentiment aligns with bullish technical indicators")
        insights['opportunity_factors'].append("Strong fundamental-technical alignment")
    elif 'Bearish' in news_sentiment and ('Downtrend' in stock_trend or price_change < 0):
        insights['correlation_analysis'].append("⚠️ Negative sentiment confirms bearish technical signals")
        insights['risk_factors'].append("Both news and technicals showing weakness")
    elif 'Bullish' in news_sentiment and 'Downtrend' in stock_trend:
        insights['correlation_analysis'].append("🔍 Divergence: Positive news vs negative technical trend - potential reversal signal")
        insights['opportunity_factors'].append("News-driven potential reversal opportunity")
    elif 'Bearish' in news_sentiment and 'Uptrend' in stock_trend:
        insights['correlation_analysis'].append("⚠️ Divergence: Negative news vs positive technical trend - monitor for weakness")
        insights['risk_factors'].append("Negative sentiment may pressure current uptrend")
    
    # Generate recommendation based on combined factors
    bullish_factors = len(insights['opportunity_factors'])
    bearish_factors = len(insights['risk_factors'])
    
    if bullish_factors > bearish_factors:
        insights['overall_recommendation'] = 'Buy/Long'
    elif bearish_factors > bullish_factors:
        insights['overall_recommendation'] = 'Sell/Short'
    else:
        insights['overall_recommendation'] = 'Hold/Monitor'
    
    return insights

def print_integrated_analysis(combined_data: Dict):
    """Print comprehensive analysis combining stock and news data"""
    print("=" * 120)
    print(f"🎯 INTEGRATED STOCK & NEWS ANALYSIS: {combined_data['symbol']}")
    print("=" * 120)
    
    # Stock analysis summary
    stock_data = combined_data['stock_analysis']
    if 'Error' not in stock_data:
        trend = stock_data.get('Trend', 'Unknown')
        price_change = stock_data.get('Price Change %', 0)
        trend_emoji = {"Strong Uptrend": "🚀", "Uptrend": "📈", "Sideways": "➡️", 
                      "Downtrend": "📉", "Strong Downtrend": "💥"}.get(trend, "❓")
        
        print(f"\n📊 TECHNICAL ANALYSIS:")
        print(f"   {trend_emoji} Trend: {trend}")
        print(f"   💰 Price Change: {price_change:+.2f}%")
        print(f"   💵 Current Price: ${stock_data.get('Close', 'N/A')}")
    
    # News analysis summary
    news_data = combined_data['news_analysis']
    if news_data.get('total_articles', 0) > 0:
        sentiment = news_data.get('sentiment_summary', {}).get('overall_sentiment', 'Neutral')
        sentiment_emoji = {"Bullish": "📈", "Bearish": "📉", "Neutral": "➡️"}.get(sentiment, "❓")
        
        print(f"\n📰 NEWS ANALYSIS:")
        print(f"   {sentiment_emoji} News Sentiment: {sentiment}")
        print(f"   📊 Total Articles: {news_data['total_articles']}")
        print(f"   📅 Last 7 Days Coverage")
    
    # Integrated insights
    insights = combined_data.get('integrated_insights', {})
    if insights:
        print(f"\n🎯 INTEGRATED INSIGHTS:")
        print(f"   🎖️ Overall Recommendation: {insights.get('overall_recommendation', 'Hold')}")
        
        if insights.get('correlation_analysis'):
            print(f"\n   🔍 Correlation Analysis:")
            for analysis in insights['correlation_analysis']:
                print(f"      {analysis}")
        
        if insights.get('opportunity_factors'):
            print(f"\n   ✅ Opportunity Factors:")
            for factor in insights['opportunity_factors']:
                print(f"      • {factor}")
        
        if insights.get('risk_factors'):
            print(f"\n   ⚠️ Risk Factors:")
            for factor in insights['risk_factors']:
                print(f"      • {factor}")
    
    print("\n" + "=" * 120)

# Usage examples and main execution
if __name__ == "__main__":
    # Example usage
    
    # Initialize news fetcher
    news_fetcher = StockNewsFetcher()
    
    # Example 1: Fetch news for a single stock
    symbol = "AAPL"
    news_data = news_fetcher.fetch_all_news(symbol, days=7)
    news_fetcher.print_news_summary(news_data)
    
    # Example 2: Save news data
    # news_fetcher.save_news_to_json(news_data)
    
    # Example 3: Create DataFrame for analysis
    # df = news_fetcher.create_news_dataframe(news_data)
    # print(df.head())
    
    # Example 4: Integrated analysis
    # combined_analysis = integrate_news_with_stock_analysis("AAPL", "1d")
    # print_integrated_analysis(combined_analysis)
    
    print("\n🎯 SETUP INSTRUCTIONS:")
    print("1. Get API keys from:")
    print("   • NewsAPI: https://newsapi.org/ (Free: 100 requests/day)")
    print("   • Alpha Vantage: https://www.alphavantage.co/ (Free: 25 requests/day)")
    print("   • Finnhub: https://finnhub.io/ (Free: 60 calls/minute)")
    print("2. Replace 'YOUR_API_KEY' placeholders with actual keys")
    print("3. Yahoo Finance works without API key")
    print("\n⚠️ Note: Some APIs may require registration and have rate limits")