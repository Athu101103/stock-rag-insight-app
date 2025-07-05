import os
import requests
import json
from typing import Dict, List
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
import streamlit as st
import pandas as pd
import torch
import plotly.express as px
import plotly.graph_objects as go
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime
import yfinance as yf
import numpy as np

# Load environment variables
load_dotenv()

# Configuration
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000/analyze_stock")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
NEWS_API_URL = os.getenv("NEWS_API_URL", "http://localhost:8000/fetch_stock_news")

# Initialize session state
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = None
if 'symbol' not in st.session_state:
    st.session_state.symbol = ""
if 'period' not in st.session_state:
    st.session_state.period = "1d"
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = None

# Initialize ChromaDB client
try:
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection("stock_analysis")
except Exception as e:
    st.error(f"Failed to initialize ChromaDB: {e}")
    collection = None

# Initialize embedding model
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
except Exception as e:
    st.error(f"Failed to initialize embedding model: {e}")
    embedding_model = None

# --- Utility Functions from stock_insights.py ---
def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    if len(prices) < period + 1:
        return pd.Series([None] * len(prices), index=prices.index)
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    if len(prices) < slow:
        return pd.Series([None] * len(prices), index=prices.index), pd.Series([None] * len(prices), index=prices.index), pd.Series([None] * len(prices), index=prices.index)
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: int = 2) -> tuple:
    if len(prices) < period:
        return pd.Series([None] * len(prices), index=prices.index), pd.Series([None] * len(prices), index=prices.index), pd.Series([None] * len(prices), index=prices.index)
    rolling_mean = prices.rolling(window=period).mean()
    rolling_std = prices.rolling(window=period).std()
    upper_band = rolling_mean + (rolling_std * std_dev)
    lower_band = rolling_mean - (rolling_std * std_dev)
    return upper_band, rolling_mean, lower_band

def fetch_historical_data(symbol: str, period: str) -> pd.DataFrame:
    try:
        interval = "5m" if period == "1d" else "15m"
        stock = yf.Ticker(symbol)
        df = stock.history(period=period, interval=interval)
        if df.empty:
            return None
        df = df.reset_index().rename(columns={'Date': 'Datetime'})
        df['Date'] = df['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df['RSI'] = calculate_rsi(df['Close'])
        macd, signal_line, _ = calculate_macd(df['Close'])
        df['MACD'] = macd
        df['MACD Signal'] = signal_line
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        upper, _, lower = calculate_bollinger_bands(df['Close'])
        df['BB Upper'] = upper
        df['BB Lower'] = lower
        df = df.dropna()
        return df
    except Exception as e:
        st.error(f"Failed to fetch historical data: {str(e)}")
        return None

# --- Utility Functions ---
def fetch_stock_analysis(symbol: str, period: str = "1d") -> Dict:
    try:
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1.0, status_forcelist=[500, 502, 503, 504])
        session.mount('http://', HTTPAdapter(max_retries=retries))
        session.mount('https://', HTTPAdapter(max_retries=retries))
        response = session.get(f"{FASTAPI_URL}?symbol={symbol}&period={period}", timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": f"Failed to fetch analysis: {str(e)}"}

def extract_chunks(analysis_json: Dict) -> List[str]:
    chunks = []
    if not analysis_json or "error" in analysis_json:
        return chunks
    for key, value in analysis_json.items():
        if isinstance(value, (int, float, str)):
            chunks.append(f"{key.replace('_', ' ').title()}: {value}")
        elif isinstance(value, list):
            chunks.append(f"{key.replace('_', ' ').title()}: {', '.join(str(item) for item in value[:3]) + '...' if len(value) > 3 else ', '.join(str(item) for item in value)}")
        elif isinstance(value, dict):
            chunks.append(f"{key.replace('_', ' ').title()}: {json.dumps(value, indent=2)}")
    return chunks

def embed_chunks(chunks: List[str]) -> List[List[float]]:
    if not chunks or not embedding_model:
        return []
    return embedding_model.encode(chunks, convert_to_tensor=False).tolist()

def store_embeddings(chunks: List[str], embeddings: List[List[float]], symbol: str):
    if not chunks or not embeddings or not collection:
        return
    try:
        collection.upsert(
            documents=chunks,
            embeddings=embeddings,
            metadatas=[{"symbol": symbol} for _ in chunks],
            ids=[f"{symbol}_{i}" for i in range(len(chunks))]
        )
    except Exception as e:
        st.error(f"Failed to store embeddings: {e}")

def rag_query(user_question: str, symbol: str) -> str:
    """Process user query using Cohere API."""
    if not COHERE_API_KEY:
        return "COHERE_API_KEY is not configured."
    if not collection or not embedding_model:
        return "Required components not initialized."
    
    query_embedding = embedding_model.encode([user_question])[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        where={"symbol": symbol}
    )
    retrieved_chunks = results["documents"][0] if results["documents"] else []

    if not retrieved_chunks:
        return "No relevant analysis found for this symbol."

    prompt = f"""Based on the following technical analysis:
{'\n'.join(retrieved_chunks)}

Answer the question: {user_question}
Provide a concise and clear response."""

    try:
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1.0, status_forcelist=[500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        headers = {
            "Authorization": f"Bearer {COHERE_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "command",
            "prompt": prompt,
            "max_tokens": 500,
            "temperature": 0.7
        }
        response = session.post("https://api.cohere.ai/v1/generate", headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result.get("generations", [{}])[0].get("text", "No response generated.").strip()
    except requests.RequestException as e:
        return f"Error generating answer: {str(e)}"

def fetch_stock_news_enhanced(symbol: str, limit: int = 5) -> List[Dict]:
    if not NEWS_API_URL:
        return []
    try:
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1.0, status_forcelist=[500, 502, 503, 504])
        session.mount('http://', HTTPAdapter(max_retries=retries))
        response = session.get(f"{NEWS_API_URL}?symbol={symbol}&limit={limit}", timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("News Items", []) if isinstance(data, dict) else data if isinstance(data, list) else []
    except Exception as e:
        return []

# --- Chart Functions ---
def create_price_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close Price', line=dict(color='#1f77b4')))
    fig.update_layout(title="Close Price", xaxis_title="Date", yaxis_title="Price (USD)", template="plotly_white")
    return fig

def create_volume_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name='Volume', marker_color='#ff7f0e'))
    fig.update_layout(title="Trading Volume", xaxis_title="Date", yaxis_title="Volume", template="plotly_white")
    return fig

def create_rsi_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], mode='lines', name='RSI (14)', line=dict(color='#2ca02c')))
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    fig.update_layout(
        title="Relative Strength Index (RSI)",
        xaxis_title="Date",
        yaxis_title="RSI",
        template="plotly_white"
    )
    return fig

def create_macd_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], mode='lines', name='MACD', line=dict(color='#d62728')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD Signal'], mode='lines', name='Signal Line', line=dict(color='#9467bd')))
    fig.update_layout(title="MACD", xaxis_title="Date", yaxis_title="MACD", template="plotly_white")
    return fig

def create_ma_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MA5'], mode='lines', name='5-Period MA', line=dict(color='#1f77b4')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MA10'], mode='lines', name='10-Period MA', line=dict(color='#ff7f0e')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MA20'], mode='lines', name='20-Period MA', line=dict(color='#2ca02c')))
    fig.update_layout(title="Moving Averages", xaxis_title="Date", yaxis_title="Price (USD)", template="plotly_white")
    return fig

def create_bollinger_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close Price', line=dict(color='#1f77b4')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['BB Upper'], mode='lines', name='Upper Band', line=dict(color='#ff7f0e', dash='dash')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['BB Lower'], mode='lines', name='Lower Band', line=dict(color='#ff7f0e', dash='dash')))
    fig.update_layout(title="Bollinger Bands", xaxis_title="Date", yaxis_title="Price (USD)", template="plotly_white")
    return fig

# --- UI Components ---
def display_analysis_and_charts():
    if not st.session_state.analysis_data or "error" in st.session_state.analysis_data:
        st.warning("No analysis data available. Please load analysis first.")
        return

    st.subheader(f"🔍 Analysis for {st.session_state.symbol} ({st.session_state.period})")
    
    metrics = {
        "Open": str(st.session_state.analysis_data.get("Open", "N/A")),
        "Close": str(st.session_state.analysis_data.get("Close", "N/A")),
        "High": str(st.session_state.analysis_data.get("High", "N/A")),
        "Low": str(st.session_state.analysis_data.get("Low", "N/A")),
        "Volume": str(st.session_state.analysis_data.get("Volume", "N/A")),
        "Price Change %": str(st.session_state.analysis_data.get("Price Change %", "N/A")),
        "5-Period MA": str(st.session_state.analysis_data.get("5-Period MA", "N/A")),
        "10-Period MA": str(st.session_state.analysis_data.get("10-Period MA", "N/A")),
        "20-Period MA": str(st.session_state.analysis_data.get("20-Period MA", "N/A")),
        "RSI (14)": str(st.session_state.analysis_data.get("RSI (14)", "N/A")),
        "MACD": str(st.session_state.analysis_data.get("MACD", "N/A")),
        "Trend": str(st.session_state.analysis_data.get("Trend", "N/A"))
    }
    
    df = pd.DataFrame({
        "Metric": metrics.keys(),
        "Value": metrics.values()
    })
    
    st.dataframe(df, use_container_width=True)
    
    st.markdown("""
    **Quick Guide to Terms:**
    - *RSI (14)*: Momentum indicator (0-100); 30-70 is neutral, <30 is oversold, >70 is overbought.
    - *MACD*: Trend indicator; positive values suggest bullish momentum, negative suggest bearish.
    - *Trend*: Shows price direction (e.g., Uptrend, Downtrend).
    - *Price Change %*: Percentage change in price over the period.
    """)

    # Charts
    st.subheader("📊 Charts")
    if st.session_state.historical_data is None or st.session_state.historical_data.empty:
        st.warning("No historical data available for charts.")
        return

    df = st.session_state.historical_data
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_price_chart(df), use_container_width=True)
        st.plotly_chart(create_rsi_chart(df), use_container_width=True)
        st.plotly_chart(create_ma_chart(df), use_container_width=True)
    with col2:
        st.plotly_chart(create_volume_chart(df), use_container_width=True)
        st.plotly_chart(create_macd_chart(df), use_container_width=True)
        st.plotly_chart(create_bollinger_chart(df), use_container_width=True)
        
def display_news():
    st.subheader("🗞️ Latest News")
    col1, col2 = st.columns([3, 1])
    with col1:
        limit = st.selectbox("Number of articles", [3, 5, 10], index=1, key="news_limit")
    with col2:
        st.markdown(
            """
            <style>
            .news-button {
                margin-top: 8px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.button("📰 Fetch News", type="primary", key="fetch_news", help="Fetch latest news articles")
        news = fetch_stock_news_enhanced(st.session_state.symbol, limit)
        if not news:
            st.warning("No news articles found.")
            return
        
        for i, article in enumerate(news, 1):
            title = article.get('title', 'No title')
            with st.expander(f"📄 {i}. {title[:80]}{'...' if len(title) > 80 else ''}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Source:** {article.get('source', 'Unknown')}")
                    st.markdown(f"**Published:** {article.get('published_at', 'N/A')}")
                with col2:
                    sentiment = article.get('sentiment_label', 'Unknown').lower()
                    sentiment_icon = {'positive': '🟢', 'negative': '🔴', 'neutral': '🟡'}.get(sentiment, '⚪')
                    st.markdown(f"**Sentiment:** {sentiment_icon} {sentiment.title()}")
                st.markdown(f"**Summary:** {article.get('summary', 'No summary available')}")
                if url := article.get('url'):
                    st.markdown(f"[Read Full Article]({url})")

def display_query():
    st.subheader("💬 Ask a Question")
    question = st.text_input("Your question", placeholder="e.g., Is this stock overbought?", key="question_input")
    if question and st.session_state.symbol:
        with st.spinner("Generating answer..."):
            answer = rag_query(question, st.session_state.symbol)
            st.markdown("**Answer:**")
            st.write(answer)

# --- Main App ---
def main():
    st.set_page_config(page_title="Stock Insights", layout="wide")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        st.session_state.symbol = st.text_input("Stock Symbol", 
                                             placeholder="e.g., AAPL, MSFT",
                                             value=st.session_state.symbol).upper()
        st.session_state.period = st.selectbox("Analysis Period", 
                                             ["1d", "5d", "1mo", "3mo", "6mo", "1y", "ytd"],
                                             index=["1d", "5d", "1mo", "3mo", "6mo", "1y", "ytd"].index(st.session_state.period))
        st.markdown(f"**Last updated:** {datetime.now().strftime('%I:%M %p IST, %B %d, %Y')}")
        
        if st.button("🔍 Load Analysis", type="primary"):
            if not st.session_state.symbol:
                st.error("Please enter a stock symbol")
            else:
                with st.spinner("Loading analysis..."):
                    st.session_state.analysis_data = fetch_stock_analysis(st.session_state.symbol, st.session_state.period)
                    st.session_state.historical_data = fetch_historical_data(st.session_state.symbol, st.session_state.period)
                    if "error" not in st.session_state.analysis_data:
                        st.success(f"Analysis loaded for {st.session_state.symbol}")
                        if st.session_state.historical_data is not None:
                            # Store embeddings
                            chunks = extract_chunks(st.session_state.analysis_data)
                            embeddings = embed_chunks(chunks)
                            store_embeddings(chunks, embeddings, st.session_state.symbol)
                    else:
                        st.error(f"Failed to load analysis: {st.session_state.analysis_data['error']}")

    # Main content
    st.title("📈 Stock Insights")
    st.markdown("A RAG-based app for stock analysis, news, and Q&A")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Analysis & Charts", "🗞️ News", "💬 Query"])
    
    with tab1:
        display_analysis_and_charts()
    with tab2:
        if st.session_state.symbol:
            display_news()
        else:
            st.info("Please enter a stock symbol and load analysis to view news.")
    with tab3:
        if st.session_state.symbol:
            display_query()
        else:
            st.info("Please enter a stock symbol and load analysis to ask questions.")

if __name__ == "__main__":
    main()
