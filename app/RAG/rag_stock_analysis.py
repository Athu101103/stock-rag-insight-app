import os
import requests
import json
from typing import Dict, List
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
import streamlit as st

# Load environment variables
load_dotenv()

# Configuration
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000/analyze_stock")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY not set in .env")

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("stock_analysis")

# Initialize embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def fetch_stock_analysis(symbol: str, period: str = "1d") -> Dict:
    """Fetch stock analysis data from FastAPI endpoint."""
    try:
        response = requests.get(f"{FASTAPI_URL}?symbol={symbol}&period={period}")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Error fetching analysis for {symbol}: {e}")
        return {"Error": str(e)}

def extract_chunks(analysis_json: Dict) -> List[str]:
    """Extract flat text chunks from JSON analysis."""
    chunks = []
    if not analysis_json or "Error" in analysis_json:
        return chunks

    for key, value in analysis_json.items():
        if isinstance(value, (int, float, str)):
            chunks.append(f"{key.capitalize()}: {value}")
        elif isinstance(value, list):
            chunks.append(f"{key.capitalize()}: {', '.join(str(item) for item in value[:3]) + '...' if len(value) > 3 else ', '.join(str(item) for item in value)}")
        elif isinstance(value, dict):
            chunks.append(f"{key.capitalize()}: {json.dumps(value, indent=2)}")
    return chunks

def embed_chunks(chunks: List[str]) -> List[List[float]]:
    """Generate embeddings for text chunks."""
    if not chunks:
        return []
    return embedding_model.encode(chunks, convert_to_tensor=False).tolist()

def store_embeddings(chunks: List[str], embeddings: List[List[float]], symbol: str):
    """Store embeddings in ChromaDB with metadata."""
    if not chunks or not embeddings:
        return
    collection.upsert(
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"symbol": symbol} for _ in chunks],
        ids=[f"{symbol}_{i}" for i in range(len(chunks))]
    )

def rag_query(user_question: str, symbol: str) -> str:
    """Process user query using Cohere API."""
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
        response = requests.post("https://api.cohere.ai/v1/generate", headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result.get("generations")[0].get("text", "No response generated.")
    except requests.RequestException as e:
        return f"Error generating answer: {str(e)}"

# Streamlit UI
def main():
    st.title("Stock Analysis Q&A")
    st.write("Explore stock technical analysis by entering a stock symbol and asking questions.")

    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        symbol = st.text_input("Stock Symbol", placeholder="e.g., AAPL, MSFT, GOOGL", value="").upper()
        period = st.selectbox("Period", options=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"], index=0)
        st.write("Last updated: 04:50 PM IST, June 20, 2025")

    # Main content
    if st.button("Load Analysis", key="load_button"):
        if not symbol:
            st.error("Please enter a stock symbol.")
        else:
            with st.spinner("Fetching and processing analysis..."):
                analysis = fetch_stock_analysis(symbol, period)
                if "Error" not in analysis:
                    # Display summary table
                    st.subheader(f"Analysis for {symbol} ({period})")
                    data = {
                        "Metric": ["Open", "Close", "High", "Low", "Volume", "Price Change %", "5-Period MA", "10-Period MA", "20-Period MA", "RSI (14)", "MACD", "Trend"],
                        "Value": [
                            str(analysis.get("Open", "N/A")),
                            str(analysis.get("Close", "N/A")),
                            str(analysis.get("High", "N/A")),
                            str(analysis.get("Low", "N/A")),
                            str(analysis.get("Volume", "N/A")),
                            str(analysis.get("Price Change %", "N/A")),
                            str(analysis.get("5-Period MA", "N/A")),
                            str(analysis.get("10-Period MA", "N/A")),
                            str(analysis.get("20-Period MA", "N/A")),
                            str(analysis.get("RSI (14)", "N/A")),
                            str(analysis.get("MACD", "N/A")),
                            str(analysis.get("Trend", "N/A"))
                        ]
                    }
                    st.table(data)

                    # Quick guide
                    st.write("""
                    **Quick Guide to Terms:**
                    - *RSI (14)*: Momentum indicator (0-100); 30-70 is neutral, below 30 is oversold, above 70 is overbought.
                    - *MACD*: Trend indicator; positive values suggest bullish momentum, negative values suggest bearish.
                    - *Trend*: Shows price direction (e.g., Uptrend, Downtrend).
                    - *Price Change %*: Percentage change in price over the period.
                    Ask a question to learn more!
                    """)

                    # Store embeddings
                    chunks = extract_chunks(analysis)
                    embeddings = embed_chunks(chunks)
                    store_embeddings(chunks, embeddings, symbol)
                    st.success(f"Analysis loaded for {symbol}!")
                else:
                    st.error(f"Failed to load analysis: {analysis['Error']}")

    # Question input
    user_question = st.text_input("Your Question", placeholder="E.g., Is AAPL overbought?", key="question_input")
    if user_question:
        with st.spinner("Generating answer..."):
            answer = rag_query(user_question, symbol)
            st.markdown("**Answer:**")
            st.write(answer)

if __name__ == "__main__":
    main()