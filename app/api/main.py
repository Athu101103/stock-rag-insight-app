from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
from typing import Dict, Any, List
from app.data.stock_insights import fetch_and_analyze_stock, load_and_analyze_csv, batch_analyze_stocks, get_available_periods, fetch_stock_news

app = FastAPI(
    title="Advanced Stock Analysis API",
    description="API for performing advanced stock intraday data analysis with technical indicators, trading signals, and stock news.",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Advanced Stock Analysis API",
        "endpoints": {
            "/analyze_stock": "Analyze a single stock via Yahoo Finance (GET)",
            "/analyze_csv": "Analyze a stock by uploading a CSV file (POST)",
            "/compare_stocks": "Compare multiple stocks via Yahoo Finance (GET)",
            "/available_periods": "List available periods for Yahoo Finance data (GET)",
            "/fetch_stock_news": "Fetch recent news for a stock via Alpha Vantage (GET)"
        },
        "disclaimer": "This analysis is for educational purposes only. Not financial advice."
    }

@app.get("/analyze_stock")
async def analyze_stock(symbol: str, period: str = "1d"):
    if not symbol:
        raise HTTPException(status_code=400, detail="Stock symbol is required.")
    
    available_periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
    if period not in available_periods:
        raise HTTPException(status_code=400, detail=f"Invalid period. Available periods: {', '.join(available_periods)}")
    
    results = fetch_and_analyze_stock(symbol, period)
    if "Error" in results:
        raise HTTPException(status_code=400, detail=results["Error"])
    
    return results

@app.post("/analyze_csv")
async def analyze_csv(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV.")
    
    content = await file.read()
    results = load_and_analyze_csv(content, file.filename)
    if "Error" in results:
        raise HTTPException(status_code=400, detail=results["Error"])
    
    return results

@app.get("/compare_stocks")
async def compare_stocks(symbols: str, period: str = "1d"):
    if not symbols:
        raise HTTPException(status_code=400, detail="At least one stock symbol is required.")
    
    available_periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
    if period not in available_periods:
        raise HTTPException(status_code=400, detail=f"Invalid period. Available periods: {', '.join(available_periods)}")
    
    symbol_list = [s.strip().upper() for s in symbols.split(",")]
    if not symbol_list:
        raise HTTPException(status_code=400, detail="Invalid symbol list format.")
    
    results = batch_analyze_stocks(symbol_list, period)
    
    comparison_data = []
    metrics = ["Close", "Price Change %", "RSI (14)", "Trend", "Volatility (20-day)"]
    
    for symbol, data in results.items():
        if "Error" not in data:
            row = {"Symbol": symbol}
            for metric in metrics:
                if metric in data:
                    row[metric] = data[metric]
                else:
                    row[metric] = "N/A"
            comparison_data.append(row)
        else:
            comparison_data.append({"Symbol": symbol, "Error": data["Error"]})
    
    return {
        "Comparison Table": comparison_data,
        "Detailed Results": results
    }

@app.get("/available_periods")
async def get_periods():
    return get_available_periods()

@app.get("/fetch_stock_news")
async def get_stock_news(symbol: str, limit: int = 5):
    if not symbol:
        raise HTTPException(status_code=400, detail="Stock symbol is required.")
    
    if limit < 1 or limit > 50:
        raise HTTPException(status_code=400, detail="Limit must be between 1 and 50.")
    
    results = fetch_stock_news(symbol, limit)
    if isinstance(results, list) and len(results) > 0 and "Error" in results[0]:
        raise HTTPException(status_code=400, detail=results[0]["Error"])
    
    return {
        "Symbol": symbol.upper(),
        "News Items": results
    }