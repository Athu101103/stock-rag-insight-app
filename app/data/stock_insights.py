import pandas as pd
import yfinance as yf
import numpy as np
import warnings
from io import BytesIO
from typing import Dict, Any, List, Tuple
import requests
from tenacity import retry, stop_after_attempt, wait_fixed
import logging
import re
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Replace with your actual Alpha Vantage API key
ALPHA_VANTAGE_API_KEY = "L5Z5ABID5UOQ744Q"  # Replace with your API key

warnings.filterwarnings('ignore')

# Helper function to convert NumPy types to Python native types
def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    return obj

def clean_and_convert(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(filter(None, map(str, col))).strip() for col in df.columns]
    else:
        df.columns = [str(col).strip() for col in df.columns]

    if df.index.name and ('date' in df.index.name.lower() or 'time' in df.index.name.lower()):
        df = df.reset_index()

    time_col = None
    for col in df.columns:
        if 'time' in col.lower() or 'date' in col.lower():
            time_col = col
            break

    if time_col is None:
        raise KeyError("No datetime column found in data.")

    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df = df.dropna(subset=[time_col])
    df = df.sort_values(time_col)
    df.rename(columns={time_col: "Datetime"}, inplace=True)

    return df

def detect_csv_format(df: pd.DataFrame) -> str:
    columns = [col.lower() for col in df.columns]
    
    has_ohlcv = all(any(required in col for col in columns) 
                   for required in ['open', 'high', 'low', 'close', 'volume'])
    
    if has_ohlcv:
        return 'ohlcv'
    else:
        price_cols = [col for col in df.columns 
                     if not any(time_word in col.lower() for time_word in ['date', 'time'])
                     and df[col].dtype in ['float64', 'int64', 'object']]
        if price_cols:
            return 'single_price'
    
    return 'unknown'

def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    if len(prices) < period + 1:
        return None
    
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else None

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
    if len(prices) < slow:
        return None, None, None
    
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    
    return macd_line.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]

def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[float, float, float]:
    if len(prices) < period:
        return None, None, None
    
    rolling_mean = prices.rolling(window=period).mean()
    rolling_std = prices.rolling(window=period).std()
    
    upper_band = rolling_mean + (rolling_std * std_dev)
    lower_band = rolling_mean - (rolling_std * std_dev)
    
    return upper_band.iloc[-1], rolling_mean.iloc[-1], lower_band.iloc[-1]

def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[float, float]:
    if len(close) < k_period:
        return None, None
    
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_period).mean()
    
    return k_percent.iloc[-1], d_percent.iloc[-1]

def detect_trend_with_reasoning(prices: pd.Series, period: int = 10) -> Tuple[str, str]:
    if len(prices) < period:
        return "Insufficient Data", "Not enough data points for trend analysis"
    
    recent_prices = prices.tail(period).reset_index(drop=True)
    x = np.arange(len(recent_prices))
    
    slope = np.polyfit(x, recent_prices, 1)[0]
    
    price_change = recent_prices.iloc[-1] - recent_prices.iloc[0]
    price_change_pct = (price_change / recent_prices.iloc[0]) * 100
    
    ma_5 = prices.tail(5).mean() if len(prices) >= 5 else prices.mean()
    ma_10 = prices.tail(10).mean() if len(prices) >= 10 else prices.mean()
    current_price = prices.iloc[-1]
    
    consecutive_ups = 0
    consecutive_downs = 0
    price_diffs = recent_prices.diff().dropna()
    
    for diff in price_diffs:
        if diff > 0:
            consecutive_ups += 1
            consecutive_downs = 0
        elif diff < 0:
            consecutive_downs += 1
            consecutive_ups = 0
    
    if slope > 0.1:
        trend = "Strong Uptrend"
        reasoning = f"Price shows strong upward momentum with slope of {slope:.4f}. "
        reasoning += f"Price increased by {price_change_pct:.2f}% over {period} periods. "
        if current_price > ma_5 > ma_10:
            reasoning += "Price is above short and long-term moving averages, confirming bullish momentum."
        if consecutive_ups >= 3:
            reasoning += f" Stock has {consecutive_ups} consecutive positive movements."
    elif slope > 0.05:
        trend = "Uptrend"
        reasoning = f"Price shows moderate upward trend with slope of {slope:.4f}. "
        reasoning += f"Price increased by {price_change_pct:.2f}% over {period} periods. "
        if current_price > ma_5:
            reasoning += "Current price is above short-term moving average, supporting the upward trend."
    elif slope > -0.05:
        trend = "Sideways"
        reasoning = f"Price is moving sideways with minimal slope of {slope:.4f}. "
        reasoning += f"Price changed by only {price_change_pct:.2f}% over {period} periods, indicating consolidation."
        reasoning += "Market is likely in a consolidation phase with no clear direction."
    elif slope > -0.1:
        trend = "Downtrend"
        reasoning = f"Price shows moderate downward trend with slope of {slope:.4f}. "
        reasoning += f"Price decreased by {abs(price_change_pct):.2f}% over {period} periods. "
        if current_price < ma_5:
            reasoning += "Current price is below short-term moving average, confirming bearish pressure."
    else:
        trend = "Strong Downtrend"
        reasoning = f"Price shows strong downward momentum with slope of {slope:.4f}. "
        reasoning += f"Price declined by {abs(price_change_pct):.2f}% over {period} periods. "
        if current_price < ma_5 < ma_10:
            reasoning += "Price is below both short and long-term moving averages, confirming bearish trend."
        if consecutive_downs >= 3:
            reasoning += f" Stock has {consecutive_downs} consecutive negative movements."
    
    return trend, reasoning

def generate_detailed_signals(df: pd.DataFrame, close_col: str) -> Tuple[List[str], List[str]]:
    signals = []
    explanations = []
    
    if len(df) < 20:
        return ["Insufficient data for signal generation"], ["Need at least 20 data points for reliable technical analysis"]
    
    prices = df[close_col]
    current_price = prices.iloc[-1]
    
    rsi = calculate_rsi(prices)
    if rsi:
        if rsi < 30:
            signals.append("RSI Oversold - Potential BUY signal")
            explanations.append(f"RSI at {rsi:.1f} indicates oversold conditions. Historically, RSI below 30 suggests the stock may be due for a bounce as selling pressure may be exhausted.")
        elif rsi > 70:
            signals.append("RSI Overbought - Potential SELL signal")
            explanations.append(f"RSI at {rsi:.1f} indicates overbought conditions. RSI above 70 suggests the stock may be overextended and due for a pullback.")
        else:
            signals.append("RSI Neutral")
            explanations.append(f"RSI at {rsi:.1f} is in neutral territory (30-70), indicating balanced buying and selling pressure.")
    
    macd, signal_line, histogram = calculate_macd(prices)
    if macd:
        if macd > signal_line and histogram > 0:
            signals.append("MACD Bullish - Potential BUY signal")
            explanations.append(f"MACD line ({macd:.4f}) is above signal line ({signal_line:.4f}) with positive histogram ({histogram:.4f}). This indicates bullish momentum as the fast moving average is pulling away from the slow one.")
        elif macd < signal_line and histogram < 0:
            signals.append("MACD Bearish - Potential SELL signal")
            explanations.append(f"MACD line ({macd:.4f}) is below signal line ({signal_line:.4f}) with negative histogram ({histogram:.4f}). This indicates bearish momentum as the faster moving average is diverging downward.")
        else:
            signals.append("MACD Neutral")
            explanations.append(f"MACD shows mixed signals with line at {macd:.4f} and signal line at {signal_line:.4f}. Momentum is unclear, suggesting a wait-and-see approach.")
    
    upper, middle, lower = calculate_bollinger_bands(prices)
    if upper and middle and lower:
        bb_position = ((current_price - lower) / (upper - lower)) * 100 if (upper - lower) != 0 else 50
        if current_price > upper:
            signals.append("Price above Bollinger Upper Band - Potential SELL signal")
            explanations.append(f"Price (${current_price:.2f}) is above the upper Bollinger Band (${upper:.2f}), suggesting the stock is statistically overbought. Band position: {bb_position:.1f}%")
        elif current_price < lower:
            signals.append("Price below Bollinger Lower Band - Potential BUY signal")
            explanations.append(f"Price (${current_price:.2f}) is below the lower Bollinger Band (${lower:.2f}), suggesting the stock is statistically oversold. Band position: {bb_position:.1f}%")
        else:
            signals.append("Price within Bollinger Bands")
            explanations.append(f"Price is within normal trading range. Position in band: {bb_position:.1f}% (0% = lower band, 100% = upper band)")
    
    if len(prices) >= 20:
        ma_5 = prices.rolling(5).mean().iloc[-1]
        ma_20 = prices.rolling(20).mean().iloc[-1]
        ma_diff_pct = ((ma_5 - ma_20) / ma_20) * 100 if ma_20 != 0 else 0
        
        if ma_5 > ma_20:
            signals.append("Short MA above Long MA - Bullish signal")
            explanations.append(f"5-period MA (${ma_5:.2f}) is {ma_diff_pct:.2f}% above 20-period MA (${ma_20:.2f}). This crossover indicates short-term momentum is stronger than long-term trend, suggesting bullish sentiment.")
        else:
            signals.append("Short MA below Long MA - Bearish signal")
            explanations.append(f"5-period MA (${ma_5:.2f}) is {abs(ma_diff_pct):.2f}% below 20-period MA (${ma_20:.2f}). This indicates short-term weakness compared to long-term trend, suggesting bearish sentiment.")
    
    volume_col = None
    for col in df.columns:
        if 'volume' in col.lower():
            volume_col = col
            break
    
    if volume_col and len(df) >= 10:
        recent_volume = df[volume_col].tail(5).mean()
        avg_volume = df[volume_col].mean()
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
        
        if volume_ratio > 1.5:
            signals.append("High Volume Confirmation")
            explanations.append(f"Recent volume is {volume_ratio:.1f}x average volume, providing strong confirmation of current price movement. High volume validates the significance of price action.")
        elif volume_ratio < 0.5:
            signals.append("Low Volume Warning")
            explanations.append(f"Recent volume is only {volume_ratio:.1f}x average volume. Low volume suggests weak conviction in current price movement and signals may be less reliable.")
    
    return signals, explanations

def get_signal_summary_note() -> str:
    note = """
📝 **Understanding Trading Signals:**

• **BUY Signals**: Suggest potential upward price movement based on technical indicators
• **SELL Signals**: Indicate possible downward pressure or overbought conditions  
• **Neutral Signals**: Show balanced conditions with no clear directional bias

⚠️ **Important Reminders:**
• Multiple confirming signals increase reliability
• Consider overall market conditions and news
• Use proper risk management and position sizing
• Signals work best when combined with fundamental analysis
• Past performance doesn't guarantee future results

🎯 **Signal Strength**: Look for confluence - when multiple indicators align, the signal becomes more reliable.
    """
    return note

def compute_advanced_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    def find_col(name: str) -> str:
        for col in df.columns:
            if name.lower() == col.lower() or name.lower() in col.lower():
                return col
        return None

    open_col = find_col('open')
    close_col = find_col('close')
    high_col = find_col('high')
    low_col = find_col('low')
    volume_col = find_col('volume')

    if None in [open_col, close_col, high_col, low_col, volume_col]:
        return {"Error": "Missing one or more required columns: Open, Close, High, Low, Volume."}

    if df.empty:
        return {"Error": "DataFrame is empty, no data to compute statistics."}

    prices = df[close_col]
    highs = df[high_col]
    lows = df[low_col]
    
    stats = {
        "Open": round(df.iloc[0][open_col], 2),
        "Close": round(prices.iloc[-1], 2),
        "High": round(highs.max(), 2),
        "Low": round(lows.min(), 2),
        "Volume": round(df[volume_col].sum(), 2),
    }
    
    price_change = stats["Close"] - stats["Open"]
    price_change_pct = (price_change / stats["Open"] * 100) if stats["Open"] != 0 else 0
    stats["Price Change"] = round(price_change, 2)
    stats["Price Change %"] = round(price_change_pct, 2)
    
    if len(prices) >= 5:
        stats["5-Period MA"] = round(prices.rolling(window=5).mean().iloc[-1], 2)
    if len(prices) >= 10:
        stats["10-Period MA"] = round(prices.rolling(window=10).mean().iloc[-1], 2)
    if len(prices) >= 20:
        stats["20-Period MA"] = round(prices.rolling(window=20).mean().iloc[-1], 2)
        stats["20-Period EMA"] = round(prices.ewm(span=20).mean().iloc[-1], 2)
    
    rsi = calculate_rsi(prices)
    if rsi:
        stats["RSI (14)"] = round(rsi, 2)
    
    macd, signal_line, histogram = calculate_macd(prices)
    if macd:
        stats["MACD"] = round(macd, 4)
        stats["MACD Signal"] = round(signal_line, 4)
        stats["MACD Histogram"] = round(histogram, 4)
    
    upper, middle, lower = calculate_bollinger_bands(prices)
    if upper:
        stats["Bollinger Upper"] = round(upper, 2)
        stats["Bollinger Middle"] = round(middle, 2)
        stats["Bollinger Lower"] = round(lower, 2)
    
    if len(prices) >= 14:
        k_percent, d_percent = calculate_stochastic(highs, lows, prices)
        if k_percent:
            stats["Stochastic %K"] = round(k_percent, 2)
            stats["Stochastic %D"] = round(d_percent, 2)
    
    if len(prices) >= 20:
        returns = prices.pct_change().dropna()
        if len(returns) > 0:
            stats["Volatility (20-day)"] = round(returns.std() * np.sqrt(252) * 100, 2)
    
    trend, trend_reasoning = detect_trend_with_reasoning(prices)
    stats["Trend"] = trend
    stats["Trend Reasoning"] = trend_reasoning
    
    recent_highs = highs.tail(20)
    recent_lows = lows.tail(20)
    stats["Resistance Level"] = round(recent_highs.quantile(0.9), 2)
    stats["Support Level"] = round(recent_lows.quantile(0.1), 2)
    
    signals, explanations = generate_detailed_signals(df, close_col)
    stats["Trading Signals"] = signals
    stats["Explanations"] = explanations
    stats["Signal Summary Note"] = get_signal_summary_note()
    
    stats["Data Points"] = len(df)
    
    # Convert NumPy types to Python native types
    stats = convert_numpy_types(stats)
    
    return stats

def compute_single_price_advanced_stats(df: pd.DataFrame) -> Dict[str, Any]:
    price_cols = [col for col in df.columns 
                 if col != 'Datetime' and 
                 not any(time_word in col.lower() for time_word in ['date', 'time'])]
    
    if not price_cols:
        return {"Error": "No price columns found in data."} 
    
    for col in price_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    all_prices = []
    for idx, row in df.iterrows():
        prices = [row[col] for col in price_cols if pd.notna(row[col])]
        if prices:
            all_prices.append(prices[-1])
        else:
            all_prices.append(None)
    
    valid_prices = [p for p in all_prices if p is not None]
    
    if not valid_prices:
        return {"Error": "No valid price data found."}
    
    price_series = pd.Series(valid_prices)
    
    stats = {
        "Latest Price": round(price_series.iloc[-1], 2),
        "Highest Price": round(price_series.max(), 2),
        "Lowest Price": round(price_series.min(), 2),
        "Average Price": round(price_series.mean(), 2),
        "Median Price": round(price_series.median(), 2),
    }
    
    if len(price_series) > 1:
        first_price = price_series.iloc[0]
        last_price = price_series.iloc[-1]
        price_change = last_price - first_price
        price_change_pct = (price_change / first_price) * 100 if first_price != 0 else 0
        
        stats["Price Change"] = round(price_change, 2)
        stats["Price Change %"] = round(price_change_pct, 2)
    
    if len(price_series) >= 5:
        stats["5-Period MA"] = round(price_series.rolling(5).mean().iloc[-1], 2)
    if len(price_series) >= 10:
        stats["10-Period MA"] = round(price_series.rolling(10).mean().iloc[-1], 2)
    if len(price_series) >= 20:
        stats["20-Period MA"] = round(price_series.rolling(20).mean().iloc[-1], 2)
        stats["20-Period EMA"] = round(price_series.ewm(span=20).mean().iloc[-1], 2)
    
    rsi = calculate_rsi(price_series)
    if rsi:
        stats["RSI (14)"] = round(rsi, 2)
    
    macd, signal_line, histogram = calculate_macd(price_series)
    if macd:
        stats["MACD"] = round(macd, 4)
        stats["MACD Signal"] = round(signal_line, 4)
        stats["MACD Histogram"] = round(histogram, 4)
    
    upper, middle, lower = calculate_bollinger_bands(price_series)
    if upper:
        stats["Bollinger Upper"] = round(upper, 2)
        stats["Bollinger Middle"] = round(middle, 2)
        stats["Bollinger Lower"] = round(lower, 2)
    
    if len(price_series) >= 20:
        returns = price_series.pct_change().dropna()
        if len(returns) > 0:
            stats["Volatility (20-day)"] = round(returns.std() * np.sqrt(252) * 100, 2)
    
    trend, trend_reasoning = detect_trend_with_reasoning(price_series)
    stats["Trend"] = trend
    stats["Trend Reasoning"] = trend_reasoning
    
    if len(price_series) >= 20:
        recent_prices = price_series.tail(20)
        stats["Resistance Level"] = round(recent_prices.quantile(0.9), 2)
        stats["Support Level"] = round(recent_prices.quantile(0.1), 2)
    
    temp_df = pd.DataFrame({'close': price_series})
    signals, explanations = generate_detailed_signals(temp_df, 'close')
    stats["Trading Signals"] = signals
    stats["Explanations"] = explanations
    stats["Signal Summary Note"] = get_signal_summary_note()
    
    stats["Data Points"] = len(price_series)
    stats["Price Columns Analyzed"] = price_cols
    
    # Convert NumPy types to Python native types
    stats = convert_numpy_types(stats)
    
    return stats

def analyze_stock_data(data: pd.DataFrame, data_format: str = None) -> Dict[str, Any]:
    try:
        df = clean_and_convert(data.copy())
        
        if data_format is None:
            data_format = detect_csv_format(df)
        
        if data_format == 'ohlcv':
            return compute_advanced_statistics(df)
        elif data_format == 'single_price':
            return compute_single_price_advanced_stats(df)
        else:
            return {"Error": "Unable to detect valid data format. Ensure data contains either OHLCV columns or price columns with datetime."}
    
    except Exception as e:
        return {"Error": f"Analysis failed: {str(e)}"}

def fetch_and_analyze_stock(symbol: str, period: str = "1d") -> Dict[str, Any]:
    try:
        interval = "5m" if period == "1d" else "15m"
        stock = yf.Ticker(symbol)
        df = stock.history(period=period, interval=interval)
        
        if df.empty:
            return {"Error": f"No data found for symbol {symbol}. This might be due to market hours or incorrect period/interval combination."}
        
        results = analyze_stock_data(df, 'ohlcv')
        results["Symbol"] = symbol.upper()
        results["Period"] = period
        results["Interval Used"] = interval
        
        # Convert NumPy types to Python native types
        results = convert_numpy_types(results)
        
        return results
    
    except Exception as e:
        return {"Error": f"Failed to fetch data for {symbol}: {str(e)}"}

def load_and_analyze_csv(content: bytes, filename: str) -> Dict[str, Any]:
    try:
        df = None
        encodings = ['utf-8', 'latin-1', 'cp1252']
        separators = [',', ';', '\t']
        
        for encoding in encodings:
            for sep in separators:
                try:
                    df = pd.read_csv(BytesIO(content), encoding=encoding, sep=sep)
                    if df.shape[1] > 1:
                        break
                except:
                    continue
            if df is not None and df.shape[1] > 1:
                break
        
        if df is None or df.empty:
            return {"Error": "Could not load CSV file. Please check file format and encoding."}
        
        results = analyze_stock_data(df)
        results["Source"] = filename
        
        # Convert NumPy types to Python native types
        results = convert_numpy_types(results)
        
        return results
    
    except Exception as e:
        return {"Error": f"Failed to load CSV: {str(e)}"}

def batch_analyze_stocks(symbols: List[str], period: str = "1mo") -> Dict[str, Dict[str, Any]]:
    results = {}
    
    for symbol in symbols:
        try:
            result = fetch_and_analyze_stock(symbol, period)
            results[symbol] = convert_numpy_types(result)
        except Exception as e:
            results[symbol] = {"Error": f"Failed to analyze {symbol}: {str(e)}"}
    
    # Convert NumPy types to Python native types
    results = convert_numpy_types(results)
    
    return results

def get_available_periods() -> Dict[str, str]:
    return {
        "1d": "1 day",
        "5d": "5 days", 
        "1mo": "1 month",
        "3mo": "3 months",
        "6mo": "6 months",
        "1y": "1 year",
        "2y": "2 years",
        "5y": "5 years",
        "10y": "10 years",
        "ytd": "Year to date",
        "max": "Maximum available"
    }

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
            return [{"Error": "Rate limit exceeded. Please try again later."}]
        response.raise_for_status()
        data = response.json()
        if "Error Message" in data:
            logger.error(f"Alpha Vantage API error: {data['Error Message']}")
            return [{"Error": f"Failed to fetch news: {data['Error Message']}"}]
        if "Information" in data and "rate limit" in data["Information"].lower():
            logger.error("Alpha Vantage rate limit exceeded")
            return [{"Error": "Rate limit exceeded. Please try again later."}]
        if "feed" not in data or not data["feed"]:
            logger.warning(f"No news found for symbol {symbol}")
            return [{"Error": f"No news found for symbol {symbol}"}]
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
        return [{"Error": f"Failed to fetch news for {symbol}: {str(e)}"}]
    except ValueError as e:
        logger.error(f"Invalid symbol {symbol}: {str(e)}")
        return [{"Error": str(e)}]
    except Exception as e:
        logger.error(f"Unexpected error fetching news for {symbol}: {str(e)}")
        return [{"Error": f"Unexpected error: {str(e)}"}]

# FastAPI application
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