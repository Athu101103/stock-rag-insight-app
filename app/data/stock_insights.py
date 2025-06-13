import pandas as pd
import yfinance as yf
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def clean_and_convert(df):
    # Flatten multi-index columns if any (e.g. from yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(filter(None, map(str, col))).strip() for col in df.columns]
    else:
        df.columns = [str(col).strip() for col in df.columns]

    # Reset index if datetime is index
    if df.index.name and ('date' in df.index.name.lower() or 'time' in df.index.name.lower()):
        df = df.reset_index()

    # Find datetime column dynamically
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

def detect_csv_format(df):
    """Detect if CSV has OHLCV format or single price column"""
    columns = [col.lower() for col in df.columns]
    
    # Check for OHLCV columns
    has_ohlcv = all(any(required in col for col in columns) 
                   for required in ['open', 'high', 'low', 'close', 'volume'])
    
    if has_ohlcv:
        return 'ohlcv'
    else:
        # Look for price columns (excluding datetime)
        price_cols = [col for col in df.columns 
                     if not any(time_word in col.lower() for time_word in ['date', 'time'])
                     and df[col].dtype in ['float64', 'int64', 'object']]
        if price_cols:
            return 'single_price'
    
    return 'unknown'

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    if len(prices) < period + 1:
        return None
    
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # Avoid division by zero
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else None

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    if len(prices) < slow:
        return None, None, None
    
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    
    return macd_line.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    if len(prices) < period:
        return None, None, None
    
    rolling_mean = prices.rolling(window=period).mean()
    rolling_std = prices.rolling(window=period).std()
    
    upper_band = rolling_mean + (rolling_std * std_dev)
    lower_band = rolling_mean - (rolling_std * std_dev)
    
    return upper_band.iloc[-1], rolling_mean.iloc[-1], lower_band.iloc[-1]

def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator"""
    if len(close) < k_period:
        return None, None
    
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_period).mean()
    
    return k_percent.iloc[-1], d_percent.iloc[-1]

def detect_trend_with_reasoning(prices, period=10):
    """Detect trend direction with detailed reasoning"""
    if len(prices) < period:
        return "Insufficient Data", "Not enough data points for trend analysis"
    
    recent_prices = prices.tail(period).reset_index(drop=True)
    x = np.arange(len(recent_prices))
    
    # Calculate slope using numpy polyfit
    slope = np.polyfit(x, recent_prices, 1)[0]
    
    # Price change metrics
    price_change = recent_prices.iloc[-1] - recent_prices.iloc[0]
    price_change_pct = (price_change / recent_prices.iloc[0]) * 100
    
    # Moving averages comparison for confirmation
    ma_5 = prices.tail(5).mean() if len(prices) >= 5 else prices.mean()
    ma_10 = prices.tail(10).mean() if len(prices) >= 10 else prices.mean()
    current_price = prices.iloc[-1]
    
    # Trend strength indicators
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
    
    # Determine trend and reasoning
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
        reasoning += f"Price changed by only {price_change_pct:.2f}% over {period} periods, indicating consolidation. "
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

def generate_detailed_signals(df, close_col):
    """Generate buy/sell signals with detailed explanations"""
    signals = []
    explanations = []
    
    if len(df) < 20:
        return ["Insufficient data for signal generation"], ["Need at least 20 data points for reliable technical analysis"]
    
    prices = df[close_col]
    current_price = prices.iloc[-1]
    
    # RSI signals with explanation
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
            explanations.append(f"RSI at {rsi:.1f} is in neutral territory (30-70), indicating balanced buying and selling pressure with no extreme conditions.")
    
    # MACD signals with explanation
    macd, signal_line, histogram = calculate_macd(prices)
    if macd and signal_line:
        if macd > signal_line and histogram > 0:
            signals.append("MACD Bullish - Potential BUY signal")
            explanations.append(f"MACD line ({macd:.4f}) is above signal line ({signal_line:.4f}) with positive histogram ({histogram:.4f}). This indicates bullish momentum as the faster moving average is pulling away from the slower one.")
        elif macd < signal_line and histogram < 0:
            signals.append("MACD Bearish - Potential SELL signal")
            explanations.append(f"MACD line ({macd:.4f}) is below signal line ({signal_line:.4f}) with negative histogram ({histogram:.4f}). This indicates bearish momentum as the faster moving average is diverging downward.")
        else:
            signals.append("MACD Neutral")
            explanations.append(f"MACD shows mixed signals with line at {macd:.4f} and signal at {signal_line:.4f}. Momentum is unclear, suggesting a wait-and-see approach.")
    
    # Bollinger Bands signals with explanation
    upper, middle, lower = calculate_bollinger_bands(prices)
    if upper and lower:
        bb_position = ((current_price - lower) / (upper - lower)) * 100
        if current_price > upper:
            signals.append("Price above Bollinger Upper Band - Potential SELL signal")
            explanations.append(f"Price (${current_price:.2f}) is above the upper Bollinger Band (${upper:.2f}), suggesting the stock is statistically overbought. Band position: {bb_position:.1f}%")
        elif current_price < lower:
            signals.append("Price below Bollinger Lower Band - Potential BUY signal")
            explanations.append(f"Price (${current_price:.2f}) is below the lower Bollinger Band (${lower:.2f}), suggesting the stock is statistically oversold. Band position: {bb_position:.1f}%")
        else:
            signals.append("Price within Bollinger Bands")
            explanations.append(f"Price is within normal trading range. Position in band: {bb_position:.1f}% (0% = lower band, 100% = upper band)")
    
    # Moving Average Crossover with explanation
    if len(prices) >= 20:
        ma_5 = prices.rolling(5).mean().iloc[-1]
        ma_20 = prices.rolling(20).mean().iloc[-1]
        ma_diff_pct = ((ma_5 - ma_20) / ma_20) * 100
        
        if ma_5 > ma_20:
            signals.append("Short MA above Long MA - Bullish signal")
            explanations.append(f"5-period MA (${ma_5:.2f}) is {ma_diff_pct:.2f}% above 20-period MA (${ma_20:.2f}). This crossover indicates short-term momentum is stronger than long-term trend, suggesting bullish sentiment.")
        else:
            signals.append("Short MA below Long MA - Bearish signal")
            explanations.append(f"5-period MA (${ma_5:.2f}) is {abs(ma_diff_pct):.2f}% below 20-period MA (${ma_20:.2f}). This indicates short-term weakness compared to long-term trend, suggesting bearish sentiment.")
    
    # Volume analysis (if available)
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

def get_signal_summary_note():
    """Generate educational note about trading signals"""
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

def compute_advanced_statistics(df):
    """Compute advanced statistics with technical indicators"""
    # Detect required columns dynamically (case-insensitive)
    def find_col(name):
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
    
    # Basic statistics
    stats = {
        "Open": round(df.iloc[0][open_col], 2),
        "Close": round(df.iloc[-1][close_col], 2),
        "High": round(df[high_col].max(), 2),
        "Low": round(df[low_col].min(), 2),
        "Volume": int(df[volume_col].sum()),
    }
    
    # Price change calculations
    price_change = stats["Close"] - stats["Open"]
    price_change_pct = (price_change / stats["Open"]) * 100
    stats["Price Change"] = round(price_change, 2)
    stats["Price Change %"] = round(price_change_pct, 2)
    
    # Moving Averages
    if len(prices) >= 5:
        stats["5-Period MA"] = round(prices.rolling(5).mean().iloc[-1], 2)
    if len(prices) >= 10:
        stats["10-Period MA"] = round(prices.rolling(10).mean().iloc[-1], 2)
    if len(prices) >= 20:
        stats["20-Period MA"] = round(prices.rolling(20).mean().iloc[-1], 2)
        stats["20-Period EMA"] = round(prices.ewm(span=20).mean().iloc[-1], 2)
    
    # RSI
    rsi = calculate_rsi(prices)
    if rsi:
        stats["RSI (14)"] = round(rsi, 2)
    
    # MACD
    macd, signal_line, histogram = calculate_macd(prices)
    if macd:
        stats["MACD"] = round(macd, 4)
        stats["MACD Signal"] = round(signal_line, 4)
        stats["MACD Histogram"] = round(histogram, 4)
    
    # Bollinger Bands
    upper, middle, lower = calculate_bollinger_bands(prices)
    if upper:
        stats["Bollinger Upper"] = round(upper, 2)
        stats["Bollinger Middle"] = round(middle, 2)
        stats["Bollinger Lower"] = round(lower, 2)
    
    # Stochastic
    if len(prices) >= 14:
        k_percent, d_percent = calculate_stochastic(highs, lows, prices)
        if k_percent:
            stats["Stochastic %K"] = round(k_percent, 2)
            stats["Stochastic %D"] = round(d_percent, 2)
    
    # Volatility
    if len(prices) >= 20:
        returns = prices.pct_change().dropna()
        if len(returns) > 0:
            stats["Volatility (20-day)"] = round(returns.std() * np.sqrt(252) * 100, 2)  # Annualized volatility %
    
    # Enhanced Trend Detection with reasoning
    trend, trend_reasoning = detect_trend_with_reasoning(prices)
    stats["Trend"] = trend
    stats["Trend Reasoning"] = trend_reasoning
    
    # Support and Resistance (simple approach)
    recent_highs = highs.tail(20)
    recent_lows = lows.tail(20)
    stats["Resistance Level"] = round(recent_highs.quantile(0.9), 2)
    stats["Support Level"] = round(recent_lows.quantile(0.1), 2)
    
    # Enhanced Trading Signals with explanations
    signals, explanations = generate_detailed_signals(df, close_col)
    stats["Trading Signals"] = signals
    stats["Signal Explanations"] = explanations
    stats["Signal Summary Note"] = get_signal_summary_note()
    
    stats["Data Points"] = len(df)
    
    return stats

def compute_single_price_advanced_stats(df):
    """Compute advanced statistics for single price column data"""
    # Get all price columns (excluding datetime)
    price_cols = [col for col in df.columns 
                 if col != 'Datetime' and 
                 not any(time_word in col.lower() for time_word in ['date', 'time'])]
    
    if not price_cols:
        return {"Error": "No price columns found in data."}
    
    # Convert price columns to numeric
    for col in price_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Create combined price series
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
    
    # Create price series for analysis
    price_series = pd.Series(valid_prices)
    
    # Calculate statistics
    stats = {
        "Latest Price": round(price_series.iloc[-1], 2),
        "Highest Price": round(price_series.max(), 2),
        "Lowest Price": round(price_series.min(), 2),
        "Average Price": round(price_series.mean(), 2),
        "Median Price": round(price_series.median(), 2),
    }
    
    # Price change calculations
    if len(price_series) > 1:
        first_price = price_series.iloc[0]
        last_price = price_series.iloc[-1]
        price_change = last_price - first_price
        price_change_pct = (price_change / first_price) * 100 if first_price != 0 else 0
        
        stats["Price Change"] = round(price_change, 2)
        stats["Price Change %"] = round(price_change_pct, 2)
    
    # Moving Averages
    if len(price_series) >= 5:
        stats["5-Period MA"] = round(price_series.rolling(5).mean().iloc[-1], 2)
    if len(price_series) >= 10:
        stats["10-Period MA"] = round(price_series.rolling(10).mean().iloc[-1], 2)
    if len(price_series) >= 20:
        stats["20-Period MA"] = round(price_series.rolling(20).mean().iloc[-1], 2)
        stats["20-Period EMA"] = round(price_series.ewm(span=20).mean().iloc[-1], 2)
    
    # RSI
    rsi = calculate_rsi(price_series)
    if rsi:
        stats["RSI (14)"] = round(rsi, 2)
    
    # MACD
    macd, signal_line, histogram = calculate_macd(price_series)
    if macd:
        stats["MACD"] = round(macd, 4)
        stats["MACD Signal"] = round(signal_line, 4)
        stats["MACD Histogram"] = round(histogram, 4)
    
    # Bollinger Bands
    upper, middle, lower = calculate_bollinger_bands(price_series)
    if upper:
        stats["Bollinger Upper"] = round(upper, 2)
        stats["Bollinger Middle"] = round(middle, 2)
        stats["Bollinger Lower"] = round(lower, 2)
    
    # Volatility
    if len(price_series) >= 20:
        returns = price_series.pct_change().dropna()
        if len(returns) > 0:
            stats["Volatility (20-day)"] = round(returns.std() * np.sqrt(252) * 100, 2)
    
    # Trend Analysis
    trend, trend_reasoning = detect_trend_with_reasoning(price_series)
    stats["Trend"] = trend
    stats["Trend Reasoning"] = trend_reasoning
    
    # Support and Resistance levels
    if len(price_series) >= 20:
        recent_prices = price_series.tail(20)
        stats["Resistance Level"] = round(recent_prices.quantile(0.9), 2)
        stats["Support Level"] = round(recent_prices.quantile(0.1), 2)
    
    # Generate trading signals (create temporary df for compatibility)
    temp_df = pd.DataFrame({'close': price_series})
    signals, explanations = generate_detailed_signals(temp_df, 'close')
    stats["Trading Signals"] = signals
    stats["Signal Explanations"] = explanations
    stats["Signal Summary Note"] = get_signal_summary_note()
    
    stats["Data Points"] = len(price_series)
    stats["Price Columns Analyzed"] = price_cols
    
    return stats

def analyze_stock_data(data, data_format=None):
    """Main function to analyze stock data regardless of format"""
    try:
        # Clean and prepare data
        df = clean_and_convert(data.copy())
        
        # Auto-detect format if not provided
        if data_format is None:
            data_format = detect_csv_format(df)
        
        print(f"Detected data format: {data_format}")
        print(f"Data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Analyze based on format
        if data_format == 'ohlcv':
            return compute_advanced_statistics(df)
        elif data_format == 'single_price':
            return compute_single_price_advanced_stats(df)
        else:
            return {"Error": "Unable to detect valid data format. Ensure data contains either OHLCV columns or price columns with datetime."}
    
    except Exception as e:
        return {"Error": f"Analysis failed: {str(e)}"}

def fetch_and_analyze_stock(symbol, period="1mo"):
    """Fetch stock data from Yahoo Finance and analyze it"""
    try:
        # Fetch data
        stock = yf.Ticker(symbol)
        df = stock.history(period=period, interval="5m")  # Use 5-minute intervals for intraday
        
        if df.empty:
            return {"Error": f"No data found for symbol {symbol}"}
        
        # Analyze the data
        results = analyze_stock_data(df, 'ohlcv')
        results["Symbol"] = symbol.upper()
        results["Period"] = period
        
        return results
    
    except Exception as e:
        return {"Error": f"Failed to fetch data for {symbol}: {str(e)}"}

def print_analysis_results(results):
    """Pretty print analysis results"""
    if "Error" in results:
        print(f"❌ Error: {results['Error']}")
        return
    
    print("=" * 80)
    print(f"📊 STOCK ANALYSIS REPORT")
    if "Symbol" in results:
        print(f"🏷️  Symbol: {results['Symbol']} | Period: {results.get('Period', 'N/A')}")
    print("=" * 80)
    
    # Basic Info
    print("\n📈 PRICE SUMMARY:")
    basic_keys = ["Open", "Close", "High", "Low", "Latest Price", "Price Change", "Price Change %"]
    for key in basic_keys:
        if key in results:
            if "%" in key:
                emoji = "📈" if results[key] > 0 else "📉" if results[key] < 0 else "➡️"
                print(f"   {emoji} {key}: {results[key]}%")
            else:
                print(f"   💰 {key}: ${results[key]}")
    
    # Technical Indicators
    print("\n🔧 TECHNICAL INDICATORS:")
    tech_keys = ["RSI (14)", "MACD", "MACD Signal", "MACD Histogram", "Stochastic %K", "Stochastic %D"]
    for key in tech_keys:
        if key in results:
            print(f"   📊 {key}: {results[key]}")
    
    # Moving Averages
    print("\n📊 MOVING AVERAGES:")
    ma_keys = ["5-Period MA", "10-Period MA", "20-Period MA", "20-Period EMA"]
    for key in ma_keys:
        if key in results:
            print(f"   📈 {key}: ${results[key]}")
    
    # Bollinger Bands
    print("\n🎯 BOLLINGER BANDS:")
    bb_keys = ["Bollinger Upper", "Bollinger Middle", "Bollinger Lower"]
    for key in bb_keys:
        if key in results:
            print(f"   🎯 {key}: ${results[key]}")
    
    # Support/Resistance
    print("\n🏗️ SUPPORT & RESISTANCE:")
    sr_keys = ["Support Level", "Resistance Level"]
    for key in sr_keys:
        if key in results:
            print(f"   🏗️ {key}: ${results[key]}")
    
    # Trend Analysis
    print(f"\n📊 TREND ANALYSIS:")
    if "Trend" in results:
        trend_emoji = {"Strong Uptrend": "🚀", "Uptrend": "📈", "Sideways": "➡️", 
                      "Downtrend": "📉", "Strong Downtrend": "💥"}.get(results["Trend"], "❓")
        print(f"   {trend_emoji} Current Trend: {results['Trend']}")
    
    if "Trend Reasoning" in results:
        print(f"   💭 Analysis: {results['Trend Reasoning']}")
    
    # Trading Signals
    print(f"\n🎯 TRADING SIGNALS:")
    if "Trading Signals" in results and "Signal Explanations" in results:
        signals = results["Trading Signals"]
        explanations = results["Signal Explanations"]
        
        for i, (signal, explanation) in enumerate(zip(signals, explanations)):
            signal_emoji = "🟢" if "BUY" in signal else "🔴" if "SELL" in signal else "🟡"
            print(f"   {signal_emoji} {signal}")
            print(f"      💡 {explanation}")
            if i < len(signals) - 1:
                print()
    
    # Educational Note
    if "Signal Summary Note" in results:
        print("\n" + results["Signal Summary Note"])
    
    # Additional Info
    other_keys = ["Volume", "Volatility (20-day)", "Data Points"]
    additional_info = []
    for key in other_keys:
        if key in results:
            if key == "Volume":
                additional_info.append(f"{key}: {results[key]:,}")
            else:
                additional_info.append(f"{key}: {results[key]}")
    
    if additional_info:
        print(f"\n📋 ADDITIONAL INFO:")
        for info in additional_info:
            print(f"   ℹ️ {info}")
    
    print("\n" + "=" * 80)

def load_and_analyze_csv(file_path):
    """Load CSV file and analyze stock data"""
    try:
        # Try different encodings and separators
        encodings = ['utf-8', 'latin-1', 'cp1252']
        separators = [',', ';', '\t']
        
        df = None
        for encoding in encodings:
            for sep in separators:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, sep=sep)
                    if df.shape[1] > 1:  # If we got multiple columns, likely correct
                        break
                except:
                    continue
            if df is not None and df.shape[1] > 1:
                break
        
        if df is None or df.empty:
            return {"Error": "Could not load CSV file. Please check file format and encoding."}
        
        print(f"Loaded CSV with shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Analyze the data
        results = analyze_stock_data(df)
        results["Source"] = file_path
        
        return results
    
    except Exception as e:
        return {"Error": f"Failed to load CSV: {str(e)}"}

def generate_insights(source="api", symbol=None, csv_file=None, period="1d"):
    """Generate stock insights based on data source"""
    try:
        if source == "api":
            if not symbol:
                return {"Error": "Stock symbol is required for API data fetching."}
            return fetch_and_analyze_stock(symbol, period)
        elif source == "csv":
            if not csv_file:
                return {"Error": "CSV file is required for CSV data analysis."}
            return load_and_analyze_csv(csv_file)
        else:
            return {"Error": "Invalid source specified. Use 'api' or 'csv'."}
    except Exception as e:
        return {"Error": f"Failed to generate insights: {str(e)}"}

def batch_analyze_stocks(symbols, period="1mo"):
    """Analyze multiple stocks at once"""
    results = {}
    
    for symbol in symbols:
        print(f"Analyzing {symbol}...")
        try:
            result = fetch_and_analyze_stock(symbol, period)
            results[symbol] = result
        except Exception as e:
            results[symbol] = {"Error": f"Failed to analyze {symbol}: {str(e)}"}
    
    return results

def compare_stocks(symbols, period="1mo"):
    """Compare key metrics across multiple stocks"""
    batch_results = batch_analyze_stocks(symbols, period)
    
    print("=" * 120)
    print("📊 STOCK COMPARISON REPORT")
    print("=" * 120)
    
    # Create comparison table
    comparison_data = []
    metrics = ["Close", "Price Change %", "RSI (14)", "Trend", "Volatility (20-day)"]
    
    for symbol, data in batch_results.items():
        if "Error" not in data:
            row = {"Symbol": symbol}
            for metric in metrics:
                if metric in data:
                    if metric == "Trend":
                        trend_emoji = {"Strong Uptrend": "🚀", "Uptrend": "📈", "Sideways": "➡️", 
                                      "Downtrend": "📉", "Strong Downtrend": "💥"}.get(data[metric], "❓")
                        row[metric] = f"{trend_emoji} {data[metric]}"
                    else:
                        row[metric] = data[metric]
                else:
                    row[metric] = "N/A"
            comparison_data.append(row)
        else:
            comparison_data.append({"Symbol": symbol, "Error": data["Error"]})
    
    # Print comparison table
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
    
    print("\n" + "=" * 120)
    
    return batch_results

# Utility functions for different data sources
def analyze_yahoo_finance(symbol, period="1mo"):
    """Simplified function to analyze Yahoo Finance data"""
    return fetch_and_analyze_stock(symbol, period)

def analyze_csv_file(file_path):
    """Simplified function to analyze CSV file"""
    return load_and_analyze_csv(file_path)

def get_available_periods():
    """Get available periods for Yahoo Finance data"""
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

def print_help():
    """Print usage instructions"""
    help_text = """
🔧 STOCK TECHNICAL ANALYSIS TOOL - USAGE GUIDE

📊 MAIN FUNCTIONS:
• analyze_yahoo_finance(symbol, period) - Analyze stock from Yahoo Finance
• analyze_csv_file(file_path) - Analyze stock data from CSV file
• compare_stocks([symbols], period) - Compare multiple stocks
• batch_analyze_stocks([symbols], period) - Analyze multiple stocks individually

📈 SUPPORTED DATA FORMATS:
• OHLCV: Open, High, Low, Close, Volume with datetime
• Single Price: Any price columns with datetime
• Auto-detection of format and columns

🎯 TECHNICAL INDICATORS INCLUDED:
• RSI (Relative Strength Index)
• MACD (Moving Average Convergence Divergence)
• Bollinger Bands
• Stochastic Oscillator
• Moving Averages (5, 10, 20 period)
• Support/Resistance levels
• Trend analysis with reasoning
• Trading signals with explanations

⏰ AVAILABLE PERIODS (Yahoo Finance):
• 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max

📋 EXAMPLE USAGE:
```python
# Analyze single stock
results = analyze_yahoo_finance("AAPL", "3mo")
print_analysis_results(results)

# Compare multiple stocks
compare_stocks(["AAPL", "GOOGL", "MSFT"], "1mo")

# Analyze CSV file
results = analyze_csv_file("stock_data.csv")
print_analysis_results(results) """