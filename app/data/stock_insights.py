import pandas as pd
import yfinance as yf

def clean_and_convert(df):
    # Flatten multi-index columns if any (e.g. from yfinance)
    df.columns = ['_'.join(filter(None, map(str, col))).strip() if isinstance(col, tuple) else str(col).strip() for col in df.columns]

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

def compute_single_price_stats(df):
    """Compute statistics for single price column data or mixed price data"""
    # Get all price columns (excluding datetime)
    price_cols = [col for col in df.columns 
                 if col != 'Datetime' and 
                 not any(time_word in col.lower() for time_word in ['date', 'time'])]
    
    if not price_cols:
        return {"Error": "No price columns found in data."}
    
    # Convert price columns to numeric, handling empty strings
    for col in price_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Create combined price series from all available price data
    all_prices = []
    for idx, row in df.iterrows():
        # Get all non-null prices for this row
        prices = [row[col] for col in price_cols if pd.notna(row[col])]
        if prices:
            # Use the last available price (most recent/important)
            all_prices.append(prices[-1])
        else:
            all_prices.append(None)
    
    # Filter out None values
    valid_prices = [p for p in all_prices if p is not None]
    
    if not valid_prices:
        return {"Error": "No valid price data found."}
    
    # Create a series for calculations
    price_series = pd.Series(valid_prices)
    moving_avg = price_series.rolling(window=min(5, len(price_series))).mean().iloc[-1]
    
    return {
        "Symbol": price_cols[0] if len(price_cols) == 1 else "Multiple Price Columns",
        "First Price": valid_prices[0],
        "Last Price": valid_prices[-1],
        "Highest Price": max(valid_prices),
        "Lowest Price": min(valid_prices),
        "Price Change": valid_prices[-1] - valid_prices[0],
        "Price Change %": round(((valid_prices[-1] - valid_prices[0]) / valid_prices[0]) * 100, 2),
        f"{min(5, len(valid_prices))}-Period Moving Average": round(moving_avg, 2),
        "Data Points": len(valid_prices),
        "Price Columns": price_cols
    }

def compute_statistics(df):
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

    moving_avg = df[close_col].rolling(window=5).mean().iloc[-1]

    return {
        "Open": df.iloc[0][open_col],
        "Close": df.iloc[-1][close_col],
        "High": df[high_col].max(),
        "Low": df[low_col].min(),
        "Volume": df[volume_col].sum(),
        "5-Period Moving Average": moving_avg
    }

def generate_insights(source="csv", csv_file=None, symbol=None):
    if source == "csv":
        if csv_file is None:
            return {"Error": "CSV file must be provided if source is 'csv'."}
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            return {"Error": f"Failed to read CSV: {str(e)}"}
            
    elif source == "api":
        if symbol is None:
            return {"Error": "Symbol must be provided if source is 'api'."}
        try:
            df = yf.download(symbol, interval="5m", period="1d")
            if df.empty:
                return {"Error": f"No data found for symbol '{symbol}'."}
        except Exception as e:
            return {"Error": f"Failed to fetch data: {str(e)}"}
    else:
        return {"Error": "Invalid source type. Use 'csv' or 'api'."}

    try:
        df = clean_and_convert(df)
    except Exception as e:
        return {"Error": f"Data cleaning failed: {str(e)}"}

    if df.empty:
        return {"Error": "No data available after cleaning."}

    # Detect format and compute appropriate statistics
    format_type = detect_csv_format(df)
    
    if format_type == 'ohlcv':
        stats = compute_statistics(df)
    elif format_type == 'single_price':
        stats = compute_single_price_stats(df)
    else:
        return {"Error": "Unsupported data format."}
    
    return stats
