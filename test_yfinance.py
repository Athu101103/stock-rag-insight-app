import yfinance as yf
import time

def test_fetch(symbol, period="1mo", interval="1d"):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period, interval=interval)
        if df.empty:
            print(f"No data found for symbol {symbol} with period={period}, interval={interval}")
        else:
            print(f"Data for {symbol} with period={period}, interval={interval}:\n{df.tail()}")
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")

if __name__ == "__main__":
    symbols = ["AAPL", "MSFT", "GOOGL"]
    for symbol in symbols:
        # Test historical data
        test_fetch(symbol, period="1mo", interval="1d")
        # Test intraday data for today
        test_fetch(symbol, period="1d", interval="5m")
        time.sleep(2)  # Avoid rate limiting