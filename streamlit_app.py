import streamlit as st
import pandas as pd
from app.data.stock_insights import generate_insights

st.set_page_config(page_title="Advanced Stock Analysis", page_icon="📈", layout="wide")

st.title("📈 Advanced Stock Intraday Data Analysis")
st.markdown("*Comprehensive technical analysis with multiple indicators and trading signals*")

# Sidebar for options
st.sidebar.header("⚙️ Configuration")
option = st.sidebar.radio("Select data source", ["Upload CSV", "Fetch from API"])

def display_analysis(insights):
    """Display analysis results in organized sections"""
    
    # Display results in organized sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Basic Statistics")
        basic_stats = ['Open', 'Close', 'High', 'Low', 'Volume', 'Latest Price', 'Price Change', 'Price Change %', 'Data Points', 'Symbol']
        
        for key in basic_stats:
            if key in insights:
                if key == 'Price Change %':
                    color = "green" if insights[key] > 0 else "red"
                    st.markdown(f"**{key}:** <span style='color:{color}'>{insights[key]}%</span>", unsafe_allow_html=True)
                elif key == 'Price Change':
                    color = "green" if insights[key] > 0 else "red"
                    st.markdown(f"**{key}:** <span style='color:{color}'>{insights[key]}</span>", unsafe_allow_html=True)
                else:
                    st.write(f"**{key}:** {insights[key]}")
    
    with col2:
        st.subheader("📈 Moving Averages")
        ma_keys = ['5-Period MA', '10-Period MA', '20-Period MA', '20-Period EMA']
        for key in ma_keys:
            if key in insights:
                st.write(f"**{key}:** {insights[key]}")
        
        if 'Volatility (20-day)' in insights:
            st.write(f"**Volatility (20-day):** {insights['Volatility (20-day)']}%")
        if 'Price Volatility %' in insights:
            st.write(f"**Price Volatility:** {insights['Price Volatility %']}%")
    
    # Technical Indicators Section
    st.subheader("🔧 Technical Indicators")
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.markdown("**Momentum Indicators**")
        if 'RSI (14)' in insights:
            rsi_val = insights['RSI (14)']
            if rsi_val > 70:
                st.markdown(f"🔴 **RSI:** {rsi_val} (Overbought)")
            elif rsi_val < 30:
                st.markdown(f"🟢 **RSI:** {rsi_val} (Oversold)")
            else:
                st.markdown(f"🟡 **RSI:** {rsi_val} (Neutral)")
        
        if all(key in insights for key in ['Stochastic %K', 'Stochastic %D']):
            st.write(f"**Stochastic %K:** {insights['Stochastic %K']}")
            st.write(f"**Stochastic %D:** {insights['Stochastic %D']}")
    
    with col4:
        st.markdown("**Trend Indicators**")
        if all(key in insights for key in ['MACD', 'MACD Signal', 'MACD Histogram']):
            st.write(f"**MACD:** {insights['MACD']}")
            st.write(f"**MACD Signal:** {insights['MACD Signal']}")
            hist_val = insights['MACD Histogram']
            color = "green" if hist_val > 0 else "red"
            st.markdown(f"**MACD Histogram:** <span style='color:{color}'>{hist_val}</span>", unsafe_allow_html=True)
    
    with col5:
        st.markdown("**Support & Resistance**")
        if all(key in insights for key in ['Bollinger Upper', 'Bollinger Middle', 'Bollinger Lower']):
            st.write(f"**BB Upper:** {insights['Bollinger Upper']}")
            st.write(f"**BB Middle:** {insights['Bollinger Middle']}")
            st.write(f"**BB Lower:** {insights['Bollinger Lower']}")
        
        if all(key in insights for key in ['Resistance Level', 'Support Level']):
            st.write(f"**Resistance:** {insights['Resistance Level']}")
            st.write(f"**Support:** {insights['Support Level']}")
    
    # Trend Analysis Section
    if 'Trend' in insights and 'Trend Reasoning' in insights:
        st.subheader("📊 Trend Analysis")
        trend = insights['Trend']
        
        # Color code trend
        if 'Strong Uptrend' in trend or 'Uptrend' in trend:
            st.markdown(f"**Current Trend:** <span style='color:green; font-weight:bold'>{trend}</span>", unsafe_allow_html=True)
        elif 'Strong Downtrend' in trend or 'Downtrend' in trend:
            st.markdown(f"**Current Trend:** <span style='color:red; font-weight:bold'>{trend}</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"**Current Trend:** <span style='color:orange; font-weight:bold'>{trend}</span>", unsafe_allow_html=True)
        
        st.write(f"**Analysis:** {insights['Trend Reasoning']}")
    
    # Enhanced Trading Signals Section
    if 'Trading Signals' in insights and 'Signal Explanations' in insights:
        st.subheader("🎯 Enhanced Trading Signals")
        
        signals = insights['Trading Signals']
        explanations = insights['Signal Explanations']
        
        # Display signals with their explanations
        for i, (signal, explanation) in enumerate(zip(signals, explanations)):
            if 'BUY' in signal.upper():
                st.success(f"🟢 **{signal}**")
                st.write(f"📝 {explanation}")
            elif 'SELL' in signal.upper():
                st.error(f"🔴 **{signal}**")
                st.write(f"📝 {explanation}")
            else:
                st.info(f"🟡 **{signal}**")
                st.write(f"📝 {explanation}")
            
            if i < len(signals) - 1:  # Add separator except for last item
                st.markdown("---")
    
    # Signal Summary Note
    if 'Signal Summary Note' in insights:
        st.subheader("📚 Trading Signal Guide")
        st.markdown(insights['Signal Summary Note'])

if option == "Upload CSV":
    st.sidebar.markdown("### 📁 File Upload")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file:
        with st.spinner("🔄 Analyzing your data..."):
            insights = generate_insights(source="csv", csv_file=uploaded_file)
        
        if "Error" in insights:
            st.error(f"❌ {insights['Error']}")
        else:
            display_analysis(insights)

elif option == "Fetch from API":
    st.sidebar.markdown("### 🌐 API Configuration")
    
    # Stock symbol input
    symbol = st.sidebar.text_input("Enter Stock Symbol", value="AAPL", placeholder="e.g., AAPL, GOOGL, TSLA")
    
    # Simple note about data source
    st.sidebar.info("📊 Fetches 1-day intraday data (5-minute intervals) from Yahoo Finance")
    
    # Fetch button
    if st.sidebar.button("🚀 Fetch & Analyze"):
        if symbol:
            with st.spinner(f"🔄 Fetching data for {symbol.upper()}..."):
                insights = generate_insights(source="api", symbol=symbol.upper(), period="1d")
            
            if "Error" in insights:
                st.error(f"❌ {insights['Error']}")
            else:
                # Symbol info
                st.info(f"📊 Analysis for **{symbol.upper()}** | 5-minute intervals | 1-day period")
                display_analysis(insights)
        else:
            st.sidebar.error("Please enter a stock symbol")

# Help section
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📖 Help")
    st.markdown("""
    **CSV Format Support:**
    - OHLCV format (Open, High, Low, Close, Volume)
    - Single price column format
    - Must include datetime/date column
    
    **API Data:**
    - Uses Yahoo Finance API
    - 5-minute intraday data
    - No API key required
    
    **Technical Indicators:**
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - Bollinger Bands
    - Stochastic Oscillator
    - Moving Averages (5, 10, 20 period)
    """)

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit | Advanced Stock Analysis Dashboard*")
st.markdown("⚠️ **Disclaimer:** This analysis is for educational purposes only. Not financial advice.")