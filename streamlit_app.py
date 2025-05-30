import streamlit as st
from app.data.stock_insights import generate_insights

st.title("Stock Intraday Data Insights")

option = st.radio("Select data source", ["Upload CSV", "Fetch from API"])

if option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        insights = generate_insights(source="csv", csv_file=uploaded_file)
        st.write("## Analysis Results")
        if "Error" in insights:
            st.error(insights["Error"])
        else:
            for key, value in insights.items():
                st.write(f"**{key}:** {value}")

else:
    st.info("📝 **Note for Indian Stocks:** For NSE stocks add '.NS' (e.g., TATAMOTORS.NS) or for BSE stocks add '.BO' (e.g., TATAMOTORS.BO)")
    
    symbol = st.text_input("Enter stock symbol", placeholder="e.g., MSFT, AAPL, TATAMOTORS.NS")
    
    if symbol:
        with st.spinner(f"Fetching data for {symbol}..."):
            insights = generate_insights(source="api", symbol=symbol)
        
        st.write("## Analysis Results")
        if "Error" in insights:
            st.error(insights["Error"])
        else:
            for key, value in insights.items():
                st.write(f"**{key}:** {value}")