"""
utils.py

Utility functions, caching, and data fetching.
"""
import yfinance as yf
import streamlit as st

# Enhance the fetch_current_price function in utils.py
@st.cache_data
def fetch_market_data(ticker: str):
    """
    Fetches extended market data including implied volatility.
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        options = stock.options
        
        if not hist.empty and options:
            current_price = float(hist["Close"].iloc[-1])
            # Get nearest expiry options chain
            opt = stock.option_chain(options[0])
            
            # Calculate average implied volatility
            avg_iv = (opt.calls['impliedVolatility'].mean() + 
                     opt.puts['impliedVolatility'].mean()) / 2
            
            return {
                'price': current_price,
                'implied_volatility': avg_iv,
                'options_chain': opt
            }
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
    return None