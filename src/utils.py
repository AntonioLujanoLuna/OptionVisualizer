"""
utils.py

Utility functions, caching, and data fetching.
"""
import yfinance as yf
import streamlit as st

@st.cache_data
def fetch_current_price(ticker: str):
    """
    Fetches the last closing price for `ticker` using yfinance.
    Returns None if fetching fails or data is empty.
    """
    try:
        data = yf.Ticker(ticker).history(period="1d")
        if not data.empty:
            return float(data["Close"].iloc[-1])
    except:
        pass
    return None
