# src/ui/pages/home.py

import streamlit as st

def render_page():
    """Render the home page with introduction and navigation."""
    st.markdown(
        """
        # Option Pricing and Analysis Tool
        
        Welcome to the option pricing and analysis tool. This application helps you:
        
        - Price options using multiple models
        - Build and analyze trading strategies
        - Manage and analyze portfolios
        - Learn about options through interactive visualizations
        
        ## Getting Started
        
        Choose a section from the sidebar to begin:
        
        - **Option Pricing**: Price individual options and analyze Greeks
        - **Strategy Builder**: Create and analyze multi-leg options strategies
        - **Portfolio Analysis**: Manage and analyze your options portfolio
        - **Education**: Learn about options through interactive examples
        
        ## Features
        
        - Multiple pricing models (Black-Scholes, Binomial, Monte Carlo)
        - Interactive strategy building and analysis
        - Real-time market data integration
        - Comprehensive risk analytics
        - Educational content and visualizations
        """
    )