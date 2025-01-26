"""
app.py

An advanced Streamlit application for option pricing visualization and analysis.
Features:
- Multiple pricing models (Black-Scholes, Binomial, Monte Carlo)
- Interactive 2D and 3D visualizations
- Real-time market data integration
- Greeks analysis and visualization
- Volatility surface exploration
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from datetime import datetime

from pricing_models import (
    option_price,
    black_scholes_greeks_call,
    black_scholes_greeks_put,
    implied_volatility_surface
)
from utils import fetch_market_data

# Visual Theme Configuration
COLORS = {
    'primary': '#2E86C1',
    'secondary': '#85C1E9',
    'accent': '#D4E6F1',
    'text': '#2C3E50',
    'background': '#F7F9F9',
    'success': '#27AE60',
    'warning': '#F39C12',
    'error': '#E74C3C'
}

def setup_page_config():
    """Configure the Streamlit page with custom settings and styling."""
    st.set_page_config(
        page_title="Advanced Option Pricing",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/yourusername/option-pricing',
            'Report a bug': "https://github.com/yourusername/option-pricing/issues",
            'About': "# Option Pricing Visualizer\nAn interactive tool for understanding options."
        }
    )
    
    # Custom CSS for better visual appeal
    st.markdown("""
        <style>
        .stApp {
            background-color: #F7F9F9;
        }
        .stButton>button {
            background-color: #2E86C1;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
        }
        .stProgress .st-bo {
            background-color: #85C1E9;
        }
        .plot-container {
            background-color: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)

def scenario_input_box(scenario_num=1, market_data=None):
    """Enhanced input box with market data integration and validation."""
    expander = st.expander(f"Scenario {scenario_num} Settings", expanded=True)
    
    with expander:
        col1, col2 = st.columns(2)
        
        with col1:
            S = st.number_input(
                f"Spot Price (S{scenario_num})",
                min_value=0.01,
                max_value=10000.0,
                value=100.0,
                step=0.1,
                key=f"S{scenario_num}"
            )
            
            if market_data and st.checkbox(f"Use Market Price ({market_data['price']:.2f})", key=f"use_market{scenario_num}"):
                S = market_data['price']
            
            K = st.number_input(
                f"Strike Price (K{scenario_num})",
                min_value=0.01,
                max_value=10000.0,
                value=100.0,
                step=0.1,
                key=f"K{scenario_num}"
            )
            
            r = st.slider(
                f"Risk-free Rate (r{scenario_num})",
                min_value=0.0,
                max_value=0.20,
                value=0.05,
                step=0.001,
                key=f"r{scenario_num}"
            )
        
        with col2:
            sigma = st.number_input(
                f"Volatility (σ{scenario_num})",
                min_value=0.01,
                max_value=2.0,
                value=0.2,
                step=0.01,
                key=f"sigma{scenario_num}"
            )
            
            if market_data and st.checkbox(f"Use Market IV ({market_data['implied_volatility']:.2%})", key=f"use_iv{scenario_num}"):
                sigma = market_data['implied_volatility']
            
            T = st.slider(
                f"Time to Maturity (T{scenario_num})",
                min_value=0.01,
                max_value=5.0,
                value=1.0,
                step=0.01,
                key=f"T{scenario_num}"
            )
        
        # Advanced parameters in a sub-expander
        with st.expander("Advanced Parameters"):
            col3, col4 = st.columns(2)
            with col3:
                steps = st.number_input(
                    f"Binomial Steps",
                    min_value=10,
                    max_value=2000,
                    value=100,
                    step=10,
                    key=f"steps{scenario_num}"
                )
            with col4:
                sims = st.number_input(
                    f"Monte Carlo Simulations",
                    min_value=1000,
                    max_value=200000,
                    value=10000,
                    step=1000,
                    key=f"sims{scenario_num}"
                )
    
    return S, K, r, sigma, T, steps, sims

def plot_option_prices(scenarios, model_choice, opt_type_choice, x_variable="spot"):
    """Generate an interactive plot comparing option prices across scenarios."""
    fig = go.Figure()
    
    for i, scenario in enumerate(scenarios, 1):
        S, K, r, sigma, T, steps, sims = scenario
        
        if x_variable == "spot":
            x_vals = np.linspace(max(1, 0.5 * S), 1.5 * S, 100)
            x_label = "Spot Price"
            current_x = S
        elif x_variable == "strike":
            x_vals = np.linspace(max(1, 0.5 * K), 1.5 * K, 100)
            x_label = "Strike Price"
            current_x = K
        else:  # time
            x_vals = np.linspace(0.01, 5.0, 100)
            x_label = "Time to Maturity"
            current_x = T
        
        for opt_type in (["call", "put"] if opt_type_choice == "Both" else [opt_type_choice.lower()]):
            prices = []
            for x in x_vals:
                if x_variable == "spot":
                    price = option_price(x, K, r, sigma, T, opt_type, model_choice, steps, sims)
                elif x_variable == "strike":
                    price = option_price(S, x, r, sigma, T, opt_type, model_choice, steps, sims)
                else:
                    price = option_price(S, K, r, sigma, x, opt_type, model_choice, steps, sims)
                prices.append(price)
            
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=prices,
                    name=f"Scenario {i} - {opt_type.title()}",
                    line=dict(
                        color=COLORS['primary'] if opt_type == "call" else COLORS['secondary'],
                        dash='solid' if i == 1 else 'dash'
                    )
                )
            )
            
            # Add vertical line for current value
            fig.add_vline(
                x=current_x,
                line_dash="dot",
                line_color=COLORS['accent'],
                annotation_text=f"Current {x_variable.title()}"
            )
    
    fig.update_layout(
        title=f"{model_choice} Option Prices vs {x_label}",
        xaxis_title=x_label,
        yaxis_title="Option Price",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def plot_greeks(scenario, opt_type):
    """Generate an interactive plot of option Greeks."""
    S, K, r, sigma, T, _, _ = scenario
    spot_range = np.linspace(max(1, 0.5 * S), 1.5 * S, 100)
    
    greeks_func = black_scholes_greeks_call if opt_type.lower() == "call" else black_scholes_greeks_put
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Delta", "Gamma", "Theta", "Vega"],
        vertical_spacing=0.15
    )
    
    greek_values = {
        'Delta': [], 'Gamma': [], 'Theta': [], 'Vega': []
    }
    
    for s in spot_range:
        greeks = greeks_func(s, K, r, sigma, T)
        for greek in greek_values:
            greek_values[greek].append(greeks[greek])
    
    positions = {
        'Delta': (1, 1), 'Gamma': (1, 2),
        'Theta': (2, 1), 'Vega': (2, 2)
    }
    
    for greek, values in greek_values.items():
        row, col = positions[greek]
        fig.add_trace(
            go.Scatter(
                x=spot_range,
                y=values,
                name=greek,
                line=dict(color=COLORS['primary'])
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        height=800,
        showlegend=False,
        template="plotly_white",
        title_text=f"Option Greeks ({opt_type})"
    )
    
    return fig

def main():
    """Main application function."""
    setup_page_config()
    
    # Title and Introduction
    st.markdown(
        "<h1 style='text-align: center; color: #2C3E50;'>Advanced Option Pricing Visualizer</h1>",
        unsafe_allow_html=True
    )
    
    st.markdown("""
        This tool provides interactive visualization and analysis of option pricing models.
        Compare different pricing methods, explore Greeks, and analyze volatility surfaces
        with real-time market data integration.
    """)
    
    # Sidebar Configuration
    st.sidebar.header("Configuration")
    
    # Market Data Integration
    ticker_symbol = st.sidebar.text_input("Stock Ticker (e.g., AAPL)", value="AAPL")
    market_data = None
    
    if st.sidebar.button("Fetch Market Data"):
        with st.spinner("Fetching market data..."):
            market_data = fetch_market_data(ticker_symbol)
            if market_data:
                st.sidebar.success(f"Successfully fetched data for {ticker_symbol}")
            else:
                st.sidebar.error("Failed to fetch market data")
    
    # Model Selection
    model_choice = st.sidebar.selectbox(
        "Pricing Model",
        ["Black-Scholes", "Binomial", "Monte Carlo"],
        help="Select the pricing model to use for calculations"
    )
    
    # Option Type
    opt_type_choice = st.sidebar.selectbox(
        "Option Type",
        ["Call", "Put", "Both"],
        help="Select which option type(s) to analyze"
    )
    
    # Analysis Mode
    analysis_mode = st.sidebar.selectbox(
        "Analysis Mode",
        ["Basic", "Advanced"],
        help="Basic mode shows price plots, Advanced includes Greeks and volatility surface"
    )
    
    # Main Content Area
    tab1, tab2, tab3 = st.tabs(["Price Analysis", "Greeks", "Volatility Surface"])
    
    with tab1:
        # Scenario Setup
        scenario1 = scenario_input_box(1, market_data)
        
        if st.checkbox("Add Second Scenario"):
            scenario2 = scenario_input_box(2, market_data)
            scenarios = [scenario1, scenario2]
        else:
            scenarios = [scenario1]
        
        # Variable to plot against
        x_variable = st.selectbox(
            "Plot Against",
            ["spot", "strike", "time"],
            format_func=lambda x: {"spot": "Spot Price", "strike": "Strike Price", "time": "Time to Maturity"}[x]
        )
        
        # Generate and display the main price plot
        price_fig = plot_option_prices(scenarios, model_choice, opt_type_choice, x_variable)
        st.plotly_chart(price_fig, use_container_width=True)
    
    with tab2:
        if model_choice == "Black-Scholes":
            st.plotly_chart(plot_greeks(scenario1, opt_type_choice), use_container_width=True)
        else:
            st.info("Greeks analysis is only available for the Black-Scholes model")
    
    with tab3:
        if analysis_mode == "Advanced":
            # Generate and display volatility surface
            surface_fig = implied_volatility_surface(scenario1[0], scenario1[1], scenario1[2])
            st.plotly_chart(surface_fig, use_container_width=True)
        else:
            st.info("Volatility surface analysis is available in Advanced mode")
    
    # Footer with additional information
    st.markdown("---")
    st.markdown("""
        ### Tips for Usage
        - Use the sidebar to configure the main parameters and analysis mode
        - Compare different scenarios by enabling the second scenario
        - Fetch real market data to compare theoretical prices with market values
        - Explore the Greeks and volatility surface in Advanced mode
    """)

if __name__ == "__main__":
    main()