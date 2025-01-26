# src/ui/pages/option_pricing.py

import streamlit as st
from ..components.option_inputs import OptionInputs
from ...models.black_scholes import BlackScholesModel

def render_page():
    """Render the option pricing analysis page."""
    st.header("Option Pricing Analysis")
    
    # Create option inputs component
    inputs = OptionInputs()
    parameters = inputs.render()
    
    # Model selection
    model = st.selectbox(
        "Pricing Model",
        ["Black-Scholes", "Binomial", "Monte Carlo"]
    )
    
    # Calculate option price
    pricing_model = BlackScholesModel()  # Or other model based on selection
    if parameters.option_type == "call":
        result = pricing_model.price_call(
            parameters.underlying_price,
            parameters.strike_price,
            parameters.risk_free_rate,
            parameters.volatility,
            parameters.time_to_expiry
        )
    else:
        result = pricing_model.price_put(
            parameters.underlying_price,
            parameters.strike_price,
            parameters.risk_free_rate,
            parameters.volatility,
            parameters.time_to_expiry
        )
    
    # Display results
    st.subheader("Option Price")
    st.metric(
        label="Price",
        value=f"${result.price:.2f}"
    )
    
    if result.greeks:
        st.subheader("Greeks")
        cols = st.columns(len(result.greeks))
        for col, (greek, value) in zip(cols, result.greeks.items()):
            col.metric(greek.capitalize(), f"{value:.4f}")