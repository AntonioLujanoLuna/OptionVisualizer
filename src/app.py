"""
app.py

Main Streamlit application combining:
- pricing models (imported from pricing_models.py)
- utilities (market data from utils.py)
- visually appealing UI
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from pricing_models import (
    option_price,
    black_scholes_greeks_call,
    black_scholes_greeks_put
)
from utils import fetch_current_price

# For matplotlib styling, you can pick a nicer style
plt.style.use("seaborn-darkgrid")

def scenario_input_box(scenario_num=1, override_spot=None):
    """
    Renders input sliders for a single scenario in an expander box.
    If override_spot is not None, that value is used for Spot if checkbox is selected.
    Returns (S, K, r, sigma, T, steps, sims) chosen by the user.
    """
    expander = st.expander(f"Scenario {scenario_num} Settings", expanded=True)
    with expander:
        S = st.slider(f"Spot (S{scenario_num})", 1.0, 1000.0, 100.0, 1.0, key=f"S{scenario_num}")
        if override_spot is not None:
            # Let user override Spot with live market data
            if st.checkbox(f"Override S{scenario_num} with market price ({override_spot:.2f})?", key=f"override{scenario_num}"):
                S = override_spot

        K = st.slider(f"Strike (K{scenario_num})", 1.0, 1000.0, 100.0, 1.0, key=f"K{scenario_num}")
        r = st.slider(f"Risk-free rate (r{scenario_num})", 0.00, 0.20, 0.05, 0.01, key=f"r{scenario_num}")
        sigma = st.slider(f"Volatility (σ{scenario_num})", 0.01, 2.0, 0.2, 0.01, key=f"sigma{scenario_num}")
        T = st.slider(f"Time to Maturity (T{scenario_num})", 0.01, 5.0, 1.0, 0.01, key=f"T{scenario_num}")

        st.markdown("**Advanced Parameters** (Binomial & Monte Carlo)")
        steps = st.number_input(f"Binomial steps (Scenario {scenario_num})", min_value=10, max_value=2000, value=100, step=10, key=f"steps{scenario_num}")
        sims = st.number_input(f"Monte Carlo simulations (Scenario {scenario_num})", min_value=1000, max_value=200000, value=10000, step=1000, key=f"sims{scenario_num}")

    return S, K, r, sigma, T, steps, sims

def main():
    st.set_page_config(
        page_title="Advanced Option Pricing",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # ----------------------------------------------------------------------
    # Title and Intro
    # ----------------------------------------------------------------------
    st.markdown("<h1 style='text-align: center; color: #3c4a5c;'>Option Pricing Visualizer</h1>", unsafe_allow_html=True)
    st.write("""  
    **Compare multiple option pricing models** (Black-Scholes, Binomial, Monte Carlo)  
    and see how key parameters (Spot \(S\), Strike \(K\), Risk-free \(r\), Volatility \(\sigma\), and Time \(T\))  
    affect European Call/Put prices and Greeks.
    """)

    # ----------------------------------------------------------------------
    # Sidebar: Market Data Fetch
    # ----------------------------------------------------------------------
    st.sidebar.header("Market Data Integration")
    ticker_symbol = st.sidebar.text_input("Stock Ticker (e.g. AAPL)", value="AAPL")
    if st.sidebar.button("Fetch Current Price"):
        current_price = fetch_current_price(ticker_symbol)
        if current_price:
            st.sidebar.success(f"Fetched price for {ticker_symbol}: {current_price:.2f}")
        else:
            current_price = None
            st.sidebar.error("Failed to fetch data. Check ticker or connection.")
    else:
        current_price = None

    # ----------------------------------------------------------------------
    # Sidebar: Model and Option Type
    # ----------------------------------------------------------------------
    st.sidebar.header("General Settings")
    model_choice = st.sidebar.selectbox("Pricing Model", ["Black-Scholes", "Binomial", "Monte Carlo"], index=0)
    opt_type_choice = st.sidebar.selectbox("Option Type to Display", ["Call", "Put", "Both"], index=0)
    scenario_count = st.sidebar.radio("Number of Scenarios", [1, 2], index=0)

    # Plot dimension
    plot_dimension = st.sidebar.radio("Plot X-axis", ["Underlying Price (S)", "Time to Maturity (T)", "Volatility (σ)"])

    # ----------------------------------------------------------------------
    # Gather inputs for each scenario
    # ----------------------------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        scenario1 = scenario_input_box(scenario_num=1, override_spot=current_price)

    scenario2 = None
    if scenario_count == 2:
        with col2:
            scenario2 = scenario_input_box(scenario_num=2, override_spot=current_price)

    # Unpack scenario 1
    S1, K1, r1, sigma1, T1, steps1, sims1 = scenario1

    # Unpack scenario 2 if needed
    if scenario2:
        S2, K2, r2, sigma2, T2, steps2, sims2 = scenario2

    # ----------------------------------------------------------------------
    # Display Option Prices & Greeks
    # ----------------------------------------------------------------------
    st.markdown("---")
    st.markdown("<h2 style='color: #2E86C1;'>Current Prices & Greeks</h2>", unsafe_allow_html=True)
    results_col1, results_col2 = st.columns(2)

    # Scenario 1 results
    with results_col1:
        st.subheader("Scenario 1")
        if opt_type_choice in ["Call", "Both"]:
            call_price_1 = option_price(
                S1, K1, r1, sigma1, T1,
                opt_type="call", model=model_choice,
                steps=steps1, sims=sims1
            )
            st.write(f"**Call Price**: {call_price_1:.4f}")

            # If Black-Scholes, show call Greeks
            if model_choice == "Black-Scholes":
                greeks_call_1 = black_scholes_greeks_call(S1, K1, r1, sigma1, T1)
                st.write(f"**Delta**: {greeks_call_1['Delta']:.4f}")
                st.write(f"**Gamma**: {greeks_call_1['Gamma']:.4f}")
                st.write(f"**Vega**:  {greeks_call_1['Vega']:.4f}")
                st.write(f"**Theta**: {greeks_call_1['Theta']:.4f}")
                st.write(f"**Rho**:   {greeks_call_1['Rho']:.4f}")

        if opt_type_choice in ["Put", "Both"]:
            put_price_1 = option_price(
                S1, K1, r1, sigma1, T1,
                opt_type="put", model=model_choice,
                steps=steps1, sims=sims1
            )
            st.write(f"**Put Price**: {put_price_1:.4f}")

            # If Black-Scholes, show put Greeks
            if model_choice == "Black-Scholes":
                greeks_put_1 = black_scholes_greeks_put(S1, K1, r1, sigma1, T1)
                st.write(f"**Delta**: {greeks_put_1['Delta']:.4f}")
                st.write(f"**Gamma**: {greeks_put_1['Gamma']:.4f}")
                st.write(f"**Vega**:  {greeks_put_1['Vega']:.4f}")
                st.write(f"**Theta**: {greeks_put_1['Theta']:.4f}")
                st.write(f"**Rho**:   {greeks_put_1['Rho']:.4f}")

    # Scenario 2 results
    if scenario_count == 2:
        with results_col2:
            st.subheader("Scenario 2")
            if opt_type_choice in ["Call", "Both"]:
                call_price_2 = option_price(
                    S2, K2, r2, sigma2, T2,
                    opt_type="call", model=model_choice,
                    steps=steps2, sims=sims2
                )
                st.write(f"**Call Price**: {call_price_2:.4f}")

                if model_choice == "Black-Scholes":
                    greeks_call_2 = black_scholes_greeks_call(S2, K2, r2, sigma2, T2)
                    st.write(f"**Delta**: {greeks_call_2['Delta']:.4f}")
                    st.write(f"**Gamma**: {greeks_call_2['Gamma']:.4f}")
                    st.write(f"**Vega**:  {greeks_call_2['Vega']:.4f}")
                    st.write(f"**Theta**: {greeks_call_2['Theta']:.4f}")
                    st.write(f"**Rho**:   {greeks_call_2['Rho']:.4f}")

            if opt_type_choice in ["Put", "Both"]:
                put_price_2 = option_price(
                    S2, K2, r2, sigma2, T2,
                    opt_type="put", model=model_choice,
                    steps=steps2, sims=sims2
                )
                st.write(f"**Put Price**: {put_price_2:.4f}")

                if model_choice == "Black-Scholes":
                    greeks_put_2 = black_scholes_greeks_put(S2, K2, r2, sigma2, T2)
                    st.write(f"**Delta**: {greeks_put_2['Delta']:.4f}")
                    st.write(f"**Gamma**: {greeks_put_2['Gamma']:.4f}")
                    st.write(f"**Vega**:  {greeks_put_2['Vega']:.4f}")
                    st.write(f"**Theta**: {greeks_put_2['Theta']:.4f}")
                    st.write(f"**Rho**:   {greeks_put_2['Rho']:.4f}")

    # ----------------------------------------------------------------------
    # Interactive Plot
    # ----------------------------------------------------------------------
    st.markdown("---")
    st.markdown("<h2 style='color: #2E86C1;'>Interactive Plot</h2>", unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(9, 6))

    def generate_plot_data(S0, K0, r0, sigma0, T0, steps0, sims0):
        """
        Returns x_vals, call_vals, put_vals for the chosen axis.
        """
        if plot_dimension == "Underlying Price (S)":
            x_vals = np.linspace(1, 1000, 100)  # from 1 to 1000 in 100 steps
            call_vals = [
                option_price(x, K0, r0, sigma0, T0, "call", model_choice, steps0, sims0)
                for x in x_vals
            ]
            put_vals = [
                option_price(x, K0, r0, sigma0, T0, "put", model_choice, steps0, sims0)
                for x in x_vals
            ]
        elif plot_dimension == "Time to Maturity (T)":
            x_vals = np.linspace(0.01, 5.0, 100)
            call_vals = [
                option_price(S0, K0, r0, sigma0, t, "call", model_choice, steps0, sims0)
                for t in x_vals
            ]
            put_vals = [
                option_price(S0, K0, r0, sigma0, t, "put", model_choice, steps0, sims0)
                for t in x_vals
            ]
        else:  # "Volatility (σ)"
            x_vals = np.linspace(0.01, 2.0, 100)
            call_vals = [
                option_price(S0, K0, r0, x, T0, "call", model_choice, steps0, sims0)
                for x in x_vals
            ]
            put_vals = [
                option_price(S0, K0, r0, x, T0, "put", model_choice, steps0, sims0)
                for x in x_vals
            ]

        return x_vals, call_vals, put_vals

    # Plot scenario 1
    x1, call1, put1 = generate_plot_data(S1, K1, r1, sigma1, T1, steps1, sims1)
    if opt_type_choice in ["Call", "Both"]:
        ax.plot(x1, call1, label="Scenario1 - Call", color="blue")
    if opt_type_choice in ["Put", "Both"]:
        ax.plot(x1, put1, label="Scenario1 - Put", color="red")

    # Vertical line for scenario 1 current point
    if plot_dimension == "Underlying Price (S)":
        ax.axvline(S1, color="blue", linestyle="--")
    elif plot_dimension == "Time to Maturity (T)":
        ax.axvline(T1, color="blue", linestyle="--")
    else:
        ax.axvline(sigma1, color="blue", linestyle="--")

    # Plot scenario 2 if selected
    if scenario_count == 2:
        x2, call2, put2 = generate_plot_data(S2, K2, r2, sigma2, T2, steps2, sims2)
        if opt_type_choice in ["Call", "Both"]:
            ax.plot(x2, call2, label="Scenario2 - Call", color="blue", linestyle=":")
        if opt_type_choice in ["Put", "Both"]:
            ax.plot(x2, put2, label="Scenario2 - Put", color="red", linestyle=":")

        # Vertical line for scenario 2
        if plot_dimension == "Underlying Price (S)":
            ax.axvline(S2, color="red", linestyle="--")
        elif plot_dimension == "Time to Maturity (T)":
            ax.axvline(T2, color="red", linestyle="--")
        else:
            ax.axvline(sigma2, color="red", linestyle="--")

    if opt_type_choice == "Both":
        ax.set_ylabel("Option Price")
    else:
        ax.set_ylabel(f"{opt_type_choice} Option Price")

    ax.set_xlabel(plot_dimension)
    ax.set_title(f"{model_choice} Prices vs {plot_dimension}")
    ax.legend()
    st.pyplot(fig)

    st.markdown("---")
    st.info("""
    **Tips for Further Exploration**:
    - Increase Binomial steps or Monte Carlo simulations in the advanced settings for more accuracy.
    - Try large volatilities, or compare different maturities side-by-side.
    - Fetch real market data and compare the theoretical price to live quotes!
    """)

if __name__ == "__main__":
    main()
