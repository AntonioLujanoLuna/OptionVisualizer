# src/ui/pages/portfolio.py

import streamlit as st
import pandas as pd
from typing import List
from ..ui.components.portfolio_table import PortfolioTable # Relative import (still works within the same package)
from ..ui.components.charts import OptionPayoffChart # Relative import (still works within the same package)
from ..analytics.risk import Position, RiskAnalyzer # Relative import (still works within the same package)
from ..models.black_scholes import BlackScholesModel # Relative import (still works within the same package)

def render_page():
    """
    Render the portfolio management and analysis page.
    
    This page allows users to:
    1. Add and manage positions
    2. View portfolio analytics
    3. Analyze risk metrics
    4. Visualize portfolio characteristics
    """
    st.header("Portfolio Management")

    # Portfolio input method selection
    input_method = st.radio(
        "Input Method",
        ["Manual Entry", "Upload Positions", "Sample Portfolio"],
        horizontal=True
    )

    if input_method == "Manual Entry":
        _render_manual_entry()
    elif input_method == "Upload Positions":
        _render_file_upload()
    else:
        _render_sample_portfolio()

    # Display portfolio analysis if positions exist
    if 'portfolio_positions' in st.session_state and st.session_state.portfolio_positions:
        _render_portfolio_analysis()

def _render_manual_entry():
    """Render manual position entry form."""
    with st.expander("Add New Position", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            option_type = st.selectbox("Option Type", ["Call", "Put"])
            strike = st.number_input("Strike Price", min_value=0.01, value=100.0)
            expiry = st.slider("Time to Expiry (Years)", 0.1, 2.0, 1.0)
        
        with col2:
            quantity = st.number_input("Quantity", value=1, step=1)
            underlying = st.number_input("Underlying Price", min_value=0.01, value=100.0)
            volatility = st.slider("Volatility", 0.1, 1.0, 0.2)

        if st.button("Add Position"):
            if 'portfolio_positions' not in st.session_state:
                st.session_state.portfolio_positions = []
            
            new_position = Position(
                option_type=option_type.lower(),
                strike=strike,
                expiry=expiry,
                quantity=quantity,
                underlying_price=underlying,
                volatility=volatility,
                risk_free_rate=0.05  # Default value
            )
            
            st.session_state.portfolio_positions.append(new_position)
            st.success("Position added successfully!")

def _render_file_upload():
    """Handle portfolio file upload."""
    uploaded_file = st.file_uploader(
        "Upload Portfolio CSV",
        type=['csv'],
        help="CSV with columns: Type,Strike,Expiry,Quantity,Underlying,Volatility"
    )
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            required_columns = ['Type', 'Strike', 'Expiry', 'Quantity', 
                              'Underlying', 'Volatility']
            
            if not all(col in df.columns for col in required_columns):
                st.error("CSV file must contain all required columns")
                return
            
            positions = []
            for _, row in df.iterrows():
                position = Position(
                    option_type=row['Type'].lower(),
                    strike=row['Strike'],
                    expiry=row['Expiry'],
                    quantity=row['Quantity'],
                    underlying_price=row['Underlying'],
                    volatility=row['Volatility'],
                    risk_free_rate=0.05
                )
                positions.append(position)
            
            st.session_state.portfolio_positions = positions
            st.success("Portfolio loaded successfully!")
        
        except Exception as e:
            st.error(f"Error loading portfolio: {str(e)}")

def _render_portfolio_analysis():
    """Render portfolio analysis section."""
    st.subheader("Portfolio Analysis")

    # Display current positions
    portfolio_table = PortfolioTable(st.session_state.portfolio_positions)
    portfolio_table.render()

    # Initialize risk analyzer with Black-Scholes model
    risk_analyzer = RiskAnalyzer(BlackScholesModel())

    # Calculate risk metrics
    risk_metrics = risk_analyzer.calculate_portfolio_risk(
        st.session_state.portfolio_positions
    )

    # Display risk metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Delta Exposure", f"{risk_metrics.delta_exposure:.2f}")
    with col2:
        st.metric("Value at Risk (95%)", f"${risk_metrics.value_at_risk:.2f}")
    with col3:
        st.metric("Expected Shortfall", f"${risk_metrics.expected_shortfall:.2f}")

    # Display payoff chart
    payoff_chart = OptionPayoffChart()
    fig = payoff_chart.create_payoff_chart(st.session_state.portfolio_positions)
    st.plotly_chart(fig, use_container_width=True)

    # Stress test results
    if risk_metrics.stress_scenarios:
        st.subheader("Stress Test Results")
        scenarios_df = pd.DataFrame.from_dict(
            risk_metrics.stress_scenarios,
            orient='index',
            columns=['P&L Impact']
        )
        st.dataframe(scenarios_df)