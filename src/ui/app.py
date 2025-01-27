# src/ui/app.py

import streamlit as st
from typing import Dict, List, Optional
import plotly.graph_objects as go
from datetime import datetime

from ..models.black_scholes import BlackScholesModel
from ..models.binomial import BinomialModel
from ..models.monte_carlo import MonteCarloModel
from ..analytics.portfolio_visualization import PortfolioVisualizer
from ..analytics.strategy_visualization import StrategyVisualizer
from ..analytics.volatility import VolatilityAnalyzer
from .education import OptionsEducation
from ..config import AppConfig

class OptionVisualizerApp:
    """
    Main application class for the Option Pricing Visualizer.
    
    This class orchestrates the entire application, providing:
    1. Clean, intuitive UI layout
    2. Interactive model parameter inputs
    3. Real-time visualization updates
    4. Educational content integration
    5. Advanced analytics features
    """
    
    def __init__(self):
        """Initialize application components and state."""
        # Initialize pricing models
        self.black_scholes = BlackScholesModel()
        self.binomial = BinomialModel()
        self.monte_carlo = MonteCarloModel()
        
        # Initialize visualization components
        self.portfolio_viz = PortfolioVisualizer()
        self.strategy_viz = StrategyVisualizer(self.black_scholes)
        self.vol_analyzer = VolatilityAnalyzer(self.black_scholes)
        
        # Initialize educational content
        self.education = OptionsEducation()
        
        # Set page configuration
        st.set_page_config(
            page_title="Option Pricing Visualizer",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Apply custom styling
        self._apply_custom_styling()
    
    def run(self):
        """Main application entry point."""
        # Display header and introduction
        self._render_header()
        
        # Create main navigation
        tab_options = [
            "Option Pricing",
            "Strategy Builder",
            "Portfolio Analysis",
            "Volatility Analysis",
            "Learning Center"
        ]
        
        selected_tab = st.sidebar.radio("Navigation", tab_options)
        
        # Render selected tab content
        if selected_tab == "Option Pricing":
            self._render_pricing_tab()
        elif selected_tab == "Strategy Builder":
            self._render_strategy_tab()
        elif selected_tab == "Portfolio Analysis":
            self._render_portfolio_tab()
        elif selected_tab == "Volatility Analysis":
            self._render_volatility_tab()
        else:  # Learning Center
            self._render_learning_tab()
    
    def _render_header(self):
        """Render application header with title and introduction."""
        st.markdown(
            """
            <h1 style='text-align: center;'>Option Pricing Visualizer</h1>
            
            <p style='text-align: center; font-size: 1.2em;'>
            An interactive tool for understanding option pricing, strategies, and risk management
            </p>
            """,
            unsafe_allow_html=True
        )
        
        with st.expander("About this Tool", expanded=False):
            st.markdown(
                """
                This tool helps you understand options through:
                - Interactive pricing model comparisons
                - Visual strategy analysis
                - Portfolio risk assessment
                - Educational content and tutorials
                
                Start by selecting a section from the sidebar navigation.
                """
            )
    
    def _render_pricing_tab(self):
        """Render the option pricing analysis tab."""
        st.header("Option Pricing Analysis")
        
        # Create two columns for input parameters
        col1, col2 = st.columns(2)
        
        with col1:
            # Basic parameters
            st.subheader("Option Parameters")
            underlying_price = st.number_input(
                "Underlying Price",
                min_value=1.0,
                value=100.0,
                step=1.0,
                help="Current price of the underlying asset"
            )
            
            strike_price = st.number_input(
                "Strike Price",
                min_value=1.0,
                value=100.0,
                step=1.0,
                help="Strike price of the option"
            )
            
            time_to_expiry = st.slider(
                "Time to Expiry (Years)",
                min_value=0.1,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="Time until option expiration in years"
            )
        
        with col2:
            # Market parameters
            st.subheader("Market Parameters")
            volatility = st.slider(
                "Volatility",
                min_value=0.1,
                max_value=1.0,
                value=0.2,
                step=0.05,
                help="Annualized volatility"
            )
            
            risk_free_rate = st.slider(
                "Risk-free Rate",
                min_value=0.0,
                max_value=0.1,
                value=0.05,
                step=0.01,
                help="Annual risk-free interest rate"
            )
        
        # Model selection and comparison
        st.subheader("Model Selection")
        models = st.multiselect(
            "Select Models to Compare",
            ["Black-Scholes", "Binomial", "Monte Carlo"],
            default=["Black-Scholes"]
        )
        
        option_type = st.radio(
            "Option Type",
            ["Call", "Put"],
            horizontal=True
        )
        
        if not models:
            st.warning("Please select at least one model to analyze.")
            return
        
        # Calculate and display results
        results = self._calculate_model_results(
            models,
            underlying_price,
            strike_price,
            time_to_expiry,
            volatility,
            risk_free_rate,
            option_type.lower()
        )
        
        self._display_pricing_results(results)
        
        # Display interactive visualizations
        st.subheader("Visual Analysis")
        visualization_type = st.selectbox(
            "Select Visualization",
            ["Price vs. Underlying", "Greeks Profile", "Time Decay"]
        )
        
        self._display_pricing_visualization(
            visualization_type,
            results,
            underlying_price,
            strike_price
        )
    
    def _calculate_model_results(self, models: List[str], S: float, K: float,
                               T: float, sigma: float, r: float,
                               option_type: str) -> Dict[str, float]:
        """Calculate option prices and Greeks using selected models."""
        results = {}
        
        for model in models:
            if model == "Black-Scholes":
                if option_type == "call":
                    result = self.black_scholes.price_call(S, K, r, sigma, T)
                else:
                    result = self.black_scholes.price_put(S, K, r, sigma, T)
            elif model == "Binomial":
                result = self.binomial.price_option(
                    S, K, r, sigma, T,
                    option_type=option_type,
                    exercise="european"
                )
            else:  # Monte Carlo
                if option_type == "call":
                    result = self.monte_carlo.price_call(S, K, r, sigma, T)
                else:
                    result = self.monte_carlo.price_put(S, K, r, sigma, T)
            
            results[model] = result
        
        return results
    
    def _display_pricing_results(self, results: Dict[str, float]):
        """Display pricing results in a clear, organized manner."""
        st.subheader("Pricing Results")
        
        # Create columns for each model
        cols = st.columns(len(results))
        
        for col, (model, result) in zip(cols, results.items()):
            with col:
                st.metric(
                    label=f"{model} Price",
                    value=f"${result.price:.2f}"
                )
                
                if result.greeks:
                    st.markdown("#### Greeks")
                    for greek, value in result.greeks.items():
                        st.metric(greek.capitalize(), f"{value:.4f}")
    
    def _display_pricing_visualization(self, viz_type: str, results: Dict,
                                    current_price: float, strike: float):
        """Create and display interactive pricing visualizations."""
        # Implementation of visualization logic...
        pass

    def _render_strategy_tab(self):
        """Render the strategy builder and analysis tab."""
        st.header("Options Strategy Builder")
        
        # Strategy selection or custom builder
        strategy_mode = st.radio(
            "Strategy Mode",
            ["Pre-defined Strategies", "Custom Strategy Builder"],
            horizontal=True
        )
        
        if strategy_mode == "Pre-defined Strategies":
            strategy = st.selectbox(
                "Select Strategy",
                [
                    "Bull Call Spread",
                    "Bear Put Spread",
                    "Iron Condor",
                    "Butterfly Spread",
                    "Calendar Spread",
                    "Straddle",
                    "Strangle"
                ]
            )
            
            # Display strategy information
            st.markdown(self.education.get_strategy_description(strategy))
            
            # Strategy parameters
            col1, col2 = st.columns(2)
            with col1:
                underlying_price = st.number_input(
                    "Underlying Price",
                    min_value=1.0,
                    value=100.0,
                    step=1.0
                )
                
                width = st.slider(
                    "Strategy Width",
                    min_value=5,
                    max_value=50,
                    value=10,
                    step=5,
                    help="Distance between strikes in the strategy"
                )
            
            with col2:
                expiry = st.slider(
                    "Days to Expiration",
                    min_value=7,
                    max_value=365,
                    value=30,
                    step=7
                )
                
                size = st.number_input(
                    "Position Size (contracts)",
                    min_value=1,
                    value=1,
                    step=1
                )
            
            # Generate and display strategy analysis
            strategy_profile = self._generate_strategy_profile(
                strategy,
                underlying_price,
                width,
                expiry,
                size
            )
            
            self._display_strategy_analysis(strategy_profile)
            
        else:  # Custom Strategy Builder
            st.subheader("Build Custom Strategy")
            
            # Initialize or get existing legs
            if 'strategy_legs' not in st.session_state:
                st.session_state.strategy_legs = []
            
            # Add new leg form
            with st.expander("Add Strategy Leg", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    option_type = st.selectbox(
                        "Option Type",
                        ["Call", "Put"],
                        key="new_leg_type"
                    )
                    
                    position = st.radio(
                        "Position",
                        ["Long", "Short"],
                        horizontal=True,
                        key="new_leg_position"
                    )
                
                with col2:
                    strike = st.number_input(
                        "Strike Price",
                        min_value=1.0,
                        value=100.0,
                        step=1.0,
                        key="new_leg_strike"
                    )
                    
                    quantity = st.number_input(
                        "Quantity",
                        min_value=1,
                        value=1,
                        step=1,
                        key="new_leg_quantity"
                    )
                
                if st.button("Add Leg"):
                    st.session_state.strategy_legs.append({
                        'type': option_type,
                        'position': position,
                        'strike': strike,
                        'quantity': quantity * (1 if position == "Long" else -1)
                    })
            
            # Display current strategy composition
            if st.session_state.strategy_legs:
                st.subheader("Current Strategy Composition")
                for i, leg in enumerate(st.session_state.strategy_legs):
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        st.write(f"{leg['position']} {leg['quantity']} {leg['type']}")
                    with col2:
                        st.write(f"Strike: {leg['strike']}")
                    with col3:
                        if st.button("Remove", key=f"remove_{i}"):
                            st.session_state.strategy_legs.pop(i)
                            st.experimental_rerun()
                
                # Analyze custom strategy
                custom_profile = self._generate_custom_strategy_profile(
                    st.session_state.strategy_legs
                )
                self._display_strategy_analysis(custom_profile)
            
            else:
                st.info("Add legs to your strategy using the form above.")
    
    def _render_portfolio_tab(self):
        """Render the portfolio analysis tab."""
        st.header("Portfolio Analysis")
        
        # Portfolio input method selection
        input_method = st.radio(
            "Input Method",
            ["Manual Entry", "Upload Positions", "Sample Portfolio"],
            horizontal=True
        )
        
        if input_method == "Manual Entry":
            # Similar to strategy builder but with more position details
            self._render_manual_portfolio_entry()
        
        elif input_method == "Upload Positions":
            uploaded_file = st.file_uploader(
                "Upload Portfolio CSV",
                type=['csv'],
                help="CSV file with columns: Type,Strike,Expiry,Quantity,Premium"
            )
            
            if uploaded_file:
                positions = self._parse_portfolio_file(uploaded_file)
                self._analyze_portfolio(positions)
        
        else:  # Sample Portfolio
            sample_name = st.selectbox(
                "Select Sample Portfolio",
                ["Covered Call Strategy", "Iron Condor Portfolio", "Delta-Neutral"]
            )
            
            positions = self._get_sample_portfolio(sample_name)
            self._analyze_portfolio(positions)