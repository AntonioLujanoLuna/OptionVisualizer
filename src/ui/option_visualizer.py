# src/ui/option_visualizer.py
import streamlit as st
from typing import Dict, List, Optional
import plotly.graph_objects as go

from src.models.black_scholes import BlackScholesModel
from src.models.binomial import BinomialModel
from src.models.monte_carlo import MonteCarloModel
from src.analytics.portfolio_visualization import PortfolioVisualizer
from src.analytics.strategy_visualization import StrategyVisualizer
from src.analytics.risk import RiskAnalyzer
from src.analytics.volatility import VolatilitySurface
from .education import OptionsEducation
from src.config import AppConfig

from .pages.home import render_page as render_home
from .pages.option_pricing import render_page as render_option_pricing
from .pages.portfolio import render_page as render_portfolio
from .pages.education import render_page as render_education
from .pages.strategy_builder import render_page as render_strategy_builder

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
        self.vol_analyzer = RiskAnalyzer(self.black_scholes)
        
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
            "Risk Analysis",
            "Learning Center"
        ]
        
        selected_tab = st.sidebar.radio("Navigation", tab_options)
        
        # Render selected tab content
        if selected_tab == "Option Pricing":
            self._render_pricing_tab()
        elif selected_tab == "Strategy Builder":
            render_strategy_builder() # Render strategy builder page
        elif selected_tab == "Portfolio Analysis":
            self._render_portfolio_tab()
        else: # Learning Center
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
                result = self.binomial.price_european(
                    S, K, r, sigma, T,
                    option_type=option_type
                )
            else: # Monte Carlo
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
            
        else: # Custom Strategy Builder
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
        
        else: # Sample Portfolio
            sample_name = st.selectbox(
                "Select Sample Portfolio",
                ["Covered Call Strategy", "Iron Condor Portfolio", "Delta-Neutral"]
            )
            
            positions = self._get_sample_portfolio(sample_name)
            self._analyze_portfolio(positions)

    def _apply_custom_styling(self):
        """Apply custom CSS styling to improve the look."""
        st.markdown(
            """
            <style>
            /* Customize Streamlit elements */
            .stApp {
                background-color: #f0f2f6; /* Light gray background */
            }
            .stTextInput, .stNumberInput, .stSelectbox, .stSlider {
                background-color: #ffffff; /* White input fields */
                border: 1px solid #ced4da; /* Light gray border */
                border-radius: 5px; /* Rounded corners */
            }
            .stButton button {
                background-color: #007bff; /* Primary blue color */
                color: white;
                border: none;
                border-radius: 5px;
                padding: 0.5rem 1rem;
            }
            .stButton button:hover {
                background-color: #0056b3; /* Darker blue on hover */
            }
            .stMetric {
                background-color: #e9ecef; /* Light gray background */
                border-radius: 5px;
                padding: 0.5rem;
            }
            .stMetric > div > div {
                font-size: 1.5rem; /* Larger font size for metric value */
            }
            .stMetric > div > div:nth-child(2) {
                font-size: 0.9rem; /* Smaller font size for label */
            }
            .stMarkdown p {
                font-size: 1.1em; /* Slightly larger paragraph text */
            }
            .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
                color: #2c3e50; /* Dark blue for headings */
            }
            </style>
            """,
            unsafe_allow_html=True
        )
    
    def _calculate_model_results(self, models, S, K, T, sigma, r, option_type):
        """Calculate option prices using the selected models."""
        results = {}
        for model_name in models:
            if model_name == "Black-Scholes":
                model = BlackScholesModel()
                if option_type == 'call':
                    results[model_name] = model.price_call(S, K, r, sigma, T)
                else:
                    results[model_name] = model.price_put(S, K, r, sigma, T)
            elif model_name == "Binomial":
                model = BinomialModel()
                # Assuming pricing for European options here
                results[model_name] = model.price_european(S, K, r, sigma, T, option_type)
            elif model_name == "Monte Carlo":
                model = MonteCarloModel()
                if option_type == 'call':
                    results[model_name] = model.price_call(S, K, r, sigma, T)
                else:
                    results[model_name] = model.price_put(S, K, r, sigma, T)
        return results

    def _display_pricing_results(self, results):
        """Display the pricing results in Streamlit metrics format."""
        st.subheader("Pricing Results")
        cols = st.columns(len(results))
        for i, (model_name, result) in enumerate(results.items()):
            with cols[i]:
                st.metric(label=f"{model_name} Price", value=f"{result.price:.2f}")
                if result.greeks:
                    st.markdown("Greeks:")
                    for greek, value in result.greeks.items():
                        st.text(f"{greek}: {value:.4f}")

    def _display_pricing_visualization(self, visualization_type, results, underlying_price, strike_price):
        """Display the selected visualization."""
        # This function can be expanded to include more sophisticated visualizations
        # using Plotly or other visualization libraries, based on the selected type.
        if visualization_type == "Price vs. Underlying":
            self._plot_price_vs_underlying(results, underlying_price, strike_price)
        elif visualization_type == "Greeks Profile":
            self._plot_greeks_profile(results, underlying_price, strike_price)
        elif visualization_type == "Time Decay":
            self._plot_time_decay(results, underlying_price, strike_price)

    def _plot_price_vs_underlying(self, results, underlying_price, strike_price):
        """Plot option price vs. underlying price for the selected models."""
        fig = go.Figure()
        price_range = np.linspace(underlying_price * 0.5, underlying_price * 1.5, 100)

        for model_name, result in results.items():
            prices = []
            if model_name == "Black-Scholes":
                for S in price_range:
                    if result.additional_info['parameters'].option_type == 'call':
                        price = self.black_scholes.price_call(S, strike_price, result.additional_info['parameters'].r,
                                                             result.additional_info['parameters'].sigma, result.additional_info['parameters'].T).price
                    else:
                        price = self.black_scholes.price_put(S, strike_price, result.additional_info['parameters'].r,
                                                            result.additional_info['parameters'].sigma, result.additional_info['parameters'].T).price
                    prices.append(price)
            elif model_name == "Binomial":
                for S in price_range:
                    prices.append(self.binomial.price_european(S, strike_price, result.additional_info['parameters'].r,
                                                            result.additional_info['parameters'].sigma, result.additional_info['parameters'].T,
                                                            result.additional_info['parameters'].option_type).price)
            elif model_name == "Monte Carlo":
                for S in price_range:
                    if result.additional_info['parameters'].option_type == 'call':
                        price = self.monte_carlo.price_call(S, strike_price, result.additional_info['parameters'].r,
                                                           result.additional_info['parameters'].sigma, result.additional_info['parameters'].T).price
                    else:
                        price = self.monte_carlo.price_put(S, strike_price, result.additional_info['parameters'].r,
                                                          result.additional_info['parameters'].sigma, result.additional_info['parameters'].T).price
                    prices.append(price)

            fig.add_trace(go.Scatter(x=price_range, y=prices, mode='lines', name=model_name))

        fig.update_layout(title='Option Price vs. Underlying Price',
                          xaxis_title='Underlying Price',
                          yaxis_title='Option Price')
        st.plotly_chart(fig)

    def _plot_greeks_profile(self, results, underlying_price, strike_price):
        """Plot Greeks profile for the selected models."""
        # This is a placeholder for the actual implementation.
        # You will need to implement the logic to calculate and plot the Greeks.
        st.info("Greek profile visualization is not yet implemented.")

    def _plot_time_decay(self, results, underlying_price, strike_price):
        """Plot option price vs. time to expiry for the selected models."""
        fig = go.Figure()
        time_range = np.linspace(0.1, 2.0, 100)  # Example time range

        for model_name, result in results.items():
            prices = []
            if model_name == "Black-Scholes":
                for T in time_range:
                    if result.additional_info['parameters'].option_type == 'call':
                        price = self.black_scholes.price_call(underlying_price, strike_price, result.additional_info['parameters'].r,
                                                             result.additional_info['parameters'].sigma, T).price
                    else:
                        price = self.black_scholes.price_put(underlying_price, strike_price, result.additional_info['parameters'].r,
                                                            result.additional_info['parameters'].sigma, T).price
                    prices.append(price)
            elif model_name == "Binomial":
                for T in time_range:
                    prices.append(self.binomial.price_european(underlying_price, strike_price, result.additional_info['parameters'].r,
                                                            result.additional_info['parameters'].sigma, T,
                                                            result.additional_info['parameters'].option_type).price)
            elif model_name == "Monte Carlo":
                for T in time_range:
                    if result.additional_info['parameters'].option_type == 'call':
                        price = self.monte_carlo.price_call(underlying_price, strike_price, result.additional_info['parameters'].r,
                                                           result.additional_info['parameters'].sigma, T).price
                    else:
                        price = self.monte_carlo.price_put(underlying_price, strike_price, result.additional_info['parameters'].r,
                                                          result.additional_info['parameters'].sigma, T).price
                    prices.append(price)

            fig.add_trace(go.Scatter(x=time_range, y=prices, mode='lines', name=model_name))

        fig.update_layout(title='Option Price vs. Time to Expiry',
                          xaxis_title='Time to Expiry (Years)',
                          yaxis_title='Option Price')
        st.plotly_chart(fig)