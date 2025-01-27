# src/ui/pages/education.py

import streamlit as st
from ..ui.components.charts import OptionPayoffChart
from ..models.black_scholes import BlackScholesModel
from ..analytics.risk import Position

def render_page():
    """
    Render the educational content page.
    
    This page provides interactive learning materials about:
    1. Basic option concepts
    2. Greeks and risk measures
    3. Trading strategies
    4. Portfolio management principles
    """
    st.header("Options Education Center")
    
    # Topic selection
    topic = st.selectbox(
        "Select Topic",
        ["Option Basics", "Greeks Explained", "Trading Strategies", "Risk Management"]
    )
    
    if topic == "Option Basics":
        _render_option_basics()
    elif topic == "Greeks Explained":
        _render_greeks_explanation()
    elif topic == "Trading Strategies":
        _render_strategy_education()
    else:
        _render_risk_management()

def _render_option_basics():
    """Render basic options education content."""
    st.subheader("Understanding Options")
    
    st.markdown("""
        Options are financial contracts that give the holder the right, but not the
        obligation, to buy (call) or sell (put) an underlying asset at a specified
        price (strike) before or at expiration.
        
        Let's explore how options work through interactive examples:
    """)
    
    # Interactive option payoff demonstration
    st.subheader("Option Payoff Demonstration")
    
    col1, col2 = st.columns(2)
    with col1:
        option_type = st.selectbox("Option Type", ["Call", "Put"])
        strike = st.number_input("Strike Price", value=100.0)
    with col2:
        premium = st.number_input("Option Premium", value=5.0)
        quantity = st.number_input("Number of Contracts", value=1)
    
    # Create demo position
    position = Position(
        option_type=option_type.lower(),
        strike=strike,
        expiry=1.0,
        quantity=quantity,
        underlying_price=100.0,
        volatility=0.2,
        risk_free_rate=0.05
    )
    
    # Display payoff diagram
    payoff_chart = OptionPayoffChart()
    fig = payoff_chart.create_payoff_chart([position])
    st.plotly_chart(fig)
    
    # Educational explanations
    st.markdown(f"""
        ### Key Points:
        
        - This {option_type.lower()} option gives you the right to 
          {'buy' if option_type.lower() == 'call' else 'sell'} at ${strike:.2f}
        - Maximum loss is limited to the premium paid: ${premium * quantity:.2f}
        - Break-even price: ${strike + premium if option_type.lower() == 'call' else strike - premium:.2f}
    """)

def _render_greeks_explanation():
    """Render interactive Greeks education content."""
    st.subheader("Understanding Option Greeks")
    
    # Create interactive Greek visualization
    model = BlackScholesModel()
    
    st.markdown("""
        Greeks measure how option prices change with respect to various factors.
        Adjust the parameters below to see how Greeks behave:
    """)
    
    # Interactive parameters
    col1, col2 = st.columns(2)
    with col1:
        spot = st.slider("Underlying Price", 80.0, 120.0, 100.0)
        strike = st.number_input("Strike Price", value=100.0)
    with col2:
        vol = st.slider("Volatility", 0.1, 0.5, 0.2)
        time = st.slider("Time to Expiry (Years)", 0.1, 2.0, 1.0)
    
    # Calculate and display Greeks
    call_greeks = model.calculate_greeks(spot, strike, 0.05, vol, time, "call")
    put_greeks = model.calculate_greeks(spot, strike, 0.05, vol, time, "put")
    
    st.subheader("Call Option Greeks")
    cols = st.columns(len(call_greeks))
    for col, (greek, value) in zip(cols, call_greeks.items()):
        col.metric(greek.capitalize(), f"{value:.4f}")
    
    # Add explanations for each Greek
    st.markdown("""
        ### Greek Interpretations:
        
        - **Delta**: Measures the rate of change in option price with respect to the underlying
        - **Gamma**: Measures the rate of change in Delta
        - **Theta**: Time decay; how much value is lost each day
        - **Vega**: Sensitivity to volatility changes
        - **Rho**: Sensitivity to interest rate changes
    """)

def _render_strategy_education():
    """Render options strategy education content."""
    st.subheader("Common Options Strategies")
    
    strategy = st.selectbox(
        "Select Strategy to Learn About",
        ["Bull Call Spread", "Iron Condor", "Covered Call", "Protective Put"]
    )
    
    # Display strategy details and interactive example
    if strategy == "Bull Call Spread":
        st.markdown("""
            A Bull Call Spread involves:
            1. Buying a call option
            2. Selling another call option with a higher strike
            
            This strategy is used when you:
            - Are moderately bullish
            - Want to reduce cost
            - Accept limited profit potential
        """)
        
        # Interactive example
        lower_strike = st.slider("Lower Strike", 90, 110, 100)
        upper_strike = st.slider("Upper Strike", lower_strike, 120, lower_strike + 5)
        
        # Create positions for visualization
        positions = [
            Position("call", lower_strike, 1.0, 1, 100.0, 0.2, 0.05),
            Position("call", upper_strike, 1.0, -1, 100.0, 0.2, 0.05)
        ]
        
        # Display payoff diagram
        payoff_chart = OptionPayoffChart()
        fig = payoff_chart.create_payoff_chart(positions)
        st.plotly_chart(fig)

# src/ui/pages/education.py (continued)

def _render_risk_management():
    """Render comprehensive risk management education content."""
    st.subheader("Portfolio Risk Management")
    
    st.markdown("""
        Understanding risk management is crucial for successful options trading. Let's explore
        the key concepts through interactive examples and practical scenarios.
        
        Risk management in options trading requires a multi-faceted approach that considers
        both market risks and position-specific characteristics. We'll examine each aspect
        and learn how to implement effective risk controls.
    """)
    
    # Risk concept selection
    risk_topic = st.selectbox(
        "Select Risk Management Topic",
        ["Position Sizing", "Greek Exposure Management", "Portfolio Diversification", 
         "Volatility Risk", "Black Swan Events"]
    )
    
    if risk_topic == "Position Sizing":
        _render_position_sizing_lesson()
    elif risk_topic == "Greek Exposure Management":
        _render_greek_management_lesson()
    elif risk_topic == "Portfolio Diversification":
        _render_diversification_lesson()
    elif risk_topic == "Volatility Risk":
        _render_volatility_risk_lesson()
    else:
        _render_black_swan_lesson()

def _render_position_sizing_lesson():
    """Teach position sizing principles through interactive examples."""
    st.markdown("""
        ### Position Sizing in Options Trading
        
        Position sizing is one of the most critical aspects of risk management. Proper
        position sizing helps protect your portfolio from unexpected market moves while
        allowing for meaningful profits when your analysis is correct.
    """)
    
    # Interactive position sizing calculator
    st.subheader("Position Size Calculator")
    
    col1, col2 = st.columns(2)
    with col1:
        account_size = st.number_input(
            "Account Size ($)",
            min_value=1000.0,
            value=100000.0,
            step=1000.0,
            help="Your total trading account size"
        )
        
        risk_percentage = st.slider(
            "Risk Per Trade (%)",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            help="Maximum percentage of account to risk on this trade"
        )
    
    with col2:
        option_premium = st.number_input(
            "Option Premium ($)",
            min_value=0.1,
            value=2.5,
            step=0.1,
            help="Cost of one option contract"
        )
        
        contract_multiplier = st.number_input(
            "Contract Multiplier",
            min_value=1,
            value=100,
            step=1,
            help="Standard equity option multiplier is 100"
        )
    
    # Calculate and display position sizing recommendations
    max_risk_amount = account_size * (risk_percentage / 100)
    max_contracts = int(max_risk_amount / (option_premium * contract_multiplier))
    
    st.markdown(f"""
        Based on your inputs:
        
        - Maximum risk amount: ${max_risk_amount:,.2f}
        - Maximum number of contracts: {max_contracts}
        - Total premium cost: ${(max_contracts * option_premium * contract_multiplier):,.2f}
        
        ### Position Sizing Guidelines
        
        When determining position size, consider:
        
        1. **Account Risk**: Never risk more than 1-2% of your account on a single trade
        2. **Correlation Risk**: Reduce position size when trading correlated assets
        3. **Volatility Adjustment**: Use smaller sizes in high volatility environments
        4. **Liquidity Considerations**: Ensure you can exit the position easily
    """)
    
    # Interactive scenario analysis
    st.subheader("Scenario Analysis")
    price_change = st.slider(
        "Simulate Price Change (%)",
        min_value=-50,
        max_value=50,
        value=0
    )
    
    # Calculate scenario outcomes
    position_value = max_contracts * option_premium * contract_multiplier
    value_change = position_value * (price_change / 100)
    new_account_value = account_size + value_change
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Position P&L", f"${value_change:,.2f}")
    col2.metric("Account Change", f"{(value_change/account_size)*100:.1f}%")
    col3.metric("New Account Value", f"${new_account_value:,.2f}")

def _render_greek_management_lesson():
    """Teach Greek exposure management through interactive scenarios."""
    st.markdown("""
        ### Managing Greek Exposures
        
        Greek exposures represent different dimensions of option risk. Understanding
        and managing these exposures is crucial for maintaining a balanced portfolio.
    """)
    
    # Interactive Greek exposure simulator
    st.subheader("Greek Exposure Simulator")
    
    # Create sample portfolio
    portfolio = [
        Position("call", 100, 1.0, 1, 100.0, 0.2, 0.05),
        Position("put", 95, 1.0, -2, 100.0, 0.2, 0.05),
        Position("call", 105, 1.0, -1, 100.0, 0.2, 0.05)
    ]
    
    # Allow users to modify market parameters
    col1, col2 = st.columns(2)
    with col1:
        price_change = st.slider(
            "Price Change (%)",
            min_value=-20,
            max_value=20,
            value=0
        )
        vol_change = st.slider(
            "Volatility Change (%)",
            min_value=-50,
            max_value=50,
            value=0
        )
    
    with col2:
        days_passed = st.slider(
            "Days Passed",
            min_value=0,
            max_value=30,
            value=0
        )
        rate_change = st.slider(
            "Rate Change (bps)",
            min_value=-50,
            max_value=50,
            value=0
        )
    
    # Calculate and display risk metrics under different scenarios
    model = BlackScholesModel()
    original_value = sum(model.price_call(p.underlying_price, p.strike, p.risk_free_rate, 
                                        p.volatility, p.expiry).price * p.quantity 
                        for p in portfolio if p.option_type == "call") + \
                    sum(model.price_put(p.underlying_price, p.strike, p.risk_free_rate,
                                      p.volatility, p.expiry).price * p.quantity 
                        for p in portfolio if p.option_type == "put")
    
    # Calculate new portfolio value under scenario
    new_portfolio = [
        Position(
            p.option_type,
            p.strike,
            p.expiry - days_passed/365,
            p.quantity,
            p.underlying_price * (1 + price_change/100),
            p.volatility * (1 + vol_change/100),
            p.risk_free_rate + rate_change/10000
        )
        for p in portfolio
    ]
    
    new_value = sum(model.price_call(p.underlying_price, p.strike, p.risk_free_rate,
                                   p.volatility, p.expiry).price * p.quantity 
                   for p in new_portfolio if p.option_type == "call") + \
                sum(model.price_put(p.underlying_price, p.strike, p.risk_free_rate,
                                  p.volatility, p.expiry).price * p.quantity 
                   for p in new_portfolio if p.option_type == "put")
    
    # Display scenario analysis results
    st.subheader("Scenario Analysis Results")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Portfolio P&L", f"${new_value - original_value:,.2f}")
    col2.metric("Percentage Change", f"{((new_value/original_value - 1) * 100):.1f}%")
    col3.metric("New Portfolio Value", f"${new_value:,.2f}")
    
    st.markdown("""
        ### Key Lessons in Greek Management
        
        1. **Delta Management**:
           - Keep overall portfolio delta aligned with your market view
           - Consider using delta hedging for neutral strategies
        
        2. **Gamma Risk**:
           - High gamma means rapid delta changes
           - Reduce gamma exposure near major events
        
        3. **Theta Decay**:
           - Time decay accelerates near expiration
           - Balance positive and negative theta positions
        
        4. **Vega Exposure**:
           - Manage volatility exposure across different strikes and expirations
           - Consider VIX products for volatility hedging
    """)

def _render_diversification_lesson():
    """Teach portfolio diversification principles through interactive examples."""
    st.markdown("""
        ### Understanding Portfolio Diversification in Options Trading
        
        Portfolio diversification in options trading goes beyond simply trading different 
        underlying assets. It involves managing exposures across multiple dimensions:
        
        - Different underlyings
        - Various expiration dates
        - Multiple strike prices
        - Different option strategies
        - Varied market conditions
        
        Let's explore how to build a well-diversified options portfolio through 
        practical examples.
    """)
    
    # Interactive portfolio builder
    st.subheader("Portfolio Diversification Simulator")
    
    # Allow users to construct a sample portfolio
    col1, col2 = st.columns(2)
    with col1:
        num_positions = st.slider(
            "Number of Positions",
            min_value=1,
            max_value=10,
            value=3,
            help="More positions can provide better diversification, but may increase complexity"
        )
        
        correlation = st.slider(
            "Average Correlation Between Positions",
            min_value=-1.0,
            max_value=1.0,
            value=0.3,
            help="Higher correlation means less diversification benefit"
        )
    
    with col2:
        strategy_mix = st.multiselect(
            "Strategy Types",
            ["Directional", "Neutral", "Volatility", "Income"],
            default=["Directional"],
            help="Different strategy types can provide strategic diversification"
        )
        
        market_scenarios = st.multiselect(
            "Market Scenarios to Test",
            ["Bull Market", "Bear Market", "High Volatility", "Low Volatility"],
            default=["Bull Market", "Bear Market"]
        )
    
    # Calculate and display diversification metrics
    portfolio_risk = calculate_portfolio_metrics(num_positions, correlation)
    
    st.markdown("### Portfolio Risk Analysis")
    col1, col2, col3 = st.columns(3)
    
    col1.metric(
        "Portfolio Risk",
        f"{portfolio_risk:.1f}%",
        help="Lower numbers indicate better diversification"
    )
    col2.metric(
        "Diversification Score",
        f"{(1 - correlation) * 100:.1f}/100",
        help="Higher scores indicate better diversification"
    )
    col3.metric(
        "Strategy Coverage",
        f"{len(strategy_mix)}/4",
        help="More strategy types can improve resilience"
    )
    
    # Show scenario analysis
    st.subheader("Scenario Analysis")
    _display_scenario_analysis(market_scenarios, strategy_mix, portfolio_risk)

def _render_volatility_risk_lesson():
    """Teach volatility risk management with interactive examples."""
    st.markdown("""
        ### Managing Volatility Risk in Options Trading
        
        Volatility is a crucial factor in options pricing and risk management. Unlike 
        stocks, options are affected by both price movement (directional risk) and 
        changes in expected volatility (volatility risk).
        
        Understanding and managing volatility risk involves:
        1. Measuring volatility exposure
        2. Understanding volatility term structure
        3. Managing vega risk
        4. Implementing volatility trading strategies
    """)
    
    # Interactive volatility analysis tool
    st.subheader("Volatility Risk Calculator")
    
    col1, col2 = st.columns(2)
    with col1:
        current_iv = st.slider(
            "Current Implied Volatility (%)",
            min_value=10,
            max_value=100,
            value=30,
            help="Current implied volatility level"
        )
        
        position_vega = st.number_input(
            "Position Vega",
            min_value=-1000.0,
            max_value=1000.0,
            value=100.0,
            help="Total portfolio vega exposure"
        )
    
    with col2:
        vol_change = st.slider(
            "Volatility Change (%)",
            min_value=-50,
            max_value=50,
            value=0,
            help="Simulate a change in implied volatility"
        )
        
        position_value = st.number_input(
            "Position Value ($)",
            min_value=0.0,
            value=10000.0,
            help="Current total position value"
        )
    
    # Calculate impact of volatility changes
    vol_pnl = calculate_volatility_pnl(position_vega, vol_change, position_value)
    
    # Display volatility risk metrics
    st.markdown("### Volatility Risk Impact")
    col1, col2, col3 = st.columns(3)
    
    col1.metric(
        "P&L Impact",
        f"${vol_pnl:,.2f}",
        f"{(vol_pnl/position_value)*100:.1f}%"
    )
    col2.metric(
        "New Position Value",
        f"${position_value + vol_pnl:,.2f}"
    )
    col3.metric(
        "New Implied Vol",
        f"{current_iv + vol_change:.1f}%",
        f"{vol_change:+.1f}%"
    )

def _render_black_swan_lesson():
    """Teach preparation for extreme market events."""
    st.markdown("""
        ### Preparing for Black Swan Events
        
        Black swan events are rare, unexpected occurrences that can have severe market 
        impacts. While we cannot predict these events, we can prepare our portfolios 
        to be more resilient when they occur.
        
        Let's explore strategies to protect against extreme market movements and 
        volatility spikes.
    """)
    
    # Interactive tail risk analysis
    st.subheader("Tail Risk Simulator")
    
    col1, col2 = st.columns(2)
    with col1:
        protection_level = st.slider(
            "Protection Level (%)",
            min_value=0,
            max_value=100,
            value=20,
            help="Percentage of portfolio protected against extreme events"
        )
        
        hedge_type = st.selectbox(
            "Hedge Type",
            ["Put Options", "VIX Calls", "Tail Risk Hedge Fund", "Mixed Strategy"]
        )
    
    with col2:
        market_drop = st.slider(
            "Market Drop Scenario (%)",
            min_value=-75,
            max_value=0,
            value=-30,
            help="Simulate a severe market decline"
        )
        
        vol_spike = st.slider(
            "Volatility Spike (%)",
            min_value=0,
            max_value=500,
            value=150,
            help="Simulate a volatility spike during market stress"
        )
    
    # Calculate impact of black swan event
    portfolio_impact = calculate_black_swan_impact(
        protection_level, hedge_type, market_drop, vol_spike
    )
    
    # Display protection analysis
    st.markdown("### Protection Analysis")
    col1, col2, col3 = st.columns(3)
    
    col1.metric(
        "Unhedged Loss",
        f"{portfolio_impact['unhedged']:,.1f}%"
    )
    col2.metric(
        "Hedged Loss",
        f"{portfolio_impact['hedged']:,.1f}%"
    )
    col3.metric(
        "Hedge Effectiveness",
        f"{portfolio_impact['effectiveness']:,.1f}%"
    )
    
    st.markdown("""
        ### Key Principles for Black Swan Protection
        
        1. **Diversification Beyond Correlation**
           - Standard correlations break down in extreme events
           - Need genuine strategy diversification
           - Consider anti-correlation assets
        
        2. **Positive Convexity**
           - Look for asymmetric payoff profiles
           - Small cost for large potential protection
           - Options-based strategies can provide convexity
        
        3. **Liquidity Management**
           - Maintain adequate cash reserves
           - Understand position liquidity under stress
           - Plan exit strategies in advance
        
        4. **Regular Stress Testing**
           - Test portfolio under extreme scenarios
           - Consider multiple risk factors simultaneously
           - Update protection strategies based on results
    """)

# Helper functions for calculations

def calculate_portfolio_metrics(num_positions: int, correlation: float) -> float:
    """Calculate portfolio risk metrics based on position count and correlation."""
    # Simple portfolio risk calculation considering diversification effects
    individual_risk = 20.0  # Assume 20% risk per position
    portfolio_risk = individual_risk * np.sqrt(
        (1/num_positions) + ((num_positions-1)/num_positions) * correlation
    )
    return portfolio_risk

def calculate_volatility_pnl(vega: float, vol_change: float, position_value: float) -> float:
    """Calculate P&L impact from volatility changes."""
    # Convert percentage change to volatility points
    vol_point_change = vol_change / 100
    
    # Calculate P&L impact
    vol_pnl = vega * vol_point_change
    return vol_pnl

def calculate_black_swan_impact(protection: float, hedge_type: str, 
                              market_drop: float, vol_spike: float) -> dict:
    """Calculate portfolio impact under black swan scenarios."""
    # Base impact from market drop
    unhedged_impact = market_drop
    
    # Calculate hedge effectiveness based on type
    hedge_effectiveness = {
        "Put Options": 0.8,
        "VIX Calls": 0.9,
        "Tail Risk Hedge Fund": 0.7,
        "Mixed Strategy": 0.85
    }
    
    # Calculate protected portion
    protection_ratio = protection / 100
    effectiveness = hedge_effectiveness[hedge_type]
    
    hedged_impact = (market_drop * (1 - protection_ratio)) + \
                   (market_drop * protection_ratio * (1 - effectiveness))
    
    return {
        "unhedged": unhedged_impact,
        "hedged": hedged_impact,
        "effectiveness": (unhedged_impact - hedged_impact) / unhedged_impact * 100
    }

def _display_scenario_analysis(scenarios: List[str], strategies: List[str], 
                             base_risk: float):
    """Display scenario analysis results for different market conditions."""
    # Create scenario matrix
    results = []
    for scenario in scenarios:
        scenario_results = []
        for strategy in strategies:
            # Calculate strategy performance in scenario
            perf = _calculate_strategy_scenario(strategy, scenario, base_risk)
            scenario_results.append(perf)
        results.append(scenario_results)
    
    # Display results as a heatmap
    fig = go.Figure(data=go.Heatmap(
        z=results,
        x=strategies,
        y=scenarios,
        colorscale='RdYlGn',
        text=[[f"{x:.1f}%" for x in row] for row in results],
        texttemplate="%{text}",
        textfont={"size":10},
        colorbar=dict(title="Returns")
    ))
    
    fig.update_layout(
        title="Strategy Performance Across Scenarios",
        xaxis_title="Strategy Type",
        yaxis_title="Market Scenario"
    )
    
    st.plotly_chart(fig)