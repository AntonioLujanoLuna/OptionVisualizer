# src/ui/education.py
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

@dataclass
class ConceptExplanation:
    """Container for educational content about an options concept."""
    title: str
    summary: str
    detailed_explanation: str
    formula: Optional[str] = None
    examples: List[Dict[str, any]] = None
    related_concepts: List[str] = None
    interactive_elements: Dict[str, any] = None

class OptionsEducation:
    """
    Educational component providing interactive learning about options concepts.
    
    This class creates visualizations and explanations to help users understand:
    1. Basic option concepts
    2. Greeks and their interpretations
    3. Volatility effects
    4. Various pricing models
    """
    
    def explain_black_scholes_assumptions(self) -> ConceptExplanation:
        """Provide detailed explanation of Black-Scholes model assumptions."""
        return ConceptExplanation(
            title="Black-Scholes Model Assumptions",
            summary="The Black-Scholes model relies on several key assumptions that simplify option pricing but may not fully reflect market reality.",
            detailed_explanation="""
            The Black-Scholes model makes the following assumptions:
            
            1. Geometric Brownian Motion (GBM):
               - Stock prices follow a continuous random walk
               - Returns are normally distributed
               - Price changes are independent
            
            2. Constant Volatility:
               - Volatility remains fixed throughout the option's life
               - In reality, volatility tends to change and exhibit clustering
            
            3. No Dividends:
               - The underlying stock pays no dividends
               - Can be adjusted for known dividend payments
            
            4. European Exercise:
               - Options can only be exercised at expiration
               - American options require different models
            
            5. No Transaction Costs:
               - Trading is frictionless with no fees or spreads
               - Real markets have various costs
            
            6. Risk-Free Rate:
               - Constant and known risk-free rate
               - Can borrow and lend at this rate
            
            Understanding these assumptions is crucial for:
            - Knowing when the model is appropriate
            - Recognizing potential pricing errors
            - Making necessary adjustments
            """,
            examples=[
                {
                    "assumption": "Constant Volatility",
                    "reality": "Volatility smile in market prices",
                    "adjustment": "Use implied volatility surface"
                }
            ],
            related_concepts=[
                "Implied Volatility",
                "Option Greeks",
                "Risk-Neutral Pricing"
            ]
        )
    
    def visualize_option_payoff(self, option_type: str, K: float, premium: float,
                              spot_range: Optional[np.ndarray] = None) -> go.Figure:
        """
        Create an interactive visualization of option payoff diagrams.
        
        This visualization shows:
        1. Payoff at expiration
        2. Current profit/loss
        3. Break-even points
        """
        if spot_range is None:
            spot_range = np.linspace(K * 0.5, K * 1.5, 100)
        
        # Calculate payoffs
        if option_type.lower() == "call":
            payoff = np.maximum(spot_range - K, 0)
        else:
            payoff = np.maximum(K - spot_range, 0)
        
        profit = payoff - premium
        
        # Create the plot
        fig = go.Figure()
        
        # Add payoff line
        fig.add_trace(go.Scatter(
            x=spot_range,
            y=payoff,
            name="Payoff at Expiration",
            line=dict(color="blue")
        ))
        
        # Add profit line
        fig.add_trace(go.Scatter(
            x=spot_range,
            y=profit,
            name="Profit/Loss",
            line=dict(color="green", dash="dash")
        ))
        
        # Add break-even line
        fig.add_hline(y=0, line=dict(color="gray", dash="dot"))
        
        # Update layout
        fig.update_layout(
            title=f"{option_type.capitalize()} Option Payoff Diagram",
            xaxis_title="Underlying Price",
            yaxis_title="Payoff/Profit",
            showlegend=True,
            hovermode="x unified"
        )
        
        return fig
    
    def explain_greeks_interactively(self) -> Dict[str, go.Figure]:
        """
        Create interactive visualizations for understanding option Greeks.
        
        Returns a dictionary of Plotly figures demonstrating:
        1. Delta and gamma relationships
        2. Theta decay patterns
        3. Vega sensitivity to volatility
        4. Rho interest rate effects
        """
        figures = {}
        
        # Delta-Gamma visualization
        fig_delta_gamma = make_subplots(
            rows=2, cols=1,
            subplot_titles=["Delta vs. Spot Price", "Gamma vs. Spot Price"]
        )
        
        # Example parameters
        S_range = np.linspace(80, 120, 100)
        K = 100
        r = 0.05
        sigma = 0.2
        T = 1.0
        
        # Calculate Greeks across spot range
        deltas_call = []
        deltas_put = []
        gammas = []
        
        for S in S_range:
            call_greeks = BlackScholesModel().calculate_greeks(
                S, K, r, sigma, T, "call"
            )
            put_greeks = BlackScholesModel().calculate_greeks(
                S, K, r, sigma, T, "put"
            )
            
            deltas_call.append(call_greeks["delta"])
            deltas_put.append(put_greeks["delta"])
            gammas.append(call_greeks["gamma"])  # Same for calls and puts
        
        # Add traces
        fig_delta_gamma.add_trace(
            go.Scatter(x=S_range, y=deltas_call, name="Call Delta"),
            row=1, col=1
        )
        fig_delta_gamma.add_trace(
            go.Scatter(x=S_range, y=deltas_put, name="Put Delta"),
            row=1, col=1
        )
        fig_delta_gamma.add_trace(
            go.Scatter(x=S_range, y=gammas, name="Gamma"),
            row=2, col=1
        )
        
        figures["delta_gamma"] = fig_delta_gamma
        
        # Similar implementations for other Greeks...
        # (Implementation abbreviated for brevity)
        
        return figures
    
    def create_interactive_tutorial(self) -> List[ConceptExplanation]:
        """
        Create a structured tutorial covering key options concepts.
        
        Returns a list of ConceptExplanation objects organized in learning order,
        from basic to advanced topics.
        """
        tutorial = []
        
        # Basic Concepts
        tutorial.append(ConceptExplanation(
            title="Introduction to Options",
            summary="Learn the fundamental concepts of options trading",
            detailed_explanation="""
            Options are financial contracts that give the holder the right,
            but not the obligation, to buy (call) or sell (put) an underlying
            asset at a specified price (strike) before or at expiration.
            
            Key Components:
            1. Premium: The price paid for the option
            2. Strike Price: The agreed-upon trading price
            3. Expiration: When the option contract ends
            4. Underlying: The asset the option is based on
            
            Options are useful for:
            - Hedging against market movements
            - Generating income 
            

            - Generating income through premium collection
            - Speculating on market movements with leverage
            - Creating sophisticated trading strategies
            
            Understanding these fundamentals is essential before moving
            to more advanced concepts like Greeks and pricing models.
            """,
            examples=[
                {
                    "scenario": "Bullish on Stock",
                    "action": "Buy Call Option",
                    "explanation": "If you believe a stock trading at $100 will rise, you might buy a call option with a strike price of $105. Your maximum loss is limited to the premium paid, while your potential profit is unlimited if the stock rises above $105 plus the premium."
                },
                {
                    "scenario": "Hedging Portfolio",
                    "action": "Buy Put Option",
                    "explanation": "If you own shares worth $10,000 and want to protect against a market decline, you could buy put options that give you the right to sell at a specified price, effectively creating a price floor for your portfolio."
                }
            ],
            related_concepts=["Moneyness", "Time Value", "Intrinsic Value"]
        ))
        
        # Option Pricing Fundamentals
        tutorial.append(ConceptExplanation(
            title="Understanding Option Pricing",
            summary="Learn what factors influence option prices and how they interact",
            detailed_explanation="""
            Option pricing is determined by several key factors that work together
            in complex ways. Understanding these factors helps make better trading
            decisions and manage risk effectively.
            
            Primary Pricing Factors:
            
            1. Underlying Price and Strike Price Relationship:
               The relationship between these prices determines the option's intrinsic
               value. An option has intrinsic value when it would be profitable to
               exercise immediately. For calls, this means the stock price is above
               the strike price; for puts, it's the reverse.
            
            2. Time to Expiration:
               Options lose value as they approach expiration, a concept known as
               time decay or theta. This decay is not linear - it accelerates as
               expiration approaches, particularly for at-the-money options.
            
            3. Volatility:
               Higher volatility increases option prices because it represents greater
               uncertainty in future stock price movements. This relationship is
               measured by vega, and it's why options are often described as
               "volatility trades."
            
            4. Interest Rates:
               While usually less significant than other factors, interest rates
               affect option prices through the cost of carrying the underlying
               position and the present value of the strike price.
            
            Understanding Price Behavior:
            
            Options exhibit complex price behavior because all these factors
            interact simultaneously. For example, an increase in volatility
            might overcome time decay, causing an option's price to rise even
            as it approaches expiration.
            """,
            formula="""
            Key Price Relationships:
            Call Premium = Max(Stock Price - Strike Price, 0) + Time Value
            Put Premium = Max(Strike Price - Stock Price, 0) + Time Value
            """,
            examples=[
                {
                    "concept": "Time Decay",
                    "scenario": "At-the-money call option with 30 days to expiration",
                    "explanation": "The option might lose value more rapidly in the final two weeks before expiration than it did in the previous two weeks, demonstrating accelerating time decay."
                },
                {
                    "concept": "Volatility Impact",
                    "scenario": "Earnings announcement approaching",
                    "explanation": "Option prices often increase as earnings approach due to higher implied volatility, even though time is passing and would normally reduce the option's value."
                }
            ]
        ))
        
        # Greeks and Risk Measures
        tutorial.append(ConceptExplanation(
            title="Option Greeks: Understanding Risk Measures",
            summary="Master the fundamental risk measures used in options trading",
            detailed_explanation="""
            Option Greeks are essential risk measures that help traders understand
            and manage their positions. Each Greek measures sensitivity to a
            different market factor.
            
            Delta: The First Line of Defense
            
            Delta measures how much an option's price changes when the underlying
            stock price changes by $1. It's often thought of as the equivalent
            stock position - a delta of 0.5 means the option behaves like owning
            50 shares of stock.
            
            More importantly, delta helps us understand probability. A call option
            with a 0.30 delta suggests approximately a 30% chance of finishing
            in-the-money at expiration. This probabilistic interpretation makes
            delta invaluable for risk management.
            
            Gamma: The Rate of Change
            
            While delta tells us our current directional risk, gamma tells us
            how quickly that risk is changing. High gamma positions can be
            dangerous because their risk profile can change rapidly with small
            moves in the underlying.
            
            Think of delta as your car's speed and gamma as its acceleration.
            Just as it's important to know both how fast you're going and how
            quickly you're speeding up or slowing down, understanding both
            delta and gamma is crucial for risk management.
            
            Theta: The Cost of Time
            
            Theta represents the daily cost of holding an option position. It's
            often called the "silent killer" because it steadily erodes option
            value regardless of market movements. Understanding theta helps
            traders decide whether to hold positions or take profits.
            
            Vega: Volatility Risk
            
            Vega measures sensitivity to volatility changes. It's particularly
            important around earnings announcements, economic events, or market
            stress when volatility can change dramatically. Long options have
            positive vega (benefit from volatility increases) while short
            options have negative vega.
            """,
            examples=[
                {
                    "greek": "Delta",
                    "position": "Long 1 Call with 0.60 delta",
                    "interpretation": "A $1 increase in the stock price would increase the option value by $0.60. The position behaves like owning 60 shares of stock."
                },
                {
                    "greek": "Gamma",
                    "position": "At-the-money option near expiration",
                    "interpretation": "High gamma means the position's delta can change dramatically with small price movements, requiring more active management."
                }
            ],
            related_concepts=[
                "Delta-Gamma Hedging",
                "Volatility Trading",
                "Risk Management Strategies"
            ]
        ))
        
        # Advanced Trading Strategies
        tutorial.append(ConceptExplanation(
            title="Option Strategies and Portfolio Management",
            summary="Learn how to combine options into sophisticated trading strategies",
            detailed_explanation="""
            Options can be combined in various ways to create positions with
            specific risk-reward characteristics. Understanding these strategies
            helps traders choose the right approach for their market outlook
            and risk tolerance.
            
            Vertical Spreads: Defined Risk Directional Trading
            
            Vertical spreads involve buying and selling options of the same type
            and expiration but different strikes. They offer defined risk and
            reward, making them popular for directional trading. The tradeoff
            is capped profit potential in exchange for reduced cost and risk.
            
            Calendar Spreads: Exploiting Time Decay
            
            Calendar spreads involve options with the same strike but different
            expirations. They profit from time decay differences between near
            and far-term options. These positions require careful management
            of both time and volatility risk.
            
            Portfolio Protection Strategies
            
            Options are powerful tools for portfolio protection. Common approaches
            include:
            
            1. Protective Puts: Buying puts to create a floor for stock positions
            2. Collars: Combining protective puts with covered calls to reduce cost
            3. VIX Hedging: Using volatility products to hedge market stress
            
            Each approach has its own cost-benefit tradeoff, and the choice
            depends on factors like market environment, cost tolerance, and
            protection needs.
            """,
            examples=[
                {
                    "strategy": "Bull Call Spread",
                    "construction": "Buy lower strike call, sell higher strike call",
                    "use_case": "Bullish outlook with defined risk and lower cost than outright call purchase"
                },
                {
                    "strategy": "Iron Condor",
                    "construction": "Sell OTM put spread and call spread",
                    "use_case": "Profit from range-bound market while limiting risk"
                }
            ]
        ))
        
        return tutorial

    def create_strategy_visualization(self, strategy_name: str,
                                   spot_range: Optional[np.ndarray] = None) -> go.Figure:
        """
        Create interactive visualizations of common option strategies.
        
        This method generates payoff and profit diagrams for complex option
        strategies, helping users understand their risk-reward characteristics.
        Each visualization includes:
        - Payoff at expiration
        - Current profit/loss
        - Break-even points
        - Component position contributions
        - Risk measures at different price points
        """
        if spot_range is None:
            spot_range = np.linspace(80, 120, 200)
        
        fig = go.Figure()
        
        if strategy_name.lower() == "bull_call_spread":
            # Example parameters for a bull call spread
            lower_strike = 100
            upper_strike = 105
            lower_premium = 3
            upper_premium = 1
            
            # Calculate individual position payoffs
            long_call = np.maximum(spot_range - lower_strike, 0) - lower_premium
            short_call = -(np.maximum(spot_range - upper_strike, 0) - upper_premium)
            total = long_call + short_call
            
            # Add traces for each component
            fig.add_trace(go.Scatter(
                x=spot_range,
                y=long_call,
                name="Long Lower Strike Call",
                line=dict(dash="dot", color="blue")
            ))
            
            fig.add_trace(go.Scatter(
                x=spot_range,
                y=short_call,
                name="Short Upper Strike Call",
                line=dict(dash="dot", color="red")
            ))
            
            fig.add_trace(go.Scatter(
                x=spot_range,
                y=total,
                name="Total Strategy",
                line=dict(color="green", width=2)
            ))
            
            # Add break-even line and annotation
            break_even = lower_strike + (lower_premium - upper_premium)
            fig.add_hline(y=0, line=dict(color="gray", dash="dash"))
            fig.add_vline(x=break_even, line=dict(color="gray", dash="dash"))
            
            fig.add_annotation(
                x=break_even,
                y=0,
                text=f"Break-even: {break_even:.2f}",
                showarrow=True,
                arrowhead=1
            )
            
            # Update layout
            fig.update_layout(
                title="Bull Call Spread Payoff Diagram",
                xaxis_title="Underlying Price",
                yaxis_title="Profit/Loss",
                showlegend=True,
                hovermode="x unified",
                annotations=[
                    dict(
                        text=f"Max Profit: {upper_strike - lower_strike - (lower_premium - upper_premium):.2f}",
                        xref="paper", yref="paper",
                        x=1.02, y=0.95,
                        showarrow=False
                    ),
                    dict(
                        text=f"Max Loss: {lower_premium - upper_premium:.2f}",
                        xref="paper", yref="paper",
                        x=1.02, y=0.85,
                        showarrow=False
                    )
                ]
            )
        
        elif strategy_name.lower() == "iron_condor":
            # Similar implementation for iron condor
            # (Implementation abbreviated for brevity)
            pass
        
        return fig
    
    def visualize_vol_surface(self) -> go.Figure:
        """
        Create an interactive 3D visualization of the volatility surface.
        
        This visualization helps users understand:
        - Volatility smile across strikes
        - Term structure of volatility
        - Put-call volatility parity
        """
        # Generate sample data
        strikes = np.linspace(80, 120, 20)
        maturities = np.linspace(0.1, 2, 20)
        K, T = np.meshgrid(strikes, maturities)
        
        # Create a sample volatility surface
        # In practice, this would use market data
        moneyness = K / 100  # Assuming current price is 100
        vol_surface = 0.2 + 0.1 * (moneyness - 1)**2 + 0.05 * np.exp(-T)
        
        # Create the 3D surface plot
        fig = go.Figure(data=[go.Surface(
            x=K,
            y=T,
            z=vol_surface,
            colorscale='Viridis'
        )])
        
        # Update layout
        fig.update_layout(
            title='Implied Volatility Surface',
            scene=dict(
                xaxis_title='Strike',
                yaxis_title='Time to Maturity',
                zaxis_title='Implied Volatility',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                )
            ),
            width=800,
            height=800
        )
        
        return fig
    
    def create_interactive_simulation(self, n_paths: int = 1000,
                                    time_steps: int = 252) -> go.Figure:
        """
        Create an interactive Monte Carlo simulation visualization.
        
        This helps users understand:
        - Path dependency of options
        - Probability distributions of outcomes
        - The relationship between volatility and price paths
        """
        # Generate price paths
        S0 = 100  # Initial price
        mu = 0.05  # Drift
        sigma = 0.2  # Volatility
        T = 1.0  # Time horizon
        dt = T / time_steps
        
        # Generate random walks
        Z = np.random.standard_normal((n_paths, time_steps))
        paths = np.zeros((n_paths, time_steps + 1))
        paths[:, 0] = S0
        
        for t in range(1, time_steps + 1):
            paths[:, t] = paths[:, t-1] * np.exp(
                (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t-1]
            )
        
        # Create visualization
        fig = go.Figure()
        
        # Add some sample paths
        time_points = np.linspace(0, T, time_steps + 1)
        for i in range(min(10, n_paths)):  # Show first 10 paths
            fig.add_trace(go.Scatter(
                x=time_points,
                y=paths[i, :],
                mode='lines',
                line=dict(width=1),
                showlegend=False,
                opacity=0.6
            ))
        
        # Add mean path
        mean_path = np.mean(paths, axis=0)
        fig.add_trace(go.Scatter(
            x=time_points,
            y=mean_path,
            mode='lines',
            name='Mean Path',
            line=dict(color='red', width=2)
        ))

        # Add confidence intervals
        upper_95 = np.percentile(paths, 97.5, axis=0)
        lower_95 = np.percentile(paths, 2.5, axis=0)
        
        fig.add_trace(go.Scatter(
            x=time_points,
            y=upper_95,
            mode='lines',
            name='95% Confidence Interval',
            line=dict(color='gray', dash='dash'),
            showlegend=True
        ))
        
        fig.add_trace(go.Scatter(
            x=time_points,
            y=lower_95,
            mode='lines',
            line=dict(color='gray', dash='dash'),
            fill='tonexty',  # Fill area between traces
            showlegend=False
        ))
        
        # Add expected value line (theoretical)
        expected_value = S0 * np.exp(mu * time_points)
        fig.add_trace(go.Scatter(
            x=time_points,
            y=expected_value,
            mode='lines',
            name='Expected Value (Theory)',
            line=dict(color='green', dash='dot', width=2)
        ))
        
        # Update layout with educational annotations
        fig.update_layout(
            title='Monte Carlo Price Path Simulation',
            xaxis_title='Time (years)',
            yaxis_title='Stock Price',
            showlegend=True,
            hovermode='x unified',
            annotations=[
                dict(
                    text='The shaded area shows the 95% confidence interval<br>'
                         'for future stock prices based on the volatility input.',
                    xref='paper', yref='paper',
                    x=1.02, y=0.95,
                    showarrow=False,
                    align='left'
                ),
                dict(
                    text=f'Parameters:<br>'
                         f'Initial Price: ${S0}<br>'
                         f'Drift: {mu:.1%}<br>'
                         f'Volatility: {sigma:.1%}',
                    xref='paper', yref='paper',
                    x=1.02, y=0.85,
                    showarrow=False,
                    align='left'
                )
            ]
        )
        
        # Add shapes to highlight important features
        fig.add_shape(
            type="rect",
            x0=0.8, x1=1.0,  # Highlight end of simulation
            y0=min(lower_95[-1], paths.min()),
            y1=max(upper_95[-1], paths.max()),
            fillcolor="rgba(255,0,0,0.1)",
            line=dict(width=0),
            layer="below"
        )
        
        return fig
    
    def create_interactive_greek_explorer(self) -> go.Figure:
        """
        Create an interactive visualization to explore how Greeks change
        with different parameters.
        
        This visualization helps users understand:
        - How Greeks evolve over time
        - The relationship between Greeks and market parameters
        - Risk management implications of Greek exposures
        """
        # Create a subplot figure with multiple Greeks
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Delta', 'Gamma', 'Theta', 'Vega'),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # Generate sample data
        S = 100  # Current stock price
        K = 100  # Strike price
        r = 0.05  # Risk-free rate
        sigma = 0.2  # Volatility
        T = 1.0  # Time to maturity
        
        # Create price range for x-axis
        spot_range = np.linspace(70, 130, 100)
        
        # Calculate Greeks at different spot prices
        deltas_call = []
        deltas_put = []
        gammas = []
        thetas = []
        vegas = []
        
        for spot in spot_range:
            # Calculate call Greeks
            call_greeks = BlackScholesModel().calculate_greeks(
                spot, K, r, sigma, T, "call"
            )
            # Calculate put Greeks
            put_greeks = BlackScholesModel().calculate_greeks(
                spot, K, r, sigma, T, "put"
            )
            
            deltas_call.append(call_greeks["delta"])
            deltas_put.append(put_greeks["delta"])
            gammas.append(call_greeks["gamma"])  # Same for calls and puts
            thetas.append(call_greeks["theta"])
            vegas.append(call_greeks["vega"])
        
        # Add traces for each Greek
        # Delta subplot
        fig.add_trace(
            go.Scatter(x=spot_range, y=deltas_call, name="Call Delta",
                      line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=spot_range, y=deltas_put, name="Put Delta",
                      line=dict(color='red')),
            row=1, col=1
        )
        
        # Gamma subplot
        fig.add_trace(
            go.Scatter(x=spot_range, y=gammas, name="Gamma",
                      line=dict(color='green')),
            row=1, col=2
        )
        
        # Theta subplot
        fig.add_trace(
            go.Scatter(x=spot_range, y=thetas, name="Theta",
                      line=dict(color='purple')),
            row=2, col=1
        )
        
        # Vega subplot
        fig.add_trace(
            go.Scatter(x=spot_range, y=vegas, name="Vega",
                      line=dict(color='orange')),
            row=2, col=2
        )
        
        # Update layout with educational annotations
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Interactive Greek Explorer",
            annotations=[
                dict(
                    text="Parameters:<br>"
                         f"Strike: ${K}<br>"
                         f"Volatility: {sigma:.1%}<br>"
                         f"Time: {T:.1f} years",
                    xref="paper", yref="paper",
                    x=1.02, y=1.0,
                    showarrow=False,
                    align='left'
                )
            ]
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Stock Price", row=2, col=1)
        fig.update_xaxes(title_text="Stock Price", row=2, col=2)
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="Value", row=2, col=1)
        
        return fig