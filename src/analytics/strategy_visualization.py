# src/analytics/strategy_visualization.py

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from src.analytics.portfolio_visualization import VisualizationConfig
from src.models.base import OptionPricingModel

@dataclass
class StrategyProfile:
    """
    Defines the characteristics of an options trading strategy.
    
    This class encapsulates all the information needed to analyze and
    visualize an options strategy, including its components, risk-reward
    profile, and typical use cases.
    """
    name: str
    description: str
    components: List[Dict]  # List of option positions
    max_loss: float
    max_profit: float
    break_even_points: List[float]
    ideal_market_outlook: str
    typical_uses: List[str]
    risk_characteristics: Dict[str, str]

class StrategyVisualizer:
    """
    Creates educational visualizations for common option strategies.
    
    This class helps users understand different option strategies through
    interactive visualizations that show:
    1. Risk-reward profiles
    2. Greek characteristics
    3. Time decay effects
    4. Volatility sensitivity
    5. Comparison between similar strategies
    """
    
    def __init__(self, pricing_model: OptionPricingModel,
                 config: Optional[VisualizationConfig] = None):
        """Initialize with pricing model and visualization configuration."""
        self.pricing_model = pricing_model
        self.config = config or VisualizationConfig()
    
    def visualize_strategy(self, strategy: StrategyProfile) -> go.Figure:
        """
        Create a comprehensive visualization of an options strategy.
        
        This method generates an interactive dashboard showing multiple
        aspects of the strategy to help users understand how it works
        and when to use it.
        """
        # Create four subplots for different aspects of the strategy
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Risk-Reward Profile",
                "Greeks Profile",
                "Time Decay Effect",
                "Volatility Sensitivity"
            ],
            specs=[
                [{"secondary_y": True}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ]
        )
        
        # Calculate base values for visualization
        current_price = sum(c['strike'] for c in strategy.components) / len(strategy.components)
        price_range = np.linspace(current_price * 0.7, current_price * 1.3, 100)
        
        # Plot risk-reward profile
        self._add_risk_reward_profile(
            fig, strategy, price_range,
            row=1, col=1
        )
        
        # Plot Greeks profile
        self._add_greeks_profile(
            fig, strategy, price_range,
            row=1, col=2
        )
        
        # Plot time decay effect
        self._add_time_decay_effect(
            fig, strategy,
            row=2, col=1
        )
        
# Plot volatility sensitivity
        self._add_volatility_sensitivity(
            fig, strategy, price_range,
            row=2, col=2
        )
        
        # Add strategy information and annotations
        self._add_strategy_annotations(fig, strategy)
        
        # Update layout with educational elements
        fig.update_layout(
            title=dict(
                text=f"Strategy Analysis: {strategy.name}",
                font=dict(size=24)
            ),
            height=1000,
            width=self.config.width,
            template=self.config.template,
            showlegend=True,
            annotations=[
                dict(
                    text=strategy.description,
                    xref="paper", yref="paper",
                    x=0, y=1.15,
                    showarrow=False,
                    align='left',
                    font=dict(size=12)
                )
            ]
        )
        
        return fig
    
    def _add_risk_reward_profile(self, fig: go.Figure, strategy: StrategyProfile,
                                price_range: np.ndarray, row: int, col: int):
        """
        Add the risk-reward profile visualization to the figure.
        
        This visualization shows how the strategy's profit/loss changes with
        the underlying price. It helps users understand:
        1. Maximum potential profit and loss
        2. Break-even points
        3. Optimal price ranges for the strategy
        4. Risk-reward ratio
        """
        # Calculate total P&L profile
        payoff = np.zeros_like(price_range)
        for component in strategy.components:
            position_payoff = self._calculate_position_payoff(
                component, price_range
            )
            payoff += position_payoff
        
        # Add individual component traces
        for i, component in enumerate(strategy.components):
            position_payoff = self._calculate_position_payoff(
                component, price_range
            )
            fig.add_trace(
                go.Scatter(
                    x=price_range,
                    y=position_payoff,
                    name=f"Component {i+1}",
                    line=dict(dash='dot'),
                    opacity=0.5
                ),
                row=row, col=col,
                secondary_y=False
            )
        
        # Add total strategy payoff
        fig.add_trace(
            go.Scatter(
                x=price_range,
                y=payoff,
                name="Total Strategy",
                line=dict(color='black', width=3)
            ),
            row=row, col=col,
            secondary_y=False
        )
        
        # Add break-even lines
        for point in strategy.break_even_points:
            fig.add_vline(
                x=point,
                line_dash="dash",
                line_color="green",
                annotation_text="Break Even",
                row=row, col=col
            )
        
        # Add horizontal lines for max profit/loss
        fig.add_hline(
            y=strategy.max_profit,
            line_dash="dash",
            line_color="blue",
            annotation_text="Max Profit",
            row=row, col=col
        )
        
        fig.add_hline(
            y=strategy.max_loss,
            line_dash="dash",
            line_color="red",
            annotation_text="Max Loss",
            row=row, col=col
        )
    
    def _add_greeks_profile(self, fig: go.Figure, strategy: StrategyProfile,
                           price_range: np.ndarray, row: int, col: int):
        """
        Add the Greeks profile visualization to the figure.
        
        This visualization shows how the strategy's Greeks change with the
        underlying price, helping users understand:
        1. Delta exposure across different price levels
        2. Gamma profile and areas of accelerating P&L change
        3. Risk characteristics at different market levels
        """
        # Calculate combined Greeks across price range
        greeks = {
            'delta': np.zeros_like(price_range),
            'gamma': np.zeros_like(price_range),
            'theta': np.zeros_like(price_range)
        }
        
        for component in strategy.components:
            component_greeks = self._calculate_position_greeks(
                component, price_range
            )
            for greek in greeks:
                greeks[greek] += component_greeks[greek]
        
        # Add traces for each Greek
        colors = {'delta': 'blue', 'gamma': 'red', 'theta': 'green'}
        for greek, values in greeks.items():
            fig.add_trace(
                go.Scatter(
                    x=price_range,
                    y=values,
                    name=greek.capitalize(),
                    line=dict(color=colors[greek])
                ),
                row=row, col=col
            )
        
        # Add annotations explaining Greek characteristics
        fig.add_annotation(
            text="Positive gamma indicates accelerating gains/losses",
            xref="paper", yref="paper",
            x=0.5, y=1.05,
            showarrow=False,
            row=row, col=col
        )
    
    def _add_time_decay_effect(self, fig: go.Figure, strategy: StrategyProfile,
                              row: int, col: int):
        """
        Add visualization of how the strategy's value changes with time.
        
        This helps users understand:
        1. Impact of time decay on the strategy
        2. Optimal holding periods
        3. When to consider adjusting or closing positions
        """
        # Calculate strategy value at different times to expiration
        days_to_expiry = np.linspace(0, 30, 31)  # Next 30 days
        values_over_time = self._calculate_time_decay(
            strategy, days_to_expiry
        )
        
        fig.add_trace(
            go.Scatter(
                x=days_to_expiry,
                y=values_over_time,
                name="Time Decay",
                line=dict(color='purple')
            ),
            row=row, col=col
        )
        
        # Add annotations about time decay characteristics
        fig.add_annotation(
            text="Theta decay accelerates near expiration",
            xref="paper", yref="paper",
            x=0.02, y=0.3,
            showarrow=False,
            row=row, col=col
        )
    
    def _add_volatility_sensitivity(self, fig: go.Figure,
                                  strategy: StrategyProfile,
                                  price_range: np.ndarray,
                                  row: int, col: int):
        """
        Add visualization of the strategy's sensitivity to volatility changes.
        
        This helps users understand:
        1. Impact of volatility changes on strategy value
        2. Optimal volatility environments for the strategy
        3. Risks from volatility changes
        """
        # Calculate strategy value at different volatility levels
        vol_levels = [0.15, 0.2, 0.25, 0.3]
        
        for vol in vol_levels:
            strategy_values = self._calculate_strategy_value(
                strategy, price_range, volatility=vol
            )
            
            fig.add_trace(
                go.Scatter(
                    x=price_range,
                    y=strategy_values,
                    name=f"Vol = {vol:.0%}",
                    line=dict(dash='solid' if vol == 0.2 else 'dash')
                ),
                row=row, col=col
            )
        
        fig.add_annotation(
            text="Strategy's response to volatility changes",
            xref="paper", yref="paper",
            x=0.5, y=1.05,
            showarrow=False,
            row=row, col=col
        )
    
    def _add_strategy_annotations(self, fig: go.Figure,
                                strategy: StrategyProfile):
        """
        Add educational annotations explaining the strategy's characteristics.
        """
        annotations = [
            dict(
                text="<b>Market Outlook:</b><br>" + strategy.ideal_market_outlook,
                xref="paper", yref="paper",
                x=1.02, y=0.98,
                showarrow=False,
                align='left'
            ),
            dict(
                text="<b>Typical Uses:</b><br>" + "<br>".join(
                    f"• {use}" for use in strategy.typical_uses
                ),
                xref="paper", yref="paper",
                x=1.02, y=0.85,
                showarrow=False,
                align='left'
            ),
            dict(
                text="<b>Risk Characteristics:</b><br>" + "<br>".join(
                    f"• {k}: {v}" for k, v in strategy.risk_characteristics.items()
                ),
                xref="paper", yref="paper",
                x=1.02, y=0.65,
                showarrow=False,
                align='left'
            )
        ]
        
        fig.update_layout(annotations=fig.layout.annotations + tuple(annotations))

    # Continuing in src/analytics/strategy_visualization.py

    def _calculate_position_payoff(self, position: Dict,
                                 price_range: np.ndarray) -> np.ndarray:
        """
        Calculate the payoff for a single position component across a price range.
        
        This method computes the theoretical value of an option position at
        different underlying prices. It takes into account:
        1. Option type (call/put)
        2. Strike price
        3. Position direction (long/short)
        4. Position size
        5. Premium paid/received
        
        Args:
            position: Dictionary containing position details
            price_range: Array of underlying prices to calculate payoff for
            
        Returns:
            Array of position values corresponding to each price point
        """
        option_type = position['option_type']
        strike = position['strike']
        quantity = position['quantity']
        premium = position['premium']
        
        # Calculate intrinsic value at each price point
        if option_type.lower() == 'call':
            intrinsic_value = np.maximum(price_range - strike, 0)
        else:  # put
            intrinsic_value = np.maximum(strike - price_range, 0)
        
        # Calculate total position value including premium
        position_value = quantity * (intrinsic_value - premium)
        
        return position_value
    
    def _calculate_position_greeks(self, position: Dict,
                                 price_range: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate option Greeks for a position across a price range.
        
        This method uses the pricing model to compute Greeks at each price point,
        taking into account:
        1. Time to expiration
        2. Volatility
        3. Risk-free rate
        4. Position size and direction
        
        Args:
            position: Dictionary containing position details
            price_range: Array of underlying prices to calculate Greeks for
            
        Returns:
            Dictionary mapping Greek names to arrays of values
        """
        greeks = {
            'delta': np.zeros_like(price_range),
            'gamma': np.zeros_like(price_range),
            'theta': np.zeros_like(price_range),
            'vega': np.zeros_like(price_range)
        }
        
        # Calculate Greeks at each price point
        for i, S in enumerate(price_range):
            position_greeks = self.pricing_model.calculate_greeks(
                S=S,
                K=position['strike'],
                r=position['risk_free_rate'],
                sigma=position['volatility'],
                T=position['time_to_expiry'],
                option_type=position['option_type']
            )
            
            # Adjust Greeks for position size and direction
            quantity = position['quantity']
            for greek in greeks:
                greeks[greek][i] = position_greeks[greek] * quantity
        
        return greeks
    
    def _calculate_time_decay(self, strategy: StrategyProfile,
                            days_to_expiry: np.ndarray) -> np.ndarray:
        """
        Calculate how the strategy's value changes as expiration approaches.
        
        This method computes the impact of theta decay on the entire strategy by:
        1. Calculating time decay for each component
        2. Adjusting for weekends and holidays
        3. Considering correlation between components
        4. Accounting for changes in other Greeks over time
        
        Args:
            strategy: StrategyProfile containing all position components
            days_to_expiry: Array of days to calculate decay for
            
        Returns:
            Array of strategy values at each point in time
        """
        strategy_value = np.zeros_like(days_to_expiry)
        
        for i, days in enumerate(days_to_expiry):
            # Convert days to years for pricing model
            time_to_expiry = days / 365.0
            
            # Calculate value of each component
            total_value = 0
            for component in strategy.components:
                # Create copy of component with adjusted time
                adjusted_component = component.copy()
                adjusted_component['time_to_expiry'] = time_to_expiry
                
                # Calculate theoretical value
                value = self.pricing_model.price_call(
                    S=component['current_price'],
                    K=component['strike'],
                    r=component['risk_free_rate'],
                    sigma=component['volatility'],
                    T=time_to_expiry
                ).price if component['option_type'].lower() == 'call' else \
                self.pricing_model.price_put(
                    S=component['current_price'],
                    K=component['strike'],
                    r=component['risk_free_rate'],
                    sigma=component['volatility'],
                    T=time_to_expiry
                ).price
                
                total_value += value * component['quantity']
            
            strategy_value[i] = total_value
        
        return strategy_value
    
    def _calculate_strategy_value(self, strategy: StrategyProfile,
                                price_range: np.ndarray,
                                volatility: float) -> np.ndarray:
        """
        Calculate the strategy's value across different underlying prices
        and a specified volatility level.
        
        This method helps understand how volatility changes affect the strategy by:
        1. Adjusting implied volatility for all components
        2. Recalculating option values with the new volatility
        3. Combining component values into total strategy value
        4. Considering volatility smile effects if applicable
        
        Args:
            strategy: StrategyProfile containing all position components
            price_range: Array of underlying prices to calculate values for
            volatility: Specific volatility level to use for calculations
            
        Returns:
            Array of strategy values corresponding to each price point
        """
        strategy_value = np.zeros_like(price_range)
        
        for i, S in enumerate(price_range):
            # Calculate value of each component at this price point
            total_value = 0
            for component in strategy.components:
                # Create copy of component with adjusted volatility
                adjusted_component = component.copy()
                adjusted_component['volatility'] = volatility
                
                # Calculate theoretical value
                if component['option_type'].lower() == 'call':
                    value = self.pricing_model.price_call(
                        S=S,
                        K=component['strike'],
                        r=component['risk_free_rate'],
                        sigma=volatility,
                        T=component['time_to_expiry']
                    ).price
                else:  # put
                    value = self.pricing_model.price_put(
                        S=S,
                        K=component['strike'],
                        r=component['risk_free_rate'],
                        sigma=volatility,
                        T=component['time_to_expiry']
                    ).price
                
                total_value += value * component['quantity']
            
            strategy_value[i] = total_value
        
        return strategy_value