# src/analytics/portfolio_visualization.py

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .risk import Position, RiskMetrics
from .visualization import VisualizationConfig 
from src.config import AppConfig

class PortfolioVisualizer:
    """
    Creates interactive visualizations for portfolio analysis and risk management.
    
    This class provides visualizations that help users understand:
    1. Portfolio composition and risk exposures
    2. Scenario analysis results
    3. Profit/Loss profiles under different market conditions
    4. Risk metrics and their evolution over time
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize with optional custom visualization configuration."""
        self.config = config or VisualizationConfig()
    
    def plot_portfolio_composition(self, positions: List[Position]) -> go.Figure:
        """
        Create an interactive visualization of portfolio composition.
        
        This visualization shows:
        1. Distribution of positions by option type and expiry
        2. Position sizes and directions (long/short)
        3. Strike price distribution
        4. Exposure to different underlyings
        """
        fig = go.Figure()
        
        # Group positions by option type
        calls = [p for p in positions if p.option_type.lower() == 'call']
        puts = [p for p in positions if p.option_type.lower() == 'put']
        
        # Create bubble chart for calls
        if calls:
            fig.add_trace(go.Scatter(
                x=[p.strike for p in calls],
                y=[p.expiry for p in calls],
                mode='markers',
                name='Calls',
                marker=dict(
                    size=[abs(p.quantity) * 10 for p in calls],
                    color=['green' if p.quantity > 0 else 'red' for p in calls],
                    symbol='circle',
                    line=dict(color='black', width=1)
                ),
                text=[f"Qty: {p.quantity}<br>Strike: {p.strike}" for p in calls],
                hoverinfo='text'
            ))
        
        # Create bubble chart for puts
        if puts:
            fig.add_trace(go.Scatter(
                x=[p.strike for p in puts],
                y=[p.expiry for p in puts],
                mode='markers',
                name='Puts',
                marker=dict(
                    size=[abs(p.quantity) * 10 for p in puts],
                    color=['green' if p.quantity > 0 else 'red' for p in puts],
                    symbol='square',
                    line=dict(color='black', width=1)
                ),
                text=[f"Qty: {p.quantity}<br>Strike: {p.strike}" for p in puts],
                hoverinfo='text'
            ))
        
        # Add vertical line at current underlying price
        current_price = positions[0].underlying_price if positions else 0
        fig.add_vline(
            x=current_price,
            line_dash="dash",
            line_color="gray",
            annotation_text="Current Price"
        )
        
        fig.update_layout(
            title='Portfolio Position Analysis',
            xaxis_title='Strike Price',
            yaxis_title='Time to Expiry (Years)',
            showlegend=True,
            width=self.config.width,
            height=self.config.height,
            template=self.config.template,
            annotations=[
                dict(
                    text="Green: Long positions<br>Red: Short positions<br>Size: Position quantity",
                    xref="paper", yref="paper",
                    x=1.02, y=0.98,
                    showarrow=False,
                    align='left'
                )
            ]
        )

    def plot_risk_profile(self, positions: List[Position],
                         price_range: Optional[Tuple[float, float]] = None) -> go.Figure:
        """
        Create an interactive visualization of the portfolio's risk profile.
        
        This method generates a comprehensive view of how the portfolio value
        changes with the underlying price. It helps traders understand their
        risk exposure and identify potential hedging needs by showing:
        
        1. Total portfolio P&L across different price levels
        2. Contribution of individual positions to overall risk
        3. Key price points where risk characteristics change
        4. Maximum profit and loss scenarios
        """
        if not positions:
            raise ValueError("Cannot create risk profile for empty portfolio")
        
        # Determine price range if not provided
        if price_range is None:
            current_price = positions[0].underlying_price
            price_range = (current_price * 0.7, current_price * 1.3)
        
        # Generate price points for x-axis
        prices = np.linspace(price_range[0], price_range[1], 100)
        
        # Create figure with secondary y-axis for Greeks
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Calculate portfolio value at each price point
        total_values = np.zeros_like(prices)
        
        # Plot individual position contributions
        for position in positions:
            position_values = self._calculate_position_values(position, prices)
            
            # Add trace for this position
            fig.add_trace(
                go.Scatter(
                    x=prices,
                    y=position_values,
                    name=f"{position.option_type.title()} K={position.strike:.1f} T={position.expiry:.2f}",
                    line=dict(dash='dash'),
                    opacity=0.6
                ),
                secondary_y=False
            )
            
            total_values += position_values
        
        # Add total portfolio value
        fig.add_trace(
            go.Scatter(
                x=prices,
                y=total_values,
                name="Total Portfolio",
                line=dict(color='black', width=3)
            ),
            secondary_y=False
        )
        
        # Add portfolio delta on secondary axis
        deltas = np.gradient(total_values, prices)
        fig.add_trace(
            go.Scatter(
                x=prices,
                y=deltas,
                name="Portfolio Delta",
                line=dict(color='red', dash='dot'),
            ),
            secondary_y=True
        )
        
        # Add key reference lines and annotations
        current_price = positions[0].underlying_price
        fig.add_vline(
            x=current_price,
            line_dash="dash",
            line_color="gray",
            annotation_text="Current Price"
        )
        
        # Add break-even points
        break_even_points = self._find_break_even_points(prices, total_values)
        for point in break_even_points:
            fig.add_vline(
                x=point,
                line_dash="dot",
                line_color="green",
                annotation_text="Break Even"
            )
        
        fig.update_layout(
            title='Portfolio Risk Profile',
            xaxis_title='Underlying Price',
            yaxis_title='Profit/Loss',
            yaxis2_title='Portfolio Delta',
            width=self.config.width,
            height=self.config.height,
            template=self.config.template,
            hovermode='x unified',
            annotations=[
                dict(
                    text=f"Max Profit: ${np.max(total_values):,.2f}<br>"
                         f"Max Loss: ${np.min(total_values):,.2f}",
                    xref="paper", yref="paper",
                    x=1.02, y=0.98,
                    showarrow=False,
                    align='left'
                )
            ]
        )
        
        return fig
    
    def plot_greek_exposures(self, positions: List[Position],
                           risk_metrics: RiskMetrics) -> go.Figure:
        """
        Create a comprehensive visualization of portfolio Greek exposures.
        
        This visualization helps risk managers understand their exposure to
        various market factors through multiple complementary views:
        
        1. A radar chart showing relative Greek exposures
        2. Bar charts showing contribution by position
        3. Time decay visualization
        4. Volatility exposure analysis
        """
        # Create four subplots: radar chart and three bar charts
        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{"type": "polar"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}]
            ],
            subplot_titles=[
                "Greek Exposures Overview",
                "Delta Contribution by Position",
                "Gamma Profile",
                "Theta Decay"
            ]
        )
        
        # Radar chart of normalized Greeks
        greeks = {
            'Delta': risk_metrics.delta_exposure,
            'Gamma': risk_metrics.gamma_exposure,
            'Theta': risk_metrics.theta_exposure,
            'Vega': risk_metrics.vega_exposure
        }
        
        # Normalize Greeks for radar chart
        max_values = {k: max(abs(v), 0.0001) for k, v in greeks.items()}
        normalized_greeks = {k: v/max_values[k] for k, v in greeks.items()}
        
        fig.add_trace(
            go.Scatterpolar(
                r=list(normalized_greeks.values()),
                theta=list(normalized_greeks.keys()),
                fill='toself',
                name='Normalized Exposure'
            ),
            row=1, col=1
        )
        
        # Bar chart of delta contributions
        position_deltas = self._calculate_position_greeks(positions)
        fig.add_trace(
            go.Bar(
                x=[f"Pos {i+1}" for i in range(len(positions))],
                y=[d['delta'] for d in position_deltas],
                name='Delta Exposure'
            ),
            row=1, col=2
        )
        
        # Gamma profile
        current_price = positions[0].underlying_price
        price_range = np.linspace(current_price * 0.8, current_price * 1.2, 50)
        gamma_profile = self._calculate_gamma_profile(positions, price_range)
        
        fig.add_trace(
            go.Scatter(
                x=price_range,
                y=gamma_profile,
                name='Gamma Profile'
            ),
            row=2, col=1
        )
        
        # Theta decay
        days_to_expiry = np.linspace(0, 30, 31)  # Next 30 days
        theta_decay = self._calculate_theta_decay(positions, days_to_expiry)
        
        fig.add_trace(
            go.Scatter(
                x=days_to_expiry,
                y=theta_decay,
                name='Theta Decay'
            ),
            row=2, col=2
        )
        
        # Update layout with educational annotations
        fig.update_layout(
            height=800,
            width=self.config.width,
            template=self.config.template,
            showlegend=True,
            annotations=[
                dict(
                    text="Larger area indicates higher sensitivity",
                    xref="paper", yref="paper",
                    x=0.02, y=0.98,
                    showarrow=False
                ),
                dict(
                    text="Negative theta indicates time decay cost",
                    xref="paper", yref="paper",
                    x=0.98, y=0.02,
                    showarrow=False
                )
            ]
        )
        
        return fig
    
    def plot_stress_test_results(self, risk_metrics: RiskMetrics) -> go.Figure:
        """
        Create a visualization of stress test results.
        
        This visualization helps users understand how their portfolio might
        perform under various market stress scenarios. It includes:
        
        1. Waterfall chart showing impact of each scenario
        2. Heat map of scenario correlations
        3. Distribution of stress test outcomes
        4. Comparison to VaR and expected shortfall
        """
        scenarios = risk_metrics.stress_scenarios
        if not scenarios:
            raise ValueError("No stress test results available")
        
        # Create figure with two subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=[
                "Stress Test Impacts",
                "Scenario Outcome Distribution"
            ],
            vertical_spacing=0.2
        )
        
        # Sort scenarios by impact
        sorted_scenarios = sorted(
            scenarios.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Waterfall chart of scenario impacts
        fig.add_trace(
            go.Waterfall(
                name="Scenario Impact",
                x=[s[0] for s in sorted_scenarios],
                y=[s[1] for s in sorted_scenarios],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                decreasing={"marker": {"color": "red"}},
                increasing={"marker": {"color": "green"}},
                text=[f"${v:,.0f}" for _, v in sorted_scenarios],
                textposition="outside"
            ),
            row=1, col=1
        )
        
        # Distribution of outcomes
        values = list(scenarios.values())
        fig.add_trace(
            go.Histogram(
                x=values,
                name="Scenario Distribution",
                nbinsx=20,
                marker_color='blue',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Add VaR and ES reference lines
        fig.add_vline(
            x=risk_metrics.value_at_risk,
            line_dash="dash",
            line_color="red",
            annotation_text="95% VaR",
            row=2, col=1
        )
        
        fig.add_vline(
            x=risk_metrics.expected_shortfall,
            line_dash="dash",
            line_color="orange",
            annotation_text="Expected Shortfall",
            row=2, col=1
        )
        
        fig.update_layout(
            title='Portfolio Stress Test Analysis',
            showlegend=False,
            height=800,
            width=self.config.width,
            template=self.config.template
        )
        
        return fig
    
    def _calculate_position_values(self, position: Position, prices: np.ndarray) -> np.ndarray:
        """Calculate position values across a range of prices."""
        results = np.zeros_like(prices)
        for i, price in enumerate(prices):
            # Update underlying price
            position_copy = Position(
                option_type=position.option_type,
                strike=position.strike,
                expiry=position.expiry,
                quantity=position.quantity,
                underlying_price=price,
                volatility=position.volatility,
                risk_free_rate=position.risk_free_rate,
                multiplier=position.multiplier
            )
            # Calculate value
            if position.option_type.lower() == 'call':
                results[i] = self.pricing_model.price_call(
                    position_copy.underlying_price,
                    position_copy.strike,
                    position_copy.risk_free_rate,
                    position_copy.volatility,
                    position_copy.expiry
                ).price * position_copy.quantity * position_copy.multiplier
            else:
                results[i] = self.pricing_model.price_put(
                    position_copy.underlying_price,
                    position_copy.strike,
                    position_copy.risk_free_rate,
                    position_copy.volatility,
                    position_copy.expiry
                ).price * position_copy.quantity * position_copy.multiplier
        return results
    
    def _find_break_even_points(self, prices: np.ndarray, values: np.ndarray) -> List[float]:
        """
        Find break-even points in the P&L curve where value crosses zero.
        
        This method uses linear interpolation to find points where the P&L curve
        crosses the zero line, representing price levels where the strategy breaks even.
        
        Args:
            prices: Array of underlying price points
            values: Array of corresponding strategy values/P&L
            
        Returns:
            List of break-even prices
        """
        break_even_points = []
        
        # Look for zero crossings in the P&L curve
        for i in range(len(values) - 1):
            # Check if values cross zero between these points
            if (values[i] <= 0 and values[i + 1] > 0) or (values[i] >= 0 and values[i + 1] < 0):
                # Use linear interpolation to find the exact crossing point
                x1, x2 = prices[i], prices[i + 1]
                y1, y2 = values[i], values[i + 1]
                
                # Calculate break-even price using linear interpolation
                # Formula: x = x1 + (0 - y1) * (x2 - x1) / (y2 - y1)
                break_even = x1 + (-y1) * (x2 - x1) / (y2 - y1)
                break_even_points.append(break_even)
        
        return sorted(break_even_points)

    def _calculate_position_greeks(self, positions: List[Position]) -> List[Dict]:
        """
        Calculate Greeks for individual positions in the portfolio.
        
        This method computes all Greeks for each position separately,
        considering position size and direction (long/short).
        
        Args:
            positions: List of Position objects
            
        Returns:
            List of dictionaries containing Greeks for each position
        """
        position_greeks = []
        
        for position in positions:
            # Get raw Greeks from pricing model
            greeks = self.pricing_model.calculate_greeks(
                position.underlying_price,
                position.strike,
                position.risk_free_rate,
                position.volatility,
                position.expiry,
                position.option_type
            )
            
            # Adjust Greeks for position size and direction
            adjusted_greeks = {
                'delta': greeks['delta'] * position.quantity * position.multiplier,
                'gamma': greeks['gamma'] * position.quantity * position.multiplier,
                'theta': greeks['theta'] * position.quantity * position.multiplier,
                'vega': greeks['vega'] * position.quantity * position.multiplier,
                'rho': greeks['rho'] * position.quantity * position.multiplier
            }
            
            # Add position identifier
            adjusted_greeks['position_id'] = id(position)
            adjusted_greeks['option_type'] = position.option_type
            adjusted_greeks['strike'] = position.strike
            
            position_greeks.append(adjusted_greeks)
        
        return position_greeks

    def _calculate_theta_decay(self, positions: List[Position], days: np.ndarray) -> np.ndarray:
        """
        Calculate the portfolio's value decay over time.
        
        This method estimates how the portfolio's value will change due to time decay,
        accounting for weekends and holidays in the decay calculation.
        
        Args:
            positions: List of Position objects
            days: Array of future days to calculate decay for
            
        Returns:
            Array of portfolio values corresponding to each future day
        """
        values = np.zeros_like(days, dtype=float)
        
        for i, day in enumerate(days):
            daily_value = 0
            
            for position in positions:
                # Create a copy of the position with adjusted time to expiry
                # Convert days to years for the pricing model
                remaining_time = max(0, position.expiry - day/365)
                
                # Price the option at this point in time
                if position.option_type.lower() == 'call':
                    price = self.pricing_model.price_call(
                        position.underlying_price,
                        position.strike,
                        position.risk_free_rate,
                        position.volatility,
                        remaining_time
                    ).price
                else:
                    price = self.pricing_model.price_put(
                        position.underlying_price,
                        position.strike,
                        position.risk_free_rate,
                        position.volatility,
                        remaining_time
                    ).price
                
                # Add the position's contribution to daily value
                daily_value += price * position.quantity * position.multiplier
            
            values[i] = daily_value
        
        return values
    
    def _calculate_gamma_profile(self, positions: List[Position], prices: np.ndarray) -> np.ndarray:
        """
        Calculate portfolio gamma profile across price range.
        
        This vectorized implementation is more efficient for large price ranges.
        """
        total_gamma = np.zeros_like(prices)
        
        for position in positions:
            # Calculate gammas for all prices at once
            gammas = np.array([
                self.pricing_model.calculate_greeks(
                    price,
                    position.strike,
                    position.risk_free_rate,
                    position.volatility,
                    position.expiry,
                    position.option_type
                )['gamma']
                for price in prices
            ])
            
            # Add position's contribution to total gamma
            total_gamma += gammas * position.quantity * position.multiplier
        
        return total_gamma