# src/ui/components/charts.py

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Optional
from ...analytics.risk import Position

class OptionPayoffChart:
    """Component for displaying option payoff and risk profiles."""
    
    @staticmethod
    def create_payoff_chart(positions: List[Position], price_range: Optional[np.ndarray] = None) -> go.Figure:
        """Create an interactive payoff diagram for multiple positions."""
        if not positions:
            raise ValueError("No positions provided for payoff chart")
        
        # Determine price range if not provided
        if price_range is None:
            current_price = positions[0].underlying_price
            price_range = np.linspace(current_price * 0.5, current_price * 1.5, 100)
        
        fig = go.Figure()
        
        # Calculate and plot individual position payoffs
        total_payoff = np.zeros_like(price_range)
        for pos in positions:
            payoff = OptionPayoffChart._calculate_position_payoff(pos, price_range)
            total_payoff += payoff
            
            fig.add_trace(go.Scatter(
                x=price_range,
                y=payoff,
                name=f"{pos.option_type.title()} K={pos.strike}",
                line=dict(dash='dash'),
                opacity=0.6
            ))
        
        # Add total payoff line
        fig.add_trace(go.Scatter(
            x=price_range,
            y=total_payoff,
            name="Total Payoff",
            line=dict(color='black', width=2)
        ))
        
        # Add current price marker
        current_price = positions[0].underlying_price
        fig.add_vline(
            x=current_price,
            line_dash="dash",
            line_color="gray",
            annotation_text="Current Price"
        )
        
        fig.update_layout(
            title="Position Payoff Analysis",
            xaxis_title="Underlying Price",
            yaxis_title="Profit/Loss",
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig

    @staticmethod
    def _calculate_position_payoff(position: Position, prices: np.ndarray) -> np.ndarray:
        """Calculate payoff for a single position across price range."""
        if position.option_type.lower() == "call":
            payoff = np.maximum(prices - position.strike, 0)
        else:
            payoff = np.maximum(position.strike - prices, 0)
        
        return payoff * position.quantity * position.multiplier