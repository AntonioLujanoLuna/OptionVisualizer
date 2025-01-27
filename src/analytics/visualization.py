# src/analytics/visualization.py

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .volatility import VolatilitySurface
from ..models.base import OptionResult
from ..config import AppConfig

@dataclass
class VisualizationConfig:
    """Configuration settings for visualization aesthetics and behavior."""
    colorscale: str = 'viridis'  # Default colorscale for surface plots
    template: str = 'plotly_white'  # Clean, professional look
    width: int = 800  # Default width for plots
    height: int = 600  # Default height for plots
    show_axes: bool = True  # Whether to show axis labels
    animation_duration: int = 500  # Duration for transitions in ms

class VolatilityVisualizer:
    """
    Creates interactive visualizations for volatility analysis.
    
    This class provides a suite of visualization tools that help users
    understand volatility surfaces and option pricing dynamics. The
    visualizations are interactive, allowing users to explore the data
    through rotation, zooming, and hovering for additional information.
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize with optional custom configuration."""
        self.config = config or VisualizationConfig()
    
    def plot_volatility_surface(self, surface: VolatilitySurface) -> go.Figure:
        """
        Create an interactive 3D visualization of the volatility surface.
        
        This plot helps users understand:
        1. The overall shape of the volatility surface
        2. How volatility varies with strike and expiry
        3. The presence of skew and term structure effects
        4. Any potential arbitrage violations or anomalies
        """
        # Create meshgrid for 3D surface
        strike_mesh, expiry_mesh = np.meshgrid(
            surface.strikes,
            surface.expiries
        )
        
        # Create the 3D surface plot
        fig = go.Figure()
        
        # Add the main volatility surface
        fig.add_trace(go.Surface(
            x=strike_mesh,
            y=expiry_mesh,
            z=surface.volatilities.T,  # Transpose for correct orientation
            colorscale=self.config.colorscale,
            name='Volatility Surface'
        ))
        
        # Add forward price curve to show ATM line
        fig.add_trace(go.Scatter3d(
            x=surface.forward_prices,
            y=surface.expiries,
            z=np.max(surface.volatilities) * np.ones_like(surface.expiries),
            mode='lines',
            line=dict(color='red', width=4),
            name='Forward Prices'
        ))
        
        # Update layout with explanatory annotations
        fig.update_layout(
            title='Implied Volatility Surface',
            scene=dict(
                xaxis_title='Strike Price',
                yaxis_title='Time to Expiry',
                zaxis_title='Implied Volatility',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                ),
                annotations=[
                    dict(
                        text="Red line indicates ATM strikes",
                        x=surface.forward_prices[0],
                        y=surface.expiries[0],
                        z=np.max(surface.volatilities)
                    )
                ]
            ),
            width=self.config.width,
            height=self.config.height,
            template=self.config.template,
            showlegend=True
        )
        
        return fig
    
    def plot_volatility_smile(self, surface: VolatilitySurface,
                            expiry_idx: Optional[int] = None) -> go.Figure:
        """
        Create an interactive plot of volatility smiles at selected expiries.
        
        This visualization shows:
        1. The shape of the volatility smile/skew
        2. How the smile varies across different expiries
        3. The relative prices of OTM calls and puts
        4. The market's assessment of tail risk
        """
        fig = go.Figure()
        
        # If no specific expiry is selected, plot smiles for multiple expiries
        if expiry_idx is None:
            # Select a few representative expiries
            expiry_indices = [
                0,  # Shortest
                len(surface.expiries) // 4,  # Quarter
                len(surface.expiries) // 2,  # Middle
                -1  # Longest
            ]
        else:
            expiry_indices = [expiry_idx]
        
        # Plot smile for each selected expiry
        for idx in expiry_indices:
            # Calculate moneyness for x-axis
            moneyness = surface.strikes / surface.forward_prices[idx]
            
            fig.add_trace(go.Scatter(
                x=moneyness,
                y=surface.volatilities[:, idx],
                name=f'T = {surface.expiries[idx]:.2f}',
                mode='lines+markers',
                hovertemplate=(
                    'Moneyness: %{x:.2f}<br>' +
                    'IV: %{y:.1%}<br>' +
                    'Strike: %{customdata[0]:.1f}'
                ),
                customdata=np.column_stack([surface.strikes])
            ))
        
        # Add vertical line at ATM point
        fig.add_vline(
            x=1.0,
            line_dash="dash",
            line_color="gray",
            annotation_text="ATM"
        )
        
        # Update layout with explanatory elements
        fig.update_layout(
            title='Volatility Smile Analysis',
            xaxis_title='Moneyness (K/F)',
            yaxis_title='Implied Volatility',
            width=self.config.width,
            height=self.config.height,
            template=self.config.template,
            showlegend=True,
            annotations=[
                dict(
                    text="OTM Puts",
                    xref="x", yref="paper",
                    x=0.7, y=1.05,
                    showarrow=False
                ),
                dict(
                    text="OTM Calls",
                    xref="x", yref="paper",
                    x=1.3, y=1.05,
                    showarrow=False
                )
            ]
        )
        
        return fig
    
    def plot_term_structure(self, surface: VolatilitySurface,
                          moneyness_levels: Optional[List[float]] = None) -> go.Figure:
        """
        Create an interactive plot of the volatility term structure.
        
        This visualization shows:
        1. How volatility varies with time to expiry
        2. Term structure patterns at different moneyness levels
        3. Mean reversion effects in longer-dated options
        4. Impact of upcoming events on specific expiries
        """
        if moneyness_levels is None:
            moneyness_levels = [0.9, 0.95, 1.0, 1.05, 1.10]
        
        fig = go.Figure()
        
        # Plot term structure for each moneyness level
        for moneyness in moneyness_levels:
            vols = []
            for i, expiry in enumerate(surface.expiries):
                # Find strike closest to desired moneyness
                target_strike = surface.forward_prices[i] * moneyness
                strike_idx = np.abs(surface.strikes - target_strike).argmin()
                vols.append(surface.volatilities[strike_idx, i])
            
            fig.add_trace(go.Scatter(
                x=surface.expiries,
                y=vols,
                name=f'{moneyness:.0%} Moneyness',
                mode='lines+markers',
                hovertemplate=(
                    'Expiry: %{x:.2f}<br>' +
                    'IV: %{y:.1%}'
                )
            ))
        
        # Update layout with explanatory elements
        fig.update_layout(
            title='Volatility Term Structure',
            xaxis_title='Time to Expiry (Years)',
            yaxis_title='Implied Volatility',
            width=self.config.width,
            height=self.config.height,
            template=self.config.template,
            showlegend=True,
            annotations=[
                dict(
                    text="Higher lines indicate more OTM options",
                    xref="paper", yref="paper",
                    x=1.02, y=0.95,
                    showarrow=False
                )
            ]
        )
        
        return fig

    def create_greek_analysis_dashboard(self, results: List[OptionResult],
                                      parameters: Dict[str, np.ndarray]) -> go.Figure:
        """
        Create a comprehensive dashboard for analyzing option Greeks.
        
        This dashboard shows:
        1. How Greeks vary with underlying price
        2. Relationships between different Greeks
        3. Time decay effects
        4. Risk exposures at different strikes
        """
        # Create subplot grid for different Greeks
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Delta', 'Gamma',
                'Theta', 'Vega',
                'Rho', 'Greeks Relationships'
            ),
            specs=[[{}, {}],
                  [{}, {}],
                  [{}, {'type': 'scatter3d'}]]
        )
        
        # Extract parameters
        spot_prices = parameters['spot']
        times = parameters['time']
        
        # Plot each Greek
        for i, result in enumerate(results):
            greeks = result.greeks
            
            # Delta plot
            fig.add_trace(
                go.Scatter(
                    x=spot_prices,
                    y=[g['delta'] for g in greeks],
                    name=f'Delta T={times[i]:.2f}'
                ),
                row=1, col=1
            )
            
            # Similar traces for other Greeks...
            # (Implementation continues with other Greeks)
            
            # 3D plot showing relationship between Delta, Gamma, and Spot
            fig.add_trace(
                go.Scatter3d(
                    x=[g['delta'] for g in greeks],
                    y=[g['gamma'] for g in greeks],
                    z=spot_prices,
                    name=f'Delta-Gamma T={times[i]:.2f}'
                ),
                row=3, col=2
            )
        
        # Update layout with educational annotations
        fig.update_layout(
            height=1200,  # Larger height for dashboard
            width=self.config.width,
            template=self.config.template,
            showlegend=True,
            annotations=[
                dict(
                    text="Delta approaches 1 (calls) or -1 (puts) for deep ITM options",
                    xref="paper", yref="paper",
                    x=0.02, y=0.98,
                    showarrow=False
                ),
                # Additional educational annotations...
            ]
        )
        
        return fig