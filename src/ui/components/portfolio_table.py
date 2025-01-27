# src/ui/components/portfolio_table.py

import streamlit as st
import pandas as pd
from typing import List
from src.analytics.risk import Position # Relative import (still works within the same package)

class PortfolioTable:
    """Interactive portfolio position display and management component."""
    
    def __init__(self, positions: List[Position]):
        self.positions = positions

    def render(self):
        """Render the portfolio positions table."""
        if not self.positions:
            st.info("No positions in portfolio. Add positions using the form above.")
            return

        # Convert positions to DataFrame for display
        data = []
        for pos in self.positions:
            data.append({
                'Type': pos.option_type.upper(),
                'Strike': f"${pos.strike:.2f}",
                'Expiry': f"{pos.expiry:.2f}y",
                'Quantity': pos.quantity,
                'Underlying': f"${pos.underlying_price:.2f}",
                'IV': f"{pos.volatility:.1%}",
                'Rate': f"{pos.risk_free_rate:.1%}"
            })
        
        df = pd.DataFrame(data)
        
        # Display the table with conditional formatting
        st.dataframe(
            df.style.apply(self._style_negative_quantities, axis=1),
            use_container_width=True
        )

    @staticmethod
    def _style_negative_quantities(row):
        """Apply red color to negative quantities (short positions)."""
        color = 'red' if row['Quantity'] < 0 else 'black'
        return [f'color: {color}'] * len(row)