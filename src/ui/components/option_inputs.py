# src/ui/components/option_inputs.py

import streamlit as st
from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class OptionParameters:
    """Container for option parameters from UI inputs."""
    underlying_price: float
    strike_price: float
    time_to_expiry: float
    volatility: float
    risk_free_rate: float
    option_type: str

class OptionInputs:
    """Reusable component for option parameter inputs."""
    
    def __init__(self, key_prefix: str = "", default_values: Optional[Dict] = None):
        """Initialize with optional key prefix for multiple instances."""
        self.key_prefix = key_prefix
        self.default_values = default_values or {
            "underlying_price": 100.0,
            "strike_price": 100.0,
            "time_to_expiry": 1.0,
            "volatility": 0.2,
            "risk_free_rate": 0.05
        }

    def render(self) -> OptionParameters:
        """Render option parameter inputs and return collected values."""
        col1, col2 = st.columns(2)
        
        with col1:
            underlying_price = st.number_input(
                "Underlying Price",
                min_value=0.01,
                value=self.default_values["underlying_price"],
                step=1.0,
                key=f"{self.key_prefix}underlying"
            )
            
            strike_price = st.number_input(
                "Strike Price",
                min_value=0.01,
                value=self.default_values["strike_price"],
                step=1.0,
                key=f"{self.key_prefix}strike"
            )
            
            time_to_expiry = st.slider(
                "Time to Expiry (Years)",
                min_value=0.1,
                max_value=5.0,
                value=self.default_values["time_to_expiry"],
                step=0.1,
                key=f"{self.key_prefix}expiry"
            )
        
        with col2:
            volatility = st.slider(
                "Volatility",
                min_value=0.05,
                max_value=1.0,
                value=self.default_values["volatility"],
                step=0.05,
                key=f"{self.key_prefix}volatility"
            )
            
            risk_free_rate = st.slider(
                "Risk-free Rate",
                min_value=0.0,
                max_value=0.1,
                value=self.default_values["risk_free_rate"],
                step=0.01,
                key=f"{self.key_prefix}rate"
            )
            
            option_type = st.selectbox(
                "Option Type",
                ["Call", "Put"],
                key=f"{self.key_prefix}type"
            )
        
        return OptionParameters(
            underlying_price=underlying_price,
            strike_price=strike_price,
            time_to_expiry=time_to_expiry,
            volatility=volatility,
            risk_free_rate=risk_free_rate,
            option_type=option_type.lower()
        )