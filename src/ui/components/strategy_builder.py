# src/ui/pages/strategy_builder.py
import streamlit as st
from dataclasses import dataclass
from typing import List, Optional
from src.ui.components.common import OptionInputs, OptionParameters

@dataclass
class StrategyLeg:
    """Represents a single leg in an options strategy."""
    option_params: OptionParameters
    quantity: int
    position: str  # "long" or "short"

class StrategyBuilder:
    """Component for building multi-leg option strategies."""
    
    def __init__(self):
        if 'strategy_legs' not in st.session_state:
            st.session_state.strategy_legs = []

    def render(self) -> List[StrategyLeg]:
        """Render the strategy builder interface."""
        st.subheader("Strategy Builder")
        
        # Add new leg form
        with st.expander("Add Strategy Leg", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                position = st.radio(
                    "Position",
                    ["Long", "Short"],
                    horizontal=True,
                    key="new_leg_position"
                )
                
                quantity = st.number_input(
                    "Quantity",
                    min_value=1,
                    value=1,
                    step=1,
                    key="new_leg_quantity"
                )
            
            with col2:
                # Use OptionInputs component for leg parameters
                leg_inputs = OptionInputs(key_prefix=f"leg_{len(st.session_state.strategy_legs)}")
                option_params = leg_inputs.render()
            
            if st.button("Add Leg"):
                st.session_state.strategy_legs.append(
                    StrategyLeg(
                        option_params=option_params,
                        quantity=quantity * (1 if position == "Long" else -1),
                        position=position.lower()
                    )
                )
                st.experimental_rerun()
        
        # Display current strategy composition
        if st.session_state.strategy_legs:
            st.subheader("Current Strategy Composition")
            for i, leg in enumerate(st.session_state.strategy_legs):
                with st.container():
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        st.write(f"{leg.position.title()} {abs(leg.quantity)} "
                                 f"{leg.option_params.option_type.title()}")
                    with col2:
                        st.write(f"Strike: ${leg.option_params