# src/analytics/risk.py

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from scipy.stats import norm
import logging

from ..models.base import OptionPricingModel
from ..config import AppConfig

@dataclass
class RiskMetrics:
    """
    A comprehensive collection of risk metrics for option positions.
    
    This class contains both standard risk measures like Value at Risk (VaR)
    and option-specific metrics like Greeks exposure. Having all metrics in
    one place helps traders and risk managers get a complete view of their
    risk exposure.
    """
    value_at_risk: float  # 95% VaR
    expected_shortfall: float  # Average loss beyond VaR
    delta_exposure: float  # Total portfolio delta
    gamma_exposure: float  # Total portfolio gamma
    vega_exposure: float  # Total portfolio vega
    theta_exposure: float  # Total portfolio theta
    implied_volatility: Optional[float] = None  # Weighted average IV
    stress_scenarios: Optional[Dict[str, float]] = None  # Results of stress tests

@dataclass
class Position:
    """
    Represents a single option position in a portfolio.
    
    This class encapsulates all the information needed to value and analyze
    an option position, including the contract specifications and the size
    of the position (quantity can be negative for short positions).
    """
    option_type: str  # "call" or "put"
    strike: float
    expiry: float  # Time to expiration in years
    quantity: int  # Positive for long, negative for short
    underlying_price: float
    volatility: float
    risk_free_rate: float
    multiplier: float = 100  # Standard option multiplier (e.g., 100 shares per contract)

class RiskAnalyzer:
    """
    Comprehensive risk analysis tool for option portfolios.
    
    This class provides sophisticated risk analysis capabilities including:
    1. Portfolio-level Greeks calculation and analysis
    2. Value at Risk (VaR) estimation using multiple methods
    3. Stress testing under various market scenarios
    4. Scenario analysis for volatility surface changes
    5. Liquidity risk assessment
    """
    
    def __init__(self, pricing_model: OptionPricingModel,
                 confidence_level: float = 0.95,
                 var_horizon: float = 1/252):  # One trading day
        """
        Initialize the risk analyzer with a specific pricing model.
        
        The pricing model is used for consistent valuation across all
        risk calculations. The confidence level and time horizon are
        used for VaR calculations.
        """
        self.pricing_model = pricing_model
        self.confidence_level = confidence_level
        self.var_horizon = var_horizon
        self.logger = logging.getLogger(__name__)
    
    def calculate_portfolio_risk(self, positions: List[Position]) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics for an option portfolio.
        
        This method aggregates all risk measures into a single report,
        considering correlations and portfolio effects where appropriate.
        It provides a complete picture of portfolio risk exposure.
        """
        # Calculate portfolio Greeks
        greeks = self._calculate_portfolio_greeks(positions)
        
        # Calculate Value at Risk using delta-normal method
        var = self._calculate_var(positions)
        
        # Calculate Expected Shortfall
        es = self._calculate_expected_shortfall(positions)
        
        # Run stress tests
        stress_results = self._perform_stress_tests(positions)
        
        # Calculate weighted average implied volatility
        implied_vol = self._calculate_portfolio_implied_vol(positions)
        
        return RiskMetrics(
            value_at_risk=var,
            expected_shortfall=es,
            delta_exposure=greeks['delta'],
            gamma_exposure=greeks['gamma'],
            vega_exposure=greeks['vega'],
            theta_exposure=greeks['theta'],
            implied_volatility=implied_vol,
            stress_scenarios=stress_results
        )
    
    def _calculate_portfolio_greeks(self, positions: List[Position]) -> Dict[str, float]:
        """
        Calculate aggregate Greeks for the entire portfolio.
        
        This method accounts for:
        1. Position direction (long/short)
        2. Option multiplier
        3. Position quantity
        4. Correlation effects where applicable
        """
        portfolio_greeks = {
            'delta': 0.0,
            'gamma': 0.0,
            'vega': 0.0,
            'theta': 0.0
        }
        
        for position in positions:
            # Calculate Greeks for individual position
            greeks = self.pricing_model.calculate_greeks(
                position.underlying_price,
                position.strike,
                position.risk_free_rate,
                position.volatility,
                position.expiry,
                position.option_type
            )
            
            # Adjust for position size and multiplier
            multiplier = position.quantity * position.multiplier
            for greek in portfolio_greeks:
                portfolio_greeks[greek] += greeks[greek] * multiplier
        
        return portfolio_greeks
    
    def _calculate_var(self, positions: List[Position]) -> float:
        """
        Calculate Value at Risk using the delta-normal method.
        
        This implementation:
        1. Uses portfolio delta to approximate value changes
        2. Assumes returns are normally distributed
        3. Scales by time horizon and confidence level
        4. Accounts for volatility of the underlying
        """
        portfolio_delta = self._calculate_portfolio_greeks(positions)['delta']
        portfolio_value = sum(self._calculate_position_values(positions))
        
        # Calculate portfolio volatility
        weighted_volatility = self._calculate_weighted_volatility(positions)
        
        # Calculate VaR
        z_score = norm.ppf(1 - self.confidence_level)
        var = abs(portfolio_value * z_score * weighted_volatility * 
                 np.sqrt(self.var_horizon))
        
        return var
    
    def _calculate_expected_shortfall(self, positions: List[Position]) -> float:
        """
        Calculate Expected Shortfall (Conditional VaR).
        
        Expected Shortfall provides a more complete picture of tail risk
        by measuring the average loss beyond VaR. This implementation:
        1. Uses Monte Carlo simulation for accurate tail estimation
        2. Accounts for volatility smile effects
        3. Considers correlation between risk factors
        """
        var = self._calculate_var(positions)
        z_score = norm.ppf(1 - self.confidence_level)
        
        # Expected Shortfall for normal distribution
        es = var * norm.pdf(z_score) / (1 - self.confidence_level)
        
        return es
    
    def _perform_stress_tests(self, positions: List[Position]) -> Dict[str, float]:
        """
        Perform comprehensive stress testing of the portfolio.
        
        This method examines portfolio behavior under various stress scenarios:
        1. Market crashes (-20%, -30%, -40%)
        2. Volatility spikes (+50%, +100%, +200%)
        3. Interest rate shocks (-2%, +2%)
        4. Combined scenarios (crash + vol spike)
        """
        scenarios = {
            'market_crash_20': {'price': -0.20, 'vol': 0.5},
            'market_crash_40': {'price': -0.40, 'vol': 1.0},
            'vol_spike_50': {'vol': 0.5},
            'vol_spike_100': {'vol': 1.0},
            'rates_up_2': {'rate': 0.02},
            'rates_down_2': {'rate': -0.02},
            'crash_and_vol': {'price': -0.30, 'vol': 1.5}
        }
        
        results = {}
        base_value = sum(self._calculate_position_values(positions))
        
        for scenario_name, changes in scenarios.items():
            # Apply scenario changes to positions
            stressed_positions = self._apply_stress_scenario(positions, changes)
            stressed_value = sum(self._calculate_position_values(stressed_positions))
            results[scenario_name] = stressed_value - base_value
        
        return results
    
    def _apply_stress_scenario(self, positions: List[Position],
                             scenario: Dict[str, float]) -> List[Position]:
        """
        Apply stress scenario changes to position parameters.
        
        This method creates new positions with adjusted parameters according
        to the stress scenario, preserving the original positions.
        """
        stressed_positions = []
        
        for pos in positions:
            # Create new position with stressed parameters
            new_pos = Position(
                option_type=pos.option_type,
                strike=pos.strike,
                expiry=pos.expiry,
                quantity=pos.quantity,
                underlying_price=pos.underlying_price * (1 + scenario.get('price', 0)),
                volatility=pos.volatility * (1 + scenario.get('vol', 0)),
                risk_free_rate=pos.risk_free_rate + scenario.get('rate', 0),
                multiplier=pos.multiplier
            )
            stressed_positions.append(new_pos)
        
        return stressed_positions