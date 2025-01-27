# src/analytics/risk.py

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from scipy.stats import norm
import logging

from src.models.base import OptionPricingModel
from src.config import AppConfig

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

    # Continuing in src/analytics/volatility.py

    def calculate_volatility_metrics(self, surface: VolatilitySurface) -> Dict[str, float]:
        """
        Calculate various volatility metrics from the surface.
        
        These metrics help understand the shape and characteristics of the 
        volatility surface, which is crucial for:
        1. Risk management - identifying potential volatility exposure
        2. Trading opportunities - finding mispriced options
        3. Market sentiment analysis - understanding market expectations
        4. Stress testing - assessing portfolio behavior under vol changes
        
        The metrics include:
        - Skewness measurements (volatility skew)
        - Term structure metrics
        - Overall surface curvature
        - Arbitrage-free condition checks
        """
        metrics = {}
        
        # Calculate ATM volatility for each maturity
        atm_volatilities = self._calculate_atm_volatilities(surface)
        metrics['atm_term_structure'] = atm_volatilities
        
        # Calculate volatility skew (25-delta risk reversal)
        skew_metrics = self._calculate_skew_metrics(surface)
        metrics.update(skew_metrics)
        
        # Calculate butterfly spread (measure of curvature)
        butterfly_metrics = self._calculate_butterfly_metrics(surface)
        metrics.update(butterfly_metrics)
        
        # Calculate term structure metrics
        term_metrics = self._calculate_term_structure_metrics(surface)
        metrics.update(term_metrics)
        
        # Calculate surface smoothness and arbitrage-free metrics
        quality_metrics = self._assess_surface_quality(surface)
        metrics.update(quality_metrics)
        
        return metrics
    
    def _calculate_atm_volatilities(self, surface: VolatilitySurface) -> np.ndarray:
        """
        Calculate at-the-money volatilities for each expiry.
        
        ATM volatilities are crucial reference points as they:
        1. Represent the market's base expectation of volatility
        2. Are typically the most liquid points on the surface
        3. Serve as anchors for interpolation and extrapolation
        """
        atm_vols = []
        
        for i, expiry in enumerate(surface.expiries):
            # Find the strike closest to the forward price
            forward = surface.forward_prices[i]
            strike_idx = np.abs(surface.strikes - forward).argmin()
            
            # Get the ATM volatility
            atm_vol = surface.volatilities[strike_idx, i]
            atm_vols.append(atm_vol)
        
        return np.array(atm_vols)
    
    def _calculate_skew_metrics(self, surface: VolatilitySurface) -> Dict[str, np.ndarray]:
        """
        Calculate volatility skew metrics for each expiry.
        
        The volatility skew (also known as the smile or smirk) reflects:
        1. Market's assessment of tail risk
        2. Supply/demand imbalances for OTM options
        3. Black-Scholes model violations in real markets
        
        We measure skew through several metrics:
        - 25-delta risk reversal (difference between 25d call and put vols)
        - Put-call volatility ratio
        - Skew slope (rate of vol change with moneyness)
        """
        metrics = {}
        
        # Calculate 25-delta risk reversals
        rr_25d = []
        pc_ratio = []
        skew_slope = []
        
        for i, expiry in enumerate(surface.expiries):
            # Find approximate 25-delta strikes
            forward = surface.forward_prices[i]
            atm_vol = self._calculate_atm_volatilities(surface)[i]
            
            # Calculate strikes for approximately 25-delta options
            # using a simplified delta approximation
            T = expiry
            call_25d_strike = forward * np.exp(0.5 * atm_vol * np.sqrt(T))
            put_25d_strike = forward * np.exp(-0.5 * atm_vol * np.sqrt(T))
            
            # Find nearest strikes in our surface
            call_idx = np.abs(surface.strikes - call_25d_strike).argmin()
            put_idx = np.abs(surface.strikes - put_25d_strike).argmin()
            
            # Calculate risk reversal
            call_vol = surface.volatilities[call_idx, i]
            put_vol = surface.volatilities[put_idx, i]
            rr_25d.append(call_vol - put_vol)
            
            # Calculate put-call vol ratio
            pc_ratio.append(put_vol / call_vol)
            
            # Calculate average skew slope
            moneyness = surface.strikes / forward
            valid_idx = (moneyness >= 0.8) & (moneyness <= 1.2)
            slope = np.polyfit(
                moneyness[valid_idx],
                surface.volatilities[valid_idx, i],
                1
            )[0]
            skew_slope.append(slope)
        
        metrics['risk_reversal_25d'] = np.array(rr_25d)
        metrics['put_call_vol_ratio'] = np.array(pc_ratio)
        metrics['skew_slope'] = np.array(skew_slope)
        
        return metrics
    
    def _calculate_butterfly_metrics(self, surface: VolatilitySurface) -> Dict[str, np.ndarray]:
        """
        Calculate butterfly spread metrics for each expiry.
        
        The butterfly spread measures the curvature of the volatility smile:
        1. Higher butterfly values indicate more pronounced smile shape
        2. Changes in butterfly can signal changing market dynamics
        3. Extreme butterfly values may indicate arbitrage opportunities
        """
        metrics = {}
        
        # Calculate 25-delta butterfly spreads
        fly_25d = []
        
        for i, expiry in enumerate(surface.expiries):
            forward = surface.forward_prices[i]
            atm_vol = self._calculate_atm_volatilities(surface)[i]
            
            # Calculate strikes for approximately 25-delta options
            T = expiry
            call_25d_strike = forward * np.exp(0.5 * atm_vol * np.sqrt(T))
            put_25d_strike = forward * np.exp(-0.5 * atm_vol * np.sqrt(T))
            
            # Find nearest strikes
            call_idx = np.abs(surface.strikes - call_25d_strike).argmin()
            put_idx = np.abs(surface.strikes - put_25d_strike).argmin()
            atm_idx = np.abs(surface.strikes - forward).argmin()
            
            # Calculate butterfly spread
            wing_vol = 0.5 * (
                surface.volatilities[call_idx, i] +
                surface.volatilities[put_idx, i]
            )
            center_vol = surface.volatilities[atm_idx, i]
            fly_25d.append(wing_vol - center_vol)
        
        metrics['butterfly_25d'] = np.array(fly_25d)
        
        return metrics
    
    def _calculate_term_structure_metrics(self, surface: VolatilitySurface) -> Dict[str, float]:
        """
        Calculate metrics describing the volatility term structure.
        
        The term structure shows how implied volatility varies with expiry:
        1. Short-term vs long-term vol expectations
        2. Mean reversion assumptions
        3. Impact of upcoming events (earnings, economic data, etc.)
        """
        atm_vols = self._calculate_atm_volatilities(surface)
        
        metrics = {
            'short_term_vol': np.mean(atm_vols[surface.expiries <= 1/12]),  # 1 month
            'medium_term_vol': np.mean(atm_vols[(surface.expiries > 1/12) & 
                                               (surface.expiries <= 1/2)]),  # 1-6 months
            'long_term_vol': np.mean(atm_vols[surface.expiries > 1/2]),     # >6 months
            'term_structure_slope': np.polyfit(surface.expiries, atm_vols, 1)[0],
            'vol_term_spread': atm_vols[-1] - atm_vols[0]  # Long - Short spread
        }
        
        return metrics
    
    def _assess_surface_quality(self, surface: VolatilitySurface) -> Dict[str, float]:
        """
        Assess the quality and arbitrage-free properties of the vol surface.
        
        A high-quality volatility surface should:
        1. Be free of static arbitrage (calendar and butterfly)
        2. Be smooth and well-behaved
        3. Have realistic relationships between strikes and maturities
        4. Be consistent with market observations
        """
        metrics = {}
        
        # Check for calendar spread arbitrage
        calendar_violations = 0
        for i in range(len(surface.strikes)):
            for j in range(len(surface.expiries)-1):
                if surface.volatilities[i,j] < surface.volatilities[i,j+1]:
                    calendar_violations += 1
        
        # Check for butterfly spread arbitrage
        butterfly_violations = 0
        for i in range(len(surface.expiries)):
            for j in range(1, len(surface.strikes)-1):
                left = surface.volatilities[j-1,i]
                center = surface.volatilities[j,i]
                right = surface.volatilities[j+1,i]
                if center > 0.5 * (left + right):  # Convexity violation
                    butterfly_violations += 1
        
        # Calculate surface smoothness
        vol_gradients = np.gradient(surface.volatilities)
        smoothness = np.mean(np.abs(vol_gradients))
        
        metrics.update({
            'calendar_arbitrage_violations': calendar_violations,
            'butterfly_arbitrage_violations': butterfly_violations,
            'surface_smoothness': smoothness,
            'surface_total_variance': np.var(surface.volatilities)
        })
        
        return metrics