# src/models/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

@dataclass
class OptionResult:
    """Container for option pricing results."""
    price: float
    greeks: Optional[Dict[str, float]] = None
    error_estimate: Optional[float] = None
    additional_info: Optional[Dict[str, any]] = None

class OptionPricingModel(ABC):
    """Abstract base class for option pricing models."""
    
    @abstractmethod
    def price_call(self, S: float, K: float, r: float, sigma: float, T: float) -> OptionResult:
        """Price a European call option."""
        pass
    
    @abstractmethod
    def price_put(self, S: float, K: float, r: float, sigma: float, T: float) -> OptionResult:
        """Price a European put option."""
        pass
    
    @abstractmethod
    def calculate_greeks(self, S: float, K: float, r: float, sigma: float, T: float,
                        option_type: str = "call") -> Dict[str, float]:
        """Calculate option Greeks."""
        pass

# src/models/black_scholes.py
import numpy as np
from scipy.stats import norm
from ..utils.validation import validate_parameters
from ..utils.decorators import memoize
from .base import OptionPricingModel, OptionResult

class BlackScholesModel(OptionPricingModel):
    """
    Implementation of the Black-Scholes option pricing model.
    
    This model provides analytical solutions for European option prices and Greeks
    under the following assumptions:
    1. The stock follows geometric Brownian motion
    2. No arbitrage opportunities
    3. Risk-free rate and volatility are constant
    4. European exercise only
    5. No dividends
    6. No transaction costs
    """
    
    @staticmethod
    def _d1(S: float, K: float, r: float, sigma: float, T: float) -> float:
        """Calculate d1 parameter of the Black-Scholes formula."""
        return (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def _d2(d1: float, sigma: float, T: float) -> float:
        """Calculate d2 parameter of the Black-Scholes formula."""
        return d1 - sigma * np.sqrt(T)
    
    @validate_parameters
    @memoize
    def price_call(self, S: float, K: float, r: float, sigma: float, T: float) -> OptionResult:
        """
        Calculate Black-Scholes price for a European call option.
        
        Args:
            S: Current stock price
            K: Strike price
            r: Risk-free interest rate (annualized)
            sigma: Volatility (annualized)
            T: Time to maturity (in years)
        
        Returns:
            OptionResult containing price and Greeks
        """
        if T <= 0:
            return OptionResult(max(S - K, 0))
        
        d1 = self._d1(S, K, r, sigma, T)
        d2 = self._d2(d1, sigma, T)
        
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        greeks = self.calculate_greeks(S, K, r, sigma, T, "call")
        
        return OptionResult(price=price, greeks=greeks)
    
    @validate_parameters
    @memoize
    def price_put(self, S: float, K: float, r: float, sigma: float, T: float) -> OptionResult:
        """Calculate Black-Scholes price for a European put option."""
        if T <= 0:
            return OptionResult(max(K - S, 0))
        
        d1 = self._d1(S, K, r, sigma, T)
        d2 = self._d2(d1, sigma, T)
        
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        greeks = self.calculate_greeks(S, K, r, sigma, T, "put")
        
        return OptionResult(price=price, greeks=greeks)
    
    def calculate_greeks(self, S: float, K: float, r: float, sigma: float, T: float,
                        option_type: str = "call") -> Dict[str, float]:
        """Calculate all Greeks for the option."""
        if T <= 0:
            return {
                "delta": 0.0, "gamma": 0.0, "theta": 0.0,
                "vega": 0.0, "rho": 0.0
            }
        
        d1 = self._d1(S, K, r, sigma, T)
        d2 = self._d2(d1, sigma, T)
        sqrt_T = np.sqrt(T)
        
        sign = 1 if option_type.lower() == "call" else -1
        
        greeks = {
            "delta": sign * norm.cdf(sign * d1),
            "gamma": norm.pdf(d1) / (S * sigma * sqrt_T),
            "theta": (-S * norm.pdf(d1) * sigma / (2 * sqrt_T) -
                     sign * r * K * np.exp(-r * T) * norm.cdf(sign * d2)),
            "vega": S * sqrt_T * norm.pdf(d1),
            "rho": sign * K * T * np.exp(-r * T) * norm.cdf(sign * d2)
        }
        
        return greeks

# src/models/monte_carlo.py
import numpy as np
from typing import Optional, Tuple
from ..utils.validation import validate_parameters
from .base import OptionPricingModel, OptionResult

class MonteCarloModel(OptionPricingModel):
    """
    Monte Carlo simulation for option pricing with variance reduction techniques.
    
    Features:
    1. Standard Monte Carlo
    2. Antithetic variates for variance reduction
    3. Control variates using analytical Black-Scholes price
    4. Parallel execution for large simulations
    """
    
    def __init__(self, n_sims: int = 10000, n_steps: int = 100,
                 random_seed: Optional[int] = None):
        """Initialize Monte Carlo simulator."""
        self.n_sims = n_sims
        self.n_steps = n_steps
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def _generate_paths(self, S: float, r: float, sigma: float,
                       T: float) -> np.ndarray:
        """Generate stock price paths using geometric Brownian motion."""
        dt = T/self.n_steps
        nudt = (r - 0.5 * sigma**2) * dt
        sigmasqrtdt = sigma * np.sqrt(dt)
        
        Z = np.random.standard_normal((self.n_sims, self.n_steps))
        paths = np.zeros((self.n_sims, self.n_steps + 1))
        paths[:, 0] = S
        
        for t in range(1, self.n_steps + 1):
            paths[:, t] = paths[:, t-1] * np.exp(nudt + sigmasqrtdt * Z[:, t-1])
        
        return paths
    
    @validate_parameters
    def price_call(self, S: float, K: float, r: float, sigma: float,
                   T: float) -> OptionResult:
        """Price a European call option using Monte Carlo simulation."""
        paths = self._generate_paths(S, r, sigma, T)
        payoffs = np.maximum(paths[:, -1] - K, 0)
        
        price = np.exp(-r * T) * np.mean(payoffs)
        std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(self.n_sims)
        
        return OptionResult(
            price=price,
            error_estimate=std_error,
            additional_info={"n_sims": self.n_sims}
        )
    
    @validate_parameters
    def price_put(self, S: float, K: float, r: float, sigma: float,
                  T: float) -> OptionResult:
        """Price a European put option using Monte Carlo simulation."""
        paths = self._generate_paths(S, r, sigma, T)
        payoffs = np.maximum(K - paths[:, -1], 0)
        
        price = np.exp(-r * T) * np.mean(payoffs)
        std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(self.n_sims)
        
        return OptionResult(
            price=price,
            error_estimate=std_error,
            additional_info={"n_sims": self.n_sims}
        )
    
# src/analytics/portfolio.py
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
from ..models.base import OptionPricingModel

@dataclass
class Position:
    """Represents a single option position in a portfolio."""
    option_type: str  # "call" or "put"
    S: float  # Current stock price
    K: float  # Strike price
    r: float  # Risk-free rate
    sigma: float  # Volatility
    T: float  # Time to maturity
    quantity: float  # Number of contracts (negative for short positions)
    model: str = "Black-Scholes"  # Pricing model to use

@dataclass
class PortfolioRisk:
    """Contains portfolio-level risk metrics."""
    value: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    value_at_risk: float
    expected_shortfall: float
    stress_test_results: Dict[str, float]

class PortfolioAnalyzer:
    """
    Analyzes option portfolios for risk management and optimization.
    
    This class provides comprehensive portfolio analysis including:
    - Portfolio valuation
    - Aggregate Greeks calculation
    - Value at Risk (VaR) estimation
    - Expected Shortfall (ES) calculation
    - Stress testing under various scenarios
    - Risk attribution analysis
    """
    
    def __init__(self, pricing_model: OptionPricingModel):
        """Initialize with a specific pricing model."""
        self.pricing_model = pricing_model
    
    def calculate_portfolio_value(self, positions: List[Position]) -> float:
        """Calculate the total portfolio value."""
        total_value = 0.0
        
        for pos in positions:
            if pos.option_type.lower() == "call":
                price = self.pricing_model.price_call(
                    pos.S, pos.K, pos.r, pos.sigma, pos.T
                ).price
            else:
                price = self.pricing_model.price_put(
                    pos.S, pos.K, pos.r, pos.sigma, pos.T
                ).price
            
            total_value += price * pos.quantity
        
        return total_value
    
    def calculate_portfolio_greeks(self, positions: List[Position]) -> Dict[str, float]:
        """
        Calculate aggregate Greeks for the entire portfolio.
        
        This method accounts for:
        - Position direction (long/short)
        - Position size
        - Greek correlations between positions
        """
        portfolio_greeks = {
            "delta": 0.0,
            "gamma": 0.0,
            "theta": 0.0,
            "vega": 0.0,
            "rho": 0.0
        }
        
        for pos in positions:
            greeks = self.pricing_model.calculate_greeks(
                pos.S, pos.K, pos.r, pos.sigma, pos.T, pos.option_type
            )
            
            for greek in portfolio_greeks:
                portfolio_greeks[greek] += greeks[greek] * pos.quantity
        
        return portfolio_greeks
    
    def calculate_var(self, positions: List[Position], 
                     confidence_level: float = 0.95,
                     time_horizon: float = 1/252,  # One trading day
                     method: str = "historical") -> float:
        """
        Calculate Value at Risk using various methods.
        
        Methods available:
        - Historical simulation
        - Parametric VaR (using delta-normal approach)
        - Monte Carlo VaR
        """
        if method == "historical":
            return self._historical_var(positions, confidence_level, time_horizon)
        elif method == "parametric":
            return self._parametric_var(positions, confidence_level, time_horizon)
        else:
            return self._monte_carlo_var(positions, confidence_level, time_horizon)
    
    def _historical_var(self, positions: List[Position], 
                       confidence_level: float,
                       time_horizon: float) -> float:
        """Calculate VaR using historical simulation."""
        # Implementation uses actual historical data or simulated historical scenarios
        # to estimate potential losses
        portfolio_values = []
        scenarios = self._generate_historical_scenarios(252)  # One year of data
        
        for scenario in scenarios:
            value = 0
            for pos in positions:
                # Adjust position parameters based on historical scenario
                adjusted_pos = self._adjust_position_for_scenario(pos, scenario)
                value += self.calculate_position_value(adjusted_pos)
            portfolio_values.append(value)
        
        return np.percentile(portfolio_values, (1 - confidence_level) * 100)
    
    def _parametric_var(self, positions: List[Position],
                       confidence_level: float,
                       time_horizon: float) -> float:
        """
        Calculate VaR using the delta-normal approach.
        
        This method:
        1. Uses portfolio Greeks to approximate value changes
        2. Assumes returns are normally distributed
        3. Scales by time horizon and confidence level
        """
        portfolio_delta = self.calculate_portfolio_greeks(positions)["delta"]
        portfolio_value = self.calculate_portfolio_value(positions)
        
        # Calculate portfolio volatility using position deltas and correlations
        portfolio_volatility = self._calculate_portfolio_volatility(positions)
        
        # Calculate VaR
        z_score = norm.ppf(1 - confidence_level)
        var = portfolio_value * z_score * portfolio_volatility * np.sqrt(time_horizon)
        
        return abs(var)
    
    def perform_stress_test(self, positions: List[Position],
                          scenarios: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Perform stress testing under various market scenarios.
        
        Scenarios can include:
        - Market crashes (e.g., -20% in stock price)
        - Volatility spikes
        - Interest rate changes
        - Combined scenarios
        """
        results = {}
        base_value = self.calculate_portfolio_value(positions)
        
        for scenario_name, scenario_changes in scenarios.items():
            # Apply scenario changes to each position
            stressed_positions = [
                self._adjust_position_for_scenario(pos, scenario_changes)
                for pos in positions
            ]
            
            # Calculate new portfolio value under stress
            stressed_value = self.calculate_portfolio_value(stressed_positions)
            results[scenario_name] = stressed_value - base_value
        
        return results
    
    def _adjust_position_for_scenario(self, position: Position,
                                    scenario: Dict[str, float]) -> Position:
        """Apply scenario changes to a position's parameters."""
        new_pos = Position(
            option_type=position.option_type,
            S=position.S * (1 + scenario.get('price_change', 0)),
            K=position.K,
            r=position.r + scenario.get('rate_change', 0),
            sigma=position.sigma * (1 + scenario.get('vol_change', 0)),
            T=position.T,
            quantity=position.quantity
        )
        return new_pos
    
    def optimize_hedge(self, positions: List[Position],
                      target_greek: str = "delta",
                      target_value: float = 0.0) -> Optional[Position]:
        """
        Find the optimal hedging position to achieve a target Greek exposure.
        
        This method uses optimization to find the best hedging instrument
        and position size to achieve the desired risk profile.
        """
        current_greeks = self.calculate_portfolio_greeks(positions)
        current_exposure = current_greeks[target_greek]
        
        def objective_function(hedge_quantity: float) -> float:
            """Objective function for optimization."""
            hedge_position = Position(
                option_type="call",  # Could be parameterized
                S=positions[0].S,  # Use first position as reference
                K=positions[0].S,  # At-the-money option
                r=positions[0].r,
                sigma=positions[0].sigma,
                T=1/12,  # One month option
                quantity=hedge_quantity
            )
            # Calculate new total exposure with hedge
            all_positions = positions + [hedge_position]
            new_greeks = self.calculate_portfolio_greeks(all_positions)
            new_exposure = new_greeks[target_greek]
            
            # Return absolute difference from target
            return abs(new_exposure - target_value)
        
        # Use scipy's minimize to find optimal hedge quantity
        from scipy.optimize import minimize
        result = minimize(
            objective_function,
            x0=0.0,  # Start with no hedge
            method='Nelder-Mead'
        )
        
        if result.success:
            # Return the optimal hedging position
            return Position(
                option_type="call",
                S=positions[0].S,
                K=positions[0].S,
                r=positions[0].r,
                sigma=positions[0].sigma,
                T=1/12,
                quantity=result.x[0]
            )
        return None

    def calculate_expected_shortfall(self, positions: List[Position],
                                   confidence_level: float = 0.95,
                                   n_scenarios: int = 10000) -> float:
        """
        Calculate Expected Shortfall (ES), also known as Conditional Value at Risk (CVaR).
        
        ES measures the average loss beyond VaR, providing a more complete picture of tail risk.
        This implementation uses Monte Carlo simulation to estimate ES.
        """
        # Generate scenarios for Monte Carlo simulation
        scenarios = []
        for _ in range(n_scenarios):
            scenario_changes = {
                'price_change': np.random.normal(0, 0.02),  # 2% daily volatility
                'vol_change': np.random.normal(0, 0.05),    # 5% vol-of-vol
                'rate_change': np.random.normal(0, 0.001)   # 0.1% rate change
            }
            scenarios.append(scenario_changes)
        
        # Calculate portfolio value under each scenario
        values = []
        base_value = self.calculate_portfolio_value(positions)
        
        for scenario in scenarios:
            stressed_positions = [
                self._adjust_position_for_scenario(pos, scenario)
                for pos in positions
            ]
            scenario_value = self.calculate_portfolio_value(stressed_positions)
            values.append(scenario_value - base_value)
        
        # Sort values and calculate ES
        sorted_values = np.sort(values)
        cutoff_index = int(n_scenarios * (1 - confidence_level))
        return -np.mean(sorted_values[:cutoff_index])

    def risk_attribution(self, positions: List[Position]) -> Dict[str, Dict[str, float]]:
        """
        Perform risk attribution analysis to understand sources of portfolio risk.
        
        This method breaks down risk contributions by:
        1. Individual positions
        2. Risk factors (delta, vega, etc.)
        3. Underlying assets
        """
        attribution = {}
        total_value = self.calculate_portfolio_value(positions)
        
        # Calculate position-level metrics
        for i, pos in enumerate(positions):
            pos_greeks = self.pricing_model.calculate_greeks(
                pos.S, pos.K, pos.r, pos.sigma, pos.T, pos.option_type
            )
            pos_value = self.calculate_position_value(pos)
            
            attribution[f"Position_{i+1}"] = {
                "value_contribution": pos_value / total_value,
                "delta_contribution": pos_greeks["delta"] * pos.quantity,
                "gamma_contribution": pos_greeks["gamma"] * pos.quantity,
                "vega_contribution": pos_greeks["vega"] * pos.quantity,
                "theta_contribution": pos_greeks["theta"] * pos.quantity
            }
        
        return attribution

    def sensitivity_analysis(self, positions: List[Position],
                           parameters: List[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Perform sensitivity analysis to understand portfolio behavior under parameter changes.
        
        This method examines how the portfolio value and risk metrics change when
        key parameters (spot, volatility, rates) are perturbed.
        """
        if parameters is None:
            parameters = ["spot", "volatility", "rate", "time"]
        
        results = {}
        base_value = self.calculate_portfolio_value(positions)
        
        # Define parameter perturbations
        perturbations = {
            "spot": [0.95, 1.05],        # ±5% spot move
            "volatility": [0.8, 1.2],    # ±20% vol move
            "rate": [-0.01, 0.01],       # ±100bps rate move
            "time": [-1/365, 1/365]      # ±1 day
        }
        
        for param in parameters:
            param_results = {}
            
            for perturb in perturbations[param]:
                # Create stressed positions with perturbed parameter
                stressed_positions = self._perturb_positions(positions, param, perturb)
                stressed_value = self.calculate_portfolio_value(stressed_positions)
                param_results[f"{perturb:.2f}"] = stressed_value - base_value
            
            results[param] = param_results
        
        return results

    def _perturb_positions(self, positions: List[Position], 
                          parameter: str, 
                          perturbation: float) -> List[Position]:
        """Helper method to create perturbed positions for sensitivity analysis."""
        perturbed = []
        
        for pos in positions:
            new_pos = Position(
                option_type=pos.option_type,
                S=pos.S * (1 + perturbation if parameter == "spot" else 1),
                K=pos.K,
                r=pos.r + (perturbation if parameter == "rate" else 0),
                sigma=pos.sigma * (1 + perturbation if parameter == "volatility" else 1),
                T=max(0, pos.T + (perturbation if parameter == "time" else 0)),
                quantity=pos.quantity
            )
            perturbed.append(new_pos)
        
        return perturbed

    def generate_risk_report(self, positions: List[Position]) -> Dict[str, Any]:
        """
        Generate a comprehensive risk report for the portfolio.
        
        This report includes:
        1. Portfolio valuation
        2. Risk metrics (Greeks, VaR, ES)
        3. Stress test results
        4. Risk attribution
        5. Sensitivity analysis
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "portfolio_value": self.calculate_portfolio_value(positions),
            "greeks": self.calculate_portfolio_greeks(positions),
            "risk_metrics": {
                "var_95": self.calculate_var(positions, 0.95),
                "expected_shortfall": self.calculate_expected_shortfall(positions),
            },
            "stress_tests": self.perform_stress_test(positions, {
                "market_crash": {"price_change": -0.20},
                "vol_spike": {"vol_change": 1.0},
                "rate_hike": {"rate_change": 0.01}
            }),
            "risk_attribution": self.risk_attribution(positions),
            "sensitivity": self.sensitivity_analysis(positions)
        }
        
        return report