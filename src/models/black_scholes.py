# src/models/black_scholes.py

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import logging
from scipy.stats import norm

from src.models.base import OptionPricingModel, OptionResult
from src.utils.validators import validate_parameters
from src.config import AppConfig

@dataclass
class BlackScholesParameters:
    """Parameters used in the Black-Scholes model."""
    S: float      # Current stock price
    K: float      # Strike price
    r: float      # Risk-free interest rate
    sigma: float  # Volatility of the underlying asset
    T: float      # Time to maturity in years
    q: float      # Dividend yield

class BlackScholesModel(OptionPricingModel):
    """
    Black-Scholes analytical model for European option pricing.
    
    This implementation provides:
    1. Pricing for European call and put options.
    2. Analytical calculation of Greeks: Delta, Gamma, Theta, Vega, and Rho.
    3. Handling of dividend yields.
    4. Robust parameter validation and error handling.
    
    The Black-Scholes model offers closed-form solutions for option prices and Greeks,
    making it highly efficient for European-style options on non-dividend-paying or
    dividend-paying assets.
    """
    
    def __init__(self, 
                 q: float = 0.0, 
                 logging_enabled: bool = True):
        """
            Initialize the Black-Scholes model with optional dividend yield.
            
            Parameters:
            - q: Dividend yield of the underlying asset (default is 0.0).
            - logging_enabled: Enable or disable logging (default is True).
        """
        self.q = q
        self.logger = logging.getLogger(__name__)
        if logging_enabled:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)
    
    @validate_parameters
    def price_call(self, S: float, K: float, r: float, sigma: float, T: float,
                   q: Optional[float] = None) -> OptionResult:
        """
        Price a European call option using the Black-Scholes formula.
        
        Parameters:
        - S: Current stock price
        - K: Strike price
        - r: Risk-free interest rate
        - sigma: Volatility of the underlying asset
        - T: Time to maturity in years
        - q: Dividend yield (optional, defaults to model's q)
        
        Returns:
        - OptionResult containing the price, Greeks, and additional information.
        """
        q = self.q if q is None else q
        params = BlackScholesParameters(S, K, r, sigma, T, q)
        self.logger.info(f"Pricing European Call with parameters: {params}")
        
        d1, d2 = self._calculate_d1_d2(params)
        call_price = (S * np.exp(-q * T) * norm.cdf(d1)) - (K * np.exp(-r * T) * norm.cdf(d2))
        
        greeks = self._calculate_greeks(d1, d2, params)
        
        return OptionResult(
            price=call_price,
            greeks=greeks,
            error_estimate=None,  # Analytical model has no simulation error
            additional_info={
                "model": "Black-Scholes",
                "parameters": params
            }
        )
    
    @validate_parameters
    def price_put(self, S: float, K: float, r: float, sigma: float, T: float,
                 q: Optional[float] = None) -> OptionResult:
        """
        Price a European put option using the Black-Scholes formula.
        
        Parameters:
        - S: Current stock price
        - K: Strike price
        - r: Risk-free interest rate
        - sigma: Volatility of the underlying asset
        - T: Time to maturity in years
        - q: Dividend yield (optional, defaults to model's q)
        
        Returns:
        - OptionResult containing the price, Greeks, and additional information.
        """
        q = self.q if q is None else q
        params = BlackScholesParameters(S, K, r, sigma, T, q)
        self.logger.info(f"Pricing European Put with parameters: {params}")
        
        d1, d2 = self._calculate_d1_d2(params)
        put_price = (K * np.exp(-r * T) * norm.cdf(-d2)) - (S * np.exp(-q * T) * norm.cdf(-d1))
        
        greeks = self._calculate_greeks(d1, d2, params)
        
        return OptionResult(
            price=put_price,
            greeks=greeks,
            error_estimate=None,  # Analytical model has no simulation error
            additional_info={
                "model": "Black-Scholes",
                "parameters": params
            }
        )
    
    def _calculate_d1_d2(self, params: BlackScholesParameters) -> Tuple[float, float]:
        """
        Calculate the d1 and d2 parameters used in the Black-Scholes formula.
        
        Parameters:
        - params: BlackScholesParameters dataclass instance.
        
        Returns:
        - A tuple containing d1 and d2.
        """
        if params.T <= 0:
            raise ValueError("Time to maturity must be positive.")
        
        d1 = (np.log(params.S / params.K) + (params.r - params.q + 0.5 * params.sigma**2) * params.T) / \
             (params.sigma * np.sqrt(params.T))
        d2 = d1 - params.sigma * np.sqrt(params.T)
        
        self.logger.debug(f"Calculated d1: {d1}, d2: {d2}")
        return d1, d2
    
    def _calculate_greeks(self, d1: float, d2: float, params: BlackScholesParameters) -> Dict[str, float]:
        """
        Calculate the Greeks for the option.
        
        Parameters:
        - d1: Calculated d1 from Black-Scholes formula
        - d2: Calculated d2 from Black-Scholes formula
        - params: BlackScholesParameters dataclass instance.
        
        Returns:
        - Dictionary containing Delta, Gamma, Theta, Vega, and Rho.
        """
        delta = np.exp(-params.q * params.T) * norm.cdf(d1) if params.option_type == "call" else \
                np.exp(-params.q * params.T) * (norm.cdf(d1) - 1)
        gamma = (np.exp(-params.q * params.T) * norm.pdf(d1)) / (params.S * params.sigma * np.sqrt(params.T))
        theta_call = (- (params.S * params.sigma * np.exp(-params.q * params.T) * norm.pdf(d1)) / (2 * np.sqrt(params.T)) 
                      - params.q * params.S * np.exp(-params.q * params.T) * norm.cdf(d1) 
                      + params.q * params.K * np.exp(-params.r * params.T) * norm.cdf(d2))
        theta_put = (- (params.S * params.sigma * np.exp(-params.q * params.T) * norm.pdf(d1)) / (2 * np.sqrt(params.T)) 
                     + params.q * params.S * np.exp(-params.q * params.T) * norm.cdf(-d1) 
                     - params.r * params.K * np.exp(-params.r * params.T) * norm.cdf(-d2))
        vega = params.S * np.exp(-params.q * params.T) * norm.pdf(d1) * np.sqrt(params.T)
        rho_call = params.K * params.T * np.exp(-params.r * params.T) * norm.cdf(d2)
        rho_put = -params.K * params.T * np.exp(-params.r * params.T) * norm.cdf(-d2)
        
        greeks = {
            "delta": self._calculate_delta(params.option_type, d1, params),
            "gamma": gamma,
            "theta": self._calculate_theta(params.option_type, params, d1, d2),
            "vega": vega,
            "rho": self._calculate_rho(params.option_type, params, d2)
        }
        
        self.logger.debug(f"Calculated Greeks: {greeks}")
        return greeks
    
    def _calculate_delta(self, option_type: str, d1: float, params: BlackScholesParameters) -> float:
        """
        Calculate Delta for the option.
        
        Parameters:
        - option_type: 'call' or 'put'
        - d1: Calculated d1 from Black-Scholes formula
        - params: BlackScholesParameters dataclass instance.
        
        Returns:
        - Delta value.
        """
        if option_type.lower() == "call":
            delta = np.exp(-params.q * params.T) * norm.cdf(d1)
        else:
            delta = np.exp(-params.q * params.T) * (norm.cdf(d1) - 1)
        self.logger.debug(f"Calculated Delta: {delta}")
        return delta
    
    def _calculate_theta(self, option_type: str, params: BlackScholesParameters, d1: float, d2: float) -> float:
        """
        Calculate Theta for the option.
        
        Parameters:
        - option_type: 'call' or 'put'
        - params: BlackScholesParameters dataclass instance.
        - d1: Calculated d1 from Black-Scholes formula
        - d2: Calculated d2 from Black-Scholes formula
        
        Returns:
        - Theta value.
        """
        term1 = -(params.S * params.sigma * np.exp(-params.q * params.T) * norm.pdf(d1)) / (2 * np.sqrt(params.T))
        if option_type.lower() == "call":
            term2 = -params.q * params.S * np.exp(-params.q * params.T) * norm.cdf(d1)
            term3 = params.q * params.K * np.exp(-params.r * params.T) * norm.cdf(d2)
            theta = term1 + term2 + term3
        else:
            term2 = params.q * params.S * np.exp(-params.q * params.T) * norm.cdf(-d1)
            term3 = -params.r * params.K * np.exp(-params.r * params.T) * norm.cdf(-d2)
            theta = term1 + term2 + term3
        self.logger.debug(f"Calculated Theta: {theta}")
        return theta
    
    def _calculate_rho(self, option_type: str, params: BlackScholesParameters, d2: float) -> float:
        """
        Calculate Rho for the option.
        
        Parameters:
        - option_type: 'call' or 'put'
        - params: BlackScholesParameters dataclass instance.
        - d2: Calculated d2 from Black-Scholes formula
        
        Returns:
        - Rho value.
        """
        if option_type.lower() == "call":
            rho = params.K * params.T * np.exp(-params.r * params.T) * norm.cdf(d2)
        else:
            rho = -params.K * params.T * np.exp(-params.r * params.T) * norm.cdf(-d2)
        self.logger.debug(f"Calculated Rho: {rho}")
        return rho
    
    def _calculate_vega(self, params: BlackScholesParameters, d1: float) -> float:
        """
        Calculate Vega for the option.
        
        Parameters:
        - params: BlackScholesParameters dataclass instance.
        - d1: Calculated d1 from Black-Scholes formula
        
        Returns:
        - Vega value.
        """
        vega = params.S * np.exp(-params.q * params.T) * norm.pdf(d1) * np.sqrt(params.T)
        self.logger.debug(f"Calculated Vega: {vega}")
        return vega
    
    def calculate_greeks(self, S: float, K: float, r: float, sigma: float,
                        T: float, option_type: str = "call",
                        q: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate option Greeks analytically.
        
        Parameters:
        - S: Current stock price
        - K: Strike price
        - r: Risk-free interest rate
        - sigma: Volatility of the underlying asset
        - T: Time to maturity in years
        - option_type: 'call' or 'put'
        - q: Dividend yield (optional, defaults to model's q)
        
        Returns:
        - Dictionary containing Delta, Gamma, Theta, Vega, and Rho.
        """
        q = self.q if q is None else q
        params = BlackScholesParameters(S, K, r, sigma, T, q)
        d1, d2 = self._calculate_d1_d2(params)
        greeks = self._calculate_greeks(d1, d2, params)
        return greeks
    
    @validate_parameters
    def implied_volatility(self, option_price: float, S: float, K: float, r: float, T: float,
                           option_type: str = "call", q: Optional[float] = None,
                           initial_guess: float = 0.2, tol: float = 1e-6,
                           max_iterations: int = 100) -> Optional[float]:
        """
        Calculate the implied volatility using the Black-Scholes model via the Newton-Raphson method.
        
        Parameters:
        - option_price: Observed market price of the option
        - S: Current stock price
        - K: Strike price
        - r: Risk-free interest rate
        - T: Time to maturity in years
        - option_type: 'call' or 'put'
        - q: Dividend yield (optional, defaults to model's q)
        - initial_guess: Initial guess for volatility (default is 0.2)
        - tol: Tolerance for convergence (default is 1e-6)
        - max_iterations: Maximum number of iterations (default is 100)
        
        Returns:
        - Implied volatility if converged, else None.
        """
        q = self.q if q is None else q
        sigma = initial_guess
        for i in range(max_iterations):
            params = BlackScholesParameters(S, K, r, sigma, T, q)
            try:
                if option_type.lower() == "call":
                    model_price = self.price_call(S, K, r, sigma, T, q).price
                else:
                    model_price = self.price_put(S, K, r, sigma, T, q).price
            except Exception as e:
                self.logger.error(f"Error in pricing during implied volatility calculation: {e}")
                return None
            
            # Vega for Newton-Raphson
            d1, _ = self._calculate_d1_d2(params)
            vega = self._calculate_vega(params, d1)
            
            price_diff = model_price - option_price
            self.logger.debug(f"Iteration {i}: sigma={sigma}, price_diff={price_diff}, vega={vega}")
            
            if abs(price_diff) < tol:
                self.logger.info(f"Converged to implied volatility: {sigma} in {i} iterations.")
                return sigma
            
            if vega == 0:
                self.logger.warning("Vega is zero. Cannot continue Newton-Raphson.")
                return None
            
            sigma -= price_diff / vega
        
        self.logger.warning("Implied volatility did not converge.")
        return None
    
    @validate_parameters
    def price_option(self, S: float, K: float, r: float, sigma: float, T: float,
                    option_type: str = "call", q: Optional[float] = None) -> OptionResult:
        """
        General method to price an option based on the option type.
        
        Parameters:
        - S: Current stock price
        - K: Strike price
        - r: Risk-free interest rate
        - sigma: Volatility of the underlying asset
        - T: Time to maturity in years
        - option_type: 'call' or 'put'
        - q: Dividend yield (optional, defaults to model's q)
        
        Returns:
        - OptionResult containing the price, Greeks, and additional information.
        """
        option_type = option_type.lower()
        if option_type == "call":
            return self.price_call(S, K, r, sigma, T, q)
        elif option_type == "put":
            return self.price_put(S, K, r, sigma, T, q)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
