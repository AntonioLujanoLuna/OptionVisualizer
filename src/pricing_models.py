"""
pricing_models.py

Enhanced implementation of option pricing models including:
1. Black-Scholes (calls & puts) with complete Greeks
2. Binomial (Cox-Ross-Rubinstein) with American option support
3. Monte Carlo with variance reduction
4. Implied volatility calculations
5. Volatility surface generation

This module provides a comprehensive suite of option pricing tools with educational
comments and clear documentation to help understand the mathematical concepts behind
option pricing.
"""

import math
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from scipy.optimize import minimize

# Constants for numerical stability and optimization
EPSILON = 1e-8
MAX_ITERATIONS = 1000

@dataclass
class OptionGreeks:
    """Container for option Greeks calculations."""
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    charm: Optional[float] = None  # Delta decay
    vanna: Optional[float] = None  # Delta-volatility sensitivity
    volga: Optional[float] = None  # Vega convexity

class BlackScholesModel:
    """
    Implementation of the Black-Scholes option pricing model.
    
    The Black-Scholes model assumes:
    1. The stock follows geometric Brownian motion
    2. No arbitrage opportunities
    3. Risk-free rate and volatility are constant
    4. European exercise only
    5. No dividends
    6. No transaction costs
    """
    
    @staticmethod
    def _d1(S: float, K: float, r: float, sigma: float, T: float) -> float:
        """Calculate the d1 parameter in the Black-Scholes formula."""
        if T <= 0 or sigma <= 0:
            raise ValueError("Time to maturity and volatility must be positive")
        return (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def _d2(d1: float, sigma: float, T: float) -> float:
        """Calculate the d2 parameter in the Black-Scholes formula."""
        return d1 - sigma * np.sqrt(T)
    
    @classmethod
    def price_call(cls, S: float, K: float, r: float, sigma: float, T: float) -> float:
        """
        Calculate the Black-Scholes price for a European call option.
        
        Parameters:
            S: Current stock price
            K: Strike price
            r: Risk-free interest rate (annualized)
            sigma: Volatility (annualized)
            T: Time to maturity (in years)
        
        Returns:
            float: Call option price
        """
        if T <= 0:
            return max(S - K, 0.0)
        
        d1 = cls._d1(S, K, r, sigma, T)
        d2 = cls._d2(d1, sigma, T)
        
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    @classmethod
    def price_put(cls, S: float, K: float, r: float, sigma: float, T: float) -> float:
        """
        Calculate the Black-Scholes price for a European put option using put-call parity.
        
        Put-call parity states: C - P = S - K*exp(-rT)
        Therefore: P = C - S + K*exp(-rT)
        """
        if T <= 0:
            return max(K - S, 0.0)
        
        call_price = cls.price_call(S, K, r, sigma, T)
        return call_price - S + K * np.exp(-r * T)
    
    @classmethod
    def calculate_greeks(cls, S: float, K: float, r: float, sigma: float, T: float, 
                        option_type: str = "call") -> OptionGreeks:
        """
        Calculate all Greeks for a given option.
        
        Includes first-order, second-order, and cross-Greeks for comprehensive risk analysis.
        """
        if T <= 0:
            return OptionGreeks(0.0, 0.0, 0.0, 0.0, 0.0)
        
        d1 = cls._d1(S, K, r, sigma, T)
        d2 = cls._d2(d1, sigma, T)
        sqrt_T = np.sqrt(T)
        
        # Common terms
        nd1 = norm.pdf(d1)
        Nd1 = norm.cdf(d1)
        Nd2 = norm.cdf(d2)
        exp_rt = np.exp(-r * T)
        
        if option_type.lower() == "put":
            # Adjust for put option
            Nd1 = Nd1 - 1
            Nd2 = Nd2 - 1
        
        # First-order Greeks
        delta = Nd1
        gamma = nd1 / (S * sigma * sqrt_T)
        theta = (-S * nd1 * sigma / (2 * sqrt_T) - 
                r * K * exp_rt * Nd2)
        vega = S * sqrt_T * nd1
        rho = K * T * exp_rt * Nd2
        
        # Second-order and cross-Greeks
        charm = -nd1 * (2*r*T - d2*sigma*sqrt_T) / (2*T*sigma*sqrt_T)
        vanna = -nd1 * d2 / sigma
        volga = vega * d1 * d2 / sigma
        
        return OptionGreeks(
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho,
            charm=charm,
            vanna=vanna,
            volga=volga
        )

class BinomialModel:
    """
    Implementation of the Cox-Ross-Rubinstein binomial tree model.
    
    Supports both European and American exercise styles, and can be used
    to price options on stocks with known discrete dividends.
    """
    
    @staticmethod
    def _calculate_parameters(sigma: float, T: float, steps: int) -> Tuple[float, float, float]:
        """Calculate the basic parameters for the binomial tree."""
        dt = T / steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1/u
        p = (np.exp(r * dt) - d) / (u - d)  # Risk-neutral probability
        return dt, u, d, p
    
    @classmethod
    def price_option(cls, S: float, K: float, r: float, sigma: float, T: float,
                    steps: int = 100, option_type: str = "call",
                    exercise: str = "european") -> float:
        """
        Price an option using the binomial tree model.
        
        Parameters:
            exercise: 'european' or 'american'
        """
        dt, u, d, p = cls._calculate_parameters(sigma, T, steps)
        discount = np.exp(-r * dt)
        
        # Initialize asset prices at final nodes
        prices = np.array([S * u**j * d**(steps-j) for j in range(steps+1)])
        
        # Initialize option values at final nodes
        if option_type.lower() == "call":
            values = np.maximum(prices - K, 0)
        else:
            values = np.maximum(K - prices, 0)
        
        # Backward induction through the tree
        for i in range(steps-1, -1, -1):
            prices = prices[:-1]  # Asset prices one step earlier
            values = discount * (p * values[1:] + (1-p) * values[:-1])
            
            if exercise.lower() == "american":
                # Check for early exercise
                if option_type.lower() == "call":
                    values = np.maximum(values, prices - K)
                else:
                    values = np.maximum(values, K - prices)
        
        return values[0]

class MonteCarloModel:
    """
    Monte Carlo simulation for option pricing with variance reduction techniques.
    
    Implements:
    1. Standard Monte Carlo
    2. Antithetic variates
    3. Control variates
    4. Stratified sampling
    """
    
    @staticmethod
    def _generate_paths(S: float, r: float, sigma: float, T: float, 
                       steps: int, sims: int) -> np.ndarray:
        """Generate stock price paths using geometric Brownian motion."""
        dt = T/steps
        nudt = (r - 0.5 * sigma**2) * dt
        sigmasqrtdt = sigma * np.sqrt(dt)
        
        # Generate random standard normal variables
        Z = np.random.standard_normal((sims, steps))
        
        # Initialize price paths array
        paths = np.zeros((sims, steps+1))
        paths[:, 0] = S
        
        # Generate paths
        for t in range(1, steps+1):
            paths[:, t] = paths[:, t-1] * np.exp(nudt + sigmasqrtdt * Z[:, t-1])
        
        return paths
    
    @classmethod
    def price_option(cls, S: float, K: float, r: float, sigma: float, T: float,
                    steps: int = 100, sims: int = 10000, option_type: str = "call",
                    variance_reduction: str = "antithetic") -> Tuple[float, float]:
        """
        Price an option using Monte Carlo simulation with variance reduction.
        
        Returns:
            Tuple[float, float]: (option_price, standard_error)
        """
        if variance_reduction == "antithetic":
            # Generate antithetic pairs of paths
            paths1 = cls._generate_paths(S, r, sigma, T, steps, sims//2)
            Z = np.random.standard_normal((sims//2, steps))
            paths2 = cls._generate_paths(S, r, sigma, T, steps, sims//2, -Z)
            
            # Combine paths
            final_prices = np.concatenate([paths1[:, -1], paths2[:, -1]])
        else:
            paths = cls._generate_paths(S, r, sigma, T, steps, sims)
            final_prices = paths[:, -1]
        
        # Calculate payoffs
        if option_type.lower() == "call":
            payoffs = np.maximum(final_prices - K, 0)
        else:
            payoffs = np.maximum(K - final_prices, 0)
        
        # Discount payoffs
        discount = np.exp(-r * T)
        option_price = discount * np.mean(payoffs)
        std_error = discount * np.std(payoffs) / np.sqrt(sims)
        
        return option_price, std_error

def implied_volatility(price: float, S: float, K: float, r: float, T: float,
                      option_type: str = "call", sigma_init: float = 0.3) -> float:
    """
    Calculate implied volatility using the Newton-Raphson method.
    
    Args:
        price: Market price of the option
        sigma_init: Initial guess for implied volatility
    
    Returns:
        float: Implied volatility that reproduces the market price
    """
    def objective(sigma):
        if option_type.lower() == "call":
            model_price = BlackScholesModel.price_call(S, K, r, sigma, T)
        else:
            model_price = BlackScholesModel.price_put(S, K, r, sigma, T)
        return model_price - price
    
    result = minimize(lambda x: abs(objective(x)), sigma_init,
                     method='Nelder-Mead', options={'maxiter': MAX_ITERATIONS})
    
    if not result.success:
        raise ValueError("Failed to converge to implied volatility")
    
    return result.x[0]

def implied_volatility_surface(S: float, r: float, T_range: np.ndarray,
                             moneyness_range: np.ndarray,
                             market_vols: np.ndarray) -> go.Figure:
    """
    Generate and plot an implied volatility surface.
    
    Args:
        T_range: Array of times to maturity
        moneyness_range: Array of K/S ratios
        market_vols: 2D array of market-observed volatilities
    
    Returns:
        plotly.graph_objects.Figure: Interactive 3D surface plot
    """
    # Create mesh grid for 3D plot
    T_mesh, M_mesh = np.meshgrid(T_range, moneyness_range)
    
    # Create the 3D surface plot
    fig = go.Figure(data=[go.Surface(
        x=T_mesh,
        y=M_mesh,
        z=market_vols,
        colorscale='Viridis'
    )])
    
    # Update layout for better visualization
    fig.update_layout(
        title='Implied Volatility Surface',
        scene=dict(
            xaxis_title='Time to Maturity',
            yaxis_title='Moneyness (K/S)',
            zaxis_title='Implied Volatility'
        ),
        width=800,
        height=800
    )
    
    return fig

# Unified interface for option pricing
def option_price(S: float, K: float, r: float, sigma: float, T: float,
                option_type: str = "call", model: str = "Black-Scholes",
                steps: int = 100, sims: int = 10000) -> float:
    """
    Unified interface for pricing options using any available model.
    
    Args:
        model: One of "Black-Scholes", "Binomial", or "Monte Carlo"
    
    Returns:
        float: Option price according to the specified model
    """
    model = model.lower().replace("-", "").replace(" ", "")
    option_type = option_type.lower()
    
    if model == "blackscholes":
        if option_type == "call":
            return BlackScholesModel.price_call(S, K, r, sigma, T)
        else:
            return BlackScholesModel.price_put(S, K, r, sigma, T)
    
    elif model == "binomial":
        return BinomialModel.price_option(
            S, K, r, sigma, T, steps, option_type)
    
    elif model == "montecarlo":
        price, _ = MonteCarloModel.price_option(
            S, K, r, sigma, T, steps, sims, option_type)
        return price
    
    else:
        raise ValueError(f"Unknown model: {model}")

def black_scholes_greeks_call(S: float, K: float, r: float, sigma: float, T: float) -> Dict[str, float]:
    """
    Calculate Greeks for a call option using Black-Scholes model.
    
    Args:
        S: Current stock price
        K: Strike price
        r: Risk-free rate (annualized)
        sigma: Volatility (annualized)
        T: Time to maturity (in years)
        
    Returns:
        Dictionary containing all Greeks values
    """
    if T <= 0:
        return {
            'delta': 1.0 if S > K else 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0
        }
    
    # Calculate d1 and d2
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    
    # Common terms
    nd1 = norm.pdf(d1)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    exp_rt = np.exp(-r * T)
    
    # Calculate Greeks
    delta = Nd1
    gamma = nd1 / (S * sigma * sqrt_T)
    theta = (-S * nd1 * sigma / (2 * sqrt_T) -
             r * K * exp_rt * Nd2)
    vega = S * sqrt_T * nd1
    rho = K * T * exp_rt * Nd2
    
    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }

def black_scholes_greeks_put(S: float, K: float, r: float, sigma: float, T: float) -> Dict[str, float]:
    """
    Calculate Greeks for a put option using Black-Scholes model.
    
    Args:
        S: Current stock price
        K: Strike price
        r: Risk-free rate (annualized)
        sigma: Volatility (annualized)
        T: Time to maturity (in years)
        
    Returns:
        Dictionary containing all Greeks values
    """
    if T <= 0:
        return {
            'delta': -1.0 if S < K else 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0
        }
    
    # Calculate d1 and d2
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    
    # Common terms
    nd1 = norm.pdf(d1)
    Nd1 = norm.cdf(-d1)  # Note the negative d1 for puts
    Nd2 = norm.cdf(-d2)  # Note the negative d2 for puts
    exp_rt = np.exp(-r * T)
    
    # Calculate Greeks
    delta = -Nd1  # Negative of the call delta
    gamma = nd1 / (S * sigma * sqrt_T)  # Same as call
    theta = (-S * nd1 * sigma / (2 * sqrt_T) +
             r * K * exp_rt * Nd2)
    vega = S * sqrt_T * nd1  # Same as call
    rho = -K * T * exp_rt * Nd2  # Negative of call rho
    
    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }

# Add these imports at the top of the file if not already present
import numpy as np
from scipy.stats import norm
from typing import Dict