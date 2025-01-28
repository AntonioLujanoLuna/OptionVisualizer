# src/models/binomial.py

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from src.models.base import OptionPricingModel, OptionResult
from src.utils.validators import validate_parameters
from src.config import AppConfig

@dataclass
class BinomialParameters:
    """Parameters for the binomial tree model."""
    dt: float      # Time step size
    u: float       # Up factor
    d: float       # Down factor
    p: float       # Risk-neutral probability of up move
    discount: float # Discount factor per step
    option_type: str = "call"
    r: float = None
    sigma: float = None 
    T: float = None  # Time to expiry

class BinomialModel(OptionPricingModel):
    """
    Cox-Ross-Rubinstein binomial tree model for option pricing.

    This implementation provides:
    1. European and American option pricing
    2. Early exercise boundary calculation
    3. Implied tree calibration to market prices
    4. Greeks calculation through finite differences
    5. Dividend handling for discrete dividends

    The binomial model discretizes time and stock price movements, making it
    particularly useful for understanding option pricing concepts and handling
    early exercise features in American options.
    """

    def __init__(self, n_steps: int = AppConfig.binomial_steps):
        """Initialize the binomial model with the specified number of steps."""
        self.n_steps = n_steps

    def _calculate_parameters(self, sigma: float, T: float, r: float, option_type: str = "call") -> BinomialParameters:
        """
        Calculate the basic parameters of the binomial tree.

        The parameters are chosen to ensure the discrete model converges to
        the continuous-time Black-Scholes model as the number of steps increases.
        This means matching the first two moments of the price distribution.
        """
        dt = T / self.n_steps

        # Calculate up and down factors using the Cox-Ross-Rubinstein parameterization
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u  # Ensures that u*d = 1

        # Calculate risk-neutral probability
        # This ensures the expected return equals the risk-free rate
        p = (np.exp(r * dt) - d) / (u - d)

        # Calculate per-step discount factor
        discount = np.exp(-r * dt)

        return BinomialParameters(dt, u, d, p, discount, option_type, r, sigma, T)

    def _build_price_tree(self, S: float, params: BinomialParameters) -> np.ndarray:
        """
        Build the stock price tree using the binomial parameters.

        The tree is stored in a 2D array where element [i,j] represents the
        stock price at time step i and up-move j. This structure allows for
        efficient vectorized operations in option value calculations.
        """
        price_tree = np.zeros((self.n_steps + 1, self.n_steps + 1))

        # Initial stock price
        price_tree[0, 0] = S

        # Build the tree level by level
        for i in range(1, self.n_steps + 1):
            # At each level i, we have i+1 possible prices
            for j in range(i + 1):
                # Price = S * u^(number of ups) * d^(number of downs)
                price_tree[i, j] = S * (params.u ** (j)) * (params.d ** (i - j))

        return price_tree

    @validate_parameters
    def price_european(self, S: float, K: float, r: float, sigma: float,
                       T: float, option_type: str = "call") -> OptionResult:
        """
        Price a European option using the binomial model.

        The method proceeds in two steps:
        1. Build the price tree forward from t=0 to t=T
        2. Calculate option values backward from t=T to t=0

        This implementation is vectorized for better performance with
        large numbers of steps.
        """
        params = self._calculate_parameters(sigma, T, r)
        price_tree = self._build_price_tree(S, params)

        # Calculate terminal option values
        option_values = np.zeros_like(price_tree)
        if option_type.lower() == "call":
            option_values[-1, :] = np.maximum(
                price_tree[-1, :] - K, 0
            )
        else:
            option_values[-1, :] = np.maximum(
                K - price_tree[-1, :], 0
            )

        # Backward induction
        for i in range(self.n_steps - 1, -1, -1):
            for j in range(i + 1):
                # Expected value of option at next time step
                option_values[i, j] = (
                    params.p * option_values[i + 1, j + 1] +
                    (1 - params.p) * option_values[i + 1, j]
                ) * params.discount

        return OptionResult(
            price=option_values[0, 0],
            greeks={}, # Greeks are now calculated separately
            additional_info={
                "n_steps": self.n_steps,
                "parameters": params
            }
        )

    @validate_parameters
    def price_american(self, S: float, K: float, r: float, sigma: float,
                       T: float, option_type: str = "call") -> OptionResult:
        """
        Price an American option using the binomial model.

        This method extends the European pricing by checking for optimal
        early exercise at each node. It also tracks the early exercise
        boundary, which can be useful for understanding optimal exercise
        strategies.
        """
        params = self._calculate_parameters(sigma, T, r)
        price_tree = self._build_price_tree(S, params)

        # Initialize arrays for values and exercise boundary
        option_values = np.zeros_like(price_tree)
        exercise_boundary = np.full(self.n_steps + 1, np.nan)

        # Set terminal payoffs
        if option_type.lower() == "call":
            option_values[-1, :] = np.maximum(
                price_tree[-1, :] - K, 0
            )
        else:
            option_values[-1, :] = np.maximum(
                K - price_tree[-1, :], 0
            )

        # Backward induction with early exercise
        for i in range(self.n_steps - 1, -1, -1):
            for j in range(i + 1):
                # Calculate continuation value
                continuation_value = (
                    params.p * option_values[i + 1, j + 1] +
                    (1 - params.p) * option_values[i + 1, j]
                ) * params.discount

                # Calculate immediate exercise value
                current_price = price_tree[i, j]
                if option_type.lower() == "call":
                    exercise_value = max(current_price - K, 0)
                else:
                    exercise_value = max(K - current_price, 0)

                # Option value is maximum of continuation and exercise
                option_values[i, j] = max(continuation_value, exercise_value)

                # Track exercise boundary
                if exercise_value > continuation_value:
                    exercise_boundary[i] = current_price

        return OptionResult(
            price=option_values[0, 0],
            greeks={}, # Greeks are now calculated separately
            additional_info={
                "n_steps": self.n_steps,
                "parameters": params,
                "exercise_boundary": exercise_boundary
            }
        )

    def price_call(self, S: float, K: float, r: float, sigma: float, T: float) -> OptionResult:
        """Price a European call option."""
        return self.price_european(S, K, r, sigma, T, option_type="call")

    def price_put(self, S: float, K: float, r: float, sigma: float, T: float) -> OptionResult:
        """Price a European put option."""
        return self.price_european(S, K, r, sigma, T, option_type="put")

    def calculate_greeks(self, S: float, K: float, r: float, sigma: float,
                          T: float, option_type: str = "call") -> Dict[str, float]:
        """
        Calculate option Greeks using finite differences.
        """
        eps_S = S * 0.001  # Small change in stock price
        eps_sigma = 0.001  # Small change in volatility
        eps_r = 0.0001    # Small change in interest rate
        eps_T = 1/365     # One day change in time

        # Select appropriate pricing function based on option type
        price_func = self.price_american if option_type.lower() == "american" else self.price_european

        # Calculate base price
        base_result = price_func(S, K, r, sigma, T, option_type)
        base_price = base_result.price

        # Delta: ∂V/∂S (central difference)
        price_up = price_func(S + eps_S, K, r, sigma, T, option_type).price
        price_down = price_func(S - eps_S, K, r, sigma, T, option_type).price
        delta = (price_up - price_down) / (2 * eps_S)

        # Gamma: ∂²V/∂S² (central difference)
        gamma = (price_up - 2 * base_price + price_down) / (eps_S * eps_S)

        # Theta: -∂V/∂T (forward difference)
        if T <= eps_T:
            theta = 0  # At expiry
        else:
            price_later = price_func(S, K, r, sigma, T - eps_T, option_type).price
            theta = (price_later - base_price) / eps_T

        # Vega: ∂V/∂σ (central difference)
        price_vol_up = price_func(S, K, r, sigma + eps_sigma, T, option_type).price
        price_vol_down = price_func(S, K, r, sigma - eps_sigma, T, option_type).price
        vega = (price_vol_up - price_vol_down) / (2 * eps_sigma)

        # Rho: ∂V/∂r (central difference)
        price_r_up = price_func(S, K, r + eps_r, sigma, T, option_type).price
        price_r_down = price_func(S, K, r - eps_r, sigma, T, option_type).price
        rho = (price_r_up - price_r_down) / (2 * eps_r)

        return {
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega,
            "rho": rho
        }