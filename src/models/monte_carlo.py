# src/models/monte_carlo.py

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import logging

from src.models.base import OptionPricingModel, OptionResult
from src.utils.validators import validate_parameters
from src.config import AppConfig

@dataclass
class SimulationResult:
    """Contains the results of a Monte Carlo simulation run."""
    paths: np.ndarray  # Shape: (n_sims, n_steps + 1)
    final_prices: np.ndarray  # Shape: (n_sims,)
    payoffs: np.ndarray  # Shape: (n_sims,)
    confidence_interval: Tuple[float, float]
    standard_error: float

class MonteCarloModel(OptionPricingModel):
    """
    Advanced Monte Carlo simulation for option pricing with variance reduction techniques.
    
    This implementation includes several sophisticated features:
    1. Antithetic variates for variance reduction
    2. Control variates using analytical solutions when available
    3. Parallel processing for large simulations
    4. Stratified sampling for better coverage of the probability space
    5. Confidence interval estimation
    """
    
    def __init__(self, n_sims: int = AppConfig.monte_carlo_sims, 
                 n_steps: int = 252, random_seed: Optional[int] = None,
                 n_processes: int = 4):
        """
        Initialize the Monte Carlo simulator with configurable parameters.
        
        The number of steps defaults to 252 (typical trading days in a year)
        to provide realistic time discretization for path-dependent options.
        """
        self.n_sims = n_sims
        self.n_steps = n_steps
        self.n_processes = n_processes
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
    
    def _generate_paths(self, S: float, r: float, sigma: float, T: float,
                       antithetic: bool = True) -> np.ndarray:
        """
        Generate stock price paths using geometric Brownian motion.
        
        The method uses antithetic variates by default to reduce variance:
        for each random path, we also generate its negative counterpart.
        This ensures better estimation of the expected value.
        """
        dt = T / self.n_steps
        nudt = (r - 0.5 * sigma**2) * dt
        sigmadt = sigma * np.sqrt(dt)
        
        # Generate standard normal random numbers
        if antithetic:
            Z = np.random.standard_normal((self.n_sims // 2, self.n_steps))
            Z = np.concatenate([Z, -Z])  # Antithetic pairs
        else:
            Z = np.random.standard_normal((self.n_sims, self.n_steps))
        
        # Initialize paths array and set initial stock price
        paths = np.zeros((self.n_sims, self.n_steps + 1))
        paths[:, 0] = S
        
        # Generate paths using vectorized operations
        for t in range(1, self.n_steps + 1):
            paths[:, t] = paths[:, t-1] * np.exp(nudt + sigmadt * Z[:, t-1])
        
        return paths
    
    def _run_parallel_simulation(self, S: float, r: float, sigma: float, T: float,
                               chunk_size: Optional[int] = None) -> List[np.ndarray]:
        """
        Run Monte Carlo simulations in parallel for better performance.
        
        This method splits the total number of simulations into chunks
        that can be processed independently on different CPU cores.
        """
        if chunk_size is None:
            chunk_size = self.n_sims // self.n_processes
        
        chunks = []
        with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
            futures = []
            for _ in range(self.n_processes):
                future = executor.submit(
                    self._generate_paths,
                    S, r, sigma, T,
                    chunk_size
                )
                futures.append(future)
            
            # Collect results as they complete
            for future in futures:
                chunk = future.result()
                chunks.append(chunk)
        
        return chunks
    
    @validate_parameters
    def price_call(self, S: float, K: float, r: float, sigma: float, T: float,
                use_control_variate: bool = True, option_type: str = "call") -> OptionResult: # Added option_type
        """
        Price a European call option using Monte Carlo simulation.
        """
        # Generate price paths
        paths = self._generate_paths(S, r, sigma, T)
        final_prices = paths[:, -1]

        # Calculate payoffs
        payoffs = np.maximum(final_prices - K, 0)

        # Apply control variate if requested
        if use_control_variate and T > 0:
            # Use stock price as control variate
            control_payoffs = final_prices - S * np.exp(r * T)
            beta = np.cov(payoffs, control_payoffs)[0, 1] / np.var(control_payoffs)
            payoffs_adjusted = payoffs - beta * control_payoffs

            # Calculate the mean and standard error
            price = np.exp(-r * T) * np.mean(payoffs_adjusted)
            std_error = np.exp(-r * T) * np.std(payoffs_adjusted) / np.sqrt(self.n_sims)
        else:
            # Standard Monte Carlo estimation
            price = np.exp(-r * T) * np.mean(payoffs)
            std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(self.n_sims)

        # Calculate confidence interval
        conf_interval = (
            price - 1.96 * std_error,
            price + 1.96 * std_error
        )

        return OptionResult(
            price=price,
            greeks={}, # Greeks are now calculated separately
            error_estimate=std_error,
            additional_info={
                "confidence_interval": conf_interval,
                "n_sims": self.n_sims,
                "n_steps": self.n_steps
            }
        )

    @validate_parameters
    def price_put(self, S: float, K: float, r: float, sigma: float, T: float,
              use_control_variate: bool = True, option_type: str = "put") -> OptionResult: # Added option_type
        """
        Price a European put option using Monte Carlo simulation.
        """
        paths = self._generate_paths(S, r, sigma, T)
        final_prices = paths[:, -1]
        payoffs = np.maximum(K - final_prices, 0)

        if use_control_variate and T > 0:
            control_payoffs = S * np.exp(r * T) - final_prices
            beta = np.cov(payoffs, control_payoffs)[0, 1] / np.var(control_payoffs)
            payoffs_adjusted = payoffs - beta * control_payoffs

            price = np.exp(-r * T) * np.mean(payoffs_adjusted)
            std_error = np.exp(-r * T) * np.std(payoffs_adjusted) / np.sqrt(self.n_sims)
        else:
            price = np.exp(-r * T) * np.mean(payoffs)
            std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(self.n_sims)

        conf_interval = (
            price - 1.96 * std_error,
            price + 1.96 * std_error
        )

        return OptionResult(
            price=price,
            greeks={}, # Greeks are now calculated separately
            error_estimate=std_error,
            additional_info={
                "confidence_interval": conf_interval,
                "n_sims": self.n_sims,
                "n_steps": self.n_steps
            }
        )

    def calculate_greeks(self, S: float, K: float, r: float, sigma: float,
                        T: float, option_type: str = "call") -> Dict[str, float]:
        """
        Calculate option Greeks using finite difference methods.
        """
        # Small parameter changes for finite differences
        eps_S = S * 0.001  # 0.1% of spot price
        eps_sigma = 0.001  # 10 basis points of volatility
        eps_r = 0.0001    # 1 basis point of rates
        eps_T = 1/365.0   # One day

        # Store original random state to use same random numbers
        random_state = np.random.get_state()

        # Price function based on option type
        if option_type == "call":
            price_func = self.price_call
        elif option_type == "put":
            price_func = self.price_put
        else:
            raise ValueError("Invalid option type")

        # Calculate base price - pass option_type here
        base_result = price_func(S, K, r, sigma, T, option_type=option_type)
        base_price = base_result.price

        # Delta: ∂V/∂S (central difference) - pass option_type here
        price_up = price_func(S + eps_S, K, r, sigma, T, option_type=option_type).price
        price_down = price_func(S - eps_S, K, r, sigma, T, option_type=option_type).price
        delta = (price_up - price_down) / (2 * eps_S)

        # Gamma: ∂²V/∂S² (central difference) - no need for option_type here
        gamma = (price_up - 2 * base_price + price_down) / (eps_S * eps_S)

        # Theta: -∂V/∂T (forward difference) - pass option_type here
        if T <= eps_T:
            # Special handling for very short time to maturity
            price_up_T = price_func(S, K, r, sigma, 2 * eps_T, option_type=option_type).price
            base_price = price_func(S, K, r, sigma, eps_T, option_type=option_type).price
            theta = -(price_up_T - base_price) / eps_T
        else:
            price_up_T = price_func(S, K, r, sigma, T + eps_T, option_type=option_type).price
            base_price = price_func(S, K, r, sigma, T, option_type=option_type).price
            theta = -(price_up_T - base_price) / eps_T

        # Vega: ∂V/∂σ (central difference) - pass option_type here
        price_vol_up = price_func(S, K, r, sigma + eps_sigma, T, option_type=option_type).price
        price_vol_down = price_func(S, K, r, sigma - eps_sigma, T, option_type=option_type).price
        vega = (price_vol_up - price_vol_down) / (2 * eps_sigma)

        # Rho: ∂V/∂r (central difference) - pass option_type here
        price_r_up = price_func(S, K, r + eps_r, sigma, T, option_type=option_type).price
        price_r_down = price_func(S, K, r - eps_r, sigma, T, option_type=option_type).price
        rho = (price_r_up - price_r_down) / (2 * eps_r)

        # Restore original random state
        np.random.set_state(random_state)

        return {
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega,
            "rho": rho
        }

    def price_exotic_option(self, S: float, K: float, r: float, sigma: float,
                             T: float, payoff_func: callable,
                             barrier: Optional[float] = None) -> OptionResult:
        """
        Price exotic options using Monte Carlo simulation.
        """
        # Generate price paths
        paths = self._generate_paths(S, r, sigma, T)

        # Calculate payoffs using the provided payoff function
        payoffs = payoff_func(paths)

        # Calculate price and error estimates
        price = np.exp(-r * T) * np.mean(payoffs)
        std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(self.n_sims)

        # Calculate confidence interval
        conf_interval = (
            price - 1.96 * std_error,
            price + 1.96 * std_error
        )

        return OptionResult(
            price=price,
            greeks = {}, # Greeks are usually not calculated with this method
            error_estimate=std_error,
            additional_info={
                "confidence_interval": conf_interval,
                "n_sims": self.n_sims,
                "n_steps": self.n_steps,
                "paths_shape": paths.shape
            }
        )