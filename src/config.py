# src/config.py
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class AppConfig:
    """Application configuration parameters."""
    default_model: str = "Black-Scholes"
    monte_carlo_sims: int = 10000
    binomial_steps: int = 100
    plot_style: str = "dark"
    cache_timeout: int = 3600  # seconds
    
    # Numerical parameters
    epsilon: float = 1e-8
    max_iterations: int = 1000
    
    # UI configuration
    default_spot: float = 100.0
    default_strike: float = 100.0
    default_rate: float = 0.05
    default_volatility: float = 0.2
    default_maturity: float = 1.0

# src/utils/validation.py
from functools import wraps
from typing import Any, Callable
import numpy as np

class ValidationError(Exception):
    """Custom exception for parameter validation errors."""
    pass

def validate_parameters(func: Callable) -> Callable:
    """Decorator to validate option pricing parameters."""
    @wraps(func)
    def wrapper(self, S: float, K: float, r: float, sigma: float, T: float, *args, **kwargs) -> Any:
        if S <= 0:
            raise ValidationError("Stock price must be positive")
        if K <= 0:
            raise ValidationError("Strike price must be positive")
        if sigma <= 0:
            raise ValidationError("Volatility must be positive")
        if T < 0:
            raise ValidationError("Time to maturity cannot be negative")
        return func(self, S, K, r, sigma, T, *args, **kwargs)
    return wrapper

# src/utils/optimization.py
from typing import Callable, Tuple
import numpy as np
from ..config import AppConfig

def newton_raphson(f: Callable, fprime: Callable, x0: float, 
                  tol: float = AppConfig.epsilon, 
                  max_iter: int = AppConfig.max_iterations) -> Tuple[float, bool]:
    """
    Implementation of Newton-Raphson method for root finding.
    
    Args:
        f: Function for which we want to find the root
        fprime: Derivative of f
        x0: Initial guess
        tol: Tolerance for convergence
        max_iter: Maximum number of iterations
    
    Returns:
        Tuple containing the root and a boolean indicating convergence
    """
    x = x0
    for _ in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:
            return x, True
        
        fpx = fprime(x)
        if abs(fpx) < tol:
            return x, False
        
        x = x - fx/fpx
    
    return x, False

# src/utils/decorators.py
from functools import lru_cache, wraps
import time
from typing import Any, Callable

def timer(func: Callable) -> Callable:
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def memoize(func: Callable) -> Callable:
    """Enhanced memoization decorator with timeout."""
    cache = {}
    
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        key = str(args) + str(kwargs)
        current_time = time.time()
        
        if key in cache:
            result, timestamp = cache[key]
            if current_time - timestamp < AppConfig.cache_timeout:
                return result
        
        result = func(*args, **kwargs)
        cache[key] = (result, current_time)
        return result
    
    return wrapper