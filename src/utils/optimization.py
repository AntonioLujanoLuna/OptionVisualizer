# src/utils/optimization.py
from typing import Callable, Tuple
import numpy as np
from src.config import AppConfig

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