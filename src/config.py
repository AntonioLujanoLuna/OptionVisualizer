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