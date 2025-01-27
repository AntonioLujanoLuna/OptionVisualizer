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

