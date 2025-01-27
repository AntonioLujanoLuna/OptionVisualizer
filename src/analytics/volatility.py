# src/analytics/volatility.py

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from scipy.interpolate import interp2d
from scipy.optimize import minimize
import pandas as pd

from src.config import AppConfig # Relative import (still works within the same package)

@dataclass
class VolatilitySurface:
    """Represents a volatility surface for option pricing."""
    strikes: np.ndarray
    expiries: np.ndarray
    volatilities: np.ndarray
    forward_prices: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        """
        Initialize interpolation function and validate surface.

        This function checks for sufficient data points and creates a 2D
        interpolation function that can be used to estimate volatility
        at any given strike and expiry within the surface's range.
        """
        if self.strikes.shape[0] < 2 or self.expiries.shape[0] < 2:
            raise ValueError("Insufficient data points for surface construction.")
        self.interpolation_func = interp2d(
            self.strikes,
            self.expiries,
            self.volatilities.T,
            kind='cubic'
        )

    def get_volatility(self, strike: float, expiry: float) -> float:
        """
        Get the interpolated volatility for a specific strike and expiry.

        Args:
            strike: Strike price of the option
            expiry: Time to expiry of the option in years

        Returns:
            Interpolated volatility at the given strike and expiry
        """
        if not (self.strikes.min() <= strike <= self.strikes.max()):
            raise ValueError(f"Strike {strike} is outside the valid range.")
        if not (self.expiries.min() <= expiry <= self.expiries.max()):
            raise ValueError(f"Expiry {expiry} is outside the valid range.")
        return self.interpolation_func(strike, expiry)[0]

    def calibrate_to_market(self, market_prices: pd.DataFrame,
                            model: 'OptionPricingModel',
                            method: str = 'Nelder-Mead') -> None:
        """
        Calibrate the volatility surface to observed market prices.

        This function adjusts the volatility surface to minimize the difference
        between model-predicted prices and market prices for a set of options.
        It's particularly useful for ensuring the surface reflects current
        market conditions.

        Args:
            market_prices: DataFrame containing market prices of options with columns
                           'strike', 'expiry', 'price'
            model: Option pricing model instance used for calibration
            method: Optimization method to use (from scipy.optimize.minimize)
        """
        def error_function(params):
            """Error function to minimize."""
            error = 0
            k = 0
            for _, row in market_prices.iterrows():
                strike_idx = np.abs(self.strikes - row['strike']).argmin()
                expiry_idx = np.abs(self.expiries - row['expiry']).argmin()

                # Update volatility in the surface
                self.volatilities[strike_idx, expiry_idx] = params[k]

                # Update interpolation function
                self.interpolation_func = interp2d(
                    self.strikes,
                    self.expiries,
                    self.volatilities.T,
                    kind='cubic'
                )

                # Calculate model price with updated volatility
                model_price = model.price_option(
                    S=model.S,
                    K=row['strike'],
                    r=model.r,
                    sigma=self.get_volatility(row['strike'], row['expiry']),
                    T=row['expiry'],
                    option_type=row['option_type']
                ).price

                error += (model_price - row['price'])**2
                k += 1
            return np.sqrt(error / len(market_prices))  # RMSE

        # Initial parameters (current volatilities)
        initial_params = self.volatilities.flatten()

        # Perform optimization
        result = minimize(
            error_function,
            initial_params,
            method=method,
            bounds=[(0.01, 1.0) for _ in range(len(initial_params))]  # Volatility bounds
        )

        if result.success:
            # Update volatilities with optimized values
            self.volatilities = result.x.reshape(self.volatilities.shape)

            # Rebuild interpolation function
            self.interpolation_func = interp2d(
                self.strikes,
                self.expiries,
                self.volatilities.T,
                kind='cubic'
            )

            print("Calibration successful.")
        else:
            print(f"Calibration failed: {result.message}")

    def visualize(self) -> None:
        """Visualize the volatility surface."""
        # Placeholder for visualization code
        pass

class VolatilityAnalyzer:
    """Analyzes and interprets volatility surfaces."""

    def __init__(self, pricing_model: 'OptionPricingModel'):
        """
        Initialize the Volatility Analyzer.

        Args:
            pricing_model: An instance of an option pricing model
        """
        self.model = pricing_model

    def calculate_implied_volatility(self, option_price: float, strike: float,
                                    expiry: float, option_type: str,
                                    method: str = 'brentq') -> float:
        """
        Calculate the implied volatility of an option using numerical methods.

        Args:
            option_price: Market price of the option
            strike: Strike price of the option
            expiry: Time to expiry of the option in years
            option_type: Type of the option ('call' or 'put')