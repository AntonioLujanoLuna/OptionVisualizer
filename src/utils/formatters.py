"""
Formatting utilities for consistent data presentation.

This module provides formatting functions for:
1. Financial values (currency, percentages)
2. Option Greeks
3. Dates and times
4. Portfolio and position data
5. Statistical metrics
"""

import locale
from datetime import datetime, date
from typing import Union, Optional, Dict, Any
from decimal import Decimal
import re

# Set locale for currency formatting
locale.setlocale(locale.LC_ALL, '')

class FinancialFormatter:
    """Handles formatting of financial values."""
    
    @staticmethod
    def currency(value: Union[float, Decimal], include_cents: bool = True) -> str:
        """
        Format a value as currency.
        
        Args:
            value: The value to format
            include_cents: Whether to include cents
            
        Returns:
            Formatted currency string
        """
        try:
            if include_cents:
                return locale.currency(float(value), grouping=True)
            return locale.currency(float(value), grouping=True, digits=0)
        except (ValueError, TypeError):
            return "$0.00" if include_cents else "$0"
    
    @staticmethod
    def percentage(value: Union[float, Decimal], decimals: int = 2) -> str:
        """Format a value as a percentage."""
        try:
            return f"{float(value):.{decimals}f}%"
        except (ValueError, TypeError):
            return "0.00%"
    
    @staticmethod
    def number(value: Union[float, Decimal], decimals: int = 2,
              use_thousands: bool = True) -> str:
        """Format a number with optional thousands separator."""
        try:
            value_float = float(value)
            if use_thousands:
                return f"{value_float:,.{decimals}f}"
            return f"{value_float:.{decimals}f}"
        except (ValueError, TypeError):
            return "0.00"
    
    @staticmethod
    def delta_value(value: Union[float, Decimal], include_sign: bool = True) -> str:
        """Format a delta (change) value with sign indicator."""
        try:
            value_float = float(value)
            if include_sign and value_float > 0:
                return f"+{value_float:,.2f}"
            return f"{value_float:,.2f}"
        except (ValueError, TypeError):
            return "+0.00" if include_sign else "0.00"

class GreeksFormatter:
    """Handles formatting of option Greeks values."""
    
    @staticmethod
    def delta(value: float) -> str:
        """Format delta with appropriate precision."""
        return f"{value:.3f}"
    
    @staticmethod
    def gamma(value: float) -> str:
        """Format gamma with appropriate precision."""
        return f"{value:.4f}"
    
    @staticmethod
    def theta(value: float) -> str:
        """Format theta with appropriate precision."""
        return f"{value:.2f}"
    
    @staticmethod
    def vega(value: float) -> str:
        """Format vega with appropriate precision."""
        return f"{value:.2f}"
    
    @staticmethod
    def rho(value: float) -> str:
        """Format rho with appropriate precision."""
        return f"{value:.3f}"
    
    @staticmethod
    def format_all(greeks: Dict[str, float]) -> Dict[str, str]:
        """Format all Greeks in a dictionary."""
        formatters = {
            'delta': GreeksFormatter.delta,
            'gamma': GreeksFormatter.gamma,
            'theta': GreeksFormatter.theta,
            'vega': GreeksFormatter.vega,
            'rho': GreeksFormatter.rho
        }
        
        return {
            greek: formatters.get(greek.lower(), str)(value)
            for greek, value in greeks.items()
        }

class DateFormatter:
    """Handles formatting of dates and times."""
    
    @staticmethod
    def format_date(d: Union[datetime, date], format_str: str = "%Y-%m-%d") -> str:
        """Format a date with specified format string."""
        try:
            return d.strftime(format_str)
        except AttributeError:
            return ""
    
    @staticmethod
    def format_expiry(expiry: Union[datetime, date, float]) -> str:
        """
        Format option expiry date/time.
        
        Handles both datetime objects and year fractions. Provides human-readable
        relative time formats (e.g., "3d" for 3 days, "2w" for 2 weeks).
        
        Args:
            expiry: Either a datetime/date object or a year fraction (e.g., 0.5 for 6 months)
            
        Returns:
            Formatted expiry string in the most appropriate unit
        """
        if isinstance(expiry, (datetime, date)):
            return expiry.strftime("%Y-%m-%d")
        try:
            # Convert year fraction to days
            days = int(float(expiry) * 365)
            if days <= 0:
                return "Expired"
            if days < 7:
                return f"{days}d"
            if days < 30:
                return f"{days//7}w"
            if days < 365:
                return f"{days//30}m"
            return f"{days//365}y"
        except (ValueError, TypeError):
            return "Invalid"
    
    @staticmethod
    def timestamp_to_age(timestamp: Union[datetime, float]) -> str:
        """
        Convert a timestamp to a human-readable age string.
        
        This is useful for showing how long a position has been held or
        when a trade was executed.
        
        Args:
            timestamp: Either a datetime object or Unix timestamp
            
        Returns:
            Human-readable age string (e.g., "2h ago", "3d ago")
        """
        try:
            if isinstance(timestamp, float):
                dt = datetime.fromtimestamp(timestamp)
            else:
                dt = timestamp
            
            delta = datetime.now() - dt
            
            # Convert timedelta to the most appropriate unit
            seconds = delta.total_seconds()
            if seconds < 60:
                return "just now"
            if seconds < 3600:
                minutes = int(seconds / 60)
                return f"{minutes}m ago"
            if seconds < 86400:
                hours = int(seconds / 3600)
                return f"{hours}h ago"
            if seconds < 604800:
                days = int(seconds / 86400)
                return f"{days}d ago"
            if seconds < 2592000:
                weeks = int(seconds / 604800)
                return f"{weeks}w ago"
            if seconds < 31536000:
                months = int(seconds / 2592000)
                return f"{months}mo ago"
            years = int(seconds / 31536000)
            return f"{years}y ago"
        except (ValueError, TypeError, AttributeError):
            return "unknown"

class PortfolioFormatter:
    """
    Handles formatting of portfolio and position data with consistent styling.
    
    This class ensures that portfolio data is presented in a clear, consistent
    manner across the application, making it easier for users to understand
    their positions and risk exposure.
    """
    
    @staticmethod
    def format_position(position: Dict[str, Any]) -> Dict[str, str]:
        """
        Format a complete position entry with all relevant fields.
        
        This method provides consistent formatting for all position attributes,
        ensuring that values are presented with appropriate precision and units.
        
        Args:
            position: Dictionary containing position details
            
        Returns:
            Dictionary with formatted position values
        """
        fin = FinancialFormatter()
        greeks = GreeksFormatter()
        dates = DateFormatter()
        
        # Create formatted position dictionary with proper precision and units
        formatted = {
            "type": position.get("option_type", "").upper(),
            "strike": fin.currency(position.get("strike", 0)),
            "expiry": dates.format_expiry(position.get("expiry", 0)),
            "quantity": str(position.get("quantity", 0)),
            "underlying": fin.currency(position.get("underlying_price", 0)),
            "premium": fin.currency(position.get("premium", 0)),
            "market_value": fin.currency(position.get("market_value", 0)),
            "unrealized_pnl": fin.delta_value(position.get("unrealized_pnl", 0)),
            "implied_vol": fin.percentage(position.get("volatility", 0)),
            "days_held": dates.timestamp_to_age(position.get("entry_date", datetime.now()))
        }
        
        # Add formatted Greeks if present
        if "greeks" in position:
            formatted.update(greeks.format_all(position["greeks"]))
        
        return formatted
    
    @staticmethod
    def format_summary(summary: Dict[str, Any]) -> Dict[str, str]:
        """
        Format portfolio summary statistics.
        
        Provides consistent formatting for portfolio-level metrics,
        making it easy to display summary information in the UI.
        
        Args:
            summary: Dictionary containing portfolio summary data
            
        Returns:
            Dictionary with formatted summary values
        """
        fin = FinancialFormatter()
        
        return {
            "total_value": fin.currency(summary.get("total_value", 0)),
            "daily_pnl": fin.delta_value(summary.get("daily_pnl", 0)),
            "total_pnl": fin.delta_value(summary.get("total_pnl", 0)),
            "daily_return": fin.percentage(summary.get("daily_return", 0)),
            "total_return": fin.percentage(summary.get("total_return", 0)),
            "realized_pnl": fin.delta_value(summary.get("realized_pnl", 0)),
            "unrealized_pnl": fin.delta_value(summary.get("unrealized_pnl", 0)),
            "margin_used": fin.currency(summary.get("margin_used", 0)),
            "buying_power": fin.currency(summary.get("buying_power", 0))
        }

class StatisticalFormatter:
    """
    Handles formatting of statistical metrics and analysis results.
    
    This class provides consistent formatting for various statistical measures
    used in portfolio analysis and risk management.
    """
    
    @staticmethod
    def format_distribution_stats(stats: Dict[str, float]) -> Dict[str, str]:
        """
        Format statistical distribution metrics.
        
        Provides appropriate formatting for measures like mean, standard
        deviation, skewness, and kurtosis.
        
        Args:
            stats: Dictionary of statistical measures
            
        Returns:
            Dictionary with formatted statistical values
        """
        fin = FinancialFormatter()
        
        return {
            "mean": fin.number(stats.get("mean", 0), decimals=3),
            "median": fin.number(stats.get("median", 0), decimals=3),
            "std_dev": fin.number(stats.get("std_dev", 0), decimals=3),
            "skewness": fin.number(stats.get("skewness", 0), decimals=3),
            "kurtosis": fin.number(stats.get("kurtosis", 0), decimals=3),
            "sharpe": fin.number(stats.get("sharpe", 0), decimals=2),
            "sortino": fin.number(stats.get("sortino", 0), decimals=2),
            "max_drawdown": fin.percentage(stats.get("max_drawdown", 0)),
            "var_95": fin.currency(stats.get("var_95", 0)),
            "cvar_95": fin.currency(stats.get("cvar_95", 0))
        }
    
    @staticmethod
    def format_correlation(value: float) -> str:
        """
        Format correlation coefficients with appropriate precision.
        
        Args:
            value: Correlation coefficient (-1 to 1)
            
        Returns:
            Formatted correlation string with color indicator
        """
        try:
            value_float = float(value)
            # Clamp value to valid range
            value_float = max(-1.0, min(1.0, value_float))
            return f"{value_float:+.2f}"
        except (ValueError, TypeError):
            return "0.00"

def format_table_value(value: Any, format_type: str = "default") -> str:
    """
    Generic formatter for table cell values.
    
    This utility function provides consistent formatting for values
    displayed in tables throughout the application.
    
    Args:
        value: The value to format
        format_type: Type of formatting to apply
        
    Returns:
        Formatted string ready for display
    """
    fin = FinancialFormatter()
    
    # Handle None/null values
    if value is None:
        return "-"
    
    # Apply formatting based on type
    format_map = {
        "currency": lambda x: fin.currency(float(x)),
        "percentage": lambda x: fin.percentage(float(x)),
        "number": lambda x: fin.number(float(x)),
        "delta": lambda x: fin.delta_value(float(x)),
        "date": lambda x: DateFormatter.format_date(x),
        "expiry": lambda x: DateFormatter.format_expiry(x),
        "age": lambda x: DateFormatter.timestamp_to_age(x)
    }
    
    try:
        if format_type in format_map:
            return format_map[format_type](value)
        return str(value)
    except (ValueError, TypeError):
        return "-"