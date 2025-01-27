"""
Input validation and business rule checking utilities.

This module provides:
1. Type validation and conversion
2. Business rule validation
3. Data format validation
4. Portfolio consistency checks
5. Custom validation decorators
"""

import re
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Union, Callable
from decimal import Decimal
from functools import wraps
import pandas as pd
import numpy as np

class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass

def validate_type(value: Any, expected_type: Union[type, tuple], 
                 field_name: str) -> None:
    """
    Validate that a value is of the expected type.
    
    Args:
        value: Value to validate
        expected_type: Expected type or tuple of types
        field_name: Name of field for error messages
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, expected_type):
        raise ValidationError(
            f"{field_name} must be of type {expected_type.__name__}, "
            f"got {type(value).__name__}"
        )

def validate_numeric(value: Any, field_name: str,
                    min_value: Optional[float] = None,
                    max_value: Optional[float] = None,
                    allow_zero: bool = True,
                    allow_negative: bool = False) -> float:
    """
    Validate and convert numeric input.
    
    Args:
        value: Value to validate
        field_name: Field name for error messages
        min_value: Optional minimum value
        max_value: Optional maximum value
        allow_zero: Whether to allow zero
        allow_negative: Whether to allow negative values
        
    Returns:
        Validated float value
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        if isinstance(value, str):
            # Remove any currency symbols or commas
            value = value.replace(', ').replace(',', '').strip()
        
        # Convert to float
        num_value = float(value)
        
        # Validate constraints
        if not allow_zero and num_value == 0:
            raise ValidationError(f"{field_name} cannot be zero")
        
        if not allow_negative and num_value < 0:
            raise ValidationError(f"{field_name} cannot be negative")
        
        if min_value is not None and num_value < min_value:
            raise ValidationError(
                f"{field_name} must be greater than or equal to {min_value}"
            )
        
        if max_value is not None and num_value > max_value:
            raise ValidationError(
                f"{field_name} must be less than or equal to {max_value}"
            )
        
        return num_value
    
    except (TypeError, ValueError) as e:
        raise ValidationError(f"{field_name} must be a valid number")

def validate_date(value: Any, field_name: str,
                 min_date: Optional[date] = None,
                 max_date: Optional[date] = None) -> date:
    """
    Validate and convert date input.
    
    Args:
        value: Date value to validate (string, datetime, or date)
        field_name: Field name for error messages
        min_date: Optional minimum allowed date
        max_date: Optional maximum allowed date
        
    Returns:
        Validated date object
    """
    try:
        if isinstance(value, datetime):
            dt = value.date()
        elif isinstance(value, date):
            dt = value
        elif isinstance(value, str):
            # Try common date formats
            for fmt in ('%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y%m%d'):
                try:
                    dt = datetime.strptime(value, fmt).date()
                    break
                except ValueError:
                    continue
            else:
                raise ValidationError(f"{field_name} is not in a recognized date format")
        else:
            raise ValidationError(f"{field_name} must be a valid date")
        
        # Validate range
        if min_date and dt < min_date:
            raise ValidationError(
                f"{field_name} must be on or after {min_date.strftime('%Y-%m-%d')}"
            )
        
        if max_date and dt > max_date:
            raise ValidationError(
                f"{field_name} must be on or before {max_date.strftime('%Y-%m-%d')}"
            )
        
        return dt
    
    except (TypeError, ValueError) as e:
        raise ValidationError(f"{field_name} must be a valid date")

def validate_option_parameters(func: Callable) -> Callable:
    """
    Decorator to validate option pricing parameters.
    
    Validates:
    - Strike price > 0
    - Time to expiry >= 0
    - Volatility > 0
    - Interest rate is reasonable
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Extract parameters from args/kwargs
        params = {}
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        params.update(bound_args.arguments)
        
        # Validate strike price
        validate_numeric(
            params.get('strike', 0),
            'Strike price',
            min_value=0.01,
            allow_zero=False
        )
        
        # Validate time to expiry
        validate_numeric(
            params.get('time_to_expiry', 0),
            'Time to expiry',
            min_value=0,
            allow_negative=False
        )
        
        # Validate volatility
        validate_numeric(
            params.get('volatility', 0),
            'Volatility',
            min_value=0.0001,
            max_value=5.0,
            allow_zero=False
        )
        
        # Validate interest rate
        validate_numeric(
            params.get('interest_rate', 0),
            'Interest rate',
            min_value=-0.1,
            max_value=0.5
        )
        
        return func(*args, **kwargs)
    
    return wrapper

def validate_portfolio_position(position: Dict[str, Any]) -> None:
    """
    Validate a portfolio position entry.
    
    Checks:
    - Required fields are present
    - Field types are correct
    - Values are within valid ranges
    - Option parameters are consistent
    """
    required_fields = {
        'symbol': str,
        'quantity': int,
        'position_type': str,
        'strike': float,
        'expiry': (date, datetime),
        'option_type': str
    }
    
    # Check required fields
    for field, field_type in required_fields.items():
        if field not in position:
            raise ValidationError(f"Missing required field: {field}")
        validate_type(position[field], field_type, field)
    
    # Validate quantity
    validate_numeric(
        position['quantity'],
        'Quantity',
        allow_zero=False
    )
    
    # Validate strike price
    validate_numeric(
        position['strike'],
        'Strike price',
        min_value=0.01,
        allow_zero=False
    )
    
    # Validate option type
    if position['option_type'].lower() not in ('call', 'put'):
        raise ValidationError("Option type must be either 'call' or 'put'")
    
    # Validate expiry is in the future
    expiry = position['expiry']
    if isinstance(expiry, datetime):
        expiry = expiry.date()
    if expiry < date.today():
        raise ValidationError("Option expiry cannot be in the past")

def validate_portfolio_margin(positions: List[Dict[str, Any]],
                           account_value: float) -> None:
    """
    Validate portfolio margin requirements.
    
    Checks:
    - Portfolio total exposure within limits
    - Individual position sizes appropriate
    - Margin requirements are met
    - Risk metrics within acceptable ranges
    """
    # Calculate total exposure
    total_exposure = sum(
        abs(pos['quantity'] * pos['strike'])
        for pos in positions
    )
    
    # Check against account value
    max_exposure = account_value * 5  # Example: 5x leverage limit
    if total_exposure > max_exposure:
        raise ValidationError(
            f"Total exposure ({total_exposure}) exceeds maximum allowed "
            f"({max_exposure}) for account size"
        )
    
    # Check individual position sizes
    for position in positions:
        position_exposure = abs(position['quantity'] * position['strike'])
        if position_exposure > account_value:
            raise ValidationError(
                f"Individual position exposure ({position_exposure}) "
                f"exceeds account value"
            )
    
    # Additional margin checks could be added here
    # This is a simplified example

def validate_data_format(df: pd.DataFrame, 
                        expected_columns: List[str],
                        column_types: Optional[Dict[str, type]] = None) -> None:
    """
    Validate DataFrame format and content.
    
    Args:
        df: DataFrame to validate
        expected_columns: List of required columns
        column_types: Optional mapping of column names to expected types
    """
    # Check for required columns
    missing_cols = set(expected_columns) - set(df.columns)
    if missing_cols:
        raise ValidationError(f"Missing required columns: {missing_cols}")
    
    # Validate column types if specified
    if column_types:
        for col, expected_type in column_types.items():
            if col in df.columns:
                # Check if column can be converted to expected type
                try:
                    df[col].astype(expected_type)
                except (ValueError, TypeError):
                    raise ValidationError(
                        f"Column '{col}' cannot be converted to {expected_type.__name__}"
                    )
    
    # Check for empty DataFrame
    if df.empty:
        raise ValidationError("DataFrame cannot be empty")
    
    # Check for missing values in required columns
    missing_values = df[expected_columns].isna().any()
    if missing_values.any():
        cols_with_missing = missing_values[missing_values].index.tolist()
        raise ValidationError(
            f"Missing values found in columns: {cols_with_missing}"
        )

def validate_csv_file(file_path: str,
                     expected_columns: List[str],
                     column_types: Optional[Dict[str, type]] = None) -> pd.DataFrame:
    """
    Validate a CSV file before processing.
    
    Args:
        file_path: Path to CSV file
        expected_columns: List of required columns
        column_types: Optional mapping of column names to expected types
        
    Returns:
        Validated DataFrame
    """
    try:
        # Try to read the CSV file
        df = pd.read_csv(file_path)
        
        # Validate format
        validate_data_format(df, expected_columns, column_types)
        
        # Additional CSV-specific checks
        if df.shape[0] > 1000000:  # Example limit
            raise ValidationError("CSV file too large (> 1M rows)")
        
        return df
        
    except pd.errors.EmptyDataError:
        raise ValidationError("CSV file is empty")
    except pd.errors.ParserError:
        raise ValidationError("Invalid CSV format")
    except FileNotFoundError:
        raise ValidationError("CSV file not found")
    except Exception as e:
        raise ValidationError(f"Error reading CSV file: {str(e)}")

class DataFrameValidator:
    """
    Validator class for DataFrame operations with custom rules.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.errors = []
    
    def validate_unique(self, columns: Union[str, List[str]]) -> 'DataFrameValidator':
        """Check if specified columns contain unique values."""
        if isinstance(columns, str):
            columns = [columns]
        
        for col in columns:
            if not self.df[col].is_unique:
                self.errors.append(f"Column '{col}' contains duplicate values")
        
        return self
    
    def validate_range(self, column: str,
                      min_value: Optional[float] = None,
                      max_value: Optional[float] = None) -> 'DataFrameValidator':
        """Validate numeric values are within specified range."""
        values = self.df[column]
        
        if min_value is not None and values.min() < min_value:
            self.errors.append(
                f"Column '{column}' contains values below minimum {min_value}"
            )
        
        if max_value is not None and values.max() > max_value:
            self.errors.append(
                f"Column '{column}' contains values above maximum {max_value}"
            )
        
        return self
    
    def validate_regex(self, column: str, pattern: str) -> 'DataFrameValidator':
        """Validate string values match regex pattern."""
        invalid_values = self.df[~self.df[column].str.match(pattern)]
        if not invalid_values.empty:
            self.errors.append(
                f"Column '{column}' contains values not matching pattern '{pattern}'"
            )
        
        return self
    
    def raise_if_invalid(self) -> None:
        """Raise ValidationError if any validations failed."""
        if self.errors:
            raise ValidationError("\n".join(self.errors))

def validate_parameters(func: Callable) -> Callable:
    """
    Generic parameter validation decorator.
    
    This decorator checks that function parameters meet specified criteria
    using type hints and optional validation rules.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        # Get type hints
        hints = get_type_hints(func)
        
        # Validate each parameter
        for name, value in bound_args.arguments.items():
            if name in hints:
                expected_type = hints[name]
                
                # Handle Optional types
                if (
                    hasattr(expected_type, '__origin__') and
                    expected_type.__origin__ is Union and
                    type(None) in expected_type.__args__
                ):
                    if value is not None:
                        expected_type = expected_type.__args__[0]
                    else:
                        continue
                
                # Validate type
                validate_type(value, expected_type, name)
                
                # Additional validation based on parameter name
                if 'price' in name.lower():
                    validate_numeric(value, name, min_value=0)
                elif 'quantity' in name.lower():
                    validate_numeric(value, name, allow_zero=False)
                elif 'date' in name.lower():
                    validate_date(value, name)
        
        return func(*args, **kwargs)
    
    return wrapper