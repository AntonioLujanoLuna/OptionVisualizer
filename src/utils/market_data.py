"""
Market data handling and processing utilities.

This module provides robust functionality for:
1. Fetching market data from various sources
2. Rate limiting and error handling
3. Data normalization and validation
4. Caching and persistence
5. Real-time data handling
"""

import asyncio
import aiohttp
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import logging
from pathlib import Path
import json
from dataclasses import dataclass
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from ..utils.cache import Cache, cached
from ..config import AppConfig

logger = logging.getLogger(__name__)

@dataclass
class MarketQuote:
    """Container for market data quotes."""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    last: float
    volume: int
    source: str
    
    @property
    def mid(self) -> float:
        """Calculate mid-point price."""
        return (self.bid + self.ask) / 2 if self.bid and self.ask else self.last
    
    @property
    def spread(self) -> float:
        """Calculate bid-ask spread."""
        return self.ask - self.bid if self.bid and self.ask else 0.0

@dataclass
class OptionQuote(MarketQuote):
    """Extended quote data for options."""
    strike: float
    expiry: datetime
    option_type: str
    underlying_price: float
    implied_volatility: float
    open_interest: int
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None

class RateLimiter:
    """
    Thread-safe rate limiter for API calls.
    
    This ensures we don't exceed API rate limits by implementing
    token bucket algorithm with configurable rates.
    """
    
    def __init__(self, max_requests: int, time_window: float):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests allowed in time window
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.tokens = max_requests
        self.last_update = time.monotonic()
        self._lock = threading.Lock()
    
    def acquire(self, tokens: int = 1) -> bool:
        """
        Attempt to acquire rate limit tokens.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            True if tokens were acquired, False if would exceed rate limit
        """
        with self._lock:
            now = time.monotonic()
            time_passed = now - self.last_update
            
            # Replenish tokens based on time passed
            self.tokens = min(
                self.max_requests,
                self.tokens + time_passed * (self.max_requests / self.time_window)
            )
            self.last_update = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    async def acquire_async(self, tokens: int = 1) -> None:
        """
        Asynchronously acquire rate limit tokens.
        
        This will wait until tokens become available.
        """
        while not self.acquire(tokens):
            await asyncio.sleep(0.1)

class MarketDataClient:
    """
    Client for fetching market data from multiple sources.
    
    Features:
    - Multiple data source support
    - Automatic failover
    - Rate limiting
    - Caching
    - Async batch operations
    """
    
    def __init__(self, config: AppConfig):
        """
        Initialize market data client.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.cache = Cache(
            max_size=1000,
            default_ttl=timedelta(minutes=15),
            disk_path=Path("cache/market_data")
        )
        
        # Initialize rate limiters for different APIs
        self.rate_limiters = {
            "yahoo": RateLimiter(max_requests=2000, time_window=3600),  # 2000/hour
            "alpha_vantage": RateLimiter(max_requests=5, time_window=60),  # 5/minute
            "iex": RateLimiter(max_requests=100, time_window=60)  # 100/minute
        }
        
        # Initialize async session
        self._session = None
        self._session_lock = threading.