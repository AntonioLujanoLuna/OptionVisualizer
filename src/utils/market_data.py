"""
Market data handling and processing utilities.

This module provides robust functionality for:
1. Fetching market data from various sources
2. Rate limiting and error handling
3. Data normalization and validation
4. Caching and persistence
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
from src.utils.cache import Cache, cached
from src.config import AppConfig

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
        self._session_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=4)
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None:
            async with self._session_lock:
                if self._session is None:
                    self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self):
        """Close client and cleanup resources."""
        if self._session:
            await self._session.close()
        self._executor.shutdown()
    
    async def get_quote(self, symbol: str, source: str = "yahoo") -> MarketQuote:
        """
        Fetch real-time quote for a symbol.
        
        Args:
            symbol: The ticker symbol
            source: Data source to use
            
        Returns:
            MarketQuote object containing latest price data
            
        Raises:
            MarketDataError: If quote cannot be fetched
        """
        cache_key = f"quote:{source}:{symbol}"
        cached_quote = self.cache.get(cache_key)
        if cached_quote:
            return cached_quote
        
        try:
            if source == "yahoo":
                await self.rate_limiters[source].acquire_async()
                quote = await self._fetch_yahoo_quote(symbol)
            elif source == "alpha_vantage":
                await self.rate_limiters[source].acquire_async()
                quote = await self._fetch_alpha_vantage_quote(symbol)
            else:
                raise ValueError(f"Unsupported data source: {source}")
            
            self.cache.set(cache_key, quote, ttl=timedelta(seconds=30))
            return quote
        
        except Exception as e:
            logger.error(f"Failed to fetch quote for {symbol} from {source}: {str(e)}")
            raise MarketDataError(f"Quote fetch failed: {str(e)}")
    
    async def get_option_chain(self, symbol: str) -> Dict[str, List[OptionQuote]]:
        """
        Fetch complete option chain for a symbol.
        
        Returns a dictionary mapping expiration dates to lists of option quotes.
        Includes both calls and puts.
        
        Args:
            symbol: The underlying symbol
            
        Returns:
            Dictionary mapping expiry dates to lists of option quotes
        """
        cache_key = f"options:{symbol}"
        cached_chain = self.cache.get(cache_key)
        if cached_chain:
            return cached_chain
        
        try:
            # Fetch using yfinance in thread pool to avoid blocking
            def fetch_chain():
                ticker = yf.Ticker(symbol)
                expirations = ticker.options
                
                chain = {}
                for expiry in expirations:
                    opts = ticker.option_chain(expiry)
                    chain[expiry] = []
                    
                    # Process calls
                    for _, row in opts.calls.iterrows():
                        chain[expiry].append(OptionQuote(
                            symbol=symbol,
                            timestamp=datetime.now(),
                            bid=row.bid,
                            ask=row.ask,
                            last=row.lastPrice,
                            volume=row.volume,
                            source="yahoo",
                            strike=row.strike,
                            expiry=datetime.strptime(expiry, "%Y-%m-%d"),
                            option_type="call",
                            underlying_price=ticker.info['regularMarketPrice'],
                            implied_volatility=row.impliedVolatility,
                            open_interest=row.openInterest,
                            delta=row.get('delta'),
                            gamma=row.get('gamma'),
                            theta=row.get('theta'),
                            vega=row.get('vega'),
                            rho=row.get('rho')
                        ))
                    
                    # Process puts
                    for _, row in opts.puts.iterrows():
                        chain[expiry].append(OptionQuote(
                            symbol=symbol,
                            timestamp=datetime.now(),
                            bid=row.bid,
                            ask=row.ask,
                            last=row.lastPrice,
                            volume=row.volume,
                            source="yahoo",
                            strike=row.strike,
                            expiry=datetime.strptime(expiry, "%Y-%m-%d"),
                            option_type="put",
                            underlying_price=ticker.info['regularMarketPrice'],
                            implied_volatility=row.impliedVolatility,
                            open_interest=row.openInterest,
                            delta=row.get('delta'),
                            gamma=row.get('gamma'),
                            theta=row.get('theta'),
                            vega=row.get('vega'),
                            rho=row.get('rho')
                        ))
                
                return chain
            
            # Execute in thread pool
            loop = asyncio.get_event_loop()
            chain = await loop.run_in_executor(self._executor, fetch_chain)
            
            # Cache for 5 minutes
            self.cache.set(cache_key, chain, ttl=timedelta(minutes=5))
            return chain
            
        except Exception as e:
            logger.error(f"Failed to fetch option chain for {symbol}: {str(e)}")
            raise MarketDataError(f"Option chain fetch failed: {str(e)}")
    
    async def get_historical_data(self, symbol: str, start_date: datetime,
                                end_date: datetime, interval: str = "1d") -> pd.DataFrame:
        """
        Fetch historical price data for a symbol.
        
        Args:
            symbol: The ticker symbol
            start_date: Start date for historical data
            end_date: End date for historical data
            interval: Data interval (1d, 1h, etc.)
            
        Returns:
            DataFrame containing historical price data
        """
        cache_key = f"history:{symbol}:{start_date}:{end_date}:{interval}"
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return cached_data
        
        try:
            def fetch_history():
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=interval
                )
                return df
            
            # Execute in thread pool
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(self._executor, fetch_history)
            
            # Cache for 1 hour if historical data
            if interval != "1m":
                self.cache.set(cache_key, data, ttl=timedelta(hours=1))
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch historical data for {symbol}: {str(e)}")
            raise MarketDataError(f"Historical data fetch failed: {str(e)}")
    
    async def _fetch_yahoo_quote(self, symbol: str) -> MarketQuote:
        """Internal method to fetch quote from Yahoo Finance."""
        async with self._get_session() as session:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            async with session.get(url) as response:
                data = await response.json()
                
                meta = data['chart']['result'][0]['meta']
                return MarketQuote(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(meta['regularMarketTime']),
                    bid=meta.get('bid', 0.0),
                    ask=meta.get('ask', 0.0),
                    last=meta['regularMarketPrice'],
                    volume=meta['regularMarketVolume'],
                    source="yahoo"
                )
    
    async def _fetch_alpha_vantage_quote(self, symbol: str) -> MarketQuote:
        """Internal method to fetch quote from Alpha Vantage."""
        api_key = self.config.get("alpha_vantage_api_key")
        if not api_key:
            raise ValueError("Alpha Vantage API key not configured")
        
        async with self._get_session() as session:
            url = f"https://www.alphavantage.co/query"
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": api_key
            }
            
            async with session.get(url, params=params) as response:
                data = await response.json()
                quote = data['Global Quote']
                
                return MarketQuote(
                    symbol=symbol,
                    timestamp=datetime.now(),  # AV doesn't provide timestamp
                    bid=float(quote.get('09. price', 0.0)),  # No bid/ask provided
                    ask=float(quote.get('09. price', 0.0)),
                    last=float(quote['05. price']),
                    volume=int(quote['06. volume']),
                    source="alpha_vantage"
                )

class MarketDataError(Exception):
    """Custom exception for market data related errors."""
    pass

class MarketDataStream:
    """
    Real-time market data streaming client.
    
    This class handles websocket connections for streaming real-time
    market data updates.
    """
    
    def __init__(self, symbols: List[str], callback: callable):
        """
        Initialize streaming client.
        
        Args:
            symbols: List of symbols to stream
            callback: Callback function for data updates
        """
        self.symbols = symbols
        self.callback = callback
        self._ws = None
        self._running = False
        self._reconnect_delay = 1.0  # Initial reconnect delay
        self._max_reconnect_delay = 60.0
    
    async def start(self):
        """Start streaming market data."""
        self._running = True
        while self._running:
            try:
                await self._connect()
                await self._stream()
            except Exception as e:
                logger.error(f"Streaming error: {str(e)}")
                if self._running:
                    await asyncio.sleep(self._reconnect_delay)
                    self._reconnect_delay = min(
                        self._reconnect_delay * 2,
                        self._max_reconnect_delay
                    )
    
    async def stop(self):
        """Stop streaming market data."""
        self._running = False
        if self._ws:
            await self._ws.close()
    
    async def _connect(self):
        """Establish websocket connection."""
        session = aiohttp.ClientSession()
        self._ws = await session.ws_connect(
            'wss://streamer.finance.yahoo.com',
            heartbeat=30
        )
        
        # Subscribe to symbols
        subscribe_msg = {
            "subscribe": self.symbols
        }
        await self._ws.send_json(subscribe_msg)
        self._reconnect_delay = 1.0  # Reset reconnect delay on successful connection
    
    async def _stream(self):
        """Handle streaming data."""
        async for msg in self._ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)
                await self.callback(data)
            elif msg.type == aiohttp.WSMsgType.CLOSED:
                break
            elif msg.type == aiohttp.WSMsgType.ERROR:
                break

def format_market_data(quotes: Union[MarketQuote, List[MarketQuote]]) -> pd.DataFrame:
    """
    Format market data quotes into a pandas DataFrame.
    
    Args:
        quotes: Single quote or list of quotes
        
    Returns:
        DataFrame containing formatted market data
    """
    if isinstance(quotes, MarketQuote):
        quotes = [quotes]
    
    data = []
    for quote in quotes:
        row = {
            'symbol': quote.symbol,
            'timestamp': quote.timestamp,
            'bid': quote.bid,
            'ask': quote.ask,
            'last': quote.last,
            'mid': quote.mid,
            'spread': quote.spread,
            'volume': quote.volume,
            'source': quote.source
        }
        
        # Add option-specific fields if available
        if isinstance(quote, OptionQuote):
            row.update({
                'strike': quote.strike,
                'expiry': quote.expiry,
                'option_type': quote.option_type,
                'underlying_price': quote.underlying_price,
                'implied_volatility': quote.implied_volatility,
                'open_interest': quote.open_interest,
                'delta': quote.delta,
                'gamma': quote.gamma,
                'theta': quote.theta,
                'vega': quote.vega,
                'rho': quote.rho
            })
        
        data.append(row)
    
    return pd.DataFrame(data)