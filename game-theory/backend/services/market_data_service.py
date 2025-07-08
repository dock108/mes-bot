import aiohttp
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import ccxt
from config import settings

logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Standardized market data structure"""
    market_id: str
    source: str
    market_type: str  # "prediction_market", "crypto"
    symbol: Optional[str] = None
    question: Optional[str] = None
    current_price: Optional[float] = None
    market_probability: Optional[float] = None
    volume_24h: Optional[float] = None
    liquidity: Optional[float] = None
    last_updated: Optional[datetime] = None
    raw_data: Optional[Dict] = None

class BaseMarketDataProvider:
    """Base class for market data providers"""
    
    def __init__(self, name: str):
        self.name = name
        self.session = None
        self.rate_limit_delay = 1.0  # seconds between requests
        self.last_request_time = 0
        
    async def _ensure_session(self):
        """Ensure aiohttp session is available"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
    
    async def _rate_limit(self):
        """Apply rate limiting"""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = asyncio.get_event_loop().time()
    
    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()

class PolymarketProvider(BaseMarketDataProvider):
    """Polymarket prediction market data provider"""
    
    def __init__(self):
        super().__init__("polymarket")
        self.base_url = "https://gamma-api.polymarket.com"
        self.rate_limit_delay = 0.5  # 2 requests per second
    
    async def get_markets(self, limit: int = 20) -> List[MarketData]:
        """Get list of active Polymarket markets"""
        try:
            await self._ensure_session()
            await self._rate_limit()
            
            url = f"{self.base_url}/markets"
            params = {
                "active": True,
                "closed": False,
                "limit": limit
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    markets = []
                    
                    for market in data:
                        market_data = MarketData(
                            market_id=market.get("id"),
                            source="polymarket",
                            market_type="prediction_market",
                            question=market.get("question"),
                            market_probability=self._parse_polymarket_probability(market),
                            volume_24h=market.get("volume"),
                            liquidity=market.get("liquidity"),
                            last_updated=datetime.utcnow(),
                            raw_data=market
                        )
                        markets.append(market_data)
                    
                    return markets
                else:
                    logger.error(f"Polymarket API error: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error fetching Polymarket data: {str(e)}")
            return []
    
    async def get_market_by_id(self, market_id: str) -> Optional[MarketData]:
        """Get specific market data by ID"""
        try:
            await self._ensure_session()
            await self._rate_limit()
            
            url = f"{self.base_url}/markets/{market_id}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    market = await response.json()
                    
                    return MarketData(
                        market_id=market.get("id"),
                        source="polymarket",
                        market_type="prediction_market",
                        question=market.get("question"),
                        market_probability=self._parse_polymarket_probability(market),
                        volume_24h=market.get("volume"),
                        liquidity=market.get("liquidity"),
                        last_updated=datetime.utcnow(),
                        raw_data=market
                    )
                else:
                    logger.error(f"Polymarket market not found: {market_id}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching Polymarket market {market_id}: {str(e)}")
            return None
    
    def _parse_polymarket_probability(self, market: Dict) -> Optional[float]:
        """Parse probability from Polymarket data"""
        try:
            # Polymarket uses token prices between 0 and 1
            outcomes = market.get("outcomes", [])
            if outcomes and len(outcomes) >= 2:
                # For binary markets, use the "Yes" outcome price
                yes_outcome = next((o for o in outcomes if o.get("name", "").lower() in ["yes", "true"]), None)
                if yes_outcome:
                    return float(yes_outcome.get("price", 0))
            return None
        except (ValueError, TypeError):
            return None

class KalshiProvider(BaseMarketDataProvider):
    """Kalshi prediction market data provider"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("kalshi")
        self.base_url = "https://trading-api.kalshi.com/trade-api/v2"
        self.api_key = api_key or getattr(settings, 'kalshi_api_key', None)
        self.rate_limit_delay = 1.0  # 1 request per second for free tier
    
    async def get_markets(self, limit: int = 20) -> List[MarketData]:
        """Get list of active Kalshi markets"""
        try:
            await self._ensure_session()
            await self._rate_limit()
            
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            url = f"{self.base_url}/markets"
            params = {
                "limit": limit,
                "status": "open"
            }
            
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    markets = []
                    
                    for market in data.get("markets", []):
                        market_data = MarketData(
                            market_id=market.get("ticker"),
                            source="kalshi",
                            market_type="prediction_market",
                            question=market.get("title"),
                            market_probability=self._parse_kalshi_probability(market),
                            volume_24h=market.get("volume_24h"),
                            liquidity=market.get("open_interest"),
                            last_updated=datetime.utcnow(),
                            raw_data=market
                        )
                        markets.append(market_data)
                    
                    return markets
                else:
                    logger.error(f"Kalshi API error: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error fetching Kalshi data: {str(e)}")
            return []
    
    def _parse_kalshi_probability(self, market: Dict) -> Optional[float]:
        """Parse probability from Kalshi data"""
        try:
            # Kalshi prices are in cents, convert to probability
            last_price = market.get("last_price")
            if last_price is not None:
                return float(last_price) / 100.0
            return None
        except (ValueError, TypeError):
            return None

class CryptoDataProvider(BaseMarketDataProvider):
    """Cryptocurrency market data provider"""
    
    def __init__(self):
        super().__init__("crypto")
        self.coingecko_url = "https://api.coingecko.com/api/v3"
        self.rate_limit_delay = 1.0  # CoinGecko free tier limit
        
        # Initialize CCXT exchange for additional data
        try:
            self.binance = ccxt.binance({
                'rateLimit': 1200,  # 50 requests per 10 seconds
                'enableRateLimit': True,
            })
        except Exception as e:
            logger.warning(f"Could not initialize Binance exchange: {e}")
            self.binance = None
    
    async def get_crypto_prices(self, symbols: List[str]) -> List[MarketData]:
        """Get cryptocurrency prices from CoinGecko"""
        try:
            await self._ensure_session()
            await self._rate_limit()
            
            # Convert symbols to CoinGecko IDs
            symbol_map = {
                "BTC": "bitcoin",
                "ETH": "ethereum", 
                "ADA": "cardano",
                "SOL": "solana",
                "MATIC": "polygon",
                "AVAX": "avalanche-2",
                "DOT": "polkadot",
                "LINK": "chainlink"
            }
            
            ids = [symbol_map.get(symbol.upper(), symbol.lower()) for symbol in symbols]
            ids_string = ",".join(ids)
            
            url = f"{self.coingecko_url}/simple/price"
            params = {
                "ids": ids_string,
                "vs_currencies": "usd",
                "include_24hr_vol": "true",
                "include_24hr_change": "true",
                "include_market_cap": "true"
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    markets = []
                    
                    for coin_id, price_data in data.items():
                        # Map back to symbol
                        symbol = next((k for k, v in symbol_map.items() if v == coin_id), coin_id.upper())
                        
                        market_data = MarketData(
                            market_id=f"{symbol}-USD",
                            source="coingecko",
                            market_type="crypto",
                            symbol=symbol,
                            current_price=price_data.get("usd"),
                            volume_24h=price_data.get("usd_24h_vol"),
                            last_updated=datetime.utcnow(),
                            raw_data=price_data
                        )
                        markets.append(market_data)
                    
                    return markets
                else:
                    logger.error(f"CoinGecko API error: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error fetching crypto data: {str(e)}")
            return []
    
    async def get_binance_price(self, symbol: str) -> Optional[MarketData]:
        """Get real-time price from Binance"""
        try:
            if not self.binance:
                return None
                
            # Format symbol for Binance (e.g., BTC/USDT -> BTCUSDT)
            binance_symbol = symbol.replace("/", "").upper()
            if not binance_symbol.endswith("USDT"):
                binance_symbol += "USDT"
            
            ticker = await asyncio.get_event_loop().run_in_executor(
                None, self.binance.fetch_ticker, binance_symbol
            )
            
            return MarketData(
                market_id=f"{symbol}-BINANCE",
                source="binance",
                market_type="crypto",
                symbol=symbol,
                current_price=ticker.get("last"),
                volume_24h=ticker.get("quoteVolume"),
                last_updated=datetime.utcnow(),
                raw_data=ticker
            )
            
        except Exception as e:
            logger.error(f"Error fetching Binance data for {symbol}: {str(e)}")
            return None

class MarketDataService:
    """Main service for aggregating market data from multiple sources"""
    
    def __init__(self):
        self.polymarket = PolymarketProvider()
        self.kalshi = KalshiProvider()
        self.crypto = CryptoDataProvider()
        self.cache = {}  # Simple in-memory cache
        self.cache_duration = timedelta(minutes=5)
    
    async def get_prediction_markets(self, limit: int = 20) -> List[MarketData]:
        """Get prediction markets from all sources"""
        markets = []
        
        # Get Polymarket data
        polymarket_data = await self.polymarket.get_markets(limit // 2)
        markets.extend(polymarket_data)
        
        # Get Kalshi data
        kalshi_data = await self.kalshi.get_markets(limit // 2)
        markets.extend(kalshi_data)
        
        return markets
    
    async def get_crypto_markets(self, symbols: List[str] = None) -> List[MarketData]:
        """Get cryptocurrency market data"""
        if symbols is None:
            symbols = ["BTC", "ETH", "ADA", "SOL", "MATIC", "AVAX"]
        
        return await self.crypto.get_crypto_prices(symbols)
    
    async def get_market_by_id(self, market_id: str, source: str) -> Optional[MarketData]:
        """Get specific market data by ID and source"""
        cache_key = f"{source}:{market_id}"
        
        # Check cache first
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if datetime.utcnow() - timestamp < self.cache_duration:
                return cached_data
        
        # Fetch fresh data
        market_data = None
        if source == "polymarket":
            market_data = await self.polymarket.get_market_by_id(market_id)
        elif source == "kalshi":
            # Kalshi would need specific implementation
            pass
        elif source in ["binance", "crypto"]:
            market_data = await self.crypto.get_binance_price(market_id)
        
        # Cache the result
        if market_data:
            self.cache[cache_key] = (market_data, datetime.utcnow())
        
        return market_data
    
    async def close(self):
        """Close all provider sessions"""
        await self.polymarket.close()
        await self.kalshi.close()
        await self.crypto.close()

# Global instance
market_data_service = MarketDataService()