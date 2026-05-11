import requests
import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

EXCHANGE_API = "https://api.crypto.com/exchange/v1"


@dataclass
class Candle:
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float


def fetch_candles(instrument: str, timeframe: str, limit: int = 50) -> list[Candle]:
    """Fetch OHLCV candles from Crypto.com Exchange."""
    url = f"{EXCHANGE_API}/public/get-candlestick"
    params = {
        "instrument_name": instrument,
        "timeframe": timeframe,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        raw = data.get("result", {}).get("data", [])
        if not raw:
            raw = data.get("data", [])
        candles = [
            Candle(
                timestamp=c.get("timestamp", c.get("t", "")),
                open=float(c.get("open", c.get("o", 0))),
                high=float(c.get("high", c.get("h", 0))),
                low=float(c.get("low", c.get("l", 0))),
                close=float(c.get("close", c.get("c", 0))),
                volume=float(c.get("volume", c.get("v", 0))),
            )
            for c in raw
        ]
        candles.sort(key=lambda c: c.timestamp)
        candles = candles[-limit:]
        logger.info("Fetched %d candles for %s (%s)", len(candles), instrument, timeframe)
        return candles
    except Exception as e:
        logger.error("Failed to fetch candles for %s/%s: %s", instrument, timeframe, e)
        return []


def fetch_ticker(instrument: str) -> Optional[dict]:
    """Fetch current ticker for an instrument."""
    url = f"{EXCHANGE_API}/public/get-tickers"
    try:
        resp = requests.get(url, params={"instrument_name": instrument}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        tickers = data.get("result", {}).get("data", data.get("data", []))
        return tickers[0] if tickers else None
    except Exception as e:
        logger.error("Failed to fetch ticker for %s: %s", instrument, e)
        return None
