import logging
from typing import Optional
from dataclasses import dataclass

import yfinance as yf

logger = logging.getLogger(__name__)

_YF_SYMBOL = "GC=F"  # Gold Futures (COMEX) — closest proxy to XAUUSD perpetual

_TF_TO_INTERVAL = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "4h": "1h", "1d": "1d",
}
_TF_TO_PERIOD = {
    "1m": "7d", "5m": "60d", "15m": "60d", "30m": "60d",
    "1h": "30d", "4h": "60d", "1d": "1y",
}


@dataclass
class Candle:
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float


def fetch_candles(instrument: str, timeframe: str, limit: int = 50) -> list[Candle]:
    """Fetch OHLCV candles via yfinance (Gold Futures GC=F)."""
    interval = _TF_TO_INTERVAL.get(timeframe, "1h")
    period = _TF_TO_PERIOD.get(timeframe, "30d")
    try:
        df = yf.Ticker(_YF_SYMBOL).history(interval=interval, period=period)
        if df.empty:
            logger.error("yfinance returned no data for %s", _YF_SYMBOL)
            return []
        df = df.tail(limit)
        candles = [
            Candle(
                timestamp=str(row.Index),
                open=float(row.Open),
                high=float(row.High),
                low=float(row.Low),
                close=float(row.Close),
                volume=float(row.Volume),
            )
            for row in df.itertuples()
        ]
        logger.info("Fetched %d candles (%s %s)", len(candles), _YF_SYMBOL, timeframe)
        return candles
    except Exception as e:
        logger.error("Failed to fetch candles for %s/%s: %s", instrument, timeframe, e)
        return []


def fetch_ticker(instrument: str) -> Optional[dict]:
    """Fetch latest gold price."""
    try:
        info = yf.Ticker(_YF_SYMBOL).fast_info
        return {"last_price": info.last_price}
    except Exception as e:
        logger.error("Failed to fetch ticker for %s: %s", instrument, e)
        return None
