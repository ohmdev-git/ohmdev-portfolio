import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import requests

logger = logging.getLogger(__name__)

EXCHANGE_API = "https://api.crypto.com/exchange/v1"
_YF_SYMBOL   = "GC%3DF"   # GC=F URL-encoded (Gold Futures)
_STOOQ_SYM   = "xauusd"


@dataclass
class Candle:
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float


# ── Source 1: Crypto.com ──────────────────────────────────────────────────────

def _from_cryptodotcom(instrument: str, timeframe: str, limit: int) -> list[Candle]:
    url = f"{EXCHANGE_API}/public/get-candlestick"
    payload = {
        "id": 1,
        "method": "public/get-candlestick",
        "params": {"instrument_name": instrument, "timeframe": timeframe},
        "nonce": int(time.time() * 1000),
    }
    resp = requests.post(url, json=payload, timeout=10)
    resp.raise_for_status()
    raw = resp.json().get("result", {}).get("data", [])
    candles = [
        Candle(
            timestamp=c.get("timestamp", c.get("t", "")),
            open=float(c.get("open",   c.get("o", 0))),
            high=float(c.get("high",   c.get("h", 0))),
            low=float(c.get("low",    c.get("l", 0))),
            close=float(c.get("close", c.get("c", 0))),
            volume=float(c.get("volume", c.get("v", 0))),
        )
        for c in raw
    ]
    candles.sort(key=lambda c: c.timestamp)
    return candles[-limit:]


# ── Source 2: Yahoo Finance ───────────────────────────────────────────────────

_YF_INTERVAL = {"1h": "1h", "4h": "1h", "1D": "1d"}
_YF_RANGE    = {"1h": "7d", "4h": "30d", "1D": "60d"}


def _from_yahoo(timeframe: str, limit: int) -> list[Candle]:
    interval = _YF_INTERVAL.get(timeframe, "1h")
    yf_range = _YF_RANGE.get(timeframe, "7d")

    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept":          "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer":         "https://finance.yahoo.com/",
        "Origin":          "https://finance.yahoo.com",
    })

    # Warm up session to pick up cookies (avoids 401 on chart endpoint)
    try:
        session.get(f"https://finance.yahoo.com/quote/{_YF_SYMBOL}", timeout=8)
    except Exception:
        pass

    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{_YF_SYMBOL}"
        f"?interval={interval}&range={yf_range}&includePrePost=false"
    )
    resp = session.get(url, timeout=15)
    resp.raise_for_status()

    result = resp.json().get("chart", {}).get("result", [])
    if not result:
        raise ValueError("Yahoo Finance returned empty chart result")

    r = result[0]
    timestamps = r.get("timestamp", [])
    quote = r.get("indicators", {}).get("quote", [{}])[0]

    candles = []
    for i, ts in enumerate(timestamps):
        try:
            o = quote["open"][i]
            h = quote["high"][i]
            l = quote["low"][i]
            c = quote["close"][i]
            v = quote.get("volume", [])[i] if i < len(quote.get("volume", [])) else 0
            if None in (o, h, l, c):
                continue
            candles.append(Candle(
                timestamp=str(ts),
                open=float(o), high=float(h),
                low=float(l),  close=float(c),
                volume=float(v or 0),
            ))
        except (IndexError, TypeError, KeyError):
            continue

    candles.sort(key=lambda c: c.timestamp)
    return candles[-limit:]


# ── Source 3: Stooq (daily CSV fallback) ─────────────────────────────────────

def _from_stooq(limit: int) -> list[Candle]:
    """Daily XAUUSD bars from stooq.com — no auth, always free."""
    url = f"https://stooq.com/q/d/l/?s={_STOOQ_SYM}&i=d"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; GoldBot/1.0)",
        "Accept": "text/csv,text/plain,*/*",
    }
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()

    lines = resp.text.strip().splitlines()
    if len(lines) < 2:
        raise ValueError("Stooq returned insufficient data")

    # CSV: Date,Open,High,Low,Close,Volume
    candles = []
    for line in lines[1:]:  # skip header
        parts = line.split(",")
        if len(parts) < 5:
            continue
        try:
            date_str = parts[0].strip()
            candles.append(Candle(
                timestamp=date_str,
                open=float(parts[1]),
                high=float(parts[2]),
                low=float(parts[3]),
                close=float(parts[4]),
                volume=float(parts[5]) if len(parts) > 5 and parts[5].strip() else 0.0,
            ))
        except (ValueError, IndexError):
            continue

    candles.sort(key=lambda c: c.timestamp)
    return candles[-limit:]


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_candles(instrument: str, timeframe: str, limit: int = 50) -> list[Candle]:
    """Fetch OHLCV candles — tries Crypto.com → Yahoo Finance → Stooq."""
    sources = [
        ("Crypto.com",    lambda: _from_cryptodotcom(instrument, timeframe, limit)),
        ("Yahoo Finance", lambda: _from_yahoo(timeframe, limit)),
        ("Stooq",         lambda: _from_stooq(limit)),
    ]
    for name, fn in sources:
        try:
            candles = fn()
            if candles:
                logger.info("Fetched %d candles from %s (%s %s)", len(candles), name, instrument, timeframe)
                return candles
            logger.warning("%s returned 0 candles — trying next source", name)
        except Exception as e:
            logger.warning("%s failed (%s) — trying next source", name, e)

    logger.error("All data sources failed — no candle data available")
    return []


def fetch_ticker(instrument: str) -> Optional[dict]:
    """Fetch current ticker from Crypto.com (best-effort)."""
    url = f"{EXCHANGE_API}/public/get-tickers"
    payload = {
        "id": 1,
        "method": "public/get-tickers",
        "params": {"instrument_name": instrument},
        "nonce": int(time.time() * 1000),
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        tickers = resp.json().get("result", {}).get("data", [])
        return tickers[0] if tickers else None
    except Exception as e:
        logger.warning("Ticker fetch failed (non-critical): %s", e)
        return None
