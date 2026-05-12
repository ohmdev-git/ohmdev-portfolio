"""AI-Trader platform integration — publish gold signals to ai4trade.ai."""
import json
import logging
import time
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

AI_TRADER_BASE = "https://ai4trade.ai/api"
_TOKEN_FILE = Path(__file__).parent / ".ai_trader_token.json"

# Simulated trade quantity (1 oz gold equivalent)
DEFAULT_QUANTITY = 1.0


def _load_token() -> Optional[str]:
    try:
        data = json.loads(_TOKEN_FILE.read_text())
        token = data.get("token")
        saved_at = data.get("saved_at", 0)
        # Tokens are valid for 24h — refresh after 23h to be safe
        if token and (time.time() - saved_at) < 82800:
            return token
    except Exception:
        pass
    return None


def _save_token(token: str) -> None:
    try:
        _TOKEN_FILE.write_text(json.dumps({"token": token, "saved_at": time.time()}))
    except Exception as e:
        logger.warning("Could not save AI-Trader token: %s", e)


def login(email: str, password: str) -> Optional[str]:
    """Login to AI-Trader and return JWT token."""
    try:
        resp = requests.post(
            f"{AI_TRADER_BASE}/claw/agents/login",
            json={"email": email, "password": password},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        # Try common token paths in the response
        token = (
            data.get("data", {}).get("token")
            or data.get("token")
            or data.get("access_token")
        )
        if token:
            _save_token(token)
            logger.info("AI-Trader: logged in successfully")
        else:
            logger.error("AI-Trader: login succeeded but no token in response: %s", data)
        return token
    except Exception as e:
        logger.error("AI-Trader login failed: %s", e)
        return None


def get_token(email: str, password: str) -> Optional[str]:
    """Return cached JWT or login fresh."""
    token = _load_token()
    if token:
        logger.debug("AI-Trader: using cached token")
        return token
    return login(email, password)


def _auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def publish_realtime_signal(
    token: str,
    symbol: str,
    action: str,
    price: float,
    quantity: float = DEFAULT_QUANTITY,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    market_type: str = "crypto",
    note: str = "",
) -> bool:
    """Publish a simulated real-time trade signal to AI-Trader."""
    payload: dict = {
        "symbol": symbol,
        "action": action,  # "buy" or "sell"
        "quantity": quantity,
        "price": price,
        "market_type": market_type,
        "note": note,
    }
    if stop_loss is not None:
        payload["stop_loss"] = stop_loss
    if take_profit is not None:
        payload["take_profit"] = take_profit

    try:
        resp = requests.post(
            f"{AI_TRADER_BASE}/signals/realtime",
            json=payload,
            headers=_auth_headers(token),
            timeout=10,
        )
        resp.raise_for_status()
        logger.info("AI-Trader: published %s %s @ %.2f", action.upper(), symbol, price)
        return True
    except requests.HTTPError as e:
        logger.error("AI-Trader publish realtime failed (%s): %s", e.response.status_code, e.response.text[:200])
        return False
    except Exception as e:
        logger.error("AI-Trader publish realtime failed: %s", e)
        return False


def publish_strategy(token: str, title: str, content: str) -> bool:
    """Publish a strategy/analysis post (no trade execution)."""
    try:
        resp = requests.post(
            f"{AI_TRADER_BASE}/signals/strategy",
            json={"title": title, "content": content},
            headers=_auth_headers(token),
            timeout=10,
        )
        resp.raise_for_status()
        logger.info("AI-Trader: published strategy '%s'", title)
        return True
    except requests.HTTPError as e:
        logger.error("AI-Trader publish strategy failed (%s): %s", e.response.status_code, e.response.text[:200])
        return False
    except Exception as e:
        logger.error("AI-Trader publish strategy failed: %s", e)
        return False


def heartbeat(token: str) -> list[dict]:
    """Poll heartbeat — returns combined list of messages and tasks."""
    try:
        resp = requests.post(
            f"{AI_TRADER_BASE}/claw/agents/heartbeat",
            headers=_auth_headers(token),
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json().get("data", {})
        messages = data.get("messages", [])
        tasks = data.get("tasks", [])
        if messages or tasks:
            logger.info("AI-Trader heartbeat: %d messages, %d tasks", len(messages), len(tasks))
        return messages + tasks
    except Exception as e:
        logger.error("AI-Trader heartbeat failed: %s", e)
        return []


def publish_signal_from_analysis(token: str, signal, cfg) -> bool:
    """Bridge between gold signal engine output and AI-Trader API.

    BUY/STRONG_BUY  → realtime trade (buy)
    SELL/STRONG_SELL → realtime trade (sell)
    HOLD            → strategy post with market analysis
    """
    from signal_engine import SignalType

    symbol = getattr(cfg, "ai_trader_symbol", "XAUUSD")
    market_type = getattr(cfg, "ai_trader_market_type", "crypto")

    if signal.signal in (SignalType.BUY, SignalType.STRONG_BUY):
        note = f"{signal.signal.value} | RSI:{signal.rsi_value:.1f} | ATR:{signal.atr_value:.2f} | " + "; ".join(signal.reasons[:3])
        return publish_realtime_signal(
            token=token,
            symbol=symbol,
            action="buy",
            price=signal.price,
            stop_loss=signal.stop_loss,
            take_profit=signal.tp1,
            market_type=market_type,
            note=note[:500],
        )

    if signal.signal in (SignalType.SELL, SignalType.STRONG_SELL):
        note = f"{signal.signal.value} | RSI:{signal.rsi_value:.1f} | ATR:{signal.atr_value:.2f} | " + "; ".join(signal.reasons[:3])
        return publish_realtime_signal(
            token=token,
            symbol=symbol,
            action="sell",
            price=signal.price,
            stop_loss=signal.stop_loss,
            take_profit=signal.tp1,
            market_type=market_type,
            note=note[:500],
        )

    # HOLD → publish as strategy analysis
    title = f"XAU/USD Market Analysis — {signal.timeframe}"
    content = signal.hold_summary()
    return publish_strategy(token=token, title=title, content=content)
