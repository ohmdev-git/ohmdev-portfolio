#!/usr/bin/env python3
"""
Gold Signal Bot — XAUUSDPERP auto-signal engine (multi-timeframe).

Usage:
    python main.py            # Run continuously (checks every CHECK_INTERVAL seconds)
    python main.py --once     # Run a single analysis and exit
    python main.py --test     # Send a test notification and exit
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import config
from fetcher import fetch_candles, fetch_ticker
from signal_engine import analyze, Signal, SignalType
from notifier import notify_all
from ai_trader import get_token, publish_signal_from_analysis, heartbeat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_STATE_FILE = Path(__file__).parent / ".signal_state.json"


def _load_state() -> dict:
    """State is keyed per-timeframe: {"5m": {"direction":..,"ts":..}, "1h": {...}}."""
    try:
        data = json.loads(_STATE_FILE.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    try:
        _STATE_FILE.write_text(json.dumps(state))
    except Exception as e:
        logger.warning("Could not save state: %s", e)


def _cooldown_ok(sig: Signal, tf: str, state: dict) -> bool:
    tf_state = state.get(tf, {})
    now = time.time()
    if sig.signal.name == tf_state.get("direction") and tf_state.get("ts", 0):
        elapsed = now - tf_state["ts"]
        cd = config.cooldown_for(tf)
        if elapsed < cd:
            remaining = int(cd - elapsed)
            logger.info("[%s] Signal %s in cooldown — %ds remaining", tf, sig.signal.name, remaining)
            return False
    return True


def run_analysis(force_notify: bool = False) -> None:
    """Fetch data, analyze, and notify — once per configured timeframe."""
    logger.info("Discord configured: %s", "YES" if config.discord_webhook_url else "NO — DISCORD_WEBHOOK_URL not set")
    logger.info("Timeframes: %s", ", ".join(config.signal_timeframes))

    state = _load_state()
    ai_token: str | None = None

    for tf in config.signal_timeframes:
        logger.info("[%s] Fetching candles for %s…", tf, config.instrument)
        candles = fetch_candles(config.instrument, tf)
        if not candles:
            logger.error("[%s] No candle data received — skipping", tf)
            continue

        signal = analyze(candles, tf, config)
        if signal is None:
            logger.warning("[%s] Could not generate signal (insufficient data)", tf)
            continue

        logger.info(
            "[%s] Signal: %s | Price: %.2f | RSI: %.1f | MACD: %+.4f",
            tf, signal.signal.name, signal.price, signal.rsi_value, signal.macd_hist,
        )

        if not (force_notify or _cooldown_ok(signal, tf, state)):
            logger.info("[%s] %s — in cooldown, no notification sent", tf, signal.signal.name)
            continue

        msg = signal.hold_summary() if signal.signal == SignalType.HOLD else signal.summary()
        results = notify_all(msg, config)
        channels = ", ".join(f"{k}={'✓' if v else '✗'}" for k, v in results.items()) or "stdout"
        logger.info("[%s] Notification sent → %s", tf, channels)
        state[tf] = {"direction": signal.signal.name, "ts": time.time()}

        if config.ai_trader_enabled and config.ai_trader_email and config.ai_trader_password:
            if ai_token is None:
                ai_token = get_token(config.ai_trader_email, config.ai_trader_password)
            if ai_token:
                ok = publish_signal_from_analysis(ai_token, signal, config)
                logger.info("[%s] AI-Trader publish: %s", tf, "✓" if ok else "✗")
            else:
                logger.warning("[%s] AI-Trader: could not obtain token — skipping publish", tf)

    _save_state(state)


def test_notification():
    """Send a dummy signal to verify notification channels are working."""
    from signal_engine import SignalType, Signal
    dummy = Signal(
        signal=SignalType.BUY,
        price=4727.30,
        entry=4727.30,
        stop_loss=4700.00,
        tp1=4768.00,
        tp2=4795.00,
        tp3=4836.00,
        risk_pips=27.30,
        rsi_value=52.5,
        ema_fast=4720.10,
        ema_slow=4710.50,
        ema_trend=4695.00,
        macd_hist=0.0025,
        atr_value=18.20,
        reasons=["Test notification — channels are working correctly"],
        timeframe="1h",
    )
    msg = "🔔 *[TEST]* " + dummy.summary()
    results = notify_all(msg, config)
    print("Results:", results)


def main():
    parser = argparse.ArgumentParser(description="Gold Signal Bot")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--test", action="store_true", help="Send test notification and exit")
    args = parser.parse_args()

    if args.test:
        logger.info("Sending test notification…")
        test_notification()
        return

    if args.once:
        run_analysis()
        return

    logger.info(
        "Gold Signal Bot started — checking %s [%s] every %ds",
        config.instrument,
        ", ".join(config.signal_timeframes),
        config.check_interval,
    )
    logger.info(
        "Channels: Telegram=%s | Discord=%s | LINE=%s",
        "✓" if config.telegram_token else "✗",
        "✓" if config.discord_webhook_url else "✗",
        "✓" if config.line_notify_token else "✗",
    )

    _ai_token: str | None = None
    if config.ai_trader_enabled and config.ai_trader_email and config.ai_trader_password:
        _ai_token = get_token(config.ai_trader_email, config.ai_trader_password)
        if _ai_token:
            logger.info("AI-Trader: connected as %s", config.ai_trader_email)

    _last_heartbeat = 0.0
    _heartbeat_interval = 45

    while True:
        try:
            run_analysis()
        except KeyboardInterrupt:
            logger.info("Interrupted — shutting down")
            break
        except Exception as e:
            logger.exception("Unexpected error in analysis cycle: %s", e)

        now = time.time()
        if _ai_token and (now - _last_heartbeat) >= _heartbeat_interval:
            try:
                heartbeat(_ai_token)
            except Exception as e:
                logger.warning("AI-Trader heartbeat error: %s", e)
            _last_heartbeat = now

        logger.info("Next check in %ds…", config.check_interval)
        try:
            time.sleep(config.check_interval)
        except KeyboardInterrupt:
            logger.info("Interrupted — shutting down")
            break


if __name__ == "__main__":
    main()
