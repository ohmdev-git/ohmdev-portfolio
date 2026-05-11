#!/usr/bin/env python3
"""
Gold Signal Bot — XAUUSDPERP auto-signal engine.

Usage:
    python main.py            # Run continuously (checks every CHECK_INTERVAL seconds)
    python main.py --once     # Run a single analysis and exit
    python main.py --test     # Send a test notification and exit

Environment variables (set in .env or export):
    TELEGRAM_BOT_TOKEN   — Telegram bot token from @BotFather
    TELEGRAM_CHAT_ID     — Your Telegram chat/group ID
    DISCORD_WEBHOOK_URL  — Discord channel webhook URL
    LINE_NOTIFY_TOKEN    — LINE Notify personal access token
    CHECK_INTERVAL       — How often to check in seconds (default 300 = 5 min)
    SIGNAL_COOLDOWN      — Min seconds between same-direction signals (default 3600)
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Support running from within the gold_signals/ directory
sys.path.insert(0, str(Path(__file__).parent))

from config import config
from fetcher import fetch_candles, fetch_ticker
from signal_engine import analyze, Signal, SignalType
from notifier import notify_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Persisted across GitHub Actions runs via cache
_STATE_FILE = Path(__file__).parent / ".signal_state.json"


def _load_state() -> dict:
    try:
        return json.loads(_STATE_FILE.read_text())
    except Exception:
        return {"direction": None, "ts": 0.0}


def _save_state(direction: str, ts: float) -> None:
    try:
        _STATE_FILE.write_text(json.dumps({"direction": direction, "ts": ts}))
    except Exception as e:
        logger.warning("Could not save state: %s", e)


def _cooldown_ok(sig: Signal) -> bool:
    state = _load_state()
    now = time.time()
    if sig.signal.name == state.get("direction") and state.get("ts", 0):
        elapsed = now - state["ts"]
        if elapsed < config.signal_cooldown:
            remaining = int(config.signal_cooldown - elapsed)
            logger.info("Signal %s in cooldown — %ds remaining", sig.signal.name, remaining)
            return False
    return True


def run_analysis(force_notify: bool = False) -> Signal | None:
    """Fetch data, analyze, and notify if signal warrants it."""
    logger.info("Discord configured: %s", "YES" if config.discord_webhook_url else "NO — DISCORD_WEBHOOK_URL not set")
    logger.info("Fetching candles for %s (%s)…", config.instrument, config.primary_tf)
    candles = fetch_candles(config.instrument, config.primary_tf)
    if not candles:
        logger.error("No candle data received — skipping cycle")
        return None

    signal = analyze(candles, config.primary_tf, config)
    if signal is None:
        logger.warning("Could not generate signal (insufficient data)")
        return None

    logger.info(
        "Signal: %s | Price: %.2f | RSI: %.1f | MACD: %+.4f",
        signal.signal.name,
        signal.price,
        signal.rsi_value,
        signal.macd_hist,
    )

    should_notify = force_notify or _cooldown_ok(signal)

    if should_notify:
        if signal.signal == SignalType.HOLD:
            msg = signal.hold_summary()
        else:
            msg = signal.summary()
        results = notify_all(msg, config)
        channels = ", ".join(f"{k}={'✓' if v else '✗'}" for k, v in results.items()) or "stdout"
        logger.info("Notification sent → %s", channels)
        _save_state(signal.signal.name, time.time())
    else:
        logger.info("%s — in cooldown, no notification sent", signal.signal.name)

    return signal


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

    # Continuous mode
    logger.info(
        "Gold Signal Bot started — checking %s every %ds",
        config.instrument,
        config.check_interval,
    )
    logger.info(
        "Channels: Telegram=%s | Discord=%s | LINE=%s",
        "✓" if config.telegram_token else "✗",
        "✓" if config.discord_webhook_url else "✗",
        "✓" if config.line_notify_token else "✗",
    )

    while True:
        try:
            run_analysis()
        except KeyboardInterrupt:
            logger.info("Interrupted — shutting down")
            break
        except Exception as e:
            logger.exception("Unexpected error in analysis cycle: %s", e)

        logger.info("Next check in %ds…", config.check_interval)
        try:
            time.sleep(config.check_interval)
        except KeyboardInterrupt:
            logger.info("Interrupted — shutting down")
            break


if __name__ == "__main__":
    main()
