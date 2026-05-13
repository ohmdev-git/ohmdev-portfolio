#!/usr/bin/env python3
"""
Autonomous Claude-powered AI trading agent for XAU/USD.

Uses Claude claude-opus-4-7 with tool use and adaptive thinking to:
  1. Fetch and analyze live XAU/USD market data
  2. Decide: BUY, SELL, or publish a market analysis (HOLD)
  3. Publish signals to AI-Trader (ai4trade.ai) using the ai_trader skill
  4. Respond intelligently to platform mentions / tasks via heartbeat

Usage:
    python agent.py             # Analyze once and exit
    python agent.py --loop      # Run continuously with heartbeat polling
    python agent.py --test      # Dry-run — analyze but don't publish
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

import anthropic
from config import config
from fetcher import fetch_candles, fetch_ticker
from signal_engine import analyze
from ai_trader import get_token, publish_realtime_signal, publish_strategy, heartbeat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL = "claude-opus-4-7"
HEARTBEAT_INTERVAL = 45  # seconds

SYSTEM_PROMPT = """You are an expert XAU/USD (gold) trading analyst and autonomous agent on the AI-Trader platform (ai4trade.ai).

Your mission:
1. Fetch live market data for XAUUSD using the fetch_market_data tool
2. Analyze technical indicators (EMA crossovers, RSI, MACD histogram, Bollinger Bands, ATR)
3. Make a decisive, well-reasoned trading call
4. Publish your decision to the AI-Trader platform

Decision framework:
- STRONG BUY (net score ≥ 5): publish_trade_signal with action="buy", include SL and TP
- BUY (net score 2–4): publish_trade_signal with action="buy"
- HOLD (net score −1 to 1): publish_market_analysis with current market context
- SELL (net score −2 to −4): publish_trade_signal with action="sell"
- STRONG SELL (net score ≤ −5): publish_trade_signal with action="sell", tight SL

Risk management rules:
- Always include stop_loss when publishing a trade signal
- Take profit should be at least 1.5× the stop-loss distance from entry
- Keep the note concise: mention the 2-3 strongest reasons for the call

When responding to platform mentions, fetch fresh market data first if the question is market-related.
Be helpful, factual, and grounded in the indicator data."""

TOOLS = [
    {
        "name": "fetch_market_data",
        "description": (
            "Fetch live XAU/USD (XAUUSDPERP) candlestick data and compute technical indicators "
            "(EMA9/21/50, RSI14, MACD 12/26/9, Bollinger Bands, ATR14). "
            "Returns current price, indicator values, pre-computed signal direction, entry/SL/TP levels, "
            "and a list of reasons. Always call this before making any trading decision."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "timeframe": {
                    "type": "string",
                    "enum": ["1h", "4h", "1D"],
                    "description": "Candlestick timeframe to analyze. Default is 1h.",
                }
            },
            "required": [],
        },
    },
    {
        "name": "publish_trade_signal",
        "description": (
            "Publish a BUY or SELL directional trade signal to the AI-Trader platform. "
            "Use this when multi-indicator confluence is strong enough to justify a live signal. "
            "The platform simulates order execution at the given price."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["buy", "sell"],
                    "description": "Trade direction",
                },
                "price": {
                    "type": "number",
                    "description": "Current market entry price",
                },
                "stop_loss": {
                    "type": "number",
                    "description": "Stop-loss price level",
                },
                "take_profit": {
                    "type": "number",
                    "description": "First take-profit price level (TP1)",
                },
                "note": {
                    "type": "string",
                    "description": "Brief reasoning for the trade (max 500 chars)",
                },
            },
            "required": ["action", "price"],
        },
    },
    {
        "name": "publish_market_analysis",
        "description": (
            "Publish a market analysis or strategy commentary to AI-Trader when there is no clear "
            "directional signal (mixed indicators / HOLD conditions). "
            "Supports Markdown formatting in content."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Title of the analysis post",
                },
                "content": {
                    "type": "string",
                    "description": "Analysis content (Markdown supported, mention key indicator values)",
                },
            },
            "required": ["title", "content"],
        },
    },
    {
        "name": "reply_to_mention",
        "description": (
            "Record a reply to a mention or task notification received from the AI-Trader platform. "
            "Use this to respond to follower questions or platform tasks."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reply": {
                    "type": "string",
                    "description": "The reply message content",
                }
            },
            "required": ["reply"],
        },
    },
]


# ── Tool implementations ───────────────────────────────────────────────────────

def _tool_fetch_market_data(timeframe: str = "1h") -> dict:
    candles = fetch_candles(config.instrument, timeframe)
    if not candles:
        return {"error": f"No candle data for {config.instrument}/{timeframe}"}

    signal = analyze(candles, timeframe, config)
    if signal is None:
        return {"error": "Insufficient candle data for analysis"}

    ticker = fetch_ticker(config.instrument)
    try:
        live_price = float(ticker.get("a", signal.price)) if ticker else signal.price
    except (TypeError, ValueError):
        live_price = signal.price

    return {
        "instrument": config.instrument,
        "timeframe": timeframe,
        "live_price": round(live_price, 2),
        "candles_analyzed": len(candles),
        "signal": signal.signal.name,
        "rsi": round(signal.rsi_value, 2),
        "ema_fast": round(signal.ema_fast, 2),
        "ema_slow": round(signal.ema_slow, 2),
        "macd_histogram": round(signal.macd_hist, 4),
        "atr": round(signal.atr_value, 2),
        "suggested_entry": round(signal.entry, 2),
        "suggested_stop_loss": round(signal.stop_loss, 2),
        "suggested_tp1": round(signal.tp1, 2),
        "suggested_tp2": round(signal.tp2, 2),
        "suggested_tp3": round(signal.tp3, 2),
        "reasons": signal.reasons,
    }


def _tool_publish_trade_signal(
    action: str,
    price: float,
    stop_loss: Optional[float],
    take_profit: Optional[float],
    note: str,
    token: str,
    dry_run: bool,
) -> dict:
    if dry_run:
        logger.info("[DRY RUN] Would publish %s @ %.2f (SL=%.2f TP=%.2f)", action.upper(), price,
                    stop_loss or 0, take_profit or 0)
        return {"status": "dry_run", "action": action, "price": price}

    ok = publish_realtime_signal(
        token=token,
        symbol=config.ai_trader_symbol,
        action=action,
        price=price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        market_type=config.ai_trader_market_type,
        note=(note or "")[:500],
    )
    return {"status": "published" if ok else "failed", "action": action, "price": price}


def _tool_publish_market_analysis(title: str, content: str, token: str, dry_run: bool) -> dict:
    if dry_run:
        logger.info("[DRY RUN] Would publish analysis: %s", title)
        return {"status": "dry_run", "title": title}

    ok = publish_strategy(token=token, title=title, content=content)
    return {"status": "published" if ok else "failed", "title": title}


def _tool_reply_to_mention(reply: str) -> dict:
    logger.info("Platform reply: %s", reply)
    return {"status": "logged", "reply": reply}


def _dispatch(name: str, inp: dict, token: Optional[str], dry_run: bool) -> dict:
    if name == "fetch_market_data":
        return _tool_fetch_market_data(inp.get("timeframe", "1h"))

    if name == "publish_trade_signal":
        if not token and not dry_run:
            return {"error": "No AI-Trader token — cannot publish"}
        return _tool_publish_trade_signal(
            action=inp["action"],
            price=float(inp["price"]),
            stop_loss=inp.get("stop_loss"),
            take_profit=inp.get("take_profit"),
            note=inp.get("note", ""),
            token=token or "",
            dry_run=dry_run,
        )

    if name == "publish_market_analysis":
        if not token and not dry_run:
            return {"error": "No AI-Trader token — cannot publish"}
        return _tool_publish_market_analysis(
            title=inp["title"],
            content=inp["content"],
            token=token or "",
            dry_run=dry_run,
        )

    if name == "reply_to_mention":
        return _tool_reply_to_mention(inp.get("reply", ""))

    return {"error": f"Unknown tool: {name}"}


# ── Agent loop ─────────────────────────────────────────────────────────────────

def run_agent(task: str, token: Optional[str], dry_run: bool = False) -> None:
    """Run one complete Claude agentic loop for the given task."""
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    messages: list[dict] = [{"role": "user", "content": task}]

    logger.info("Agent task: %s", task[:120])

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            for block in response.content:
                if hasattr(block, "text") and block.text:
                    logger.info("Agent: %s", block.text[:500])
            break

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            logger.info("→ tool: %s(%s)", block.name, json.dumps(block.input)[:200])
            try:
                result = _dispatch(block.name, block.input, token=token, dry_run=dry_run)
            except Exception as exc:
                result = {"error": str(exc)}

            logger.info("← result: %s", json.dumps(result)[:300])
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result),
            })

        messages.append({"role": "user", "content": tool_results})


def process_events(events: list[dict], token: Optional[str], dry_run: bool = False) -> None:
    """Pass heartbeat events to the agent for intelligent handling."""
    for event in events:
        event_type = event.get("type", "message")
        content = (
            event.get("content")
            or event.get("message")
            or event.get("text")
            or json.dumps(event)
        )
        task = (
            f"You received a platform {event_type}: \"{content}\"\n\n"
            "If this is a market question or trade request, fetch current market data first "
            "then reply with a grounded answer. "
            "If it asks for a new signal, analyze and publish appropriately."
        )
        run_agent(task, token, dry_run=dry_run)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Claude AI Trading Agent — XAU/USD")
    parser.add_argument("--loop", action="store_true", help="Continuous heartbeat polling mode")
    parser.add_argument("--test", "--dry-run", dest="dry_run", action="store_true",
                        help="Dry run — analyze but do not publish to AI-Trader")
    args = parser.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.error("ANTHROPIC_API_KEY not set — cannot run Claude agent")
        sys.exit(0)

    dry_run = args.dry_run
    token: Optional[str] = None

    if config.ai_trader_enabled and config.ai_trader_email and config.ai_trader_password:
        token = get_token(config.ai_trader_email, config.ai_trader_password)
        if token:
            logger.info("AI-Trader: authenticated as %s", config.ai_trader_email)
        else:
            logger.warning("AI-Trader: could not obtain token — switching to dry run")
            dry_run = True
    else:
        logger.warning("AI-Trader not configured — running in dry-run mode")
        dry_run = True

    initial_task = (
        "Analyze the current XAU/USD (gold) market and publish an appropriate signal.\n\n"
        "Steps:\n"
        "1. Call fetch_market_data for the 1h timeframe\n"
        "2. Review the returned indicators carefully\n"
        "3. Decide: is this a BUY, SELL, or HOLD based on indicator confluence?\n"
        "4. Publish the appropriate signal (trade or market analysis)"
    )

    logger.info("Claude agent starting (model=%s, dry_run=%s, loop=%s)", MODEL, dry_run, args.loop)
    run_agent(initial_task, token, dry_run=dry_run)

    if not args.loop:
        return

    logger.info("Entering heartbeat loop (interval=%ds)…", HEARTBEAT_INTERVAL)
    last_hb = 0.0

    while True:
        try:
            now = time.time()
            if token and (now - last_hb) >= HEARTBEAT_INTERVAL:
                events = heartbeat(token)
                if events:
                    process_events(events, token, dry_run=dry_run)
                last_hb = now
            time.sleep(10)
        except KeyboardInterrupt:
            logger.info("Interrupted — shutting down")
            break
        except Exception as exc:
            logger.exception("Heartbeat loop error: %s", exc)
            time.sleep(30)


if __name__ == "__main__":
    main()
