"""Gold signal engine — generates BUY/SELL/HOLD signals with entry, SL, and TP levels."""
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from fetcher import Candle
from indicators import ema, rsi, macd, bollinger_bands, atr

logger = logging.getLogger(__name__)


class SignalType(Enum):
    STRONG_BUY = "STRONG BUY 🔥"
    BUY = "BUY 🟢"
    HOLD = "HOLD ⏳"
    SELL = "SELL 🔴"
    STRONG_SELL = "STRONG SELL 💥"


@dataclass
class Signal:
    signal: SignalType
    price: float
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    risk_pips: float
    rsi_value: float
    ema_fast: float
    ema_slow: float
    ema_trend: float
    macd_hist: float
    atr_value: float
    reasons: list[str]
    timeframe: str

    def risk_reward_str(self) -> str:
        risk = abs(self.entry - self.stop_loss)
        if risk == 0:
            return "N/A"
        r1 = abs(self.tp1 - self.entry) / risk
        r2 = abs(self.tp2 - self.entry) / risk
        r3 = abs(self.tp3 - self.entry) / risk
        return f"TP1={r1:.1f}R | TP2={r2:.1f}R | TP3={r3:.1f}R"

    def summary(self) -> str:
        direction = "LONG" if self.signal in (SignalType.BUY, SignalType.STRONG_BUY) else "SHORT"
        lines = [
            f"🥇 *XAU/USD SIGNAL — {self.signal.value}*",
            f"📊 Timeframe: `{self.timeframe}`",
            f"💵 Price: `${self.price:,.2f}`",
            "",
            f"📍 *Entry*: `${self.entry:,.2f}`",
            f"🛑 *Stop Loss*: `${self.stop_loss:,.2f}`",
            f"🎯 *TP1*: `${self.tp1:,.2f}`",
            f"🎯 *TP2*: `${self.tp2:,.2f}`",
            f"🎯 *TP3*: `${self.tp3:,.2f}`",
            f"📐 Risk/Reward: `{self.risk_reward_str()}`",
            "",
            f"📈 *Indicators*",
            f"• RSI({14}): `{self.rsi_value:.1f}`",
            f"• EMA Fast/Slow: `{self.ema_fast:,.2f}` / `{self.ema_slow:,.2f}`",
            f"• MACD Histogram: `{self.macd_hist:+.4f}`",
            f"• ATR: `{self.atr_value:.2f}`",
            "",
            "📝 *Reasons*",
        ] + [f"• {r}" for r in self.reasons]
        return "\n".join(lines)


def _last_valid(values: list[Optional[float]]) -> Optional[float]:
    for v in reversed(values):
        if v is not None:
            return v
    return None


def _prev_valid(values: list[Optional[float]]) -> Optional[float]:
    found = 0
    for v in reversed(values):
        if v is not None:
            found += 1
            if found == 2:
                return v
    return None


def analyze(candles: list[Candle], timeframe: str, cfg=None) -> Optional[Signal]:
    """Run all indicators and return a Signal (or None if not enough data)."""
    if cfg is None:
        from config import config as cfg

    # Need at least EMA21 seed + 2 bars to compute crossover
    min_required = cfg.ema_slow + 2
    if len(candles) < min_required:
        logger.warning("Not enough candles (%d, need %d) for analysis", len(candles), min_required)
        return None

    closes = [c.close for c in candles]
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]

    ema_fast_series = ema(closes, cfg.ema_fast)
    ema_slow_series = ema(closes, cfg.ema_slow)
    ema_trend_series = ema(closes, cfg.ema_trend)
    rsi_series = rsi(closes, cfg.rsi_period)
    macd_result = macd(closes, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
    bb = bollinger_bands(closes, cfg.bb_period, cfg.bb_std)
    atr_series = atr(highs, lows, closes, cfg.atr_period)

    # Latest values
    price = closes[-1]
    ef = _last_valid(ema_fast_series)
    es = _last_valid(ema_slow_series)
    et = _last_valid(ema_trend_series)
    rsi_val = _last_valid(rsi_series)
    mhist = _last_valid(macd_result.histogram)
    mhist_prev = _prev_valid(macd_result.histogram)
    bb_upper = _last_valid(bb.upper)
    bb_lower = _last_valid(bb.lower)
    atr_val = _last_valid(atr_series)

    # et (EMA50) and mhist (MACD) may be None when candle count is low — handled below
    if any(v is None for v in [ef, es, rsi_val, atr_val]):
        logger.warning("Insufficient indicator data (EMA9/21, RSI, ATR required)")
        return None

    # Previous EMA values for crossover detection
    ef_prev = _prev_valid(ema_fast_series)
    es_prev = _prev_valid(ema_slow_series)

    reasons: list[str] = []
    bull_score = 0
    bear_score = 0

    # --- EMA trend ---
    if ef > es:
        reasons.append(f"EMA{cfg.ema_fast} >{cfg.ema_slow} (bullish)")
        bull_score += 2
    else:
        reasons.append(f"EMA{cfg.ema_fast} <{cfg.ema_slow} (bearish)")
        bear_score += 2

    # Crossover bonus
    if ef_prev is not None and es_prev is not None:
        if ef > es and ef_prev <= es_prev:
            reasons.append(f"Golden cross EMA{cfg.ema_fast}×{cfg.ema_slow} just happened!")
            bull_score += 2
        elif ef < es and ef_prev >= es_prev:
            reasons.append(f"Death cross EMA{cfg.ema_fast}×{cfg.ema_slow} just happened!")
            bear_score += 2

    # --- Price vs trend EMA (optional — skipped when candle count is low) ---
    if et is not None:
        if price > et:
            reasons.append(f"Price above EMA{cfg.ema_trend} (uptrend)")
            bull_score += 1
        else:
            reasons.append(f"Price below EMA{cfg.ema_trend} (downtrend)")
            bear_score += 1

    # --- RSI ---
    if rsi_val < cfg.rsi_oversold:
        reasons.append(f"RSI {rsi_val:.1f} — oversold (buy zone)")
        bull_score += 2
    elif rsi_val > cfg.rsi_overbought:
        reasons.append(f"RSI {rsi_val:.1f} — overbought (sell zone)")
        bear_score += 2
    elif 40 <= rsi_val <= 60:
        reasons.append(f"RSI {rsi_val:.1f} — neutral")
    elif rsi_val > 60:
        reasons.append(f"RSI {rsi_val:.1f} — bullish momentum")
        bull_score += 1
    else:
        reasons.append(f"RSI {rsi_val:.1f} — bearish momentum")
        bear_score += 1

    # --- MACD (optional — needs 26+ candles) ---
    if mhist is not None:
        if mhist > 0:
            reasons.append("MACD histogram positive (bullish momentum)")
            bull_score += 1
            if mhist_prev is not None and mhist_prev <= 0:
                reasons.append("MACD histogram just crossed above zero!")
                bull_score += 1
        else:
            reasons.append("MACD histogram negative (bearish momentum)")
            bear_score += 1
            if mhist_prev is not None and mhist_prev >= 0:
                reasons.append("MACD histogram just crossed below zero!")
                bear_score += 1

    # --- Bollinger Bands ---
    if bb_lower is not None and price <= bb_lower:
        reasons.append("Price at/below lower Bollinger Band (oversold)")
        bull_score += 1
    elif bb_upper is not None and price >= bb_upper:
        reasons.append("Price at/above upper Bollinger Band (overbought)")
        bear_score += 1

    # --- Determine signal type ---
    net = bull_score - bear_score
    if net >= 5:
        sig = SignalType.STRONG_BUY
    elif net >= 2:
        sig = SignalType.BUY
    elif net <= -5:
        sig = SignalType.STRONG_SELL
    elif net <= -2:
        sig = SignalType.SELL
    else:
        sig = SignalType.HOLD

    # --- Compute SL and TP ---
    is_long = sig in (SignalType.BUY, SignalType.STRONG_BUY)
    sl_distance = atr_val * 1.5  # 1.5 × ATR for stop loss

    if is_long:
        stop_loss = price - sl_distance
        tp1 = price + sl_distance * cfg.risk_reward_tp1
        tp2 = price + sl_distance * cfg.risk_reward_tp2
        tp3 = price + sl_distance * cfg.risk_reward_tp3
    else:
        stop_loss = price + sl_distance
        tp1 = price - sl_distance * cfg.risk_reward_tp1
        tp2 = price - sl_distance * cfg.risk_reward_tp2
        tp3 = price - sl_distance * cfg.risk_reward_tp3

    return Signal(
        signal=sig,
        price=price,
        entry=price,
        stop_loss=round(stop_loss, 2),
        tp1=round(tp1, 2),
        tp2=round(tp2, 2),
        tp3=round(tp3, 2),
        risk_pips=sl_distance,
        rsi_value=rsi_val,
        ema_fast=ef,
        ema_slow=es,
        ema_trend=et if et is not None else 0.0,
        macd_hist=mhist if mhist is not None else 0.0,
        atr_value=atr_val,
        reasons=reasons,
        timeframe=timeframe,
    )
