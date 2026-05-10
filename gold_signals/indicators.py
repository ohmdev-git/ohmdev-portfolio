"""Pure-Python technical indicators — no external dependencies required."""
from dataclasses import dataclass
from typing import Optional


def _ema(values: list[float], period: int) -> list[Optional[float]]:
    result: list[Optional[float]] = [None] * len(values)
    k = 2.0 / (period + 1)
    # Seed with simple average of first `period` values
    if len(values) < period:
        return result
    seed = sum(values[:period]) / period
    result[period - 1] = seed
    for i in range(period, len(values)):
        result[i] = values[i] * k + result[i - 1] * (1 - k)
    return result


def _sma(values: list[float], period: int) -> list[Optional[float]]:
    result: list[Optional[float]] = [None] * len(values)
    for i in range(period - 1, len(values)):
        result[i] = sum(values[i - period + 1 : i + 1]) / period
    return result


def ema(values: list[float], period: int) -> list[Optional[float]]:
    return _ema(values, period)


def rsi(values: list[float], period: int = 14) -> list[Optional[float]]:
    result: list[Optional[float]] = [None] * len(values)
    if len(values) <= period:
        return result
    gains, losses = [], []
    for i in range(1, len(values)):
        delta = values[i] - values[i - 1]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))
    # Initial average gain/loss
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(values)):
        idx = i - period  # index into gains/losses
        avg_gain = (avg_gain * (period - 1) + gains[idx]) / period
        avg_loss = (avg_loss * (period - 1) + losses[idx]) / period
        if avg_loss == 0:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100.0 - 100.0 / (1 + rs)
    return result


@dataclass
class MACDResult:
    macd: list[Optional[float]]
    signal: list[Optional[float]]
    histogram: list[Optional[float]]


def macd(
    values: list[float], fast: int = 12, slow: int = 26, signal_period: int = 9
) -> MACDResult:
    ema_fast = _ema(values, fast)
    ema_slow = _ema(values, slow)
    macd_line: list[Optional[float]] = []
    for f, s in zip(ema_fast, ema_slow):
        if f is None or s is None:
            macd_line.append(None)
        else:
            macd_line.append(f - s)

    # Compute signal line as EMA of macd_line (skip Nones)
    valid_indices = [i for i, v in enumerate(macd_line) if v is not None]
    signal_line: list[Optional[float]] = [None] * len(values)
    if len(valid_indices) >= signal_period:
        valid_vals = [macd_line[i] for i in valid_indices]
        valid_ema = _ema(valid_vals, signal_period)
        for j, vi in enumerate(valid_indices):
            signal_line[vi] = valid_ema[j]

    histogram: list[Optional[float]] = []
    for m, s in zip(macd_line, signal_line):
        if m is None or s is None:
            histogram.append(None)
        else:
            histogram.append(m - s)

    return MACDResult(macd=macd_line, signal=signal_line, histogram=histogram)


@dataclass
class BBResult:
    upper: list[Optional[float]]
    middle: list[Optional[float]]
    lower: list[Optional[float]]


def bollinger_bands(values: list[float], period: int = 20, std_dev: float = 2.0) -> BBResult:
    middle = _sma(values, period)
    upper: list[Optional[float]] = [None] * len(values)
    lower: list[Optional[float]] = [None] * len(values)
    for i in range(period - 1, len(values)):
        window = values[i - period + 1 : i + 1]
        mean = middle[i]
        variance = sum((x - mean) ** 2 for x in window) / period
        std = variance ** 0.5
        upper[i] = mean + std_dev * std
        lower[i] = mean - std_dev * std
    return BBResult(upper=upper, middle=middle, lower=lower)


def atr(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> list[Optional[float]]:
    """Average True Range."""
    trs: list[float] = []
    for i in range(len(highs)):
        if i == 0:
            trs.append(highs[i] - lows[i])
        else:
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            trs.append(tr)
    result: list[Optional[float]] = [None] * len(highs)
    if len(trs) < period:
        return result
    result[period - 1] = sum(trs[:period]) / period
    for i in range(period, len(trs)):
        result[i] = (result[i - 1] * (period - 1) + trs[i]) / period
    return result
