"""Walk-forward backtester for the XAU/USD signal engine.

Usage:
    python backtest.py          # print summary to stdout
    python backtest.py --html   # generate docs/index.html
"""
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))

import requests

from fetcher import Candle
from signal_engine import SignalType, analyze

# ── Constants ─────────────────────────────────────────────────────────────────

WARMUP      = 55   # bars needed before first signal (covers EMA50 + MACD warm-up)
MIN_GAP     = 4    # minimum bars between consecutive signals
MAX_HOLDING = 48   # force-exit after this many bars

_YF_SYMBOL = "GC%3DF"

# ── Data fetching ─────────────────────────────────────────────────────────────

def _fetch_yahoo_60d_hourly() -> list[Candle]:
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

    try:
        session.get(f"https://finance.yahoo.com/quote/{_YF_SYMBOL}", timeout=8)
    except Exception:
        pass

    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{_YF_SYMBOL}"
        f"?interval=1h&range=60d&includePrePost=false"
    )
    resp = session.get(url, timeout=15)
    resp.raise_for_status()

    result = resp.json().get("chart", {}).get("result", [])
    if not result:
        raise ValueError("Yahoo Finance returned empty result")

    r          = result[0]
    timestamps = r.get("timestamp", [])
    quote      = r.get("indicators", {}).get("quote", [{}])[0]

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
    return candles


def _fetch_stooq_fallback() -> list[Candle]:
    url = "https://stooq.com/q/d/l/?s=xauusd&i=d"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; GoldBot/1.0)",
        "Accept": "text/csv,text/plain,*/*",
    }
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()

    lines = resp.text.strip().splitlines()
    if len(lines) < 2:
        raise ValueError("Stooq returned insufficient data")

    candles = []
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) < 5:
            continue
        try:
            candles.append(Candle(
                timestamp=parts[0].strip(),
                open=float(parts[1]),
                high=float(parts[2]),
                low=float(parts[3]),
                close=float(parts[4]),
                volume=float(parts[5]) if len(parts) > 5 and parts[5].strip() else 0.0,
            ))
        except (ValueError, IndexError):
            continue

    candles.sort(key=lambda c: c.timestamp)
    return candles


def fetch_backtest_data() -> list[Candle]:
    try:
        candles = _fetch_yahoo_60d_hourly()
        if candles:
            print(f"[data] Yahoo Finance: {len(candles)} hourly bars")
            return candles
    except Exception as e:
        print(f"[data] Yahoo Finance failed: {e}")

    try:
        candles = _fetch_stooq_fallback()
        if candles:
            print(f"[data] Stooq fallback: {len(candles)} daily bars")
            return candles
    except Exception as e:
        print(f"[data] Stooq failed: {e}")

    return []


# ── Trade simulation ──────────────────────────────────────────────────────────

@dataclass
class Trade:
    bar_index:   int
    timestamp:   str
    direction:   str        # "LONG" or "SHORT"
    signal_type: str
    entry:       float
    stop_loss:   float
    tp1:         float
    tp2:         float
    tp3:         float
    exit_price:  float = 0.0
    result:      str   = ""    # win_tp1/win_tp2/win_tp3/loss/timeout
    pnl_r:       float = 0.0


def _simulate_trade(trade: Trade, candles: list[Candle], start_bar: int) -> Trade:
    """Walk forward bar-by-bar; check SL first (pessimistic), then best TP hit."""
    is_long = trade.direction == "LONG"
    risk    = abs(trade.entry - trade.stop_loss)

    for i in range(start_bar, min(start_bar + MAX_HOLDING, len(candles))):
        bar = candles[i]
        low, high = bar.low, bar.high

        # Pessimistic: SL checked before TP within same bar
        if is_long:
            if low <= trade.stop_loss:
                trade.exit_price = trade.stop_loss
                trade.result     = "loss"
                trade.pnl_r      = -1.0
                return trade
            if high >= trade.tp3:
                trade.exit_price = trade.tp3
                trade.result     = "win_tp3"
                trade.pnl_r      = round((trade.tp3 - trade.entry) / risk, 2)
                return trade
            if high >= trade.tp2:
                trade.exit_price = trade.tp2
                trade.result     = "win_tp2"
                trade.pnl_r      = round((trade.tp2 - trade.entry) / risk, 2)
                return trade
            if high >= trade.tp1:
                trade.exit_price = trade.tp1
                trade.result     = "win_tp1"
                trade.pnl_r      = round((trade.tp1 - trade.entry) / risk, 2)
                return trade
        else:
            if high >= trade.stop_loss:
                trade.exit_price = trade.stop_loss
                trade.result     = "loss"
                trade.pnl_r      = -1.0
                return trade
            if low <= trade.tp3:
                trade.exit_price = trade.tp3
                trade.result     = "win_tp3"
                trade.pnl_r      = round((trade.entry - trade.tp3) / risk, 2)
                return trade
            if low <= trade.tp2:
                trade.exit_price = trade.tp2
                trade.result     = "win_tp2"
                trade.pnl_r      = round((trade.entry - trade.tp2) / risk, 2)
                return trade
            if low <= trade.tp1:
                trade.exit_price = trade.tp1
                trade.result     = "win_tp1"
                trade.pnl_r      = round((trade.entry - trade.tp1) / risk, 2)
                return trade

    # Timeout — exit at last bar close
    last = candles[min(start_bar + MAX_HOLDING - 1, len(candles) - 1)]
    trade.exit_price = last.close
    trade.result     = "timeout"
    if is_long:
        trade.pnl_r = round((last.close - trade.entry) / risk, 2)
    else:
        trade.pnl_r = round((trade.entry - last.close) / risk, 2)
    return trade


# ── Walk-forward engine ───────────────────────────────────────────────────────

def run_backtest(candles: list[Candle]) -> list[Trade]:
    trades:       list[Trade] = []
    last_signal_bar = -MIN_GAP  # allow signal at bar WARMUP on first iteration

    end_bar = len(candles) - MAX_HOLDING  # leave room for trade resolution

    for i in range(WARMUP, end_bar):
        if i - last_signal_bar < MIN_GAP:
            continue

        # Strict look-ahead prevention: pass only candles[0:i+1]
        signal = analyze(candles[:i + 1], timeframe="1h")
        if signal is None:
            continue
        if signal.signal in (SignalType.HOLD,):
            continue

        direction = (
            "LONG"  if signal.signal in (SignalType.BUY, SignalType.STRONG_BUY)
            else "SHORT"
        )

        trade = Trade(
            bar_index   = i,
            timestamp   = candles[i].timestamp,
            direction   = direction,
            signal_type = signal.signal.name,
            entry       = signal.entry,
            stop_loss   = signal.stop_loss,
            tp1         = signal.tp1,
            tp2         = signal.tp2,
            tp3         = signal.tp3,
        )

        _simulate_trade(trade, candles, start_bar=i + 1)
        trades.append(trade)
        last_signal_bar = i

    return trades


# ── Metrics ───────────────────────────────────────────────────────────────────

@dataclass
class Metrics:
    total_trades:  int
    win_rate:      float
    profit_factor: float
    total_r:       float
    max_drawdown:  float
    avg_win_r:     float
    avg_loss_r:    float
    equity_curve:  list[float]


def compute_metrics(trades: list[Trade]) -> Metrics:
    if not trades:
        return Metrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, [])

    wins   = [t for t in trades if t.pnl_r > 0]
    losses = [t for t in trades if t.pnl_r <= 0]

    win_rate      = len(wins) / len(trades)
    gross_profit  = sum(t.pnl_r for t in wins)
    gross_loss    = abs(sum(t.pnl_r for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    total_r       = sum(t.pnl_r for t in trades)
    avg_win_r     = gross_profit / len(wins)  if wins   else 0.0
    avg_loss_r    = gross_loss   / len(losses) if losses else 0.0

    # Equity curve and max drawdown
    equity  = 0.0
    peak    = 0.0
    max_dd  = 0.0
    curve   = []
    for t in trades:
        equity += t.pnl_r
        curve.append(round(equity, 4))
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd

    return Metrics(
        total_trades  = len(trades),
        win_rate      = round(win_rate, 4),
        profit_factor = round(profit_factor, 4),
        total_r       = round(total_r, 4),
        max_drawdown  = round(max_dd, 4),
        avg_win_r     = round(avg_win_r, 4),
        avg_loss_r    = round(avg_loss_r, 4),
        equity_curve  = curve,
    )


# ── HTML report generation ────────────────────────────────────────────────────

def _fmt_ts(ts: str) -> str:
    """Format a Unix timestamp string or date string for display."""
    try:
        unix = int(float(ts))
        dt   = datetime.fromtimestamp(unix, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, OSError):
        return ts[:16] if len(ts) >= 16 else ts


def _color(condition: bool) -> str:
    return "#26a641" if condition else "#f85149"


def generate_html(trades: list[Trade], metrics: Metrics, candles: list[Candle]) -> str:
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if candles:
        period_start = _fmt_ts(candles[0].timestamp)
        period_end   = _fmt_ts(candles[-1].timestamp)
        data_period  = f"{period_start} → {period_end}"
    else:
        data_period = "N/A"

    if not trades:
        return _insufficient_data_html(now, data_period)

    # ── Prepare chart data ────────────────────────────────────────────────────
    equity_labels = [str(i + 1) for i in range(len(metrics.equity_curve))]
    equity_data   = metrics.equity_curve

    outcome_counts = {k: 0 for k in ("win_tp1", "win_tp2", "win_tp3", "loss", "timeout")}
    for t in trades:
        outcome_counts[t.result] = outcome_counts.get(t.result, 0) + 1

    signal_counts: dict[str, int] = {}
    for t in trades:
        signal_counts[t.signal_type] = signal_counts.get(t.signal_type, 0) + 1

    long_count  = sum(1 for t in trades if t.direction == "LONG")
    short_count = sum(1 for t in trades if t.direction == "SHORT")

    # Trade log rows
    rows_html = ""
    for idx, t in enumerate(trades, 1):
        dir_arrow = "▲" if t.direction == "LONG" else "▼"
        dir_color = "#26a641" if t.direction == "LONG" else "#f85149"
        sig_label = t.signal_type.replace("_", " ")

        result_colors = {
            "win_tp1": "#26a641", "win_tp2": "#388bfd",
            "win_tp3": "#ffd700", "loss": "#f85149", "timeout": "#8b949e",
        }
        res_color = result_colors.get(t.result, "#8b949e")
        res_label = t.result.replace("_", " ").upper()

        pnl_color = "#26a641" if t.pnl_r > 0 else ("#f85149" if t.pnl_r < 0 else "#8b949e")
        pnl_sign  = "+" if t.pnl_r > 0 else ""

        rows_html += f"""
        <tr>
          <td style="color:#8b949e">{idx}</td>
          <td>{_fmt_ts(t.timestamp)}</td>
          <td style="color:{dir_color};font-weight:700">{dir_arrow} {t.direction}</td>
          <td style="color:#e6edf3">{sig_label}</td>
          <td>${t.entry:,.2f}</td>
          <td style="color:#f85149">${t.stop_loss:,.2f}</td>
          <td style="color:#26a641">${t.tp1:,.2f}</td>
          <td>${t.exit_price:,.2f}</td>
          <td><span style="background:{res_color}22;color:{res_color};padding:2px 8px;border-radius:4px;font-size:0.75rem;font-weight:600">{res_label}</span></td>
          <td style="color:{pnl_color};font-weight:600">{pnl_sign}{t.pnl_r:.2f}R</td>
        </tr>"""

    # ── Metric card values ────────────────────────────────────────────────────
    wr_color  = _color(metrics.win_rate > 0.50)
    pf_color  = _color(metrics.profit_factor > 1.5)
    tr_color  = _color(metrics.total_r > 0)

    wr_pct    = f"{metrics.win_rate * 100:.1f}%"
    pf_str    = f"{metrics.profit_factor:.2f}" if metrics.profit_factor != float("inf") else "∞"
    tr_sign   = "+" if metrics.total_r > 0 else ""

    # ── JSON payloads for Chart.js ────────────────────────────────────────────
    equity_labels_json = json.dumps(equity_labels)
    equity_data_json   = json.dumps(equity_data)

    outcome_data_json  = json.dumps([
        outcome_counts["win_tp1"], outcome_counts["win_tp2"],
        outcome_counts["win_tp3"], outcome_counts["loss"], outcome_counts["timeout"],
    ])

    tp_labels = ["TP1 Wins", "TP2 Wins", "TP3 Wins"]
    tp_values = [outcome_counts["win_tp1"], outcome_counts["win_tp2"], outcome_counts["win_tp3"]]

    sig_labels_json = json.dumps(list(signal_counts.keys()))
    sig_values_json = json.dumps(list(signal_counts.values()))
    tp_values_json  = json.dumps(tp_values)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>XAU/USD Signal Backtest Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  :root {{
    --bg:      #0d1117;
    --card:    #161b22;
    --border:  #30363d;
    --text:    #e6edf3;
    --muted:   #8b949e;
    --gold:    #ffd700;
    --green:   #26a641;
    --red:     #f85149;
    --blue:    #388bfd;
  }}
  html {{ font-size: 16px; }}
  body {{ background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; padding: 24px; }}

  .header {{ text-align: center; padding: 32px 0 24px; border-bottom: 1px solid var(--border); margin-bottom: 32px; }}
  .header h1 {{ font-size: 2rem; font-weight: 700; color: var(--gold); letter-spacing: -0.5px; }}
  .header p  {{ color: var(--muted); margin-top: 8px; font-size: 0.875rem; }}
  .header .updated {{ margin-top: 6px; font-size: 0.75rem; color: var(--border); }}

  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 28px; }}
  .card  {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }}
  .card .label {{ font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin-bottom: 8px; }}
  .card .value {{ font-size: 1.5rem; font-weight: 700; line-height: 1; }}
  .card .sub   {{ font-size: 0.75rem; color: var(--muted); margin-top: 4px; }}

  .row {{ display: grid; gap: 16px; margin-bottom: 16px; }}
  .row-2 {{ grid-template-columns: 2fr 1fr; }}
  .row-3 {{ grid-template-columns: repeat(3, 1fr); }}
  @media (max-width: 900px) {{ .row-2, .row-3 {{ grid-template-columns: 1fr; }} }}

  .panel {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 20px; }}
  .panel h3 {{ font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.07em; color: var(--muted); margin-bottom: 16px; }}
  .chart-wrap {{ position: relative; }}
  .chart-wrap.tall {{ height: 260px; }}
  .chart-wrap.medium {{ height: 220px; }}

  .table-wrap {{ overflow-x: auto; margin-top: 16px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.8rem; }}
  thead th {{ text-align: left; padding: 8px 12px; color: var(--muted); font-weight: 600; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; border-bottom: 1px solid var(--border); }}
  tbody td {{ padding: 8px 12px; border-bottom: 1px solid #21262d; color: var(--muted); white-space: nowrap; }}
  tbody tr:hover {{ background: #1c2128; }}
  tbody tr:last-child td {{ border-bottom: none; }}
</style>
</head>
<body>

<div class="header">
  <h1>🥇 XAU/USD Signal Backtest Report</h1>
  <p>EMA 9/21/50 · RSI 14 · MACD 12/26/9 · Bollinger Bands 20,2 · ATR 14 (SL=1.5×ATR · TP1=1.5R · TP2=2.5R · TP3=4R)</p>
  <p class="updated">Last updated: {now} · Data: {data_period}</p>
</div>

<div class="cards">
  <div class="card">
    <div class="label">Total Trades</div>
    <div class="value" style="color:var(--text)">{metrics.total_trades}</div>
    <div class="sub">signals traded</div>
  </div>
  <div class="card">
    <div class="label">Win Rate</div>
    <div class="value" style="color:{wr_color}">{wr_pct}</div>
    <div class="sub">{sum(1 for t in trades if t.pnl_r > 0)} wins / {sum(1 for t in trades if t.pnl_r <= 0)} losses</div>
  </div>
  <div class="card">
    <div class="label">Profit Factor</div>
    <div class="value" style="color:{pf_color}">{pf_str}</div>
    <div class="sub">gross profit / gross loss</div>
  </div>
  <div class="card">
    <div class="label">Total R</div>
    <div class="value" style="color:{tr_color}">{tr_sign}{metrics.total_r:.2f}R</div>
    <div class="sub">cumulative R-multiples</div>
  </div>
  <div class="card">
    <div class="label">Max Drawdown</div>
    <div class="value" style="color:var(--red)">{metrics.max_drawdown:.2f}R</div>
    <div class="sub">peak-to-trough</div>
  </div>
  <div class="card">
    <div class="label">Avg Win</div>
    <div class="value" style="color:var(--green)">+{metrics.avg_win_r:.2f}R</div>
    <div class="sub">per winning trade</div>
  </div>
  <div class="card">
    <div class="label">Avg Loss</div>
    <div class="value" style="color:var(--red)">-{metrics.avg_loss_r:.2f}R</div>
    <div class="sub">per losing trade</div>
  </div>
  <div class="card">
    <div class="label">Data Period</div>
    <div class="value" style="color:var(--text);font-size:0.95rem">{len(candles)}</div>
    <div class="sub">hourly bars (~60 days)</div>
  </div>
</div>

<div class="row row-2">
  <div class="panel">
    <h3>Equity Curve (cumulative R)</h3>
    <div class="chart-wrap tall"><canvas id="equityChart"></canvas></div>
  </div>
  <div class="panel">
    <h3>Trade Outcomes</h3>
    <div class="chart-wrap tall"><canvas id="outcomeChart"></canvas></div>
  </div>
</div>

<div class="row row-3">
  <div class="panel">
    <h3>Win by TP Level</h3>
    <div class="chart-wrap medium"><canvas id="tpChart"></canvas></div>
  </div>
  <div class="panel">
    <h3>Signal Type Breakdown</h3>
    <div class="chart-wrap medium"><canvas id="signalChart"></canvas></div>
  </div>
  <div class="panel">
    <h3>Long vs Short</h3>
    <div class="chart-wrap medium"><canvas id="dirChart"></canvas></div>
  </div>
</div>

<div class="panel">
  <h3>Trade Log</h3>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>#</th><th>Time</th><th>Direction</th><th>Signal</th>
          <th>Entry</th><th>SL</th><th>TP1</th><th>Exit</th><th>Result</th><th>P&amp;L (R)</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>
</div>

<script>
Chart.defaults.color = '#8b949e';
Chart.defaults.borderColor = '#30363d';

const gridColor = '#21262d';
const tickColor = '#8b949e';

// ── Equity Curve ──────────────────────────────────────────────────────────────
new Chart(document.getElementById('equityChart'), {{
  type: 'line',
  data: {{
    labels: {equity_labels_json},
    datasets: [{{
      label: 'Equity (R)',
      data: {equity_data_json},
      borderColor: '#ffd700',
      backgroundColor: 'rgba(255,215,0,0.08)',
      borderWidth: 2,
      pointRadius: 0,
      pointHoverRadius: 4,
      fill: true,
      tension: 0.3,
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }}, tooltip: {{ mode: 'index', intersect: false }} }},
    scales: {{
      x: {{ display: false }},
      y: {{
        grid: {{ color: gridColor }},
        ticks: {{ color: tickColor, callback: v => v + 'R' }},
      }}
    }}
  }}
}});

// ── Trade Outcomes doughnut ───────────────────────────────────────────────────
new Chart(document.getElementById('outcomeChart'), {{
  type: 'doughnut',
  data: {{
    labels: ['TP1 Hit', 'TP2 Hit', 'TP3 Hit', 'Stop Loss', 'Timeout'],
    datasets: [{{
      data: {outcome_data_json},
      backgroundColor: ['#26a641','#388bfd','#ffd700','#f85149','#8b949e'],
      borderColor: '#161b22',
      borderWidth: 2,
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{ position: 'bottom', labels: {{ padding: 12, font: {{ size: 11 }} }} }},
    }}
  }}
}});

// ── TP Bar Chart ──────────────────────────────────────────────────────────────
new Chart(document.getElementById('tpChart'), {{
  type: 'bar',
  data: {{
    labels: ['TP1', 'TP2', 'TP3'],
    datasets: [{{
      label: 'Wins',
      data: {tp_values_json},
      backgroundColor: ['#26a641','#388bfd','#ffd700'],
      borderRadius: 4,
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ grid: {{ display: false }}, ticks: {{ color: tickColor }} }},
      y: {{ grid: {{ color: gridColor }}, ticks: {{ color: tickColor, stepSize: 1 }} }},
    }}
  }}
}});

// ── Signal Type doughnut ──────────────────────────────────────────────────────
new Chart(document.getElementById('signalChart'), {{
  type: 'doughnut',
  data: {{
    labels: {sig_labels_json},
    datasets: [{{
      data: {sig_values_json},
      backgroundColor: ['#ffd700','#26a641','#388bfd','#f85149','#da3633'],
      borderColor: '#161b22',
      borderWidth: 2,
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{ position: 'bottom', labels: {{ padding: 10, font: {{ size: 10 }} }} }},
    }}
  }}
}});

// ── Long vs Short doughnut ────────────────────────────────────────────────────
new Chart(document.getElementById('dirChart'), {{
  type: 'doughnut',
  data: {{
    labels: ['Long', 'Short'],
    datasets: [{{
      data: [{long_count}, {short_count}],
      backgroundColor: ['#26a641','#f85149'],
      borderColor: '#161b22',
      borderWidth: 2,
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{ position: 'bottom', labels: {{ padding: 12, font: {{ size: 11 }} }} }},
    }}
  }}
}});
</script>
</body>
</html>
"""


def _insufficient_data_html(now: str, data_period: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>XAU/USD Backtest Report</title>
<style>
  body {{ background: #0d1117; color: #e6edf3; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         display: flex; align-items: center; justify-content: center; min-height: 100vh; text-align: center; }}
  .box {{ background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 48px 64px; }}
  h1   {{ color: #ffd700; font-size: 1.5rem; margin-bottom: 16px; }}
  p    {{ color: #8b949e; }}
  .ts  {{ margin-top: 24px; font-size: 0.75rem; color: #30363d; }}
</style>
</head>
<body>
  <div class="box">
    <h1>🥇 XAU/USD Signal Backtest Report</h1>
    <p>Insufficient data to run backtest — no trades were generated.</p>
    <p>Data period: {data_period}</p>
    <p class="ts">Last updated: {now}</p>
  </div>
</body>
</html>
"""


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="XAU/USD walk-forward backtester")
    parser.add_argument("--html", action="store_true", help="Generate HTML report")
    args = parser.parse_args()

    candles = fetch_backtest_data()
    if not candles:
        print("ERROR: no candle data available — all sources failed")
        sys.exit(1)

    print(f"[backtest] Running on {len(candles)} bars …")
    trades  = run_backtest(candles)
    metrics = compute_metrics(trades)

    # ── stdout summary ────────────────────────────────────────────────────────
    print("\n══════════════════════════════════════════")
    print("  XAU/USD Walk-Forward Backtest Summary")
    print("══════════════════════════════════════════")
    print(f"  Bars used    : {len(candles)}")
    print(f"  Total trades : {metrics.total_trades}")

    if metrics.total_trades == 0:
        print("  No trades were generated (market ranged or all HOLD signals)")
    else:
        pf_str = f"{metrics.profit_factor:.2f}" if metrics.profit_factor != float("inf") else "∞"
        print(f"  Win rate     : {metrics.win_rate * 100:.1f}%")
        print(f"  Profit factor: {pf_str}")
        print(f"  Total R      : {metrics.total_r:+.2f}R")
        print(f"  Max drawdown : {metrics.max_drawdown:.2f}R")
        print(f"  Avg win      : +{metrics.avg_win_r:.2f}R")
        print(f"  Avg loss     : -{metrics.avg_loss_r:.2f}R")
    print("══════════════════════════════════════════\n")

    if args.html:
        out_dir = os.path.join(os.path.dirname(__file__), "..", "docs")
        out_dir = os.path.normpath(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "index.html")

        html = generate_html(trades, metrics, candles)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"[report] HTML written to: {out_path}")


if __name__ == "__main__":
    main()
