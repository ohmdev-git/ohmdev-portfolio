import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    # Crypto.com exchange API
    exchange_base_url: str = "https://api.crypto.com/exchange/v1"
    instrument: str = "XAUUSDPERP"

    # Timeframes to analyze
    primary_tf: str = "1h"   # Main signal timeframe (used by agent.py)
    higher_tf: str = "4h"    # Trend confirmation

    # Timeframes the bot sends signals for — each analyzed & notified independently.
    # Override via env, e.g. SIGNAL_TIMEFRAMES="5m,1h,4h"
    signal_timeframes: list = field(default_factory=lambda: [
        tf.strip() for tf in os.getenv("SIGNAL_TIMEFRAMES", "5m,1h").split(",") if tf.strip()
    ])

    # Technical indicator settings
    ema_fast: int = 9
    ema_slow: int = 21
    ema_trend: int = 50
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0

    # Risk management
    atr_period: int = 14
    risk_reward_tp1: float = 1.5
    risk_reward_tp2: float = 2.5
    risk_reward_tp3: float = 4.0

    # Scheduler interval (seconds)
    check_interval: int = int(os.getenv("CHECK_INTERVAL", "300"))  # 5 minutes

    # Notification channels (enabled via env)
    telegram_token: Optional[str] = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN"))
    telegram_chat_id: Optional[str] = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID"))
    discord_webhook_url: Optional[str] = field(default_factory=lambda: os.getenv("DISCORD_WEBHOOK_URL"))
    line_notify_token: Optional[str] = field(default_factory=lambda: os.getenv("LINE_NOTIFY_TOKEN"))

    # Signal cooldown - don't resend same signal within N seconds (fallback default)
    signal_cooldown: int = int(os.getenv("SIGNAL_COOLDOWN", "3600"))  # 1 hour

    # Per-timeframe cooldown (seconds) — shorter TFs fire more often.
    # Falls back to signal_cooldown for any timeframe not listed here.
    signal_cooldowns: dict = field(default_factory=lambda: {"5m": 900, "15m": 1800, "1h": 3600, "4h": 14400})

    def cooldown_for(self, tf: str) -> int:
        return self.signal_cooldowns.get(tf, self.signal_cooldown)

    # ── AI-Trader (ai4trade.ai) ────────────────────────────────────────────────
    ai_trader_enabled: bool = os.getenv("AI_TRADER_ENABLED", "false").lower() == "true"
    ai_trader_email: Optional[str] = field(default_factory=lambda: os.getenv("AI_TRADER_EMAIL"))
    ai_trader_password: Optional[str] = field(default_factory=lambda: os.getenv("AI_TRADER_PASSWORD"))
    ai_trader_symbol: str = os.getenv("AI_TRADER_SYMBOL", "XAUUSD")
    ai_trader_market_type: str = os.getenv("AI_TRADER_MARKET_TYPE", "crypto")


config = Config()
