"""Multi-channel notifier — Telegram, Discord, LINE Notify."""
import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)


def _escape_markdown(text: str) -> str:
    """Escape Telegram MarkdownV2 special chars."""
    specials = r"_*[]()~`>#+-=|{}.!"
    for ch in specials:
        text = text.replace(ch, f"\\{ch}")
    return text


def send_telegram(token: str, chat_id: str, message: str) -> bool:
    """Send message via Telegram Bot API (Markdown parse mode)."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        logger.info("Telegram: sent successfully")
        return True
    except Exception as e:
        logger.error("Telegram send failed: %s", e)
        return False


def send_discord(webhook_url: str, message: str, username: str = "Gold Signal Bot") -> bool:
    """Send message to a Discord channel via webhook."""
    # Discord doesn't render Telegram markdown the same — strip * and `
    clean = message.replace("*", "**").replace("`", "`")
    payload = {
        "username": username,
        "content": clean,
    }
    try:
        resp = requests.post(webhook_url, json=payload, timeout=10)
        resp.raise_for_status()
        logger.info("Discord: sent successfully")
        return True
    except Exception as e:
        logger.error("Discord send failed: %s", e)
        return False


def send_line(token: str, message: str) -> bool:
    """Send message via LINE Notify."""
    # Strip Telegram markdown asterisks/backticks for plain text
    clean = message.replace("*", "").replace("`", "")
    url = "https://notify-api.line.me/api/notify"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        resp = requests.post(url, headers=headers, data={"message": "\n" + clean}, timeout=10)
        resp.raise_for_status()
        logger.info("LINE Notify: sent successfully")
        return True
    except Exception as e:
        logger.error("LINE Notify send failed: %s", e)
        return False


def notify_all(message: str, cfg=None) -> dict[str, bool]:
    """Send to all configured channels. Returns per-channel success status."""
    if cfg is None:
        from config import config as cfg

    results: dict[str, bool] = {}

    if cfg.telegram_token and cfg.telegram_chat_id:
        results["telegram"] = send_telegram(cfg.telegram_token, cfg.telegram_chat_id, message)
    else:
        logger.debug("Telegram not configured — skipping")

    if cfg.discord_webhook_url:
        results["discord"] = send_discord(cfg.discord_webhook_url, message)
    else:
        logger.debug("Discord not configured — skipping")

    if cfg.line_notify_token:
        results["line"] = send_line(cfg.line_notify_token, message)
    else:
        logger.debug("LINE Notify not configured — skipping")

    if not results:
        logger.warning("No notification channel configured. Set at least one of: "
                       "TELEGRAM_BOT_TOKEN+TELEGRAM_CHAT_ID, DISCORD_WEBHOOK_URL, LINE_NOTIFY_TOKEN")
        # Print to stdout so the signal is never silently lost
        print("\n" + "=" * 60)
        print(message)
        print("=" * 60 + "\n")

    return results
