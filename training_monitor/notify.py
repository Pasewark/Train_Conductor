"""Telegram and Pushover notification helpers."""

import os
import sys

try:
    import requests
except ImportError:
    requests = None


def send_telegram(message: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise RuntimeError("Telegram env vars not set (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)")
    if requests is None:
        raise RuntimeError("Missing dependency: requests (pip install requests)")

    MAX_LEN = 4000
    if len(message) > MAX_LEN:
        message = message[:MAX_LEN] + "\n\n[message truncated]"

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    resp = requests.post(
        url,
        data={"chat_id": chat_id, "text": message, "disable_web_page_preview": True},
        timeout=10,
    )
    resp.raise_for_status()


def send_pushover(message: str) -> None:
    user_key = os.getenv("PUSHOVER_USER_KEY")
    app_token = os.getenv("PUSHOVER_APP_TOKEN")
    if not user_key or not app_token:
        raise RuntimeError("Pushover env vars not set (PUSHOVER_USER_KEY, PUSHOVER_APP_TOKEN)")
    if requests is None:
        raise RuntimeError("Missing dependency: requests (pip install requests)")

    url = "https://api.pushover.net/1/messages.json"
    resp = requests.post(
        url,
        data={
            "token": app_token,
            "user": user_key,
            "title": "Training Monitor",
            "message": message,
        },
        timeout=10,
    )
    resp.raise_for_status()


def notify(message: str, level: str = "info", allow_telegram: bool = True) -> None:
    prefix = "🚨 " if level == "alert" else "ℹ️ " if level == "info" else ""
    full_message = prefix + message

    try:
        if allow_telegram and os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"):
            send_telegram(full_message)
            return
        if os.getenv("PUSHOVER_USER_KEY") and os.getenv("PUSHOVER_APP_TOKEN"):
            send_pushover(full_message)
            return
    except Exception as e:
        print(f"[monitor] Notification failed: {e}", file=sys.stderr)

    print(full_message, file=sys.stderr)
