"""Interactive Telegram chat: reply to the bot to ask questions about training."""

import os
import sys
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional

try:
    import requests
except ImportError:
    requests = None

from .analyzer import LLMAnalyzer


class TelegramChatHandler:
    """
    Polls Telegram for incoming messages and responds with LLM-generated answers
    grounded in the current training context (metrics, system state, analyses).

    Runs on a background thread inside TrainingMonitor.
    """

    CHAT_SYSTEM_PROMPT = """\
You are an expert ML engineer monitoring a live neural network training run. \
The user is messaging you via Telegram or the web chat to ask questions about the ongoing training.

You have access to the current training context (metrics, system state, recent analyses, config). \
Use this context to answer the user's questions accurately and concisely.

Keep responses concise — they'll be read on a phone. Use numbers and specifics when relevant.
If you don't have enough information to answer, say so.

You can write and read long-term memories with memory_write and memory_read.
Use memory_search to find relevant memories by text query.
Memory categories: bugs, issues, suggestions, anomalies, messages_from_user, other."""

    def __init__(
        self,
        analyzer: LLMAnalyzer,
        poll_interval_sec: float = 2.0,
        max_chat_history: int = 20,
        tool_manager: Optional[Any] = None,
        llm_lock: Optional[threading.Lock] = None,
    ):
        self.analyzer = analyzer
        self.poll_interval = poll_interval_sec
        self.max_chat_history = max_chat_history
        self.tool_manager = tool_manager
        self.llm_lock = llm_lock

        self.token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

        if not self.token or not self.chat_id:
            raise ValueError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set")

        self._last_update_id: int = 0
        self._chat_history: Deque[Dict[str, str]] = deque(maxlen=max_chat_history)
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # Set by TrainingMonitor to share live state
        self.get_context_fn: Optional[Callable[[], str]] = None

        # Callback to notify the monitor of each message exchange.
        # Signature: on_exchange_fn(role: str, text: str, msg_type: str)
        #   role: "user" or "model"
        #   msg_type: "user_message" or "chat_response"
        self.on_exchange_fn: Optional[Callable[[str, str, str], None]] = None

    # --------------------------------------------------------------------- #
    #  Telegram API helpers
    # --------------------------------------------------------------------- #

    def _telegram_api(self, method: str, data: Optional[Dict] = None) -> Optional[Any]:
        url = f"https://api.telegram.org/bot{self.token}/{method}"
        try:
            resp = requests.post(url, json=data or {}, timeout=15)
            resp.raise_for_status()
            result = resp.json()
            if result.get("ok"):
                return result.get("result")
            # Log the actual error description from Telegram
            desc = result.get("description", "unknown error")
            print(f"[telegram-chat] API {method} returned ok=false: {desc}", file=sys.stderr)
            return None
        except Exception as e:
            print(f"[telegram-chat] API error ({method}): {e}", file=sys.stderr)
            return None

    def _send_reply(self, text: str, reply_to_message_id: Optional[int] = None) -> None:
        MAX_LEN = 4000
        if len(text) > MAX_LEN:
            text = text[:MAX_LEN] + "\n\n[truncated]"

        data: Dict[str, Any] = {
            "chat_id": self.chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }
        if reply_to_message_id:
            data["reply_to_message_id"] = reply_to_message_id

        self._telegram_api("sendMessage", data)

    # --------------------------------------------------------------------- #
    #  Polling
    # --------------------------------------------------------------------- #

    def _delete_webhook(self) -> None:
        """
        Delete any existing webhook so getUpdates works.

        If a webhook was ever set on this bot (even by another app), the
        Telegram API silently ignores getUpdates calls.  This is the #1
        cause of "bot receives nothing".
        """
        result = self._telegram_api("deleteWebhook", {"drop_pending_updates": False})
        if result is not None:
            print(f"[telegram-chat] Webhook cleared", file=sys.stderr)

    def _consume_old_updates(self) -> None:
        """Consume any pending updates so we don't reply to old messages on startup."""
        result = self._telegram_api("getUpdates", {"timeout": 0, "limit": 100})
        if result:
            for update in result:
                uid = update.get("update_id", 0)
                if uid >= self._last_update_id:
                    self._last_update_id = uid + 1
            print(f"[telegram-chat] Consumed {len(result)} old update(s)", file=sys.stderr)

    def _poll_updates(self) -> List[Dict[str, Any]]:
        result = self._telegram_api("getUpdates", {
            "offset": self._last_update_id,
            "timeout": 5,
            "limit": 10,
            "allowed_updates": ["message"],
        })
        if not result:
            return []

        messages: List[Dict[str, Any]] = []
        for update in result:
            uid = update.get("update_id", 0)
            if uid >= self._last_update_id:
                self._last_update_id = uid + 1

            msg = update.get("message", {})
            if str(msg.get("chat", {}).get("id", "")) != str(self.chat_id):
                continue
            text = msg.get("text", "").strip()
            if text:
                messages.append({
                    "text": text,
                    "message_id": msg.get("message_id"),
                    "from": msg.get("from", {}).get("first_name", "User"),
                })

        return messages

    # --------------------------------------------------------------------- #
    #  Message handling
    # --------------------------------------------------------------------- #

    def _build_context_prompt(self) -> str:
        if self.get_context_fn:
            return self.get_context_fn()
        return "(No training context available yet.)"

    def _handle_message(self, user_text: str, message_id: Optional[int] = None) -> None:
        self._chat_history.append({"role": "user", "content": user_text})

        # Notify monitor of the user message
        if self.on_exchange_fn:
            self.on_exchange_fn("user", user_text, "user_message")

        context = self._build_context_prompt()

        # The training context goes in the system message to avoid
        # consecutive user messages, which some APIs reject.
        system_content = (
            f"{self.CHAT_SYSTEM_PROMPT}\n\n"
            f"## Current Training Context\n\n{context}\n\n## END CONTEXT"
        )

        messages: list[Dict[str, str]] = [
            {"role": "system", "content": system_content},
        ]
        for msg in self._chat_history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        try:
            if self.llm_lock:
                with self.llm_lock:
                    reply_text = self.analyzer._call_api_for_messages(
                        messages,
                        tool_manager=self.tool_manager,
                    )
            else:
                reply_text = self.analyzer._call_api_for_messages(
                    messages,
                    tool_manager=self.tool_manager,
                )
        except Exception as e:
            print(f"[telegram-chat] LLM error: {e}", file=sys.stderr)
            reply_text = f"Sorry, I couldn't process that: {e}"

        self._chat_history.append({"role": "assistant", "content": reply_text})
        self._send_reply(reply_text, reply_to_message_id=message_id)

        # Notify monitor of the model response
        if self.on_exchange_fn:
            self.on_exchange_fn("model", reply_text, "chat_response")

    # --------------------------------------------------------------------- #
    #  Thread lifecycle
    # --------------------------------------------------------------------- #

    def _poll_loop(self) -> None:
        try:
            self._delete_webhook()
            self._consume_old_updates()
        except Exception as e:
            print(f"[telegram-chat] Startup error (continuing anyway): {e}", file=sys.stderr)

        print(f"[telegram-chat] Listening for messages in chat {self.chat_id}", file=sys.stderr)

        while self._running:
            try:
                messages = self._poll_updates()
                for msg in messages:
                    self._handle_message(msg["text"], msg.get("message_id"))
            except Exception as e:
                print(f"[telegram-chat] Poll error: {e}", file=sys.stderr)

            time.sleep(self.poll_interval)

        print(f"[telegram-chat] Polling stopped", file=sys.stderr)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="telegram-chat")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
