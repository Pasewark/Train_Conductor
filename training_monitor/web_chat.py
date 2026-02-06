"""Chainlit-based web chat for monitor conversation history."""

from __future__ import annotations

import asyncio
import collections
import json
import os
import sqlite3
import time
import uuid
import mimetypes
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import chainlit as cl
from starlette.responses import JSONResponse

try:
    from training_monitor.utils import DEFAULT_PROJECT_NAME, sanitize_experiment_name
except Exception:
    import sys
    from pathlib import Path as _Path

    _repo_root = _Path(__file__).resolve().parents[1]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))
    from training_monitor.utils import DEFAULT_PROJECT_NAME, sanitize_experiment_name


_POLL_INTERVAL_SEC = float(os.getenv("TM_WEB_POLL_INTERVAL", "1.0"))
_include_env = os.getenv("TM_WEB_INCLUDE_ANALYSES")
if _include_env is None:
    _INCLUDE_ANALYSES = False
else:
    _INCLUDE_ANALYSES = _include_env.lower() in ("1", "true", "yes")

# ------------------------------------------------------------------ #
#  Debug panel infrastructure
# ------------------------------------------------------------------ #

_DEBUG_LOG_BUFFER: collections.deque = collections.deque(maxlen=500)
_DEBUG_SESSION_STATE: dict = {}
_ACTIVE_EXPERIMENT: Optional[str] = None


def _debug_log(level: str, message: str) -> None:
    _DEBUG_LOG_BUFFER.append({
        "ts": time.time(),
        "level": level,
        "msg": message,
    })


def _snapshot_session_state() -> None:
    try:
        _DEBUG_SESSION_STATE.update({
            "db_path": cl.user_session.get("db_path"),
            "experiment": cl.user_session.get("experiment"),
            "last_id": cl.user_session.get("last_id", 0),
            "model_override": cl.user_session.get("model_override"),
            "reasoning_override": cl.user_session.get("reasoning_override"),
            "chat_profile": cl.user_session.get("chat_profile"),
            "has_thinking_msg": cl.user_session.get("thinking_msg") is not None,
            "current_tool_call": cl.user_session.get("current_tool_call"),
        })
        # Load and include monitor init args with monitor. prefix
        db_path_str = cl.user_session.get("db_path")
        experiment = cl.user_session.get("experiment")
        if db_path_str and experiment:
            init_args = _load_monitor_init_args(Path(db_path_str), experiment)
            if init_args:
                for key, value in init_args.items():
                    _DEBUG_SESSION_STATE[f"monitor.{key}"] = value
    except Exception:
        pass


from chainlit.server import app as _chainlit_app  # noqa: E402
from chainlit.config import config as _cl_config  # noqa: E402

# .chainlit/ is gitignored, so config.toml isn't synced across machines.
# Set custom_js/custom_css programmatically to ensure they're always active.
if not _cl_config.ui.custom_css:
    _cl_config.ui.custom_css = "/public/custom.css"
if not _cl_config.ui.custom_js:
    _cl_config.ui.custom_js = "/public/debug-panel.js"


@_chainlit_app.get("/api/debug")
async def _debug_api():
    return JSONResponse({
        "logs": list(_DEBUG_LOG_BUFFER),
        "state": dict(_DEBUG_SESSION_STATE),
    })

# Move our route before the catch-all /{full_path:path} so it isn't shadowed.
_debug_route = _chainlit_app.routes.pop()
for _i, _r in enumerate(_chainlit_app.routes):
    if getattr(_r, "path", "") == "/{full_path:path}":
        _chainlit_app.routes.insert(_i, _debug_route)
        break


def _parse_int_env(name: str) -> Optional[int]:
    value = os.getenv(name)
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


_HISTORY_LIMIT = _parse_int_env("TM_WEB_HISTORY_LIMIT")
_ATTACH_MAX_CHARS = 8000
_ATTACH_MAX_TOTAL_CHARS = 20000


def _parse_csv_env(name: str) -> List[str]:
    value = os.getenv(name, "")
    if not value:
        return []
    return [entry.strip() for entry in value.split(",") if entry.strip()]


def _resolve_db_path() -> Optional[Path]:
    db_env = os.getenv("TM_DB_PATH")
    if db_env:
        return Path(db_env).expanduser()

    root_dir = Path(os.getenv("TM_ROOT_DIR", "ai_logger")).expanduser()
    project_env = os.getenv("TM_PROJECT")
    experiment_env = os.getenv("TM_EXPERIMENT")

    if experiment_env:
        project_name = project_env or DEFAULT_PROJECT_NAME
        return root_dir / project_name / sanitize_experiment_name(experiment_env) / "training_monitor.db"

    if project_env:
        candidates = list((root_dir / project_env).glob("*/training_monitor.db"))
    else:
        candidates = list(root_dir.glob("*/*/training_monitor.db"))

    if not candidates:
        return None

    try:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    except FileNotFoundError:
        return None


def _resolve_experiment(db_path: Path) -> str:
    if _ACTIVE_EXPERIMENT:
        return _ACTIVE_EXPERIMENT

    exp_env = os.getenv("TM_EXPERIMENT")
    if exp_env:
        return sanitize_experiment_name(exp_env)

    try:
        with sqlite3.connect(str(db_path), timeout=1) as conn:
            row = conn.execute(
                "SELECT experiment FROM conversation_messages ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            if row and row[0]:
                return str(row[0])
    except sqlite3.OperationalError:
        pass

    return db_path.parent.name


def _latest_db_experiment(db_path: Path) -> Optional[str]:
    """Query the most recent experiment from the DB (ignores env vars)."""
    try:
        with sqlite3.connect(str(db_path), timeout=1) as conn:
            row = conn.execute(
                "SELECT experiment FROM conversation_messages "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row and row[0]:
                return str(row[0])
    except sqlite3.OperationalError:
        pass
    return None


def _safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return cleaned or "attachment"


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def _read_text_file(path: Path, limit: int) -> str:
    try:
        data = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        try:
            data = path.read_text(encoding="latin-1", errors="ignore")
        except Exception:
            return ""
    return _truncate_text(data, limit)


def _read_pdf_text(path: Path, limit: int) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except Exception:
            return ""

    try:
        reader = PdfReader(str(path))
    except Exception:
        return ""

    chunks: List[str] = []
    total = 0
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        if not page_text:
            continue
        remaining = limit - total
        if remaining <= 0:
            break
        page_text = _truncate_text(page_text, remaining)
        chunks.append(page_text)
        total += len(page_text)
        if total >= limit:
            break
    return "\n".join(chunks)


def _load_monitor_settings(db_path: Path, experiment: str) -> Optional[Dict[str, Optional[str]]]:
    try:
        with sqlite3.connect(str(db_path), timeout=1) as conn:
            row = conn.execute(
                "SELECT model, reasoning_effort FROM monitor_settings "
                "WHERE experiment = ? ORDER BY timestamp DESC LIMIT 1",
                (experiment,),
            ).fetchone()
        if not row:
            return None
        return {
            "model": row[0],
            "reasoning_effort": row[1],
        }
    except sqlite3.OperationalError:
        return None


def _load_monitor_init_args(db_path: Path, experiment: str) -> Optional[Dict[str, Any]]:
    """Load monitor init_args from the database."""
    try:
        with sqlite3.connect(str(db_path), timeout=1) as conn:
            # Check if init_args column exists
            cols = {row[1] for row in conn.execute("PRAGMA table_info(monitor_settings)")}
            if "init_args" not in cols:
                return None
            row = conn.execute(
                "SELECT init_args FROM monitor_settings "
                "WHERE experiment = ? ORDER BY timestamp DESC LIMIT 1",
                (experiment,),
            ).fetchone()
        if not row or not row[0]:
            return None
        return json.loads(row[0])
    except (sqlite3.OperationalError, json.JSONDecodeError):
        return None


def _build_model_options(current_model: Optional[str]) -> List[str]:
    options = _parse_csv_env("TM_WEB_MODEL_OPTIONS")
    if not options:
        options = ["gpt-5.2", "gpt-5.2-pro", "gpt-5-mini", "gpt-5-nano"]
    options = _sort_models_best_first(options)
    if current_model:
        if current_model not in options:
            options = [current_model] + options
    return options


def _build_reasoning_options(
    model: Optional[str],
    current_reasoning: Optional[str],
) -> List[str]:
    options = _parse_csv_env("TM_WEB_REASONING_OPTIONS")
    if not options:
        if _is_reasoning_model_name(model):
            options = ["xhigh", "high", "medium", "low", "default"]
        else:
            options = ["default"]
    if "default" not in options:
        options.append("default")
    if current_reasoning and current_reasoning not in options:
        options = [current_reasoning] + options
    options = _sort_reasoning_best_first(options)
    return options


def _is_reasoning_model_name(model: Optional[str]) -> bool:
    if not model:
        return False
    lower = model.lower()
    return any(x in lower for x in ("gpt-5", "o1", "o3"))


def _sort_models_best_first(options: List[str]) -> List[str]:
    return sorted(options, key=_model_sort_key)


def _model_sort_key(model: str) -> Any:
    lower = model.lower()
    tokens = [t for t in re.split(r"[^a-z0-9]+", lower) if t]
    size_rank = 1
    if any(t in tokens for t in ("pro", "max", "xl", "xlarge")):
        size_rank = 0
    elif any(t in tokens for t in ("mini", "small", "lite")):
        size_rank = 2
    elif any(t in tokens for t in ("nano",)):
        size_rank = 3
    elif any(t in tokens for t in ("micro", "tiny")):
        size_rank = 4
    version_nums = [int(t) for t in re.findall(r"\d+", lower)]
    while len(version_nums) < 3:
        version_nums.append(0)
    version_nums = version_nums[:3]
    version_key = tuple(-n for n in version_nums)
    return (size_rank, version_key, lower)


def _sort_reasoning_best_first(options: List[str]) -> List[str]:
    return sorted(options, key=_reasoning_sort_key)


def _reasoning_sort_key(option: str) -> Any:
    rank = {
        "xhigh": 0,
        "high": 1,
        "medium": 2,
        "low": 3,
        "default": 4,
    }
    return (rank.get(option, 99), option)


def _parse_profile_name(profile_name: str):
    """Parse 'model (reasoning)' into (model, reasoning_effort or None)."""
    m = re.match(r"^(.+?)\s+\((\w+)\)$", profile_name)
    if m:
        return m.group(1), m.group(2)
    return profile_name, None


def _model_has_reasoning_profiles(model: str) -> bool:
    """Whether a model should get expanded reasoning-effort chat profiles."""
    env_models = _parse_csv_env("TM_WEB_REASONING_MODELS")
    if env_models:
        return model in env_models
    if not _is_reasoning_model_name(model):
        return False
    lower = model.lower()
    for suffix in ("-pro", "-mini", "-nano", "-micro"):
        if lower.endswith(suffix):
            return False
    return True


@cl.set_chat_profiles
async def _chat_profiles():
    db_path = _resolve_db_path()
    current_model = None
    current_reasoning = None
    if db_path and db_path.exists():
        experiment = _resolve_experiment(db_path)
        settings = _load_monitor_settings(db_path, experiment) or {}
        current_model = settings.get("model")
        current_reasoning = settings.get("reasoning_effort")

    model_options = _build_model_options(current_model)

    # Determine the profile name that matches the current settings
    default_profile = current_model or model_options[0]
    if (
        current_reasoning
        and current_reasoning != "default"
        and _model_has_reasoning_profiles(default_profile)
    ):
        default_profile = f"{current_model} ({current_reasoning})"

    profiles: List[cl.ChatProfile] = []
    for model in model_options:
        if _model_has_reasoning_profiles(model):
            r = current_reasoning if model == current_model else None
            reasoning_opts = _build_reasoning_options(model, r)
            for effort in reasoning_opts:
                if effort == "default":
                    profiles.append(
                        cl.ChatProfile(
                            name=model,
                            markdown_description=f"Use **{model}** with default reasoning",
                        )
                    )
                    continue
                profiles.append(
                    cl.ChatProfile(
                        name=f"{model} ({effort})",
                        markdown_description=f"Use **{model}** with **{effort}** reasoning effort",
                    )
                )
        else:
            profiles.append(
                cl.ChatProfile(
                    name=model,
                    markdown_description=f"Use **{model}**",
                )
            )

    # Move the current/default profile to the front of the list
    for i, p in enumerate(profiles):
        if p.name == default_profile:
            profiles.insert(0, profiles.pop(i))
            break

    return profiles


def _query_messages(
    db_path: Path,
    experiment: str,
    since_id: Optional[int],
    include_analyses: bool,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if not db_path.exists():
        return []

    query = (
        "SELECT id, timestamp, role, msg_type, content "
        "FROM conversation_messages WHERE experiment = ?"
    )
    params: List[Any] = [experiment]

    if since_id is not None:
        query += " AND id > ?"
        params.append(since_id)

    if not include_analyses:
        query += " AND msg_type IN (?, ?, ?)"
        params.extend(["user_message", "chat_response", "notification"])

    if limit and limit > 0:
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
    else:
        query += " ORDER BY id ASC"

    try:
        with sqlite3.connect(str(db_path), timeout=1) as conn:
            conn.row_factory = sqlite3.Row
            rows = [dict(r) for r in conn.execute(query, params).fetchall()]
        if limit and limit > 0:
            rows.reverse()
        return rows
    except sqlite3.OperationalError as exc:
        _debug_log("error", f"Error querying messages: {exc}")
        return []


def _ensure_web_chat_tables(db_path: Path) -> None:
    try:
        with sqlite3.connect(str(db_path), timeout=1) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS web_chat_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT,
                    experiment TEXT,
                    timestamp REAL,
                    content TEXT,
                    status TEXT,
                    response TEXT,
                    response_timestamp REAL,
                    model TEXT,
                    reasoning_effort TEXT,
                    attachments TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_web_chat_req_id
                    ON web_chat_requests(request_id);
                CREATE INDEX IF NOT EXISTS idx_web_chat_exp_id
                    ON web_chat_requests(experiment, id);
            """)
            conn.commit()
            _ensure_web_chat_request_columns(conn)
    except sqlite3.OperationalError:
        return


def _ensure_web_chat_request_columns(conn: sqlite3.Connection) -> None:
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(web_chat_requests)")}
        if "model" not in cols:
            conn.execute("ALTER TABLE web_chat_requests ADD COLUMN model TEXT")
        if "reasoning_effort" not in cols:
            conn.execute("ALTER TABLE web_chat_requests ADD COLUMN reasoning_effort TEXT")
        if "attachments" not in cols:
            conn.execute("ALTER TABLE web_chat_requests ADD COLUMN attachments TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        return


def _process_attachments(
    db_path: Path,
    elements: Optional[List[Any]],
) -> List[Dict[str, Any]]:
    if not elements:
        return []

    uploads_dir = db_path.parent / "web_uploads"
    try:
        uploads_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    attachments: List[Dict[str, Any]] = []
    total_chars = 0

    for element in elements:
        path_str = getattr(element, "path", None)
        if not path_str:
            continue
        src_path = Path(path_str)
        if not src_path.exists():
            continue

        name = getattr(element, "name", None) or src_path.name
        name = _safe_filename(name)
        mime = getattr(element, "mime", None) or mimetypes.guess_type(name)[0] or "application/octet-stream"
        size = getattr(element, "size", None)
        if size is None:
            try:
                size = src_path.stat().st_size
            except Exception:
                size = None

        dest_path = src_path
        if uploads_dir.exists():
            dest_name = f"{int(time.time())}_{name}"
            dest_path = uploads_dir / dest_name
            try:
                shutil.copy2(src_path, dest_path)
            except Exception:
                dest_path = src_path

        attachment: Dict[str, Any] = {
            "name": name,
            "mime": mime,
            "size": size,
            "path": str(dest_path),
        }

        text_content = ""
        remaining = max(_ATTACH_MAX_TOTAL_CHARS - total_chars, 0)
        per_limit = min(_ATTACH_MAX_CHARS, remaining)

        if per_limit > 0:
            if mime.startswith("text/") or mime in (
                "application/json",
                "application/x-yaml",
                "application/yaml",
                "application/xml",
                "text/plain",
            ):
                text_content = _read_text_file(dest_path, per_limit)
            elif mime == "application/pdf":
                text_content = _read_pdf_text(dest_path, per_limit)

        if text_content:
            attachment["text"] = text_content
            total_chars += len(text_content)
        attachments.append(attachment)

        if total_chars >= _ATTACH_MAX_TOTAL_CHARS:
            break

    return attachments


def _enqueue_web_chat_request(
    db_path: Path,
    experiment: str,
    content: str,
    model: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    attachments: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    try:
        _ensure_web_chat_tables(db_path)
        request_id = str(uuid.uuid4())
        attachments_json = None
        if attachments:
            try:
                attachments_json = json.dumps(attachments)
            except Exception:
                attachments_json = None
        with sqlite3.connect(str(db_path), timeout=1) as conn:
            conn.execute(
                "INSERT INTO web_chat_requests "
                "(request_id, experiment, timestamp, content, status, model, reasoning_effort, attachments) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    request_id,
                    experiment,
                    time.time(),
                    content,
                    "pending",
                    model,
                    reasoning_effort,
                    attachments_json,
                ),
            )
            conn.commit()
        _debug_log("info", f"Enqueued request: {request_id}")
        return request_id
    except sqlite3.OperationalError as exc:
        _debug_log("error", f"Error enqueuing request: {exc}")
        return None


def _format_author(role: str, msg_type: str) -> str:
    role_lower = (role or "").lower()
    if msg_type == "notification":
        return "Monitor"
    if role_lower == "user":
        return "User"
    if role_lower == "model":
        if msg_type == "analysis":
            return "Model (Analysis)"
        return "Model"
    if role:
        return role
    return "Unknown"


def _message_ui_type(role: str, msg_type: str) -> Optional[str]:
    role_lower = (role or "").lower()
    if role_lower == "user" or msg_type == "user_message":
        return "user_message"
    return "assistant_message"


def _should_skip_row(row: Dict[str, Any]) -> bool:
    msg_type = str(row.get("msg_type", ""))
    if msg_type != "user_message":
        return False
    content = str(row.get("content", ""))
    timestamp = float(row.get("timestamp", 0.0) or 0.0)
    sent_cache = cl.user_session.get("sent_cache", [])
    if not sent_cache:
        return False
    for idx, cached in enumerate(list(sent_cache)):
        cached_text = cached.get("content", "")
        cached_ts = cached.get("timestamp", 0.0)
        if cached_text == content and abs(timestamp - cached_ts) < 5.0:
            sent_cache.pop(idx)
            cl.user_session.set("sent_cache", sent_cache)
            return True
    return False


# ------------------------------------------------------------------ #
#  Activity / thinking indicators
# ------------------------------------------------------------------ #

_ACTIVITY_LABELS = {
    "analyzing": "Analysis in progress",
    "responding": "Thinking",
    "generating_code_summary": "Generating code summary",
    "generating_metric_descriptions": "Generating metric descriptions",
    "system_check": "Running system check",
}

_TOOL_DISPLAY_NAMES = {
    "snapshot_manifest": "Listing files",
    "snapshot_search": "Searching",
    "snapshot_read": "Reading",
    "snapshot_line_count": "Checking file",
    "snapshot_read_many": "Reading files",
}


def _friendly_tool_call(raw: str) -> str:
    """Convert internal tool call string to a user-friendly display name.

    Input formats: ``"snapshot_read: src/model.py"`` or ``"snapshot_manifest"``.
    """
    name, _, detail = raw.partition(": ")
    friendly = _TOOL_DISPLAY_NAMES.get(name, name)
    if detail:
        return f"{friendly}: {detail}"
    return friendly


async def _animate_dots(msg: cl.Message, initial_label: str) -> None:
    """Cycle dots on a message, reading label + pending count from session."""
    dots = 1
    try:
        while True:
            await asyncio.sleep(0.5)
            dots = (dots % 3) + 1
            label = cl.user_session.get("thinking_label", initial_label)
            tool_call = cl.user_session.get("current_tool_call")
            pending = cl.user_session.get("pending_request_count", 0)
            dots_str = "." * dots + " " * (3 - dots)
            if tool_call:
                display = _friendly_tool_call(tool_call)
                line = f"`{label}{dots_str}`  `{display}`"
            else:
                line = f"`{label}{dots_str}`"
            if pending > 0:
                req_word = "request" if pending == 1 else "requests"
                line += f"\n`+{pending} {req_word} waiting`"
            msg.content = line
            await msg.update()
    except asyncio.CancelledError:
        return


async def _clear_activity_indicator() -> None:
    """Remove any active thinking/activity indicator."""
    anim_task = cl.user_session.get("thinking_anim_task")
    if anim_task:
        anim_task.cancel()
        cl.user_session.set("thinking_anim_task", None)
    thinking_msg = cl.user_session.get("thinking_msg")
    if thinking_msg:
        await thinking_msg.remove()
        cl.user_session.set("thinking_msg", None)
    cl.user_session.set("current_tool_call", None)
    _debug_log("info", "Cleared activity indicator")


async def _show_activity_indicator(label: str) -> None:
    """Show an animated indicator message (e.g. 'Analysis in progress.')."""
    cl.user_session.set("thinking_label", label)
    if cl.user_session.get("thinking_msg"):
        return  # Already showing — animation will pick up new label
    _debug_log("info", f"Showing indicator: {label}")
    msg = cl.Message(content=f"`{label}.  `", author="Model")
    await msg.send()
    anim_task = asyncio.create_task(_animate_dots(msg, label))
    cl.user_session.set("thinking_msg", msg)
    cl.user_session.set("thinking_anim_task", anim_task)


def _read_monitor_activity(db_path: Path, experiment: str) -> Dict[str, Optional[str]]:
    """Read the current monitor activity from the DB."""
    try:
        with sqlite3.connect(str(db_path), timeout=1) as conn:
            row = conn.execute(
                "SELECT status, tool_call FROM monitor_activity WHERE experiment = ?",
                (experiment,),
            ).fetchone()
        if row:
            return {"status": row[0] or None, "tool_call": row[1] or None}
    except sqlite3.OperationalError:
        pass
    return {"status": None, "tool_call": None}


async def _send_row(row: Dict[str, Any]) -> None:
    if _should_skip_row(row):
        return
    role = str(row.get("role", ""))
    msg_type = str(row.get("msg_type", ""))

    _debug_log("info", f"Displaying message: {msg_type} ({role})")

    # Clear any active indicator when a non-user message arrives
    if msg_type != "user_message":
        await _clear_activity_indicator()

    # Decrement pending count when a chat response arrives
    if msg_type == "chat_response":
        count = cl.user_session.get("pending_request_count", 0)
        cl.user_session.set("pending_request_count", max(0, count - 1))

    author = _format_author(role, msg_type)
    content = str(row.get("content", ""))
    ui_type = _message_ui_type(role, msg_type)
    try:
        msg = cl.Message(content=content, author=author, type=ui_type)
    except TypeError:
        msg = cl.Message(content=content, author=author)
        try:
            msg.type = ui_type
        except Exception:
            pass
    await msg.send()


async def _poll_updates() -> None:
    try:
        while True:
            await asyncio.sleep(_POLL_INTERVAL_SEC)
            db_path_str = cl.user_session.get("db_path")
            experiment = cl.user_session.get("experiment")
            last_id = cl.user_session.get("last_id", 0)

            if not db_path_str or not experiment:
                continue

            db_path = Path(db_path_str)
            rows = _query_messages(
                db_path=db_path,
                experiment=experiment,
                since_id=last_id,
                include_analyses=_INCLUDE_ANALYSES,
                limit=None,
            )

            # Check monitor activity and show/hide indicator + tool calls
            activity_info = _read_monitor_activity(db_path, experiment)
            activity = activity_info.get("status")
            tool_call = activity_info.get("tool_call")

            pending = cl.user_session.get("pending_request_count", 0)

            if activity and activity != "idle":
                label = _ACTIVITY_LABELS.get(activity, "Processing")
                _debug_log("info", f"Monitor activity: {activity}")
                if tool_call:
                    _debug_log("info", f"Tool call: {tool_call}")
                await _show_activity_indicator(label)
                cl.user_session.set("current_tool_call", tool_call)
            elif pending > 0:
                # Idle but requests are queued — keep indicator alive
                await _show_activity_indicator("Processing")
                cl.user_session.set("current_tool_call", None)
            elif not rows:
                # Only clear activity indicator if monitor went idle AND no new
                # messages arrived (messages clear it via _send_row already)
                if activity == "idle" and cl.user_session.get("thinking_msg"):
                    await _clear_activity_indicator()

            if rows:
                _debug_log("info", f"Polled {len(rows)} new messages, last_id={last_id}")

                for row in rows:
                    await _send_row(row)
                    last_id = max(last_id, int(row.get("id", last_id)))

                cl.user_session.set("last_id", last_id)

            # Detect experiment transitions
            try:
                latest_exp = _latest_db_experiment(db_path)
                if latest_exp and latest_exp != experiment:
                    global _ACTIVE_EXPERIMENT
                    _ACTIVE_EXPERIMENT = latest_exp
                    _DEBUG_SESSION_STATE["experiment_changed"] = latest_exp
                    _debug_log("info", f"Experiment transition: {experiment} -> {latest_exp}")
            except Exception:
                pass

            _snapshot_session_state()

            # Once the session has caught up to the new experiment, clear the flag
            if _DEBUG_SESSION_STATE.get("experiment_changed") == experiment:
                _DEBUG_SESSION_STATE.pop("experiment_changed", None)
    except asyncio.CancelledError:
        return


async def _initialize_session(db_path: Path) -> bool:
    if cl.user_session.get("poll_task"):
        return True
    if not db_path.exists():
        return False

    experiment = _resolve_experiment(db_path)
    cl.user_session.set("db_path", str(db_path))
    cl.user_session.set("experiment", experiment)
    cl.user_session.set("sent_cache", [])

    rows = _query_messages(
        db_path=db_path,
        experiment=experiment,
        since_id=None,
        include_analyses=_INCLUDE_ANALYSES,
        limit=_HISTORY_LIMIT,
    )

    last_id = 0
    for row in rows:
        await _send_row(row)
        last_id = max(last_id, int(row.get("id", last_id)))

    cl.user_session.set("last_id", last_id)

    if not rows:
        await cl.Message(
            content=(
                "Waiting for messages... "
                "This page mirrors monitor chat content with richer rendering."
            )
        ).send()

    poll_task = asyncio.create_task(_poll_updates())
    cl.user_session.set("poll_task", poll_task)
    _debug_log("info", f"Connected to DB: {db_path}")
    return True


async def _wait_for_db() -> None:
    try:
        while True:
            await asyncio.sleep(_POLL_INTERVAL_SEC)
            _debug_log("debug", "Waiting for DB...")
            db_path = _resolve_db_path()
            if db_path is None:
                continue
            if await _initialize_session(db_path):
                return
    except asyncio.CancelledError:
        return


@cl.on_chat_start
async def _on_chat_start() -> None:
    # Parse model/reasoning from selected chat profile
    profile_name = cl.user_session.get("chat_profile")
    if profile_name:
        model, reasoning = _parse_profile_name(profile_name)
        cl.user_session.set("model_override", model)
        cl.user_session.set("reasoning_override", reasoning)

    _debug_log("info", f"Session started | profile={profile_name}")
    _snapshot_session_state()

    db_path = _resolve_db_path()
    if db_path is None or not db_path.exists():
        await cl.Message(
            content=(
                "No training monitor database found. "
                "Set `TM_DB_PATH`, or set `TM_ROOT_DIR` + `TM_PROJECT` + `TM_EXPERIMENT`."
            )
        ).send()
        wait_task = asyncio.create_task(_wait_for_db())
        cl.user_session.set("wait_task", wait_task)
        return

    await _initialize_session(db_path)


@cl.on_chat_end
async def _on_chat_end() -> None:
    _debug_log("info", "Session ended")
    wait_task = cl.user_session.get("wait_task")
    if wait_task:
        wait_task.cancel()
    poll_task = cl.user_session.get("poll_task")
    if poll_task:
        poll_task.cancel()

@cl.on_message
async def _on_message(message: cl.Message) -> None:
    db_path_str = cl.user_session.get("db_path")
    experiment = cl.user_session.get("experiment")
    if not db_path_str or not experiment:
        await cl.Message(
            content="Web chat is not connected to a training DB yet."
        ).send()
        return

    content = (message.content or "").strip()
    if not content:
        return

    _debug_log("info", f"User message received: {content[:80]}")

    db_path = Path(db_path_str)
    attachments = _process_attachments(db_path, getattr(message, "elements", None))

    model_override = cl.user_session.get("model_override")
    reasoning_override = cl.user_session.get("reasoning_override")
    req_id = _enqueue_web_chat_request(
        db_path,
        experiment,
        content,
        model=str(model_override) if model_override else None,
        reasoning_effort=str(reasoning_override) if reasoning_override else None,
        attachments=attachments,
    )
    if not req_id:
        await cl.Message(
            content="Failed to enqueue message. Is the monitor running?"
        ).send()
        return

    # Reposition indicator below the user message
    await _clear_activity_indicator()

    count = cl.user_session.get("pending_request_count", 0)
    cl.user_session.set("pending_request_count", count + 1)

    # Determine current activity label
    label = "Thinking"
    activity_info = _read_monitor_activity(db_path, experiment)
    activity = activity_info.get("status")
    if activity and activity != "idle":
        label = _ACTIVITY_LABELS.get(activity, "Processing")
        cl.user_session.set("current_tool_call", activity_info.get("tool_call"))

    await _show_activity_indicator(label)

    sent_cache = cl.user_session.get("sent_cache", [])
    sent_cache.append({"content": content, "timestamp": time.time()})
    if len(sent_cache) > 50:
        sent_cache = sent_cache[-50:]
    cl.user_session.set("sent_cache", sent_cache)
