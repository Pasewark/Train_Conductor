"""TrainingMonitor — the main long-running monitoring process."""

import base64
import difflib
import json
import mimetypes
import os
import queue
import socket
import sqlite3
import sys
import threading
import time
from collections import deque
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

try:
    import zmq
except ImportError:
    zmq = None

try:
    import requests
except ImportError:
    requests = None

from .analyzer import LLMAnalyzer
from .memory_tools import CombinedToolManager, MemoryToolManager
from .notify import notify
from .snapshot_tools import MultiRootSnapshotToolManager, SnapshotToolManager
from .system_metrics import collect_system_metrics
from .telegram_chat import TelegramChatHandler
from .types import SystemMetric, TrainingMetric
from .utils import (
    create_code_snapshot,
    create_code_snapshot_from_roots,
    default_experiment_dir,
    default_project_dir,
    ensure_dir,
    hash_directory,
    iter_snapshot_files,
    safe_json_dumps,
    sanitize_experiment_name,
    sanitize_project_name,
    SNAPSHOT_DIFF_FILENAME,
)


class TrainingMonitor:
    """
    Main monitoring process:
    - Receives metrics via ZMQ
    - Collects system metrics
    - Persists to SQLite
    - Runs periodic LLM analysis
    - Optionally enables interactive Telegram chat

    Everything is stored under root_dir/project/experiment_name/ once the experiment
    name is known.
    """

    def __init__(
        self,
        zmq_port: int = 5555,
        root_dir: str = "ai_logger",
        project: Optional[str] = None,
        experiment: Optional[str] = None,
        db_path: str = "",
        analysis_interval_min: float = 5.0,
        analysis_interval_steps: Optional[int] = None,
        system_poll_interval_sec: float = 10.0,
        max_prompt_len: int = 8000,
        openai_model: str = "gpt-5.2",
        openai_base_url: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        gpus: Optional[List[int]] = None,
        recovery_max_age_min: float = 60.0,
        notify_all_analyses: bool = True,
        sig_figs: int = 5,
        idle_timeout_min: Optional[float] = None,
        telegram_chat: bool = False,
        max_conversation_history_tokens: int = 5000,
        regenerate_code_analysis: bool = False,
        visible_directories: Optional[List[str]] = None,
        code_snapshot_paths: Optional[List[str]] = None,
        code_snapshot_manifest: Optional[str] = None,
    ):
        self.zmq_port = zmq_port
        self.root_dir = Path(root_dir)
        ensure_dir(self.root_dir)
        self.project_name = sanitize_project_name(project)
        self.project_dir = default_project_dir(self.root_dir, self.project_name)
        ensure_dir(self.project_dir)
        self.memory_dir = self.project_dir / "memories"
        ensure_dir(self.memory_dir)
        self.project_cache_dir = self.project_dir / ".cache"
        ensure_dir(self.project_cache_dir)
        self._last_snapshot_meta_path = self.project_cache_dir / "last_snapshot.json"

        self.analysis_interval = analysis_interval_min * 60
        self.analysis_interval_steps = analysis_interval_steps
        self.system_poll_interval = system_poll_interval_sec
        self.max_prompt_len = max_prompt_len
        self.gpus = gpus
        self.recovery_max_age_min = recovery_max_age_min
        self.notify_all_analyses = notify_all_analyses
        self.telegram_enabled = telegram_chat
        self.idle_timeout = idle_timeout_min * 60 if idle_timeout_min else None
        self.max_conversation_history_tokens = max_conversation_history_tokens
        self.regenerate_code_analysis = regenerate_code_analysis
        self.visible_directories = list(visible_directories or [])
        self.code_snapshot_paths = list(code_snapshot_paths or [])
        self.code_snapshot_manifest = code_snapshot_manifest

        # State
        self.training_metrics: Deque[TrainingMetric] = deque(maxlen=10000)
        self.system_metrics: Deque[SystemMetric] = deque(maxlen=1000)
        self.last_analysis_time: Optional[float] = None
        self.last_analysis_step: Optional[int] = None
        self.last_analysis_request_time: Optional[float] = None
        self.last_analysis_request_step: Optional[int] = None
        self.last_update_time: Optional[float] = None
        self.is_idle: bool = False
        self.current_experiment: Optional[str] = sanitize_experiment_name(experiment) if experiment else None
        self.running = True
        self.first_metric_received = False

        # Unified conversation history: analyses, user messages, and chat responses.
        # Each entry: {"timestamp": float, "role": "model"|"user",
        #              "type": "analysis"|"chat_response"|"user_message",
        #              "text": str}
        self.conversation_history: Deque[Dict[str, Any]] = deque(maxlen=500)
        self._analysis_count: int = 0

        # Storage (db + run_dir) — opened lazily if experiment unknown
        self.run_dir: Optional[Path] = None
        self.db_path: Optional[Path] = None
        self.db_conn: Optional[sqlite3.Connection] = None
        self._explicit_db_path = Path(db_path) if db_path else None

        # Code summary (generated once from snapshot, included in all analysis prompts)
        self.code_summary: Optional[str] = None

        # Metric descriptions (generated once from snapshot, maps metric name -> description)
        self.metric_descriptions: Optional[Dict[str, str]] = None
        self.snapshot_tool_manager: Optional[SnapshotToolManager] = None
        self.analysis_tool_manager: Optional[Any] = None
        self.code_diff_text: Optional[str] = None

        # Concurrency primitives
        self._state_lock = threading.Lock()
        self._db_lock = threading.Lock()
        self._llm_lock = threading.Lock()
        self._analysis_queue: queue.Queue = queue.Queue()
        self._analysis_worker: Optional[threading.Thread] = None
        self._analysis_worker_running = threading.Event()
        self._analysis_in_flight = threading.Event()
        self._analysis_queue_alerted = False
        self._web_chat_thread: Optional[threading.Thread] = None
        self._web_chat_running = threading.Event()

        # Memory tools
        self.memory_tool_manager = MemoryToolManager(
            memory_dir=self.memory_dir,
            get_experiment=self._get_experiment_name,
        )

        # Components
        self.analyzer = LLMAnalyzer(
            model=openai_model,
            base_url=openai_base_url,
            reasoning_effort=reasoning_effort,
            sig_figs=sig_figs,
        )

        # ZMQ
        if zmq is None:
            raise RuntimeError("zmq not installed: pip install pyzmq")

        self.zmq_ctx = zmq.Context()
        self.zmq_socket = self.zmq_ctx.socket(zmq.PULL)
        self.zmq_socket.bind(f"tcp://*:{zmq_port}")
        self.zmq_socket.setsockopt(zmq.RCVTIMEO, 100)

        self.hostname = socket.gethostname()
        self.experiment_config: Optional[Dict[str, Any]] = None

        # Telegram interactive chat
        self.telegram_chat_handler: Optional[TelegramChatHandler] = None
        if telegram_chat:
            try:
                self.telegram_chat_handler = TelegramChatHandler(
                    analyzer=self.analyzer,
                    tool_manager=None,
                    llm_lock=self._llm_lock,
                )
                self.telegram_chat_handler.get_context_fn = self._build_chat_context
                self.telegram_chat_handler.on_exchange_fn = (
                    lambda role, text, msg_type: self._on_telegram_exchange(
                        role,
                        text,
                        msg_type,
                        source="telegram",
                    )
                )
            except ValueError as e:
                print(f"[monitor] Telegram chat disabled: {e}", file=sys.stderr)

        # Store init args for debug panel exposure
        self._init_args = self._collect_init_args(
            zmq_port=zmq_port,
            root_dir=root_dir,
            project=project,
            experiment=experiment,
            db_path=db_path,
            analysis_interval_min=analysis_interval_min,
            analysis_interval_steps=analysis_interval_steps,
            system_poll_interval_sec=system_poll_interval_sec,
            max_prompt_len=max_prompt_len,
            openai_model=openai_model,
            openai_base_url=openai_base_url,
            reasoning_effort=reasoning_effort,
            gpus=gpus,
            recovery_max_age_min=recovery_max_age_min,
            notify_all_analyses=notify_all_analyses,
            sig_figs=sig_figs,
            idle_timeout_min=idle_timeout_min,
            telegram_chat=telegram_chat,
            max_conversation_history_tokens=max_conversation_history_tokens,
            regenerate_code_analysis=regenerate_code_analysis,
            visible_directories=visible_directories,
            code_snapshot_paths=code_snapshot_paths,
            code_snapshot_manifest=code_snapshot_manifest,
        )

        # If experiment was provided, open storage immediately
        if self.current_experiment:
            self._open_experiment_storage(self.current_experiment)

    def _collect_init_args(self, **kwargs) -> Dict[str, Any]:
        """Collect all __init__ arguments into a serializable dict for debugging."""
        result: Dict[str, Any] = {}
        for key, value in kwargs.items():
            # Convert Path objects to strings for JSON serialization
            if isinstance(value, Path):
                result[key] = str(value)
            elif isinstance(value, list):
                # Convert any Path objects in lists
                result[key] = [str(v) if isinstance(v, Path) else v for v in value]
            else:
                result[key] = value
        return result

    # ------------------------------------------------------------------ #
    #  Telegram exchange callback
    # ------------------------------------------------------------------ #

    def _log_chat_exchange(self, role: str, text: str, msg_type: str, source: str) -> None:
        role_label = "User" if role == "user" else "Model" if role == "model" else role
        source_label = source or "unknown"
        print(f"\n{'='*60}", file=sys.stderr)
        print(
            f"[monitor] Chat {role_label} ({msg_type}) from {source_label}:",
            file=sys.stderr,
        )
        print(f"{'='*60}", file=sys.stderr)
        print(text, file=sys.stderr)
        print(f"{'='*60}\n", file=sys.stderr)

    def _on_telegram_exchange(
        self,
        role: str,
        text: str,
        msg_type: str,
        source: str = "unknown",
    ) -> None:
        """Record/log a user or model exchange from web/telegram chat."""
        text_str = str(text)
        self._log_chat_exchange(role=role, text=text_str, msg_type=msg_type, source=source)
        entry: Dict[str, Any] = {
            "timestamp": time.time(),
            "role": role,
            "type": msg_type,
            "text": text_str,
        }
        with self._state_lock:
            self.conversation_history.append(entry)
        self._persist_conversation_message(entry)

    # ------------------------------------------------------------------ #
    #  Code snapshot (CLI support)
    # ------------------------------------------------------------------ #

    def _snapshot_dir_is_empty(self, path: Path) -> bool:
        if not path.exists():
            return True
        if not path.is_dir():
            return True
        for _ in path.rglob("*"):
            return False
        return True

    def _truncate_attachment_text(self, text: str, limit: int = 8000) -> str:
        if len(text) <= limit:
            return text
        return text[:limit] + "\n...[truncated]"

    def _encode_image_as_data_url(self, path: Path, mime: Optional[str] = None, max_bytes: int = 30_000_000) -> Optional[str]:
        try:
            size = path.stat().st_size
        except Exception:
            return None
        if size > max_bytes:
            return None
        if not mime:
            mime = mimetypes.guess_type(path.name)[0]
        if not mime or not mime.startswith("image/"):
            return None
        try:
            data = path.read_bytes()
        except Exception:
            return None
        encoded = base64.b64encode(data).decode("utf-8")
        return f"data:{mime};base64,{encoded}"

    def _maybe_create_code_snapshot(self) -> None:
        if self.run_dir is None:
            return

        code_dir = self.run_dir / "code"
        if not self._snapshot_dir_is_empty(code_dir):
            return

        manifest_path = None
        if self.code_snapshot_manifest:
            manifest_path = Path(self.code_snapshot_manifest).expanduser()

        roots: List[Path] = []
        for entry in self.code_snapshot_paths:
            if not entry:
                continue
            try:
                roots.append(Path(entry).expanduser())
            except Exception:
                continue

        if roots and manifest_path is not None:
            warning = (
                "[monitor] Warning: both code_snapshot_manifest and code_snapshot_paths "
                "were provided. Using code_snapshot_paths and ignoring the manifest."
            )
            print(warning, file=sys.stderr)
            if os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"):
                self._notify(warning, level="info")

        if roots:
            n = create_code_snapshot_from_roots(
                roots,
                code_dir,
                base_dir=Path.cwd(),
            )
        elif manifest_path is not None:
            n = create_code_snapshot(manifest_path, code_dir)
        else:
            return

        if n > 0:
            print(f"[monitor] Code snapshot: {n} file(s) → {code_dir}", file=sys.stderr)
            return

        if self._snapshot_dir_is_empty(code_dir):
            warning = (
                "[monitor] Warning: code snapshot directory is empty "
                f"at {code_dir}. LLM analysis will lack code context. "
                "To include code, pass code_snapshot_paths=[...] with files/dirs, "
                "or pass code_snapshot_manifest=... (snapshot_files.txt), "
                "or use code_snapshot_all/code_snapshot_manifest via tm.init(). "
                "You can also set visible_directories=[...] for live, read-only access."
            )
            print(warning, file=sys.stderr)
            if os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"):
                self._notify(warning, level="info")

    # ------------------------------------------------------------------ #
    #  Code diffs (snapshot vs previous snapshot)
    # ------------------------------------------------------------------ #

    def _load_last_snapshot_meta(self) -> Optional[Dict[str, Any]]:
        if not self._last_snapshot_meta_path.exists():
            return None
        try:
            raw = self._last_snapshot_meta_path.read_text(encoding="utf-8").strip()
        except Exception:
            return None
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        return data

    def _save_last_snapshot_meta(self, snapshot_hash: str, code_dir: Path) -> None:
        try:
            payload = {
                "hash": snapshot_hash,
                "path": str(code_dir),
                "updated_at": time.time(),
            }
            ensure_dir(self.project_cache_dir)
            self._last_snapshot_meta_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except Exception as e:
            print(f"[monitor] Failed to save snapshot metadata: {e}", file=sys.stderr)

    def _read_text_lines(self, path: Path) -> Optional[List[str]]:
        try:
            return path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            return None
        except Exception:
            return None

    def _diff_snapshots(self, prev_dir: Path, curr_dir: Path) -> str:
        prev_files = iter_snapshot_files(prev_dir, exclude_files=[SNAPSHOT_DIFF_FILENAME])
        curr_files = iter_snapshot_files(curr_dir, exclude_files=[SNAPSHOT_DIFF_FILENAME])
        all_files = sorted(set(prev_files) | set(curr_files))

        diff_lines: List[str] = []

        for rel in all_files:
            prev_path = prev_dir / rel
            curr_path = curr_dir / rel
            prev_exists = prev_path.exists()
            curr_exists = curr_path.exists()

            prev_lines = self._read_text_lines(prev_path) if prev_exists else []
            curr_lines = self._read_text_lines(curr_path) if curr_exists else []

            if prev_lines is None or curr_lines is None:
                diff_lines.append(f"Binary or unreadable file changed: {rel}")
                continue

            if prev_lines == curr_lines:
                continue

            diff = difflib.unified_diff(
                prev_lines,
                curr_lines,
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
                lineterm="",
            )
            diff_lines.extend(diff)

        return "\n".join(diff_lines).strip()

    def _write_diff_file(self, code_dir: Path, diff_text: str, prev_dir: Optional[Path]) -> None:
        diff_path = code_dir / SNAPSHOT_DIFF_FILENAME
        header_lines = [
            "# AUTO-GENERATED FILE - NOT PART OF THE TRAINING CODE.",
            "# Snapshot diff between current snapshot and the previous snapshot.",
        ]
        if prev_dir is not None:
            header_lines.append(f"# Previous snapshot: {prev_dir}")
        header_lines.append(f"# Current snapshot: {code_dir}")
        header_lines.append(f"# Generated at: {datetime.now().isoformat()}")
        header_lines.append("#")
        if not diff_text:
            header_lines.append("# No changes detected or no previous snapshot available.")
        header = "\n".join(header_lines)

        try:
            payload = header + ("\n" + diff_text if diff_text else "\n")
            diff_path.write_text(payload, encoding="utf-8")
        except Exception as e:
            print(f"[monitor] Failed to write snapshot diff: {e}", file=sys.stderr)

    def _generate_code_diff(self) -> None:
        """Generate a diff vs the previous snapshot and store it on disk + in memory."""
        self.code_diff_text = None
        if self.run_dir is None:
            return
        code_dir = self.run_dir / "code"
        if not code_dir.is_dir():
            return

        prev_meta = self._load_last_snapshot_meta()
        prev_dir = None
        prev_hash = None
        if prev_meta:
            prev_path = prev_meta.get("path")
            if isinstance(prev_path, str):
                prev_dir = Path(prev_path)
                if not prev_dir.is_dir():
                    prev_dir = None
            prev_hash = prev_meta.get("hash")

        snapshot_hash = hash_directory(code_dir)
        if snapshot_hash is None:
            self._write_diff_file(code_dir, "", prev_dir)
            return

        if prev_hash == snapshot_hash:
            self._write_diff_file(code_dir, "", prev_dir)
            self._save_last_snapshot_meta(snapshot_hash, code_dir)
            return

        if prev_dir is None or prev_dir.resolve() == code_dir.resolve():
            self._write_diff_file(code_dir, "", prev_dir)
            self._save_last_snapshot_meta(snapshot_hash, code_dir)
            return

        try:
            diff_text = self._diff_snapshots(prev_dir, code_dir)
        except Exception as e:
            print(f"[monitor] Failed to generate snapshot diff: {e}", file=sys.stderr)
            diff_text = ""

        self._write_diff_file(code_dir, diff_text, prev_dir)
        self.code_diff_text = diff_text or None
        self._save_last_snapshot_meta(snapshot_hash, code_dir)

    def _get_code_diff_for_prompt(self, max_lines: int = 1000) -> Optional[str]:
        if not self.code_diff_text:
            return None
        lines = self.code_diff_text.splitlines()
        if len(lines) <= max_lines:
            diff_body = self.code_diff_text
            truncated_note = ""
        else:
            diff_body = "\n".join(lines[:max_lines])
            remaining = len(lines) - max_lines
            truncated_note = f"\n... (diff truncated to {max_lines} lines; {remaining} more lines not shown)"

        return diff_body + truncated_note

    # ------------------------------------------------------------------ #
    #  Storage
    # ------------------------------------------------------------------ #

    def _open_experiment_storage(self, experiment: str) -> None:
        experiment = sanitize_experiment_name(experiment)
        self.current_experiment = experiment

        self.run_dir = default_experiment_dir(self.root_dir, experiment, self.project_name)
        ensure_dir(self.run_dir)

        self._maybe_create_code_snapshot()
        self._generate_code_diff()

        if self._explicit_db_path is not None:
            dbp = self._explicit_db_path
            if not dbp.is_absolute():
                dbp = self.run_dir / dbp
        else:
            dbp = self.run_dir / "training_monitor.db"

        ensure_dir(dbp.parent)

        try:
            if self.db_conn is not None:
                self.db_conn.close()
        except Exception:
            pass

        self.db_path = dbp
        self.db_conn = sqlite3.connect(str(dbp), check_same_thread=False)
        self._init_db()
        self._persist_monitor_settings(experiment)

        # Load experiment config from config.json (written by AILogger) or from DB
        if self.experiment_config is None:
            config_path = self.run_dir / "config.json"
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        loaded_config = json.load(f)
                    with self._state_lock:
                        self.experiment_config = loaded_config
                    self._persist_experiment_config(experiment, loaded_config)
                except Exception:
                    pass
            if self.experiment_config is None:
                loaded_config = self._load_experiment_config(experiment)
                with self._state_lock:
                    self.experiment_config = loaded_config

        # Load code summary from disk if it exists (e.g. recovery / restart)
        if self.code_summary is None:
            summary_path = self.run_dir / "code_summary.md"
            if summary_path.exists():
                try:
                    loaded_summary = summary_path.read_text(encoding="utf-8")
                    with self._state_lock:
                        self.code_summary = loaded_summary
                    print(f"[monitor] Loaded existing code summary from {summary_path}", file=sys.stderr)
                except Exception:
                    pass

        if self.snapshot_tool_manager is None:
            self.snapshot_tool_manager = SnapshotToolManager(self.run_dir / "code")

        self.analysis_tool_manager = self._build_analysis_tool_manager()
        if self.telegram_chat_handler:
            self.telegram_chat_handler.tool_manager = self.analysis_tool_manager or self.snapshot_tool_manager

        # Load metric descriptions from disk if they exist
        if self.metric_descriptions is None:
            desc_path = self.run_dir / "metric_descriptions.json"
            if desc_path.exists():
                try:
                    loaded_desc = json.loads(desc_path.read_text(encoding="utf-8"))
                    with self._state_lock:
                        self.metric_descriptions = loaded_desc
                    print(f"[monitor] Loaded existing metric descriptions from {desc_path}", file=sys.stderr)
                except Exception:
                    pass

    def _build_analysis_tool_manager(self) -> Optional[Any]:
        if self.run_dir is None:
            base_manager = self.snapshot_tool_manager
            if base_manager is None:
                return CombinedToolManager(
                    base_manager=None,
                    extra_managers=[self.memory_tool_manager],
                )
            return CombinedToolManager(
                base_manager=base_manager,
                extra_managers=[self.memory_tool_manager],
            )

        roots: Dict[str, Path] = {}
        labels: Dict[str, str] = {}

        code_dir = self.run_dir / "code"
        if code_dir.is_dir():
            roots["code"] = code_dir
            labels["code"] = "code snapshot"

        if self.memory_dir.is_dir():
            roots["memories"] = self.memory_dir
            labels["memories"] = "memories"

        used_names = set(roots.keys())
        for entry in self.visible_directories:
            if not entry:
                continue
            try:
                visible_path = Path(entry).expanduser()
                if not visible_path.is_absolute():
                    visible_path = (Path.cwd() / visible_path)
                visible_path = visible_path.resolve()
            except Exception:
                continue

            if not visible_path.is_dir():
                print(f"[monitor] Visible directory not found: {visible_path}", file=sys.stderr)
                continue

            base = sanitize_experiment_name(visible_path.name) or "dir"
            root_name = f"visible_{base}"
            if root_name in used_names:
                suffix = 2
                while f"{root_name}_{suffix}" in used_names:
                    suffix += 1
                root_name = f"{root_name}_{suffix}"
            used_names.add(root_name)
            roots[root_name] = visible_path
            labels[root_name] = f"visible: {visible_path.name}"

        if not roots:
            return CombinedToolManager(
                base_manager=None,
                extra_managers=[self.memory_tool_manager],
            )

        if len(roots) == 1 and "code" in roots:
            base_manager = self.snapshot_tool_manager
            return CombinedToolManager(
                base_manager=base_manager,
                extra_managers=[self.memory_tool_manager],
            )

        base_manager = MultiRootSnapshotToolManager(roots=roots, root_labels=labels)
        return CombinedToolManager(
            base_manager=base_manager,
            extra_managers=[self.memory_tool_manager],
        )

    def _build_chat_context(self) -> str:
        """Build a training context summary for the Telegram chat handler."""
        with self._state_lock:
            training_metrics = list(self.training_metrics)
            system_metrics = list(self.system_metrics)
            conversation_history = list(self.conversation_history)
            experiment = self.current_experiment or "unknown"
            config = self.experiment_config
            code_summary = self.code_summary
            metric_descriptions = (
                dict(self.metric_descriptions) if self.metric_descriptions is not None else None
            )
        memory_summaries = self._get_memory_summaries()
        return self.analyzer.build_prompt(
            training_metrics=training_metrics,
            system_metrics=system_metrics,
            max_prompt_len=self.max_prompt_len,
            experiment_name=experiment,
            conversation_history=conversation_history,
            max_conversation_history_tokens=self.max_conversation_history_tokens,
            config=config,
            code_summary=code_summary,
            metric_descriptions=metric_descriptions,
            memory_summaries=memory_summaries,
        )

    def _get_experiment_name(self) -> str:
        with self._state_lock:
            return self.current_experiment or "unknown"

    def _get_memory_summaries(self) -> Dict[str, List[str]]:
        try:
            return self.memory_tool_manager.recent_summaries(limit=50)
        except Exception:
            return {}

    def _init_db(self):
        if self.db_conn is None:
            return
        self.db_conn.executescript("""
            CREATE TABLE IF NOT EXISTS training_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment TEXT,
                step INTEGER,
                timestamp REAL,
                metrics TEXT
            );
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                data TEXT
            );
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment TEXT,
                timestamp REAL,
                prompt TEXT,
                response TEXT,
                alerted INTEGER
            );
            CREATE TABLE IF NOT EXISTS experiment_configs (
                experiment TEXT PRIMARY KEY,
                timestamp REAL,
                config TEXT
            );
            CREATE TABLE IF NOT EXISTS monitor_settings (
                experiment TEXT PRIMARY KEY,
                timestamp REAL,
                model TEXT,
                reasoning_effort TEXT,
                init_args TEXT
            );
            CREATE TABLE IF NOT EXISTS conversation_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment TEXT,
                timestamp REAL,
                role TEXT,
                msg_type TEXT,
                content TEXT
            );
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
            CREATE INDEX IF NOT EXISTS idx_training_exp_step
                ON training_metrics(experiment, step);
            CREATE INDEX IF NOT EXISTS idx_system_ts
                ON system_metrics(timestamp);
            CREATE INDEX IF NOT EXISTS idx_conv_exp_ts
                ON conversation_messages(experiment, timestamp);
            CREATE INDEX IF NOT EXISTS idx_web_chat_req_id
                ON web_chat_requests(request_id);
            CREATE INDEX IF NOT EXISTS idx_web_chat_exp_id
                ON web_chat_requests(experiment, id);
            CREATE TABLE IF NOT EXISTS monitor_activity (
                experiment TEXT PRIMARY KEY,
                status TEXT,
                description TEXT,
                tool_call TEXT,
                timestamp REAL
            );
        """)
        self.db_conn.commit()
        self._ensure_web_chat_request_columns()
        self._ensure_monitor_settings_columns()

    def _ensure_monitor_settings_columns(self) -> None:
        """Add init_args column to monitor_settings if it doesn't exist (migration)."""
        if self.db_conn is None:
            return
        try:
            with self._db_lock:
                cols = {
                    row[1] for row in self.db_conn.execute("PRAGMA table_info(monitor_settings)")
                }
                if "init_args" not in cols:
                    self.db_conn.execute("ALTER TABLE monitor_settings ADD COLUMN init_args TEXT")
                self.db_conn.commit()
        except Exception as e:
            print(f"[monitor] Error ensuring monitor_settings columns: {e}", file=sys.stderr)

    def _ensure_web_chat_request_columns(self) -> None:
        if self.db_conn is None:
            return
        try:
            with self._db_lock:
                cols = {
                    row[1] for row in self.db_conn.execute("PRAGMA table_info(web_chat_requests)")
                }
                if "model" not in cols:
                    self.db_conn.execute("ALTER TABLE web_chat_requests ADD COLUMN model TEXT")
                if "reasoning_effort" not in cols:
                    self.db_conn.execute("ALTER TABLE web_chat_requests ADD COLUMN reasoning_effort TEXT")
                if "attachments" not in cols:
                    self.db_conn.execute("ALTER TABLE web_chat_requests ADD COLUMN attachments TEXT")
                self.db_conn.commit()
        except Exception as e:
            print(f"[monitor] Error ensuring web_chat_requests columns: {e}", file=sys.stderr)

    def _persist_experiment_config(self, experiment: str, config: Dict[str, Any]) -> None:
        if self.db_conn is None:
            return
        try:
            with self._db_lock:
                self.db_conn.execute(
                    "INSERT INTO experiment_configs (experiment, timestamp, config) VALUES (?, ?, ?) "
                    "ON CONFLICT(experiment) DO UPDATE SET timestamp=excluded.timestamp, config=excluded.config",
                    (experiment, time.time(), safe_json_dumps(config)),
                )
                self.db_conn.commit()
        except Exception as e:
            print(f"[monitor] Error persisting config: {e}", file=sys.stderr)

    def _write_activity(self, status: str, description: str = "", tool_call: str = "") -> None:
        """Write current monitor activity status to DB for the web UI to poll."""
        if self.db_conn is None:
            return
        experiment = None
        with self._state_lock:
            experiment = self.current_experiment
        if not experiment:
            return
        try:
            with self._db_lock:
                self.db_conn.execute(
                    "INSERT INTO monitor_activity "
                    "(experiment, status, description, tool_call, timestamp) "
                    "VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(experiment) DO UPDATE SET "
                    "status=excluded.status, description=excluded.description, "
                    "tool_call=excluded.tool_call, timestamp=excluded.timestamp",
                    (experiment, status, description, tool_call, time.time()),
                )
                self.db_conn.commit()
        except Exception:
            pass

    def _make_tool_call_reporter(self, status: str = "analyzing") -> Callable:
        """Create a callback that records each tool call in activity + stderr logs."""
        def on_tool_call(name: str, args: Dict[str, Any]) -> None:
            summary = name
            # Extract the most informative argument for display
            for key in ("path", "file_path", "query", "pattern", "key", "name"):
                if key in args and args[key]:
                    summary = f"{name}: {args[key]}"
                    break
            print(f"[monitor] Tool call ({status}): {summary}", file=sys.stderr)
            self._write_activity(status, tool_call=summary)
        return on_tool_call

    def _persist_monitor_settings(self, experiment: str) -> None:
        if self.db_conn is None:
            return
        try:
            init_args_json = None
            if hasattr(self, "_init_args") and self._init_args:
                init_args_json = safe_json_dumps(self._init_args)
            with self._db_lock:
                self.db_conn.execute(
                    "INSERT INTO monitor_settings (experiment, timestamp, model, reasoning_effort, init_args) "
                    "VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(experiment) DO UPDATE SET "
                    "timestamp=excluded.timestamp, model=excluded.model, "
                    "reasoning_effort=excluded.reasoning_effort, init_args=excluded.init_args",
                    (
                        experiment,
                        time.time(),
                        self.analyzer.model,
                        self.analyzer.reasoning_effort,
                        init_args_json,
                    ),
                )
                self.db_conn.commit()
        except Exception as e:
            print(f"[monitor] Error persisting monitor settings: {e}", file=sys.stderr)

    def _load_experiment_config(self, experiment: str) -> Optional[Dict[str, Any]]:
        try:
            if self.db_conn is None:
                return None
            with self._db_lock:
                cur = self.db_conn.execute(
                    "SELECT config FROM experiment_configs WHERE experiment = ? LIMIT 1",
                    (experiment,),
                )
                row = cur.fetchone()
            if not row:
                return None
            return json.loads(row[0])
        except Exception as e:
            print(f"[monitor] Error loading config: {e}", file=sys.stderr)
            return None

    def _persist_training_metric(self, metric: TrainingMetric, experiment: str):
        if self.db_conn is None:
            return
        with self._db_lock:
            self.db_conn.execute(
                "INSERT INTO training_metrics (experiment, step, timestamp, metrics) VALUES (?, ?, ?, ?)",
                (experiment, metric.step, metric.timestamp, json.dumps(metric.metrics)),
            )
            self.db_conn.commit()

    def _persist_system_metric(self, metric: SystemMetric):
        if self.db_conn is None:
            return
        with self._db_lock:
            self.db_conn.execute(
                "INSERT INTO system_metrics (timestamp, data) VALUES (?, ?)",
                (metric.timestamp, json.dumps(asdict(metric))),
            )
            self.db_conn.commit()

    def _persist_analysis(
        self,
        experiment: str,
        prompt: str,
        response: str,
        alerted: bool,
        db_conn: Optional[sqlite3.Connection] = None,
    ):
        conn = db_conn or self.db_conn
        if conn is None:
            return
        with self._db_lock:
            conn.execute(
                "INSERT INTO analyses (experiment, timestamp, prompt, response, alerted) VALUES (?, ?, ?, ?, ?)",
                (experiment, time.time(), prompt, response, int(alerted)),
            )
            conn.commit()

    def _persist_conversation_message(
        self,
        entry: Dict[str, Any],
        experiment: Optional[str] = None,
        db_conn: Optional[sqlite3.Connection] = None,
    ) -> None:
        """Persist a single conversation history entry to the DB."""
        conn = db_conn or self.db_conn
        if conn is None:
            return
        experiment_name = experiment or (self.current_experiment or "unknown")
        try:
            with self._db_lock:
                conn.execute(
                    "INSERT INTO conversation_messages (experiment, timestamp, role, msg_type, content) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        experiment_name,
                        entry["timestamp"],
                        entry["role"],
                        entry["type"],
                        entry["text"],
                    ),
                )
                conn.commit()
        except Exception as e:
            print(f"[monitor] Error persisting conversation message: {e}", file=sys.stderr)

    def _notify(self, message: str, level: str = "info", record: bool = True) -> None:
        notify(message, level=level, allow_telegram=self.telegram_enabled)
        if not record:
            return

        prefix = "🚨 " if level == "alert" else "ℹ️ " if level == "info" else ""
        entry: Dict[str, Any] = {
            "timestamp": time.time(),
            "role": "system",
            "type": "notification",
            "text": prefix + message,
        }
        with self._state_lock:
            experiment = self.current_experiment or "unknown"
        self._persist_conversation_message(entry, experiment=experiment)

    def _analyses_jsonl_path(self, experiment: str, run_dir: Optional[Path] = None) -> Optional[Path]:
        target_dir = run_dir or self.run_dir
        if target_dir is None:
            return None
        return target_dir / "analyses.jsonl"

    def _append_analysis_to_json(
        self,
        experiment: str,
        analysis: str,
        alerted: bool,
        step_range: Optional[Tuple[int, int]] = None,
        run_dir: Optional[Path] = None,
    ):
        if not experiment or experiment == "unknown":
            return
        json_path = self._analyses_jsonl_path(experiment, run_dir=run_dir)
        if json_path is None:
            return
        try:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "unix_timestamp": time.time(),
                "experiment": experiment,
                "step_min": step_range[0] if step_range else None,
                "step_max": step_range[1] if step_range else None,
                "alerted": alerted,
                "analysis": analysis,
            }
            ensure_dir(json_path.parent)
            with open(json_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"[monitor] Error writing to JSON: {e}", file=sys.stderr)

    # ------------------------------------------------------------------ #
    #  Recovery
    # ------------------------------------------------------------------ #

    def _recover_from_db(self, experiment: str, max_age_minutes: float = 60.0) -> tuple[int, int, int]:
        if self.db_conn is None:
            return 0, 0, 0
        try:
            min_timestamp = time.time() - (max_age_minutes * 60)

            with self._db_lock:
                cursor = self.db_conn.execute(
                    """SELECT step, timestamp, metrics FROM training_metrics
                       WHERE experiment = ? AND timestamp > ?
                       ORDER BY step ASC""",
                    (experiment, min_timestamp),
                )
                training_rows = list(cursor)

                cursor = self.db_conn.execute(
                    """SELECT timestamp, data FROM system_metrics
                       WHERE timestamp > ?
                       ORDER BY timestamp ASC""",
                    (min_timestamp,),
                )
                system_rows = list(cursor)

            count = 0
            with self._state_lock:
                for row in training_rows:
                    step, timestamp, metrics_json = row
                    self.training_metrics.append(TrainingMetric(
                        step=step, timestamp=timestamp, metrics=json.loads(metrics_json),
                    ))
                    count += 1

            sys_count = 0
            with self._state_lock:
                for row in system_rows:
                    timestamp, data_json = row
                    data = json.loads(data_json)
                    self.system_metrics.append(SystemMetric(
                        timestamp=data["timestamp"],
                        gpus=data["gpus"],
                        cpu_percent=data["cpu_percent"],
                        ram_percent=data["ram_percent"],
                        ram_used_gb=data["ram_used_gb"],
                        ram_total_gb=data["ram_total_gb"],
                    ))
                    sys_count += 1

            # Recover conversation history from conversation_messages table
            conv_count = 0
            try:
                with self._db_lock:
                    cursor = self.db_conn.execute(
                        """SELECT timestamp, role, msg_type, content
                           FROM conversation_messages
                           WHERE experiment = ? AND timestamp > ?
                           ORDER BY timestamp ASC""",
                        (experiment, min_timestamp),
                    )
                    conv_rows = list(cursor)

                with self._state_lock:
                    for row in conv_rows:
                        timestamp, role, msg_type, content = row
                        self.conversation_history.append({
                            "timestamp": timestamp,
                            "role": role,
                            "type": msg_type,
                            "text": content,
                        })
                        if msg_type == "analysis":
                            self._analysis_count += 1
                        conv_count += 1
            except sqlite3.OperationalError:
                # Table might not exist yet (upgrading from older version).
                # Fall back to loading from the analyses table.
                with self._db_lock:
                    cursor = self.db_conn.execute(
                        """SELECT timestamp, response, alerted FROM analyses
                           WHERE experiment = ? AND timestamp > ?
                           ORDER BY timestamp ASC""",
                        (experiment, min_timestamp),
                    )
                    analysis_rows = list(cursor)

                with self._state_lock:
                    for row in analysis_rows:
                        timestamp, response, alerted = row
                        self.conversation_history.append({
                            "timestamp": timestamp,
                            "role": "model",
                            "type": "analysis",
                            "text": response,
                        })
                        self._analysis_count += 1
                        conv_count += 1

            return count, sys_count, conv_count

        except Exception as e:
            print(f"[monitor] Error recovering from DB: {e}", file=sys.stderr)
            return 0, 0, 0

    # ------------------------------------------------------------------ #
    #  ZMQ message handling
    # ------------------------------------------------------------------ #

    def _receive_zmq_messages(self):
        while self.running:
            try:
                msg = self.zmq_socket.recv_json()

                now = time.time()
                resumed = False
                with self._state_lock:
                    self.last_update_time = now
                    if self.is_idle:
                        self.is_idle = False
                        resumed = True
                if resumed:
                    print(f"\n[monitor] Training activity resumed", file=sys.stderr)

                if msg.get("type") == "training_metric":
                    experiment = sanitize_experiment_name(msg.get("experiment") or "unknown")

                    if "config" in msg and msg["config"] is not None:
                        with self._state_lock:
                            self.experiment_config = msg["config"]
                            config_snapshot = self.experiment_config
                        self._persist_experiment_config(experiment, config_snapshot)

                    # First metric: open storage if needed, then recover
                    if not self.first_metric_received:
                        self.first_metric_received = True
                        if self.db_conn is None:
                            self._open_experiment_storage(experiment)

                        print(f"\n[monitor] Experiment: {experiment}", file=sys.stderr)
                        print(f"[monitor] Run dir: {self.run_dir}", file=sys.stderr)
                        print(f"[monitor] DB: {self.db_path}", file=sys.stderr)

                        train_count, sys_count, conv_count = self._recover_from_db(
                            experiment, self.recovery_max_age_min,
                        )
                        if train_count > 0:
                            with self._state_lock:
                                steps = [m.step for m in self.training_metrics]
                            print(f"[monitor] Recovered {train_count} training metrics, {conv_count} conversation messages", file=sys.stderr)
                            print(f"[monitor] Step range: {min(steps)} - {max(steps)}", file=sys.stderr)
                            self._notify(
                                f"Monitor recovered on {self.hostname}\n"
                                f"Experiment: {experiment}\n"
                                f"Recovered {train_count} training metrics, {conv_count} conversation messages",
                                level="info",
                            )
                        else:
                            print(f"[monitor] No recovery data found, starting fresh", file=sys.stderr)

                        loaded_config = self._load_experiment_config(experiment)
                        with self._state_lock:
                            self.current_experiment = experiment
                            self.experiment_config = loaded_config

                    # Experiment change mid-run
                    elif experiment and self.current_experiment and experiment != self.current_experiment:
                        print(f"\n[monitor] New experiment detected: {experiment}", file=sys.stderr)
                        print(f"[monitor] Previous: {self.current_experiment}", file=sys.stderr)

                        with self._state_lock:
                            self.training_metrics.clear()
                            self.conversation_history.clear()
                            self._analysis_count = 0
                            self.last_analysis_time = None
                            self.last_analysis_step = None
                            self.last_analysis_request_time = None
                            self.last_analysis_request_step = None
                            self.code_summary = None
                            self.metric_descriptions = None
                            self.code_diff_text = None
                        self._clear_analysis_queue()

                        self._open_experiment_storage(experiment)
                        self._notify(
                            f"New experiment started on {self.hostname}\n"
                            f"Experiment: {experiment}\n"
                            f"Previous: {self.current_experiment}",
                            level="info",
                        )
                        loaded_config = self._load_experiment_config(experiment)
                        with self._state_lock:
                            self.current_experiment = experiment
                            self.experiment_config = loaded_config

                    metric = TrainingMetric(
                        step=msg["step"],
                        timestamp=msg["timestamp"],
                        metrics=msg["metrics"],
                    )
                    with self._state_lock:
                        self.training_metrics.append(metric)
                    self._persist_training_metric(metric, experiment)

                    if msg.get("force_analysis"):
                        override: Dict[str, Any] = {}
                        if msg.get("openai_model"):
                            override["openai_model"] = msg.get("openai_model")
                        if msg.get("reasoning_effort") is not None:
                            override["reasoning_effort"] = msg.get("reasoning_effort")
                        self._enqueue_analysis_request(reason="force", overrides=override or None)

                elif msg.get("type") == "reset":
                    print(f"\n[monitor] Received reset, clearing buffers", file=sys.stderr)
                    with self._state_lock:
                        self.training_metrics.clear()
                        self.conversation_history.clear()
                        self._analysis_count = 0
                        self.last_analysis_time = None
                        self.last_analysis_step = None
                        self.last_analysis_request_time = None
                        self.last_analysis_request_step = None
                        self.current_experiment = sanitize_experiment_name(msg.get("experiment") or "unknown")
                        self.first_metric_received = False
                        self.code_summary = None
                        self.metric_descriptions = None
                        self.code_diff_text = None
                    self._clear_analysis_queue()

                elif msg.get("type") == "shutdown":
                    print(f"\n[monitor] Received shutdown request", file=sys.stderr)
                    self.running = False
                    return

            except zmq.Again:
                break
            except Exception as e:
                print(f"[monitor] ZMQ receive error: {e}", file=sys.stderr)
                break

    # ------------------------------------------------------------------ #
    #  System metrics collection
    # ------------------------------------------------------------------ #

    def _collect_system_metrics(self):
        try:
            metric = collect_system_metrics(gpus=self.gpus)
            with self._state_lock:
                self.system_metrics.append(metric)
            self._persist_system_metric(metric)
        except Exception as e:
            print(f"[monitor] System metrics error: {e}", file=sys.stderr)

    # ------------------------------------------------------------------ #
    #  Code summary
    # ------------------------------------------------------------------ #

    def _get_snapshot_cache_dir(self) -> Optional[Path]:
        """
        Return the shared cache directory for the current code snapshot,
        based on a SHA-256 hash of the snapshot contents.

        Layout: <root_dir>/.cache/<hash>/

        Returns None if run_dir is missing or the code directory doesn't
        exist / is empty.
        """
        if self.run_dir is None:
            return None
        code_dir = self.run_dir / "code"
        snapshot_hash = hash_directory(code_dir)
        if snapshot_hash is None:
            return None
        cache_dir = self.root_dir / ".cache" / snapshot_hash
        ensure_dir(cache_dir)
        return cache_dir

    def _generate_code_summary(self) -> None:
        """
        Use the snapshot tool manager to let the LLM explore the code
        snapshot via tools (snapshot_manifest, snapshot_read, etc.) and
        produce a structured summary.  The result is persisted to
        <run_dir>/code_summary.md and cached under
        <root_dir>/.cache/<hash>/code_summary.md for reuse across runs.

        Skips silently if:
        - A summary was already loaded (e.g. from a previous run)
        - run_dir is not set or code/ doesn't exist
        - snapshot_tool_manager is not available
        """
        with self._state_lock:
            if self.code_summary is not None:
                return

        if self.run_dir is None:
            return

        code_dir = self.run_dir / "code"
        summary_path = self.run_dir / "code_summary.md"

        # Check per-run file (e.g. monitor restart within same experiment)
        if not self.regenerate_code_analysis and summary_path.exists():
            try:
                loaded_summary = summary_path.read_text(encoding="utf-8")
                with self._state_lock:
                    self.code_summary = loaded_summary
                print(f"[monitor] Loaded existing code summary from {summary_path}", file=sys.stderr)
                return
            except Exception:
                pass

        if not code_dir.is_dir():
            return

        # Check shared cache (reuse across experiments with same code)
        cache_dir = self._get_snapshot_cache_dir()
        if not self.regenerate_code_analysis and cache_dir is not None:
            cached_summary = cache_dir / "code_summary.md"
            if cached_summary.exists():
                try:
                    loaded_summary = cached_summary.read_text(encoding="utf-8")
                    with self._state_lock:
                        self.code_summary = loaded_summary
                    # Copy into run_dir for quick access on restarts
                    try:
                        summary_path.write_text(loaded_summary, encoding="utf-8")
                    except Exception:
                        pass
                    print(f"[monitor] Loaded cached code summary from {cached_summary}", file=sys.stderr)
                    return
                except Exception:
                    pass

        if self.snapshot_tool_manager is None:
            return

        print(f"[monitor] Generating code summary from {code_dir} ...", file=sys.stderr)

        try:
            self._write_activity("generating_code_summary", "Generating code summary")
            code_diff_prompt = self._get_code_diff_for_prompt()
            summary = self.analyzer.generate_code_summary(
                tool_manager=self.snapshot_tool_manager,
                code_diff=code_diff_prompt,
                on_tool_call=self._make_tool_call_reporter("generating_code_summary"),
            )
            with self._state_lock:
                self.code_summary = summary

            # Persist to run_dir
            try:
                summary_path.write_text(summary, encoding="utf-8")
                print(f"[monitor] Code summary saved to {summary_path}", file=sys.stderr)
            except Exception as e:
                print(f"[monitor] Failed to save code summary: {e}", file=sys.stderr)

            # Persist to shared cache
            if cache_dir is not None:
                try:
                    (cache_dir / "code_summary.md").write_text(summary, encoding="utf-8")
                    print(f"[monitor] Code summary cached to {cache_dir / 'code_summary.md'}", file=sys.stderr)
                except Exception as e:
                    print(f"[monitor] Failed to cache code summary: {e}", file=sys.stderr)

            print(f"\n{'='*60}", file=sys.stderr)
            print(f"Code Summary", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)
            print(summary, file=sys.stderr)
            print(f"{'='*60}\n", file=sys.stderr)
            self._write_activity("idle")

        except Exception as e:
            print(f"[monitor] Code summary generation failed: {e}", file=sys.stderr)
            self._write_activity("idle")

    # ------------------------------------------------------------------ #
    #  Metric descriptions
    # ------------------------------------------------------------------ #

    def _generate_metric_descriptions(self) -> None:
        """
        Use the snapshot tool manager to let the LLM explore the code
        snapshot via tools and extract metric name -> description mappings.
        The result is persisted to <run_dir>/metric_descriptions.json and
        cached under <root_dir>/.cache/<hash>/metric_descriptions.json
        for reuse across runs.

        Skips silently if:
        - Descriptions were already loaded (e.g. from a previous run)
        - run_dir is not set or code/ doesn't exist
        - snapshot_tool_manager is not available
        """
        with self._state_lock:
            if self.metric_descriptions is not None:
                return

        if self.run_dir is None:
            return

        code_dir = self.run_dir / "code"
        desc_path = self.run_dir / "metric_descriptions.json"

        # Check per-run file
        if not self.regenerate_code_analysis and desc_path.exists():
            try:
                loaded_desc = json.loads(
                    desc_path.read_text(encoding="utf-8")
                )
                with self._state_lock:
                    self.metric_descriptions = loaded_desc
                print(f"[monitor] Loaded existing metric descriptions from {desc_path}", file=sys.stderr)
                return
            except Exception:
                pass

        if not code_dir.is_dir():
            return

        # Check shared cache
        cache_dir = self._get_snapshot_cache_dir()
        if not self.regenerate_code_analysis and cache_dir is not None:
            cached_desc = cache_dir / "metric_descriptions.json"
            if cached_desc.exists():
                try:
                    loaded_desc = json.loads(
                        cached_desc.read_text(encoding="utf-8")
                    )
                    with self._state_lock:
                        self.metric_descriptions = loaded_desc
                    # Copy into run_dir
                    try:
                        desc_path.write_text(
                            json.dumps(loaded_desc, indent=2, sort_keys=True),
                            encoding="utf-8",
                        )
                    except Exception:
                        pass
                    print(f"[monitor] Loaded cached metric descriptions from {cached_desc}", file=sys.stderr)
                    return
                except Exception:
                    pass

        if self.snapshot_tool_manager is None:
            return

        print(f"[monitor] Generating metric descriptions from {code_dir} ...", file=sys.stderr)

        try:
            self._write_activity("generating_metric_descriptions", "Generating metric descriptions")
            with self._state_lock:
                code_summary = self.code_summary
            code_diff_prompt = self._get_code_diff_for_prompt()
            descriptions = self.analyzer.generate_metric_descriptions(
                tool_manager=self.snapshot_tool_manager,
                code_summary=code_summary,
                code_diff=code_diff_prompt,
                on_tool_call=self._make_tool_call_reporter("generating_metric_descriptions"),
            )
            with self._state_lock:
                self.metric_descriptions = descriptions

            desc_json = json.dumps(descriptions, indent=2, sort_keys=True)

            # Persist to run_dir
            try:
                desc_path.write_text(desc_json, encoding="utf-8")
                print(f"[monitor] Metric descriptions saved to {desc_path}", file=sys.stderr)
            except Exception as e:
                print(f"[monitor] Failed to save metric descriptions: {e}", file=sys.stderr)

            # Persist to shared cache
            if cache_dir is not None:
                try:
                    (cache_dir / "metric_descriptions.json").write_text(desc_json, encoding="utf-8")
                    print(f"[monitor] Metric descriptions cached to {cache_dir / 'metric_descriptions.json'}", file=sys.stderr)
                except Exception as e:
                    print(f"[monitor] Failed to cache metric descriptions: {e}", file=sys.stderr)

            print(f"\n{'='*60}", file=sys.stderr)
            print(f"Metric Descriptions", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)
            for k, v in sorted(descriptions.items()):
                print(f"  {k}: {v}", file=sys.stderr)
            print(f"{'='*60}\n", file=sys.stderr)
            self._write_activity("idle")

        except Exception as e:
            print(f"[monitor] Metric descriptions generation failed: {e}", file=sys.stderr)
            self._write_activity("idle")

    def _persist_metric_descriptions(self, descriptions: Optional[Dict[str, str]] = None) -> None:
        """Save metric descriptions to run_dir and shared cache."""
        if descriptions is None:
            with self._state_lock:
                if self.metric_descriptions is None:
                    return
                descriptions = dict(self.metric_descriptions)

        desc_json = json.dumps(descriptions, indent=2, sort_keys=True)

        if self.run_dir is not None:
            desc_path = self.run_dir / "metric_descriptions.json"
            try:
                desc_path.write_text(desc_json, encoding="utf-8")
            except Exception as e:
                print(f"[monitor] Failed to save metric descriptions: {e}", file=sys.stderr)

        cache_dir = self._get_snapshot_cache_dir()
        if cache_dir is not None:
            try:
                (cache_dir / "metric_descriptions.json").write_text(desc_json, encoding="utf-8")
            except Exception as e:
                print(f"[monitor] Failed to cache metric descriptions: {e}", file=sys.stderr)

    def _fill_missing_metric_descriptions(self) -> None:
        """
        Check if training metrics contain keys without descriptions.
        If so, prompt the LLM to generate descriptions for only the
        missing metrics.  Repeat up to 5 times until all metrics have
        descriptions or no new descriptions are generated.

        Updates self.metric_descriptions in place and persists to disk
        and shared cache after each successful round.
        """
        if self.snapshot_tool_manager is None:
            return

        for round_num in range(1, 6):  # max 5 rounds
            with self._state_lock:
                if self.metric_descriptions is None:
                    self.metric_descriptions = {}
                existing = dict(self.metric_descriptions)
                code_summary = self.code_summary
                # Collect all metric keys from training data
                all_keys: set[str] = set()
                for m in self.training_metrics:
                    all_keys.update(m.metrics.keys())

            # Find missing
            missing = sorted(all_keys - set(existing.keys()))

            if not missing:
                return

            print(
                f"[monitor] Filling missing metric descriptions "
                f"(round {round_num}/5): {len(missing)} missing — {missing}",
                file=sys.stderr,
            )

            try:
                self._write_activity("generating_metric_descriptions", "Filling missing metric descriptions")
                with self._llm_lock:
                    new_descriptions = self.analyzer.generate_missing_metric_descriptions(
                        missing_metrics=missing,
                        existing_descriptions=existing,
                        tool_manager=self.snapshot_tool_manager,
                        code_summary=code_summary,
                        on_tool_call=self._make_tool_call_reporter("generating_metric_descriptions"),
                    )
                self._write_activity("idle")
            except Exception as e:
                self._write_activity("idle")
                print(
                    f"[monitor] Round {round_num}/5: error generating "
                    f"missing descriptions: {e}",
                    file=sys.stderr,
                )
                return

            if not new_descriptions:
                print(
                    f"[monitor] Round {round_num}/5: no new descriptions "
                    f"generated, stopping",
                    file=sys.stderr,
                )
                return

            # Merge new descriptions into existing set
            with self._state_lock:
                if self.metric_descriptions is None:
                    self.metric_descriptions = {}
                self.metric_descriptions.update(new_descriptions)
                merged = dict(self.metric_descriptions)

            print(
                f"[monitor] Round {round_num}/5: added "
                f"{len(new_descriptions)} description(s):",
                file=sys.stderr,
            )
            for k, v in sorted(new_descriptions.items()):
                print(f"  {k}: {v}", file=sys.stderr)

            # Persist after each successful round
            self._persist_metric_descriptions(descriptions=merged)

        # Report any still-missing after all rounds
        with self._state_lock:
            all_keys = set()
            for m in self.training_metrics:
                all_keys.update(m.metrics.keys())
            known = set(self.metric_descriptions.keys()) if self.metric_descriptions else set()
        still_missing = sorted(all_keys - known)
        if still_missing:
            print(
                f"[monitor] After 5 rounds, {len(still_missing)} metric(s) "
                f"still lack descriptions: {still_missing}",
                file=sys.stderr,
            )

    # ------------------------------------------------------------------ #
    #  Analysis queue + worker
    # ------------------------------------------------------------------ #

    def _start_analysis_worker(self) -> None:
        if self._analysis_worker and self._analysis_worker.is_alive():
            return
        self._analysis_worker_running.set()
        self._analysis_worker = threading.Thread(
            target=self._analysis_worker_loop,
            daemon=True,
            name="analysis-worker",
        )
        self._analysis_worker.start()

    def _stop_analysis_worker(self) -> None:
        self._analysis_worker_running.clear()
        try:
            self._analysis_queue.put_nowait(None)
        except Exception:
            pass
        if self._analysis_worker:
            self._analysis_worker.join(timeout=5)
            self._analysis_worker = None

    def _analysis_worker_loop(self) -> None:
        while self._analysis_worker_running.is_set():
            try:
                request = self._analysis_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if request is None:
                self._analysis_queue.task_done()
                break

            self._analysis_in_flight.set()
            try:
                self._run_analysis_request(request)
            finally:
                self._analysis_in_flight.clear()
                self._analysis_queue.task_done()
                self._update_analysis_queue_alert_state()

    def _clear_analysis_queue(self) -> None:
        while True:
            try:
                self._analysis_queue.get_nowait()
                self._analysis_queue.task_done()
            except queue.Empty:
                break
        self._analysis_queue_alerted = False

    def _update_analysis_queue_alert_state(self) -> None:
        try:
            qsize = self._analysis_queue.qsize()
        except Exception:
            return

        if qsize > 3 and not self._analysis_queue_alerted:
            with self._state_lock:
                experiment = self.current_experiment or "unknown"
            self._notify(
                f"Analysis queue backlog on {self.hostname}\n"
                f"Experiment: {experiment}\n"
                f"Pending analyses: {qsize}",
                level="alert",
            )
            self._analysis_queue_alerted = True
        elif qsize <= 3 and self._analysis_queue_alerted:
            self._analysis_queue_alerted = False

    def _start_web_chat_worker(self) -> None:
        if self._web_chat_thread and self._web_chat_thread.is_alive():
            return
        self._web_chat_running.set()
        self._web_chat_thread = threading.Thread(
            target=self._web_chat_loop,
            daemon=True,
            name="web-chat",
        )
        self._web_chat_thread.start()

    def _stop_web_chat_worker(self) -> None:
        self._web_chat_running.clear()
        if self._web_chat_thread:
            self._web_chat_thread.join(timeout=5)
            self._web_chat_thread = None

    def _web_chat_loop(self) -> None:
        last_id = 0
        while self.running and self._web_chat_running.is_set():
            try:
                if self.db_conn is None or not self.current_experiment:
                    time.sleep(1.0)
                    continue

                with self._db_lock:
                    rows = list(self.db_conn.execute(
                        """SELECT id, request_id, experiment, content, model, reasoning_effort, attachments
                           FROM web_chat_requests
                           WHERE experiment = ? AND status = 'pending' AND id > ?
                           ORDER BY id ASC
                           LIMIT 20""",
                        (self.current_experiment, last_id),
                    ))

                if not rows:
                    time.sleep(0.5)
                    continue

                for row in rows:
                    req_id, request_id, experiment, content, model_override, reasoning_override, attachments_json = row
                    if req_id > last_id:
                        last_id = req_id

                    with self._db_lock:
                        cur = self.db_conn.execute(
                            "UPDATE web_chat_requests SET status = 'processing' "
                            "WHERE id = ? AND status = 'pending'",
                            (req_id,),
                        )
                        self.db_conn.commit()
                    if cur.rowcount == 0:
                        continue

                    user_text = str(content or "").strip()
                    if not user_text:
                        with self._db_lock:
                            self.db_conn.execute(
                                "UPDATE web_chat_requests SET status = 'done', "
                                "response = ?, response_timestamp = ? WHERE id = ?",
                                ("(Empty message ignored.)", time.time(), req_id),
                            )
                            self.db_conn.commit()
                        continue

                    if attachments_json:
                        try:
                            attachments = json.loads(attachments_json)
                        except Exception:
                            attachments = []
                    else:
                        attachments = []

                    user_text_display = user_text
                    user_text_for_llm = user_text

                    image_parts: List[Dict[str, Any]] = []
                    if attachments:
                        summary_lines = ["[Attachments]"]
                        detail_lines: List[str] = []
                        for attachment in attachments:
                            name = attachment.get("name") or "attachment"
                            mime = attachment.get("mime") or "application/octet-stream"
                            size = attachment.get("size")
                            size_str = f"{size} bytes" if isinstance(size, int) else "unknown size"
                            summary = f"- {name} ({mime}, {size_str})"

                            text = attachment.get("text")
                            if text:
                                summary += " (text extracted)"
                                detail_lines.append(f"### {name}\n{self._truncate_attachment_text(str(text))}")
                            elif str(mime).startswith("image/"):
                                path_str = attachment.get("path")
                                data_url = None
                                if path_str:
                                    data_url = self._encode_image_as_data_url(Path(path_str), mime=str(mime))
                                if data_url:
                                    image_parts.append({"type": "input_image", "image_url": data_url})
                                    summary += " (image attached)"
                                else:
                                    summary += " (image too large or unreadable)"
                            elif mime == "application/pdf":
                                summary += " (pdf attached; text extraction unavailable)"
                            summary_lines.append(summary)

                        user_text_display = user_text_display + "\n\n" + "\n".join(summary_lines)
                        if detail_lines:
                            user_text_for_llm = (
                                user_text_for_llm
                                + "\n\n[Attachment Text]\n"
                                + "\n\n".join(detail_lines)
                            )

                    self._on_telegram_exchange("user", user_text_display, "user_message", source="web")

                    context = self._build_chat_context()
                    system_content = (
                        f"{TelegramChatHandler.CHAT_SYSTEM_PROMPT}\n\n"
                        f"## Current Training Context\n\n{context}\n\n## END CONTEXT"
                    )
                    user_content_parts: List[Dict[str, Any]] = [
                        {"type": "input_text", "text": user_text_for_llm},
                    ]
                    user_content_parts.extend(image_parts)

                    messages: list[Dict[str, Any]] = [
                        {"role": "system", "content": [{"type": "input_text", "text": system_content}]},
                        {"role": "user", "content": user_content_parts},
                    ]

                    if reasoning_override is not None and str(reasoning_override).strip().lower() == "default":
                        reasoning_override = None

                    try:
                        _tool_reporter = self._make_tool_call_reporter("responding")
                        if self._llm_lock:
                            with self._llm_lock:
                                self._write_activity("responding", "Responding to user message")
                                reply_text = self.analyzer._call_api_for_messages(
                                    messages,
                                    tool_manager=self.analysis_tool_manager or self.snapshot_tool_manager,
                                    openai_model_override=str(model_override) if model_override else None,
                                    reasoning_effort_override=str(reasoning_override) if reasoning_override else None,
                                    on_tool_call=_tool_reporter,
                                )
                        else:
                            self._write_activity("responding", "Responding to user message")
                            reply_text = self.analyzer._call_api_for_messages(
                                messages,
                                tool_manager=self.analysis_tool_manager or self.snapshot_tool_manager,
                                openai_model_override=str(model_override) if model_override else None,
                                reasoning_effort_override=str(reasoning_override) if reasoning_override else None,
                                on_tool_call=_tool_reporter,
                            )
                    except Exception as e:
                        reply_text = f"Sorry, I couldn't process that: {e}"
                    finally:
                        self._write_activity("idle")

                    self._on_telegram_exchange("model", reply_text, "chat_response", source="web")

                    with self._db_lock:
                        self.db_conn.execute(
                            "UPDATE web_chat_requests SET status = 'done', "
                            "response = ?, response_timestamp = ? WHERE id = ?",
                            (reply_text, time.time(), req_id),
                        )
                        self.db_conn.commit()

            except Exception as e:
                print(f"[monitor] Web chat error: {e}", file=sys.stderr)
                time.sleep(1.0)

    def _enqueue_analysis_request(
        self,
        reason: str,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        request = {
            "requested_at": time.time(),
            "reason": reason,
            "overrides": overrides or {},
        }
        with self._state_lock:
            self.last_analysis_request_time = request["requested_at"]
            if self.training_metrics:
                self.last_analysis_request_step = max(m.step for m in self.training_metrics)
        self._analysis_queue.put(request)
        self._update_analysis_queue_alert_state()

    def _run_analysis_request(self, request: Dict[str, Any]) -> None:
        now = time.time()
        overrides = request.get("overrides") or {}

        with self._state_lock:
            if len(self.training_metrics) == 0:
                return

        print(f"[monitor] Running LLM analysis...", file=sys.stderr)

        # Fill in descriptions for any newly-observed metrics before analysis
        self._fill_missing_metric_descriptions()

        with self._state_lock:
            experiment = self.current_experiment or "unknown"
            training_metrics = list(self.training_metrics)
            system_metrics = list(self.system_metrics)
            conversation_history = list(self.conversation_history)
            config = self.experiment_config
            code_summary = self.code_summary
            metric_descriptions = (
                dict(self.metric_descriptions) if self.metric_descriptions is not None else None
            )
            tool_manager = self.analysis_tool_manager or self.snapshot_tool_manager
            max_prompt_len = self.max_prompt_len
            max_conv_tokens = self.max_conversation_history_tokens
            run_dir = self.run_dir
            db_path = self.db_path
        memory_summaries = self._get_memory_summaries()

        try:
            self._write_activity("analyzing", "Running periodic analysis")
            with self._llm_lock:
                analysis, should_alert = self.analyzer.analyze(
                    training_metrics=training_metrics,
                    system_metrics=system_metrics,
                    max_prompt_len=max_prompt_len,
                    experiment_name=experiment,
                    conversation_history=conversation_history,
                    max_conversation_history_tokens=max_conv_tokens,
                    config=config,
                    code_summary=code_summary,
                    metric_descriptions=metric_descriptions,
                    memory_summaries=memory_summaries,
                    tool_manager=tool_manager,
                    openai_model_override=overrides.get("openai_model"),
                    reasoning_effort_override=overrides.get("reasoning_effort"),
                    on_tool_call=self._make_tool_call_reporter("analyzing"),
                )

            # Add analysis to unified conversation history (only if experiment hasn't changed)
            analysis_entry: Dict[str, Any] = {
                "timestamp": now,
                "role": "model",
                "type": "analysis",
                "text": analysis,
            }

            with self._state_lock:
                same_experiment = (self.current_experiment or "unknown") == experiment
                if same_experiment:
                    self.conversation_history.append(analysis_entry)
                    self._analysis_count += 1

            step_range = None
            if training_metrics:
                steps = [m.step for m in training_metrics]
                step_range = (min(steps), max(steps))

            self._append_analysis_to_json(
                experiment,
                analysis,
                should_alert,
                step_range,
                run_dir=run_dir,
            )

            # Build prompt for persistence (exclude the analysis we just added)
            prompt = self.analyzer.build_prompt(
                training_metrics,
                system_metrics,
                max_prompt_len,
                experiment,
                conversation_history=conversation_history,
                max_conversation_history_tokens=max_conv_tokens,
                config=config,
                code_summary=code_summary,
                metric_descriptions=metric_descriptions,
                memory_summaries=memory_summaries,
            )

            # Persist analysis + conversation message using a dedicated connection
            analysis_conn: Optional[sqlite3.Connection] = None
            try:
                if db_path is not None:
                    analysis_conn = sqlite3.connect(str(db_path), check_same_thread=False)
                self._persist_analysis(experiment, prompt, analysis, should_alert, db_conn=analysis_conn)
                self._persist_conversation_message(
                    analysis_entry,
                    experiment=experiment,
                    db_conn=analysis_conn,
                )
            finally:
                if analysis_conn is not None:
                    try:
                        analysis_conn.close()
                    except Exception:
                        pass

            print(f"\n{'='*60}", file=sys.stderr)
            with self._state_lock:
                count = self._analysis_count
            print(f"Analysis at {datetime.now().strftime('%H:%M:%S')} (#{count})", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)
            print(analysis, file=sys.stderr)
            print(f"{'='*60}\n", file=sys.stderr)

            if should_alert:
                self._notify(
                    f"Training Alert on {self.hostname}\nExperiment: {experiment}\nTime: {datetime.now().strftime('%H:%M:%S')}\n\n{analysis}",
                    level="alert",
                )
            elif self.notify_all_analyses:
                self._notify(
                    f"Training Analysis on {self.hostname}\nExperiment: {experiment}\nTime: {datetime.now().strftime('%H:%M:%S')}\n\n{analysis}",
                    level="info",
                )

            with self._state_lock:
                if (self.current_experiment or "unknown") == experiment:
                    self.last_analysis_time = now
                    if step_range:
                        self.last_analysis_step = step_range[1]

            self._write_activity("idle")

        except Exception as e:
            print(f"[monitor] Analysis error: {e}", file=sys.stderr)
            self._write_activity("idle")
            with self._state_lock:
                if (self.current_experiment or "unknown") == experiment:
                    self.last_analysis_time = now
                    if training_metrics:
                        self.last_analysis_step = max(m.step for m in training_metrics)

    # ------------------------------------------------------------------ #
    #  LLM analysis
    # ------------------------------------------------------------------ #

    def _maybe_run_analysis(self):
        now = time.time()
        with self._state_lock:
            has_metrics = len(self.training_metrics) > 0
            last_request_time = self.last_analysis_request_time
            last_request_step = self.last_analysis_request_step
            current_step = max((m.step for m in self.training_metrics), default=None)
            last_update_time = self.last_update_time
            is_idle = self.is_idle

        if not has_metrics:
            return

        time_ready = (
            last_request_time is None
            or now - last_request_time >= self.analysis_interval
        )

        step_ready = False
        if self.analysis_interval_steps and current_step is not None:
            if last_request_step is None:
                step_ready = True
            elif current_step - last_request_step >= self.analysis_interval_steps:
                step_ready = True

        if not time_ready and not step_ready:
            return

        if self.idle_timeout and last_update_time:
            time_since_update = now - last_update_time
            if time_since_update >= self.idle_timeout:
                if not is_idle:
                    with self._state_lock:
                        if not self.is_idle:
                            self.is_idle = True
                    print(
                        f"\n[monitor] No training updates for {time_since_update/60:.1f} min, pausing analyses",
                        file=sys.stderr,
                    )
                return

        reason = "interval" if time_ready else "step"
        self._enqueue_analysis_request(reason=reason)

    # ------------------------------------------------------------------ #
    #  Initial system check
    # ------------------------------------------------------------------ #

    def _run_initial_system_check(self):
        print(f"[monitor] Running initial system check...", file=sys.stderr)

        system_samples = []
        for _ in range(3):
            try:
                sample = collect_system_metrics(gpus=self.gpus)
                system_samples.append(sample)
                time.sleep(1)
            except Exception as e:
                print(f"[monitor] System sample error: {e}", file=sys.stderr)

        if not system_samples:
            print(f"[monitor] Could not collect system metrics for initial check", file=sys.stderr)
            return

        latest = system_samples[-1]

        gpu_info = []
        for gpu in latest.gpus:
            gpu_info.append(
                f"  GPU {gpu['index']}: {gpu['util_percent']}% util, "
                f"{gpu['mem_used_gb']:.1f}/{gpu['mem_total_gb']:.1f} GB mem, "
                f"{gpu['temp_c']}°C, {gpu['power_w']}W"
            )

        config_section = ""
        if self.experiment_config:
            try:
                cfg_text = safe_json_dumps(self.experiment_config)
                if len(cfg_text) > 3000:
                    cfg_text = cfg_text[:3000] + "\n... (truncated)"
                config_section = f"\nExperiment Configuration:\n{cfg_text}\n"
            except Exception:
                pass

        experiment_line = (
            f"Experiment: {self.current_experiment}"
            if self.current_experiment
            else "Experiment: (pending)"
        )
        project_line = f"Project: {self.project_name}"

        system_summary = f"""\
System Status at Monitor Startup
================================
Host: {self.hostname}
{project_line}
{experiment_line}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

GPUs:
{chr(10).join(gpu_info) if gpu_info else "  No GPUs detected"}

CPU: {latest.cpu_percent}%
RAM: {latest.ram_used_gb:.1f}/{latest.ram_total_gb:.1f} GB ({latest.ram_percent}%)
{config_section}
Monitor Configuration:
- Analysis interval: {self.analysis_interval/60:.1f} min
- ZMQ port: {self.zmq_port}
- Root dir: {self.root_dir}
"""

        startup_prompt = """\
You are checking the system status before a training run begins.

Review the system information and provide:
1. A brief assessment of system readiness (GPUs available, memory headroom, any concerns)
2. Note anything unusual (high baseline GPU usage, memory pressure, thermal issues)
3. If an experiment configuration is provided, briefly note any observations about the training setup \
(e.g. model size vs available VRAM, batch size considerations)
4. Confirm the system looks ready for training, or flag any issues

Keep it concise - just 3-5 sentences."""

        try:
            self._write_activity("system_check", "Running initial system check")
            messages = [
                {"role": "system", "content": startup_prompt},
                {"role": "user", "content": system_summary},
            ]
            analysis = self.analyzer._call_api_for_messages(
                messages,
                max_tokens=512,
                tool_manager=self.snapshot_tool_manager,
                on_tool_call=self._make_tool_call_reporter("system_check"),
            )
            self._write_activity("idle")

            print(f"\n{'='*60}", file=sys.stderr)
            print(f"Initial System Check", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)
            print(analysis, file=sys.stderr)
            print(f"{'='*60}\n", file=sys.stderr)

            self._notify(
                f"Training Monitor - System Check\nHost: {self.hostname}\n\n{analysis}",
                level="info",
            )

        except Exception as e:
            print(f"[monitor] Initial system check failed: {e}", file=sys.stderr)
            self._write_activity("idle")

    # ------------------------------------------------------------------ #
    #  Main loop
    # ------------------------------------------------------------------ #

    def run(self):
        try:
            print(f"[monitor] Starting on port {self.zmq_port}", file=sys.stderr)
            print(f"[monitor] Root dir: {self.root_dir}", file=sys.stderr)
            print(f"[monitor] Project: {self.project_name}", file=sys.stderr)
            if self.current_experiment:
                print(f"[monitor] Experiment: {self.current_experiment}", file=sys.stderr)
                print(f"[monitor] Run dir: {self.run_dir}", file=sys.stderr)
                print(f"[monitor] DB: {self.db_path}", file=sys.stderr)
            else:
                print(f"[monitor] Experiment: (waiting for first metric)", file=sys.stderr)

            print(f"[monitor] Analysis interval: {self.analysis_interval/60:.1f} min", file=sys.stderr)
            if self.analysis_interval_steps:
                print(f"[monitor] Analysis interval: {self.analysis_interval_steps} steps", file=sys.stderr)
            print(f"[monitor] Model: {self.analyzer.model}", file=sys.stderr)
            if self.analyzer.reasoning_effort:
                print(f"[monitor] Reasoning effort: {self.analyzer.reasoning_effort}", file=sys.stderr)
            if self.gpus:
                print(f"[monitor] Monitoring GPUs: {self.gpus}", file=sys.stderr)
            else:
                print(f"[monitor] Monitoring GPUs: all", file=sys.stderr)
            print(f"[monitor] Notify all analyses: {self.notify_all_analyses}", file=sys.stderr)
            print(
                f"[monitor] Idle timeout: {(self.idle_timeout/60):.1f} min"
                if self.idle_timeout
                else "[monitor] Idle timeout: disabled",
                file=sys.stderr,
            )
            print(
                f"[monitor] Telegram chat: {'enabled' if self.telegram_chat_handler else 'disabled'}",
                file=sys.stderr,
            )
            web_port = os.getenv("TM_WEBPAGE_PORT")
            if web_port:
                print(f"[monitor] Web chat: enabled on localhost:{web_port}", file=sys.stderr)
            print(
                f"[monitor] Conversation history token limit: {self.max_conversation_history_tokens}",
                file=sys.stderr,
            )

            gpu_str = f"GPUs {self.gpus}" if self.gpus else "all GPUs"
            self._notify(
                f"Training Monitor started on {self.hostname}\n"
                f"ZMQ port: {self.zmq_port}\n"
                f"Monitoring: {gpu_str}\n"
                f"Analysis interval: {self.analysis_interval/60:.1f} min\n"
                f"Root dir: {self.root_dir}\n"
                f"Project: {self.project_name}"
                + ("\nTelegram chat: enabled (reply to messages to chat)" if self.telegram_chat_handler else "")
                + (f"\nWeb chat: enabled on localhost:{web_port}" if web_port else ""),
                level="info",
            )

            # Generate code summary and metric descriptions at startup
            self._generate_code_summary()
            self._generate_metric_descriptions()

            self._start_analysis_worker()
            self._start_web_chat_worker()

            if self.telegram_chat_handler:
                self.telegram_chat_handler.start()

            last_system_poll = 0

            while self.running:
                now = time.time()

                self._receive_zmq_messages()

                if now - last_system_poll >= self.system_poll_interval:
                    self._collect_system_metrics()
                    last_system_poll = now

                self._maybe_run_analysis()

                time.sleep(0.1)
        finally:
            try:
                self.stop()
            except Exception:
                pass

    def stop(self):
        self.running = False
        self._stop_web_chat_worker()
        try:
            self._stop_analysis_worker()
        except Exception:
            pass
        try:
            if self.telegram_chat_handler:
                self.telegram_chat_handler.stop()
        except Exception:
            pass
        try:
            if hasattr(self, "zmq_socket") and self.zmq_socket is not None:
                self.zmq_socket.close()
        except Exception:
            pass
        try:
            if hasattr(self, "zmq_ctx") and self.zmq_ctx is not None:
                self.zmq_ctx.term()
        except Exception:
            pass
        try:
            if self.db_conn is not None:
                self.db_conn.close()
        except Exception:
            pass
