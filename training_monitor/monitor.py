"""TrainingMonitor — the main long-running monitoring process."""

import json
import socket
import sqlite3
import sys
import time
from collections import deque
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

try:
    import zmq
except ImportError:
    zmq = None

try:
    import requests
except ImportError:
    requests = None

from .analyzer import LLMAnalyzer
from .notify import notify
from .snapshot_tools import SnapshotToolManager
from .system_metrics import collect_system_metrics
from .telegram_chat import TelegramChatHandler
from .types import SystemMetric, TrainingMetric
from .utils import (
    default_experiment_dir,
    ensure_dir,
    hash_directory,
    safe_json_dumps,
    sanitize_experiment_name,
)


class TrainingMonitor:
    """
    Main monitoring process:
    - Receives metrics via ZMQ
    - Collects system metrics
    - Persists to SQLite
    - Runs periodic LLM analysis
    - Optionally enables interactive Telegram chat

    Everything is stored under root_dir/experiment_name/ once the experiment
    name is known.
    """

    def __init__(
        self,
        zmq_port: int = 5555,
        root_dir: str = "ai_logger",
        experiment: Optional[str] = None,
        db_path: str = "",
        analysis_interval_min: float = 5.0,
        analysis_interval_steps: Optional[int] = None,
        system_poll_interval_sec: float = 10.0,
        max_prompt_len: int = 8000,
        openai_model: str = "gpt-4o",
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
    ):
        self.zmq_port = zmq_port
        self.root_dir = Path(root_dir)
        ensure_dir(self.root_dir)

        self.analysis_interval = analysis_interval_min * 60
        self.analysis_interval_steps = analysis_interval_steps
        self.system_poll_interval = system_poll_interval_sec
        self.max_prompt_len = max_prompt_len
        self.gpus = gpus
        self.recovery_max_age_min = recovery_max_age_min
        self.notify_all_analyses = notify_all_analyses
        self.idle_timeout = idle_timeout_min * 60 if idle_timeout_min else None
        self.max_conversation_history_tokens = max_conversation_history_tokens
        self.regenerate_code_analysis = regenerate_code_analysis

        # State
        self.training_metrics: Deque[TrainingMetric] = deque(maxlen=10000)
        self.system_metrics: Deque[SystemMetric] = deque(maxlen=1000)
        self.last_analysis_time: Optional[float] = None
        self.last_analysis_step: Optional[int] = None
        self.last_update_time: Optional[float] = None
        self.is_idle: bool = False
        self.force_analysis_pending: bool = False
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
                    tool_manager=self.snapshot_tool_manager,
                )
                self.telegram_chat_handler.get_context_fn = self._build_chat_context
                self.telegram_chat_handler.on_exchange_fn = self._on_telegram_exchange
            except ValueError as e:
                print(f"[monitor] Telegram chat disabled: {e}", file=sys.stderr)

        # If experiment was provided, open storage immediately
        if self.current_experiment:
            self._open_experiment_storage(self.current_experiment)

    # ------------------------------------------------------------------ #
    #  Telegram exchange callback
    # ------------------------------------------------------------------ #

    def _on_telegram_exchange(self, role: str, text: str, msg_type: str) -> None:
        """Called by TelegramChatHandler when a user message or model response occurs."""
        entry: Dict[str, Any] = {
            "timestamp": time.time(),
            "role": role,
            "type": msg_type,
            "text": text,
        }
        self.conversation_history.append(entry)
        self._persist_conversation_message(entry)

    # ------------------------------------------------------------------ #
    #  Storage
    # ------------------------------------------------------------------ #

    def _open_experiment_storage(self, experiment: str) -> None:
        experiment = sanitize_experiment_name(experiment)
        self.current_experiment = experiment

        self.run_dir = default_experiment_dir(self.root_dir, experiment)
        ensure_dir(self.run_dir)

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

        # Load experiment config from config.json (written by AILogger) or from DB
        if self.experiment_config is None:
            config_path = self.run_dir / "config.json"
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        self.experiment_config = json.load(f)
                    self._persist_experiment_config(experiment, self.experiment_config)
                except Exception:
                    pass
            if self.experiment_config is None:
                self.experiment_config = self._load_experiment_config(experiment)

        # Load code summary from disk if it exists (e.g. recovery / restart)
        if self.code_summary is None:
            summary_path = self.run_dir / "code_summary.md"
            if summary_path.exists():
                try:
                    self.code_summary = summary_path.read_text(encoding="utf-8")
                    print(f"[monitor] Loaded existing code summary from {summary_path}", file=sys.stderr)
                except Exception:
                    pass

        if self.snapshot_tool_manager is None:
            self.snapshot_tool_manager = SnapshotToolManager(self.run_dir / "code")
        if self.telegram_chat_handler and self.snapshot_tool_manager:
            self.telegram_chat_handler.tool_manager = self.snapshot_tool_manager

        # Load metric descriptions from disk if they exist
        if self.metric_descriptions is None:
            desc_path = self.run_dir / "metric_descriptions.json"
            if desc_path.exists():
                try:
                    self.metric_descriptions = json.loads(desc_path.read_text(encoding="utf-8"))
                    print(f"[monitor] Loaded existing metric descriptions from {desc_path}", file=sys.stderr)
                except Exception:
                    pass

    def _build_chat_context(self) -> str:
        """Build a training context summary for the Telegram chat handler."""
        return self.analyzer.build_prompt(
            training_metrics=list(self.training_metrics),
            system_metrics=list(self.system_metrics),
            max_prompt_len=self.max_prompt_len,
            experiment_name=self.current_experiment or "unknown",
            conversation_history=list(self.conversation_history),
            max_conversation_history_tokens=self.max_conversation_history_tokens,
            config=self.experiment_config,
            code_summary=self.code_summary,
            metric_descriptions=self.metric_descriptions,
        )

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
            CREATE TABLE IF NOT EXISTS conversation_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment TEXT,
                timestamp REAL,
                role TEXT,
                msg_type TEXT,
                content TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_training_exp_step
                ON training_metrics(experiment, step);
            CREATE INDEX IF NOT EXISTS idx_system_ts
                ON system_metrics(timestamp);
            CREATE INDEX IF NOT EXISTS idx_conv_exp_ts
                ON conversation_messages(experiment, timestamp);
        """)
        self.db_conn.commit()

    def _persist_experiment_config(self, experiment: str, config: Dict[str, Any]) -> None:
        try:
            self.db_conn.execute(
                "INSERT INTO experiment_configs (experiment, timestamp, config) VALUES (?, ?, ?) "
                "ON CONFLICT(experiment) DO UPDATE SET timestamp=excluded.timestamp, config=excluded.config",
                (experiment, time.time(), safe_json_dumps(config)),
            )
            self.db_conn.commit()
        except Exception as e:
            print(f"[monitor] Error persisting config: {e}", file=sys.stderr)

    def _load_experiment_config(self, experiment: str) -> Optional[Dict[str, Any]]:
        try:
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
        self.db_conn.execute(
            "INSERT INTO training_metrics (experiment, step, timestamp, metrics) VALUES (?, ?, ?, ?)",
            (experiment, metric.step, metric.timestamp, json.dumps(metric.metrics)),
        )
        self.db_conn.commit()

    def _persist_system_metric(self, metric: SystemMetric):
        if self.db_conn is None:
            return
        self.db_conn.execute(
            "INSERT INTO system_metrics (timestamp, data) VALUES (?, ?)",
            (metric.timestamp, json.dumps(asdict(metric))),
        )
        self.db_conn.commit()

    def _persist_analysis(self, experiment: str, prompt: str, response: str, alerted: bool):
        if self.db_conn is None:
            return
        self.db_conn.execute(
            "INSERT INTO analyses (experiment, timestamp, prompt, response, alerted) VALUES (?, ?, ?, ?, ?)",
            (experiment, time.time(), prompt, response, int(alerted)),
        )
        self.db_conn.commit()

    def _persist_conversation_message(self, entry: Dict[str, Any]) -> None:
        """Persist a single conversation history entry to the DB."""
        if self.db_conn is None:
            return
        experiment = self.current_experiment or "unknown"
        try:
            self.db_conn.execute(
                "INSERT INTO conversation_messages (experiment, timestamp, role, msg_type, content) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    experiment,
                    entry["timestamp"],
                    entry["role"],
                    entry["type"],
                    entry["text"],
                ),
            )
            self.db_conn.commit()
        except Exception as e:
            print(f"[monitor] Error persisting conversation message: {e}", file=sys.stderr)

    def _analyses_jsonl_path(self, experiment: str) -> Optional[Path]:
        if self.run_dir is None:
            return None
        return self.run_dir / "analyses.jsonl"

    def _append_analysis_to_json(
        self,
        experiment: str,
        analysis: str,
        alerted: bool,
        step_range: Optional[Tuple[int, int]] = None,
    ):
        if not experiment or experiment == "unknown":
            return
        json_path = self._analyses_jsonl_path(experiment)
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

            cursor = self.db_conn.execute(
                """SELECT step, timestamp, metrics FROM training_metrics
                   WHERE experiment = ? AND timestamp > ?
                   ORDER BY step ASC""",
                (experiment, min_timestamp),
            )
            count = 0
            for row in cursor:
                step, timestamp, metrics_json = row
                self.training_metrics.append(TrainingMetric(
                    step=step, timestamp=timestamp, metrics=json.loads(metrics_json),
                ))
                count += 1

            cursor = self.db_conn.execute(
                """SELECT timestamp, data FROM system_metrics
                   WHERE timestamp > ?
                   ORDER BY timestamp ASC""",
                (min_timestamp,),
            )
            sys_count = 0
            for row in cursor:
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
                cursor = self.db_conn.execute(
                    """SELECT timestamp, role, msg_type, content
                       FROM conversation_messages
                       WHERE experiment = ? AND timestamp > ?
                       ORDER BY timestamp ASC""",
                    (experiment, min_timestamp),
                )
                for row in cursor:
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
                cursor = self.db_conn.execute(
                    """SELECT timestamp, response, alerted FROM analyses
                       WHERE experiment = ? AND timestamp > ?
                       ORDER BY timestamp ASC""",
                    (experiment, min_timestamp),
                )
                for row in cursor:
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
                self.last_update_time = now

                if self.is_idle:
                    self.is_idle = False
                    print(f"\n[monitor] Training activity resumed", file=sys.stderr)

                if msg.get("type") == "training_metric":
                    experiment = sanitize_experiment_name(msg.get("experiment") or "unknown")

                    if "config" in msg and msg["config"] is not None:
                        self.experiment_config = msg["config"]
                        self._persist_experiment_config(experiment, self.experiment_config)

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
                            steps = [m.step for m in self.training_metrics]
                            print(f"[monitor] Recovered {train_count} training metrics, {conv_count} conversation messages", file=sys.stderr)
                            print(f"[monitor] Step range: {min(steps)} - {max(steps)}", file=sys.stderr)
                            notify(
                                f"Monitor recovered on {self.hostname}\n"
                                f"Experiment: {experiment}\n"
                                f"Recovered {train_count} training metrics, {conv_count} conversation messages",
                                level="info",
                            )
                        else:
                            print(f"[monitor] No recovery data found, starting fresh", file=sys.stderr)

                        self.current_experiment = experiment
                        self.experiment_config = self._load_experiment_config(experiment)

                    # Experiment change mid-run
                    elif experiment and self.current_experiment and experiment != self.current_experiment:
                        print(f"\n[monitor] New experiment detected: {experiment}", file=sys.stderr)
                        print(f"[monitor] Previous: {self.current_experiment}", file=sys.stderr)

                        self.training_metrics.clear()
                        self.conversation_history.clear()
                        self._analysis_count = 0
                        self.last_analysis_time = None
                        self.last_analysis_step = None
                        self.force_analysis_pending = False
                        self.code_summary = None
                        self.metric_descriptions = None

                        self._open_experiment_storage(experiment)
                        notify(
                            f"New experiment started on {self.hostname}\n"
                            f"Experiment: {experiment}\n"
                            f"Previous: {self.current_experiment}",
                            level="info",
                        )
                        self.current_experiment = experiment
                        self.experiment_config = self._load_experiment_config(experiment)

                    metric = TrainingMetric(
                        step=msg["step"],
                        timestamp=msg["timestamp"],
                        metrics=msg["metrics"],
                    )
                    self.training_metrics.append(metric)
                    self._persist_training_metric(metric, experiment)

                    if msg.get("force_analysis"):
                        self.force_analysis_pending = True

                elif msg.get("type") == "reset":
                    print(f"\n[monitor] Received reset, clearing buffers", file=sys.stderr)
                    self.training_metrics.clear()
                    self.conversation_history.clear()
                    self._analysis_count = 0
                    self.last_analysis_time = None
                    self.last_analysis_step = None
                    self.current_experiment = sanitize_experiment_name(msg.get("experiment") or "unknown")
                    self.first_metric_received = False

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
        if self.code_summary is not None:
            return

        if self.run_dir is None:
            return

        code_dir = self.run_dir / "code"
        summary_path = self.run_dir / "code_summary.md"

        # Check per-run file (e.g. monitor restart within same experiment)
        if not self.regenerate_code_analysis and summary_path.exists():
            try:
                self.code_summary = summary_path.read_text(encoding="utf-8")
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
                    self.code_summary = cached_summary.read_text(encoding="utf-8")
                    # Copy into run_dir for quick access on restarts
                    try:
                        summary_path.write_text(self.code_summary, encoding="utf-8")
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
            summary = self.analyzer.generate_code_summary(
                tool_manager=self.snapshot_tool_manager,
            )
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

        except Exception as e:
            print(f"[monitor] Code summary generation failed: {e}", file=sys.stderr)

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
        if self.metric_descriptions is not None:
            return

        if self.run_dir is None:
            return

        code_dir = self.run_dir / "code"
        desc_path = self.run_dir / "metric_descriptions.json"

        # Check per-run file
        if not self.regenerate_code_analysis and desc_path.exists():
            try:
                self.metric_descriptions = json.loads(
                    desc_path.read_text(encoding="utf-8")
                )
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
                    self.metric_descriptions = json.loads(
                        cached_desc.read_text(encoding="utf-8")
                    )
                    # Copy into run_dir
                    try:
                        desc_path.write_text(
                            json.dumps(self.metric_descriptions, indent=2, sort_keys=True),
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
            descriptions = self.analyzer.generate_metric_descriptions(
                tool_manager=self.snapshot_tool_manager,
            )
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

        except Exception as e:
            print(f"[monitor] Metric descriptions generation failed: {e}", file=sys.stderr)

    def _persist_metric_descriptions(self) -> None:
        """Save current metric descriptions to run_dir and shared cache."""
        if self.metric_descriptions is None:
            return

        desc_json = json.dumps(self.metric_descriptions, indent=2, sort_keys=True)

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

        if self.metric_descriptions is None:
            self.metric_descriptions = {}

        for round_num in range(1, 6):  # max 5 rounds
            # Collect all metric keys from training data
            all_keys: set[str] = set()
            for m in self.training_metrics:
                all_keys.update(m.metrics.keys())

            # Find missing
            missing = sorted(all_keys - set(self.metric_descriptions.keys()))

            if not missing:
                return

            print(
                f"[monitor] Filling missing metric descriptions "
                f"(round {round_num}/5): {len(missing)} missing — {missing}",
                file=sys.stderr,
            )

            try:
                new_descriptions = self.analyzer.generate_missing_metric_descriptions(
                    missing_metrics=missing,
                    existing_descriptions=self.metric_descriptions,
                    tool_manager=self.snapshot_tool_manager,
                )
            except Exception as e:
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
            self.metric_descriptions.update(new_descriptions)

            print(
                f"[monitor] Round {round_num}/5: added "
                f"{len(new_descriptions)} description(s):",
                file=sys.stderr,
            )
            for k, v in sorted(new_descriptions.items()):
                print(f"  {k}: {v}", file=sys.stderr)

            # Persist after each successful round
            self._persist_metric_descriptions()

        # Report any still-missing after all rounds
        all_keys = set()
        for m in self.training_metrics:
            all_keys.update(m.metrics.keys())
        still_missing = sorted(all_keys - set(self.metric_descriptions.keys()))
        if still_missing:
            print(
                f"[monitor] After 5 rounds, {len(still_missing)} metric(s) "
                f"still lack descriptions: {still_missing}",
                file=sys.stderr,
            )

    # ------------------------------------------------------------------ #
    #  LLM analysis
    # ------------------------------------------------------------------ #

    def _maybe_run_analysis(self):
        now = time.time()

        force_now = self.force_analysis_pending
        if force_now:
            self.force_analysis_pending = False
            print(f"[monitor] Forced analysis requested", file=sys.stderr)

        if not force_now:
            time_ready = (
                self.last_analysis_time is None
                or now - self.last_analysis_time >= self.analysis_interval
            )

            step_ready = False
            if self.analysis_interval_steps and self.training_metrics:
                current_step = max(m.step for m in self.training_metrics)
                if self.last_analysis_step is None:
                    step_ready = True
                elif current_step - self.last_analysis_step >= self.analysis_interval_steps:
                    step_ready = True

            if not time_ready and not step_ready:
                return

        if len(self.training_metrics) == 0:
            return

        if not force_now:
            if self.idle_timeout and self.last_update_time:
                time_since_update = now - self.last_update_time
                if time_since_update >= self.idle_timeout:
                    if not self.is_idle:
                        self.is_idle = True
                        print(f"\n[monitor] No training updates for {time_since_update/60:.1f} min, pausing analyses", file=sys.stderr)
                    return

        print(f"[monitor] Running LLM analysis...", file=sys.stderr)

        # Fill in descriptions for any newly-observed metrics before analysis
        self._fill_missing_metric_descriptions()

        try:
            experiment = self.current_experiment or "unknown"

            analysis, should_alert = self.analyzer.analyze(
                training_metrics=list(self.training_metrics),
                system_metrics=list(self.system_metrics),
                max_prompt_len=self.max_prompt_len,
                experiment_name=experiment,
                conversation_history=list(self.conversation_history),
                max_conversation_history_tokens=self.max_conversation_history_tokens,
                config=self.experiment_config,
                code_summary=self.code_summary,
                metric_descriptions=self.metric_descriptions,
                tool_manager=self.snapshot_tool_manager,
            )

            # Add analysis to unified conversation history
            analysis_entry: Dict[str, Any] = {
                "timestamp": now,
                "role": "model",
                "type": "analysis",
                "text": analysis,
            }
            self.conversation_history.append(analysis_entry)
            self._persist_conversation_message(analysis_entry)
            self._analysis_count += 1

            step_range = None
            if self.training_metrics:
                steps = [m.step for m in self.training_metrics]
                step_range = (min(steps), max(steps))

            self._append_analysis_to_json(experiment, analysis, should_alert, step_range)

            # Build prompt for persistence (exclude the analysis we just added)
            prompt = self.analyzer.build_prompt(
                list(self.training_metrics),
                list(self.system_metrics),
                self.max_prompt_len,
                experiment,
                conversation_history=list(self.conversation_history)[:-1],
                max_conversation_history_tokens=self.max_conversation_history_tokens,
                config=self.experiment_config,
                code_summary=self.code_summary,
                metric_descriptions=self.metric_descriptions,
            )
            self._persist_analysis(experiment, prompt, analysis, should_alert)

            print(f"\n{'='*60}", file=sys.stderr)
            print(f"Analysis at {datetime.now().strftime('%H:%M:%S')} (#{self._analysis_count})", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)
            print(analysis, file=sys.stderr)
            print(f"{'='*60}\n", file=sys.stderr)

            if should_alert:
                notify(
                    f"Training Alert on {self.hostname}\nExperiment: {experiment}\n\n{analysis}",
                    level="alert",
                )
            elif self.notify_all_analyses:
                notify(
                    f"Training Analysis on {self.hostname}\nExperiment: {experiment}\n\n{analysis}",
                    level="info",
                )

            self.last_analysis_time = now
            if step_range:
                self.last_analysis_step = step_range[1]

        except Exception as e:
            print(f"[monitor] Analysis error: {e}", file=sys.stderr)
            self.last_analysis_time = now
            if self.training_metrics:
                self.last_analysis_step = max(m.step for m in self.training_metrics)

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

        system_summary = f"""\
System Status at Monitor Startup
================================
Host: {self.hostname}
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
            messages = [
                {"role": "system", "content": startup_prompt},
                {"role": "user", "content": system_summary},
            ]
            analysis = self.analyzer._call_api_for_messages(
                messages,
                max_tokens=512,
                tool_manager=self.snapshot_tool_manager,
            )

            print(f"\n{'='*60}", file=sys.stderr)
            print(f"Initial System Check", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)
            print(analysis, file=sys.stderr)
            print(f"{'='*60}\n", file=sys.stderr)

            notify(
                f"Training Monitor - System Check\nHost: {self.hostname}\n\n{analysis}",
                level="info",
            )

        except Exception as e:
            print(f"[monitor] Initial system check failed: {e}", file=sys.stderr)

    # ------------------------------------------------------------------ #
    #  Main loop
    # ------------------------------------------------------------------ #

    def run(self):
        try:
            print(f"[monitor] Starting on port {self.zmq_port}", file=sys.stderr)
            print(f"[monitor] Root dir: {self.root_dir}", file=sys.stderr)
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
            print(
                f"[monitor] Conversation history token limit: {self.max_conversation_history_tokens}",
                file=sys.stderr,
            )

            gpu_str = f"GPUs {self.gpus}" if self.gpus else "all GPUs"
            notify(
                f"Training Monitor started on {self.hostname}\n"
                f"ZMQ port: {self.zmq_port}\n"
                f"Monitoring: {gpu_str}\n"
                f"Analysis interval: {self.analysis_interval/60:.1f} min\n"
                f"Root dir: {self.root_dir}"
                + ("\nTelegram chat: enabled (reply to messages to chat)" if self.telegram_chat_handler else ""),
                level="info",
            )

            # Generate code summary and metric descriptions before the initial system check
            self._generate_code_summary()
            self._generate_metric_descriptions()

            self._run_initial_system_check()

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