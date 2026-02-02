"""AILogger — lightweight client for training scripts to send metrics."""

import atexit
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import zmq
except ImportError:
    zmq = None

from .utils import (
    create_code_snapshot,
    default_experiment_dir,
    ensure_dir,
    find_available_port,
    port_is_listening,
    port_is_available,
    resolve_under,
    safe_json_dumps,
    sanitize_experiment_name,
    should_spawn_monitor,
)


_ACTIVE_LOGGER: Optional["AILogger"] = None


def init(**kwargs: Any) -> "AILogger":
    """Initialize the global logger (wandb-style)."""
    global _ACTIVE_LOGGER
    _ACTIVE_LOGGER = AILogger.init(**kwargs)
    return _ACTIVE_LOGGER


def get_logger() -> Optional["AILogger"]:
    return _ACTIVE_LOGGER


def log(metrics: Dict[str, Any], step: int, force_analysis: bool = False) -> None:
    if _ACTIVE_LOGGER is None:
        raise RuntimeError("training_monitor.init(...) must be called before log().")
    _ACTIVE_LOGGER.log(metrics, step, force_analysis=force_analysis)


def reset(new_experiment_name: Optional[str] = None) -> None:
    if _ACTIVE_LOGGER is None:
        raise RuntimeError("training_monitor.init(...) must be called before reset().")
    _ACTIVE_LOGGER.reset(new_experiment_name=new_experiment_name)


def close() -> None:
    global _ACTIVE_LOGGER
    if _ACTIVE_LOGGER is None:
        return
    _ACTIVE_LOGGER.close()
    _ACTIVE_LOGGER = None


class AILogger:
    """
    Lightweight logger for use in training scripts.

    Writes everything under root_dir/experiment_name/ and can
    auto-start the monitor subprocess.
    """

    def __init__(
        self,
        zmq_port: int = 5555,
        db_path: Optional[str] = None,
        experiment_name: Optional[str] = None,
        *,
        root_dir: str = "ai_logger",
        config: Optional[Dict[str, Any]] = None,
        # autostart-related
        start_monitor: bool | str = False,
        monitor_host: str = "127.0.0.1",
        monitor_args: Optional[Dict[str, Any]] = None,
        monitor_log_path: Optional[str] = None,
        terminate_monitor_on_close: bool = True,
        wait_for_monitor: bool = True,
    ):
        self.experiment_name = sanitize_experiment_name(experiment_name or f"exp_{int(time.time())}")
        self.root_dir = Path(root_dir)
        self.run_dir = default_experiment_dir(self.root_dir, self.experiment_name)
        ensure_dir(self.run_dir)

        self.config = config or None
        self._config_sent = False

        # Write config for monitor to pick up at startup (before spawning it)
        if self.config:
            try:
                with open(self.run_dir / "config.json", "w") as f:
                    f.write(safe_json_dumps(self.config))
            except Exception:
                pass

        self.zmq_socket = None
        self.db_conn = None

        self._monitor_proc: Optional[subprocess.Popen] = None
        self._monitor_started_by_us: bool = False
        self._terminate_monitor_on_close = terminate_monitor_on_close
        self._monitor_host = monitor_host
        self._zmq_port = zmq_port

        # Resolve optional local backup DB under run_dir (if relative)
        db_path_resolved: Optional[Path] = resolve_under(self.run_dir, db_path)

        # Resolve monitor log path under run_dir
        monitor_log_path_resolved: Optional[Path] = resolve_under(self.run_dir, monitor_log_path)

        # Optionally start monitor first (so port is bound and receiving)
        if start_monitor and should_spawn_monitor(start_monitor):
            if not port_is_available(zmq_port):
                new_port = find_available_port(zmq_port + 1)
                if new_port is None:
                    print(
                        f"[AILogger] ZMQ port {zmq_port} is unavailable; "
                        "continuing with the requested port.",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"[AILogger] ZMQ port {zmq_port} is unavailable; "
                        f"using port {new_port} for this run.",
                        file=sys.stderr,
                    )
                    self._zmq_port = new_port
            self._ensure_monitor_running(
                host=monitor_host,
                port=self._zmq_port,
                monitor_args=monitor_args or {},
                monitor_log_path=str(monitor_log_path_resolved) if monitor_log_path_resolved else None,
                wait_for_monitor=wait_for_monitor,
            )

        # Setup ZMQ
        if zmq is not None:
            try:
                ctx = zmq.Context.instance()
                self.zmq_socket = ctx.socket(zmq.PUSH)
                self.zmq_socket.setsockopt(zmq.SNDHWM, 10000)
                self.zmq_socket.setsockopt(zmq.LINGER, 1000)
                self.zmq_socket.connect(f"tcp://{monitor_host}:{self._zmq_port}")
            except Exception as e:
                print(f"[AILogger] ZMQ setup failed: {e}", file=sys.stderr)

        # Setup SQLite direct-write backup
        if db_path_resolved:
            import sqlite3
            ensure_dir(db_path_resolved.parent)
            self.db_conn = sqlite3.connect(str(db_path_resolved), check_same_thread=False)
            self._init_db()

        atexit.register(self.close)

    @classmethod
    def init(
        cls,
        *,
        experiment_name: str,
        zmq_port: int = 5555,
        root_dir: str = "ai_logger",
        config: Optional[Dict[str, Any]] = None,
        training_command: Optional[str] = None,

        # Monitor configuration
        monitor_db_path: str = "training_monitor.db",
        analysis_interval_min: float = 5.0,
        system_poll_sec: float = 10.0,
        max_prompt_len: int = 8000,
        openai_model: str = "gpt-4o",
        openai_base_url: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        gpus: Optional[List[int]] = None,
        notify_all_analyses: bool = True,
        sig_figs: int = 5,
        idle_timeout_min: Optional[float] = None,
        max_conversation_history_tokens: int = 5000,

        # behavior knobs
        start_monitor: bool | str = "auto",
        monitor_host: str = "127.0.0.1",
        monitor_log_path: Optional[str] = None,
        terminate_monitor_on_close: bool = True,
        telegram_chat: bool = False,

        # optional local backup db for the training process itself
        local_backup_db_path: Optional[str] = None,

        # code snapshot
        code_snapshot_manifest: Optional[str] = "snapshot_files.txt",

        # code analysis caching
        regenerate_code_analysis: bool = False,
    ) -> "AILogger":
        """
        wandb-like entrypoint.  Starts monitor if needed, then returns a logger.

        Layout:
          run_dir = root_dir/experiment_name/
          monitor db default: run_dir/training_monitor.db
          analyses jsonl:     run_dir/analyses.jsonl
          monitor log:        run_dir/training_monitor.log
          code snapshot:      run_dir/code/
        """
        exp_sanitized = sanitize_experiment_name(experiment_name)
        root = Path(root_dir)
        run_dir = default_experiment_dir(root, exp_sanitized)
        ensure_dir(run_dir)

        default_log_path = monitor_log_path or "training_monitor.log"

        config_with_command = dict(config) if config else {}
        if training_command is not None:
            config_with_command["training_command"] = training_command
        elif "training_command" not in config_with_command:
            config_with_command["training_command"] = " ".join(sys.argv)

        monitor_args: Dict[str, Any] = {
            "root_dir": str(root_dir),
            "config": config_with_command,
            "experiment": exp_sanitized,
            "db_path": monitor_db_path,
            "analysis_interval_min": analysis_interval_min,
            "system_poll_sec": system_poll_sec,
            "max_prompt_len": max_prompt_len,
            "openai_model": openai_model,
            "openai_base_url": openai_base_url,
            "reasoning_effort": reasoning_effort,
            "gpus": gpus,
            "notify_all_analyses": notify_all_analyses,
            "sig_figs": sig_figs,
            "idle_timeout_min": idle_timeout_min,
            "telegram_chat": telegram_chat,
            "max_conversation_history_tokens": max_conversation_history_tokens,
            "regenerate_code_analysis": regenerate_code_analysis,
        }

        # Code snapshot
        if code_snapshot_manifest is not None:
            code_dir = run_dir / "code"
            n = create_code_snapshot(Path(code_snapshot_manifest), code_dir)
            if n > 0:
                print(f"[AILogger] Code snapshot: {n} file(s) → {code_dir}", file=sys.stderr)

        return cls(
            zmq_port=zmq_port,
            db_path=local_backup_db_path,
            experiment_name=exp_sanitized,
            root_dir=root_dir,
            config=config_with_command,
            start_monitor=start_monitor,
            monitor_host=monitor_host,
            monitor_args=monitor_args,
            monitor_log_path=default_log_path,
            terminate_monitor_on_close=terminate_monitor_on_close,
            wait_for_monitor=True,
        )

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _init_db(self):
        if self.db_conn:
            self.db_conn.execute("""
                CREATE TABLE IF NOT EXISTS training_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment TEXT,
                    step INTEGER,
                    timestamp REAL,
                    metrics TEXT
                )
            """)
            self.db_conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_exp_step
                ON training_metrics(experiment, step)
            """)
            self.db_conn.commit()

    def _ensure_monitor_running(
        self,
        host: str,
        port: int,
        monitor_args: Dict[str, Any],
        monitor_log_path: Optional[str],
        wait_for_monitor: bool,
    ) -> None:
        if port_is_listening(host, port):
            return

        # The monitor is invoked as `python -m training_monitor ...`
        cmd = [
            sys.executable, "-u", "-m", "training_monitor",
            "--zmq-port", str(port),
            "--root-dir", str(monitor_args.get("root_dir", "ai_logger")),
            "--experiment", str(monitor_args.get("experiment", self.experiment_name)),
            "--db-path", str(monitor_args.get("db_path", "")),
            "--analysis-interval-min", str(monitor_args.get("analysis_interval_min", 5.0)),
            "--system-poll-sec", str(monitor_args.get("system_poll_sec", 10.0)),
            "--max-prompt-len", str(monitor_args.get("max_prompt_len", 8000)),
            "--openai-model", str(monitor_args.get("openai_model", "gpt-4o")),
            "--sig-figs", str(monitor_args.get("sig_figs", 5)),
            "--max-conversation-history-tokens", str(monitor_args.get("max_conversation_history_tokens", 5000)),
        ]

        if monitor_args.get("openai_base_url"):
            cmd += ["--openai-base-url", str(monitor_args["openai_base_url"])]
        if monitor_args.get("reasoning_effort"):
            cmd += ["--reasoning-effort", str(monitor_args["reasoning_effort"])]

        gpus = monitor_args.get("gpus")
        if gpus is not None:
            cmd += ["--gpus", ",".join(str(x) for x in gpus)]
        if monitor_args.get("notify_all_analyses", True):
            cmd += ["--notify-all-analyses"]
        else:
            cmd += ["--no-notify-all-analyses"]
        if monitor_args.get("idle_timeout_min") is not None:
            cmd += ["--idle-timeout-min", str(monitor_args["idle_timeout_min"])]
        if monitor_args.get("telegram_chat"):
            cmd += ["--telegram-chat"]
        if monitor_args.get("regenerate_code_analysis"):
            cmd += ["--regenerate-code-analysis"]

        stdout = None
        stderr = None
        f = None
        if monitor_log_path:
            logp = Path(monitor_log_path)
            ensure_dir(logp.parent)
            f = open(logp, "a", buffering=1)
            stdout = f
            stderr = f

        try:
            self._monitor_proc = subprocess.Popen(
                cmd,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
                env=os.environ.copy(),
            )
            self._monitor_started_by_us = True
        except Exception as e:
            print(f"[AILogger] Failed to spawn monitor: {e}", file=sys.stderr)
            try:
                if f:
                    f.close()
            except Exception:
                pass
            return

        if wait_for_monitor:
            for _ in range(50):
                if port_is_listening(host, port):
                    return
                time.sleep(0.1)
            print(f"[AILogger] Monitor did not become ready on {host}:{port}", file=sys.stderr)

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def log(self, metrics: Dict[str, Any], step: int, force_analysis: bool = False) -> None:
        timestamp = time.time()
        msg: Dict[str, Any] = {
            "type": "training_metric",
            "experiment": self.experiment_name,
            "step": step,
            "timestamp": timestamp,
            "metrics": metrics,
            "force_analysis": force_analysis,
        }

        if (self.config is not None) and (not self._config_sent):
            msg["config"] = self.config
            self._config_sent = True

        if self.zmq_socket:
            try:
                self.zmq_socket.send_json(msg, zmq.NOBLOCK)
            except zmq.Again:
                pass
            except Exception as e:
                print(f"[AILogger] ZMQ send error: {e}", file=sys.stderr)

        if self.db_conn:
            try:
                self.db_conn.execute(
                    "INSERT INTO training_metrics (experiment, step, timestamp, metrics) VALUES (?, ?, ?, ?)",
                    (self.experiment_name, step, timestamp, json.dumps(metrics)),
                )
                self.db_conn.commit()
            except Exception as e:
                print(f"[AILogger] SQLite write error: {e}", file=sys.stderr)

    def reset(self, new_experiment_name: Optional[str] = None) -> None:
        if new_experiment_name:
            self.experiment_name = sanitize_experiment_name(new_experiment_name)

        if self.zmq_socket:
            try:
                self.zmq_socket.send_json({
                    "type": "reset",
                    "timestamp": time.time(),
                    "experiment": self.experiment_name,
                }, zmq.NOBLOCK)
            except Exception:
                pass

    def close(self):
        if self._monitor_started_by_us and self._terminate_monitor_on_close and self.zmq_socket:
            try:
                self.zmq_socket.send_json({"type": "shutdown", "timestamp": time.time()}, zmq.NOBLOCK)
            except Exception:
                pass

        if self.zmq_socket:
            try:
                self.zmq_socket.close()
            except Exception:
                pass
            self.zmq_socket = None

        if self.db_conn:
            try:
                self.db_conn.close()
            except Exception:
                pass
            self.db_conn = None

        if self._monitor_started_by_us and self._terminate_monitor_on_close and self._monitor_proc is not None:
            try:
                self._monitor_proc.terminate()
            except Exception:
                pass
            self._monitor_proc = None
            self._monitor_started_by_us = False