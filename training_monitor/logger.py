"""AILogger — lightweight client for training scripts to send metrics."""

import atexit
import importlib.util
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

try:
    import wandb
except ImportError:
    wandb = None

from .utils import (
    create_code_snapshot,
    create_code_snapshot_from_roots,
    default_experiment_dir,
    default_project_dir,
    ensure_dir,
    find_available_port,
    port_is_listening,
    port_is_available,
    resolve_under,
    safe_json_dumps,
    sanitize_experiment_name,
    sanitize_project_name,
    should_spawn_monitor,
)
from .notify import notify


_ACTIVE_LOGGER: Optional["AILogger"] = None
_NO_LOGGER_WARNED = False


class _NoOpLogger:
    """Fallback logger that never raises (used when init fails)."""

    def __init__(self, reason: Optional[str] = None):
        self.reason = reason or "unknown error"

    def log(self, *args: Any, **kwargs: Any) -> None:
        pass

    def reset(self, *args: Any, **kwargs: Any) -> None:
        pass

    def close(self) -> None:
        pass


def _snapshot_dir_is_empty(path: Path) -> bool:
    if not path.exists():
        return True
    for entry in path.rglob("*"):
        if entry.is_file():
            return False
    return True


def _supports_osc8(stream: Any) -> bool:
    try:
        if not hasattr(stream, "isatty") or not stream.isatty():
            return False
    except Exception:
        return False
    term = os.environ.get("TERM", "")
    if term.lower() == "dumb":
        return False
    return True


def _format_terminal_link(url: str, label: Optional[str] = None, stream: Any = sys.stderr) -> str:
    if not _supports_osc8(stream):
        return url
    esc = "\x1b"
    text = label or url
    return f"{esc}]8;;{url}{esc}\\{text}{esc}]8;;{esc}\\"


def _find_chainlit_app_root() -> Optional[Path]:
    try:
        here = Path(__file__).resolve()
    except Exception:
        return None
    for parent in here.parents:
        if (parent / "public").is_dir():
            return parent
    return None


def init(**kwargs: Any) -> "AILogger":
    """Initialize the global logger (wandb-style)."""
    global _ACTIVE_LOGGER
    try:
        _ACTIVE_LOGGER = AILogger.init(**kwargs)
    except Exception as e:
        print(f"[AILogger] init failed: {e}", file=sys.stderr)
        _ACTIVE_LOGGER = _NoOpLogger(reason=str(e))  # type: ignore[assignment]
    return _ACTIVE_LOGGER


def get_logger() -> Optional["AILogger"]:
    return _ACTIVE_LOGGER


def log(
    metrics: Dict[str, Any],
    step: int,
    force_analysis: bool = False,
    openai_model: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    **wandb_kwargs: Any,
) -> None:
    global _NO_LOGGER_WARNED
    if _ACTIVE_LOGGER is None:
        if not _NO_LOGGER_WARNED:
            print("[AILogger] log() called before init(); ignoring.", file=sys.stderr)
            _NO_LOGGER_WARNED = True
        return
    try:
        _ACTIVE_LOGGER.log(
            metrics,
            step,
            force_analysis=force_analysis,
            openai_model=openai_model,
            reasoning_effort=reasoning_effort,
            **wandb_kwargs,
        )
    except Exception as e:
        print(f"[AILogger] log() failed: {e}", file=sys.stderr)


def reset(new_experiment_name: Optional[str] = None) -> None:
    if _ACTIVE_LOGGER is None:
        return
    try:
        _ACTIVE_LOGGER.reset(new_experiment_name=new_experiment_name)
    except Exception as e:
        print(f"[AILogger] reset() failed: {e}", file=sys.stderr)


def close() -> None:
    global _ACTIVE_LOGGER
    if _ACTIVE_LOGGER is None:
        return
    try:
        _ACTIVE_LOGGER.close()
    except Exception as e:
        print(f"[AILogger] close() failed: {e}", file=sys.stderr)
    _ACTIVE_LOGGER = None


class AILogger:
    """
    Lightweight logger for use in training scripts.

    Writes everything under root_dir/project/experiment_name/ and can
    auto-start the monitor subprocess.
    """

    def __init__(
        self,
        zmq_port: int = 5555,
        db_path: Optional[str] = None,
        experiment_name: Optional[str] = None,
        *,
        root_dir: str = "ai_logger",
        project: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        # autostart-related
        start_monitor: bool | str = False,
        monitor_host: str = "127.0.0.1",
        monitor_args: Optional[Dict[str, Any]] = None,
        monitor_log_path: Optional[str] = None,
        terminate_monitor_on_close: bool = True,
        wait_for_monitor: bool = True,
        use_wandb: bool = False,
        webpage: bool = True,
        webpage_port: Optional[int] = None,
    ):
        self.project_name = sanitize_project_name(project)
        self.experiment_name = sanitize_experiment_name(experiment_name or f"exp_{int(time.time())}")
        self.root_dir = Path(root_dir)
        self.project_dir = default_project_dir(self.root_dir, self.project_name)
        self.run_dir = default_experiment_dir(self.root_dir, self.experiment_name, self.project_name)
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
        self._use_wandb = use_wandb
        self._web_proc: Optional[subprocess.Popen] = None
        self._web_started_by_us = False
        self._web_port: Optional[int] = None

        # Resolve optional local backup DB under run_dir (if relative)
        db_path_resolved: Optional[Path] = resolve_under(self.run_dir, db_path)

        # Resolve monitor log path under run_dir
        monitor_log_path_resolved: Optional[Path] = resolve_under(self.run_dir, monitor_log_path)

        # Optional web UI preflight (so startup notice can include the port)
        web_port: Optional[int] = None
        web_enabled = webpage and should_spawn_monitor("auto")
        if web_enabled:
            if importlib.util.find_spec("chainlit") is None:
                print(
                    "[AILogger] Chainlit is not installed; skipping web UI. "
                    "Install with: pip install -e \".[web]\"",
                    file=sys.stderr,
                )
                web_enabled = False
            else:
                candidate_port = webpage_port or 8000
                if not port_is_available(candidate_port, host="127.0.0.1"):
                    candidate_port = find_available_port(candidate_port + 1, host="127.0.0.1")
                web_port = candidate_port
                if web_port is None:
                    print(
                        "[AILogger] No available port found for Chainlit; skipping web UI.",
                        file=sys.stderr,
                    )
                    web_enabled = False

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
                extra_env={"TM_WEBPAGE_PORT": str(web_port)} if web_port else None,
            )

        # Optionally start Chainlit web chat (read-only mirror)
        if web_enabled and web_port is not None:
            monitor_db_path = None
            if monitor_args is not None:
                monitor_db_path = monitor_args.get("db_path")
            if not monitor_db_path:
                monitor_db_path = "training_monitor.db"
            resolved_db_path = resolve_under(self.run_dir, str(monitor_db_path))
            if resolved_db_path is None:
                resolved_db_path = self.run_dir / "training_monitor.db"
            self._ensure_webpage_running(
                host="127.0.0.1",
                port=web_port,
                db_path=resolved_db_path,
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
        name: Optional[str] = None,
        experiment_name: Optional[str] = None,
        project: Optional[str] = None,
        zmq_port: int = 5555,
        root_dir: str = "ai_logger",
        config: Optional[Dict[str, Any]] = None,
        training_command: Optional[str] = None,
        use_wandb: bool = False,
        visible_directories: Optional[List[str]] = None,

        # Monitor configuration
        monitor_db_path: str = "training_monitor.db",
        analysis_interval_min: float = 5.0,
        system_poll_sec: float = 10.0,
        max_prompt_len: int = 8000,
        openai_model: str = "gpt-5.2",
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
        webpage: bool = True,
        webpage_port: Optional[int] = None,

        # optional local backup db for the training process itself
        local_backup_db_path: Optional[str] = None,

        # code snapshot (if code_snapshot_all is True, ignore manifest)
        code_snapshot_manifest: Optional[str] = "snapshot_files.txt",
        code_snapshot_paths: Optional[List[str]] = None,
        code_snapshot_all: bool = False,

        # code analysis caching
        regenerate_code_analysis: bool = False,
        # extra args forwarded to wandb.init when use_wandb=True
        **wandb_kwargs: Any,
    ) -> "AILogger":
        """
        wandb-like entrypoint.  Starts monitor if needed, then returns a logger.

        Layout:
          run_dir = root_dir/project/experiment_name/
          monitor db default: run_dir/training_monitor.db
          analyses jsonl:     run_dir/analyses.jsonl
          monitor log:        run_dir/training_monitor.log
          code snapshot:      run_dir/code/
        """
        run_name = name or experiment_name
        if not run_name:
            raise ValueError("name must be provided (or experiment_name for compatibility).")
        exp_sanitized = sanitize_experiment_name(run_name)
        project_name = sanitize_project_name(project)
        root = Path(root_dir)
        run_dir = default_experiment_dir(root, exp_sanitized, project_name)
        ensure_dir(run_dir)

        default_log_path = monitor_log_path or "training_monitor.log"

        config_with_command = dict(config) if config else {}
        if training_command is not None:
            config_with_command["training_command"] = training_command
        elif "training_command" not in config_with_command:
            config_with_command["training_command"] = " ".join(sys.argv)

        if use_wandb:
            if wandb is None:
                raise RuntimeError("wandb is not installed. Install it or set use_wandb=False.")
            wandb_init_kwargs = dict(wandb_kwargs)
            wandb_init_kwargs.setdefault("name", run_name)
            if config_with_command and "config" not in wandb_init_kwargs:
                wandb_init_kwargs["config"] = config_with_command
            wandb.init(**wandb_init_kwargs)

        monitor_args: Dict[str, Any] = {
            "root_dir": str(root_dir),
            "project": project_name,
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
        if visible_directories:
            resolved_visible: List[str] = []
            for entry in visible_directories:
                if not entry:
                    continue
                try:
                    resolved = str(Path(entry).expanduser().resolve())
                except Exception:
                    continue
                resolved_visible.append(resolved)
            if resolved_visible:
                monitor_args["visible_directories"] = resolved_visible

        # Code snapshot
        snapshot_roots: List[Path] = []
        if code_snapshot_paths:
            for entry in code_snapshot_paths:
                if not entry:
                    continue
                try:
                    snapshot_roots.append(Path(entry).expanduser())
                except Exception:
                    continue

        if code_snapshot_all:
            code_dir = run_dir / "code"
            n = create_code_snapshot_from_roots(
                [Path(".")],
                code_dir,
                exclude_dirs=[Path(root_dir)],
            )
            if n > 0:
                print(f"[AILogger] Code snapshot: {n} file(s) → {code_dir}", file=sys.stderr)
            elif _snapshot_dir_is_empty(code_dir):
                warning = (
                    "[AILogger] Warning: code snapshot directory is empty "
                    f"at {code_dir}. LLM analysis will lack code context. "
                    "To include code, either set code_snapshot_all=True, "
                    "or pass code_snapshot_paths=[...] with files/dirs, "
                    "or create a snapshot_files.txt manifest listing files/dirs "
                    "and pass code_snapshot_manifest=... . "
                    "You can also set visible_directories=[...] for live, read-only access."
                )
                print(warning, file=sys.stderr)
                if telegram_chat and os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"):
                    notify(warning, level="info", allow_telegram=True)
        elif snapshot_roots:
            code_dir = run_dir / "code"
            if code_snapshot_manifest is not None:
                warning = (
                    "[AILogger] Warning: both code_snapshot_manifest and code_snapshot_paths "
                    "were provided. Using code_snapshot_paths and ignoring the manifest."
                )
                print(warning, file=sys.stderr)
                if telegram_chat and os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"):
                    notify(warning, level="info", allow_telegram=True)
            n = create_code_snapshot_from_roots(
                snapshot_roots,
                code_dir,
                base_dir=Path.cwd(),
            )
            if n > 0:
                print(f"[AILogger] Code snapshot: {n} file(s) → {code_dir}", file=sys.stderr)
            elif _snapshot_dir_is_empty(code_dir):
                warning = (
                    "[AILogger] Warning: code snapshot directory is empty "
                    f"at {code_dir}. LLM analysis will lack code context. "
                    "To include code, either set code_snapshot_all=True, "
                    "or pass code_snapshot_paths=[...] with files/dirs, "
                    "or create a snapshot_files.txt manifest listing files/dirs "
                    "and pass code_snapshot_manifest=... . "
                    "You can also set visible_directories=[...] for live, read-only access."
                )
                print(warning, file=sys.stderr)
                if telegram_chat and os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"):
                    notify(warning, level="info", allow_telegram=True)
        elif code_snapshot_manifest is not None:
            code_dir = run_dir / "code"
            n = create_code_snapshot(Path(code_snapshot_manifest), code_dir)
            if n > 0:
                print(f"[AILogger] Code snapshot: {n} file(s) → {code_dir}", file=sys.stderr)
            elif _snapshot_dir_is_empty(code_dir):
                warning = (
                    "[AILogger] Warning: code snapshot directory is empty "
                    f"at {code_dir}. LLM analysis will lack code context. "
                    "To include code, either set code_snapshot_all=True, "
                    "or pass code_snapshot_paths=[...] with files/dirs, "
                    "or create a snapshot_files.txt manifest listing files/dirs "
                    "and pass code_snapshot_manifest=... . "
                    "You can also set visible_directories=[...] for live, read-only access."
                )
                print(warning, file=sys.stderr)
                if telegram_chat and os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"):
                    notify(warning, level="info", allow_telegram=True)

        return cls(
            zmq_port=zmq_port,
            db_path=local_backup_db_path,
            experiment_name=exp_sanitized,
            root_dir=root_dir,
            project=project_name,
            config=config_with_command,
            start_monitor=start_monitor,
            monitor_host=monitor_host,
            monitor_args=monitor_args,
            monitor_log_path=default_log_path,
            terminate_monitor_on_close=terminate_monitor_on_close,
            wait_for_monitor=True,
            use_wandb=use_wandb,
            webpage=webpage,
            webpage_port=webpage_port,
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
        extra_env: Optional[Dict[str, str]] = None,
    ) -> None:
        if port_is_listening(host, port):
            return

        # The monitor is invoked as `python -m training_monitor ...`
        cmd = [
            sys.executable, "-u", "-m", "training_monitor",
            "--zmq-port", str(port),
            "--root-dir", str(monitor_args.get("root_dir", "ai_logger")),
            "--project", str(monitor_args.get("project", "default_project")),
            "--experiment", str(monitor_args.get("experiment", self.experiment_name)),
            "--db-path", str(monitor_args.get("db_path", "")),
            "--analysis-interval-min", str(monitor_args.get("analysis_interval_min", 5.0)),
            "--system-poll-sec", str(monitor_args.get("system_poll_sec", 10.0)),
            "--max-prompt-len", str(monitor_args.get("max_prompt_len", 8000)),
            "--openai-model", str(monitor_args.get("openai_model", "gpt-5.2")),
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
        if monitor_args.get("visible_directories"):
            cmd += ["--visible-directories", ",".join(str(x) for x in monitor_args["visible_directories"])]

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
            env = os.environ.copy()
            if extra_env:
                env.update(extra_env)
            self._monitor_proc = subprocess.Popen(
                cmd,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
                env=env,
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

    def _ensure_webpage_running(
        self,
        host: str,
        port: Optional[int],
        db_path: Path,
    ) -> bool:
        if importlib.util.find_spec("chainlit") is None:
            print(
                "[AILogger] Chainlit is not installed; skipping web UI. "
                "Install with: pip install -e \".[web]\"",
                file=sys.stderr,
            )
            return False

        if port is None:
            candidate_port = 8000
            if not port_is_available(candidate_port, host=host):
                candidate_port = find_available_port(candidate_port + 1, host=host)
            if candidate_port is None:
                print(
                    "[AILogger] No available port found for Chainlit; skipping web UI.",
                    file=sys.stderr,
                )
                return False
        else:
            candidate_port = port
            if not port_is_available(candidate_port, host=host):
                print(
                    f"[AILogger] Port {candidate_port} is unavailable; skipping web UI.",
                    file=sys.stderr,
                )
                return False

        app_path = (Path(__file__).resolve().parent / "web_chat.py").as_posix()
        env = os.environ.copy()
        env["TM_DB_PATH"] = str(db_path)
        env.setdefault("TM_ROOT_DIR", str(self.root_dir))
        env.setdefault("TM_PROJECT", self.project_name)
        env.setdefault("TM_EXPERIMENT", self.experiment_name)
        app_root = _find_chainlit_app_root()
        if app_root is not None:
            env.setdefault("CHAINLIT_APP_ROOT", str(app_root))

        cmd = [
            sys.executable,
            "-u",
            "-m",
            "chainlit",
            "run",
            app_path,
            "--host",
            host,
            "--port",
            str(candidate_port),
        ]

        try:
            self._web_proc = subprocess.Popen(
                cmd,
                stdout=None,
                stderr=None,
                start_new_session=True,
                env=env,
            )
            self._web_started_by_us = True
            self._web_port = candidate_port
            url = f"http://{host}:{candidate_port}"
            link = _format_terminal_link(url, stream=sys.stderr)
            print(f"[AILogger] Web chat running at {link}", file=sys.stderr)
            return True
        except Exception as e:
            print(f"[AILogger] Failed to spawn Chainlit: {e}", file=sys.stderr)
            return False

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def log(
        self,
        metrics: Dict[str, Any],
        step: int,
        force_analysis: bool = False,
        openai_model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        **wandb_kwargs: Any,
    ) -> None:
        timestamp = time.time()
        msg: Dict[str, Any] = {
            "type": "training_metric",
            "experiment": self.experiment_name,
            "step": step,
            "timestamp": timestamp,
            "metrics": metrics,
            "force_analysis": force_analysis,
        }
        if force_analysis:
            if openai_model:
                model = str(openai_model).strip()
                if model:
                    msg["openai_model"] = model
            if reasoning_effort is not None:
                effort = str(reasoning_effort).strip()
                if effort:
                    msg["reasoning_effort"] = effort

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

        if self._use_wandb and wandb is not None:
            try:
                wandb_log_kwargs = dict(wandb_kwargs)
                wandb_log_kwargs.setdefault("step", step)
                wandb.log(metrics, **wandb_log_kwargs)
            except Exception as e:
                print(f"[AILogger] wandb.log failed: {e}", file=sys.stderr)

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
        if self._use_wandb and wandb is not None:
            try:
                wandb.finish()
            except Exception as e:
                print(f"[AILogger] wandb.finish failed: {e}", file=sys.stderr)

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

        if self._web_started_by_us and self._web_proc is not None:
            try:
                self._web_proc.terminate()
            except Exception:
                pass
            self._web_proc = None
            self._web_started_by_us = False
