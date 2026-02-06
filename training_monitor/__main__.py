"""CLI entry point: ``python -m training_monitor``."""

import argparse
import signal
import sys
from typing import List, Optional

from .monitor import TrainingMonitor


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Training Monitor with LLM Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--zmq-port", type=int, default=5555,
                        help="ZMQ port for receiving metrics")
    parser.add_argument("--root-dir", type=str, default="ai_logger",
                        help="Root output directory (project/experiment subdirs created here)")
    parser.add_argument("--project", type=str, default="default_project",
                        help="Project name (used as a subdirectory under root-dir)")
    parser.add_argument("--experiment", type=str, default="",
                        help="Experiment name (optional; if omitted, taken from first metric)")
    parser.add_argument("--db-path", type=str, default="",
                        help="SQLite database path. If relative, resolved under <root-dir>/<project>/<experiment>/ . "
                             "If empty, defaults to <root-dir>/<project>/<experiment>/training_monitor.db")

    parser.add_argument("--analysis-interval-min", type=float, default=5.0,
                        help="Minutes between LLM analyses")
    parser.add_argument("--analysis-interval-steps", type=int, default=None,
                        help="Steps between LLM analyses (default: disabled, use time-based only)")
    parser.add_argument("--system-poll-sec", type=float, default=10.0,
                        help="Seconds between system metric polls")
    parser.add_argument("--max-prompt-len", type=int, default=8000,
                        help="Maximum tokens in LLM prompt")
    parser.add_argument("--openai-model", type=str, default="gpt-5.2",
                        help="OpenAI model name (or local model if using custom base URL)")
    parser.add_argument("--openai-base-url", type=str, default=None,
                        help="Custom OpenAI-compatible API base URL (for local models)")
    parser.add_argument("--reasoning-effort", type=str, default=None,
                        choices=["none", "low", "medium", "high", "xhigh"],
                        help="Reasoning effort for OpenAI reasoning models")
    parser.add_argument("--gpus", type=str, default="",
                        help="Comma-separated GPU indices to monitor (e.g., '0,1,2'). Default: all GPUs")
    parser.add_argument("--recovery-max-age-min", type=float, default=60.0,
                        help="Max age (minutes) of metrics to recover from SQLite on restart")
    parser.add_argument("--notify-all-analyses", action=argparse.BooleanOptionalAction, default=True,
                        help="Send all analyses to Telegram/Pushover, not just alerts (default: enabled)")
    parser.add_argument("--sig-figs", type=int, default=5,
                        help="Significant figures for metrics in LLM prompt")
    parser.add_argument("--idle-timeout-min", type=float, default=None,
                        help="Stop analyses if no training updates for this many minutes (default: disabled)")
    parser.add_argument("--telegram-chat", action="store_true",
                        help="Enable Telegram usage (notifications + interactive chat). "
                             "Requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars.")
    parser.add_argument("--max-conversation-history-tokens", type=int, default=5000,
                        help="Maximum estimated tokens of conversation history (analyses + chat messages) "
                             "to include in each LLM analysis prompt")
    parser.add_argument("--regenerate-code-analysis", action="store_true",
                        help="Force regeneration of the code summary and metric descriptions, "
                             "ignoring any cached versions from previous runs")
    parser.add_argument("--code-snapshot-manifest", type=str, default="",
                        help="Path to a snapshot manifest (same format as snapshot_files.txt)")
    parser.add_argument("--code-snapshot-paths", type=str, default="",
                        help="Comma-separated list of file/dir paths to snapshot into run_dir/code")
    parser.add_argument("--visible-directories", type=str, default="",
                        help="Comma-separated list of directories to expose via snapshot tools "
                             "(read-only, live view; not copied)")

    args = parser.parse_args()

    gpus: Optional[List[int]] = None
    if args.gpus.strip():
        gpus = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]

    exp = args.experiment.strip() or None
    project = args.project.strip() or None

    visible_dirs = []
    if args.visible_directories.strip():
        visible_dirs = [d.strip() for d in args.visible_directories.split(",") if d.strip()]

    snapshot_paths = []
    if args.code_snapshot_paths.strip():
        snapshot_paths = [p.strip() for p in args.code_snapshot_paths.split(",") if p.strip()]

    manifest_path = args.code_snapshot_manifest.strip() or None

    monitor = TrainingMonitor(
        zmq_port=args.zmq_port,
        root_dir=args.root_dir,
        project=project,
        experiment=exp,
        db_path=args.db_path,
        analysis_interval_min=args.analysis_interval_min,
        analysis_interval_steps=args.analysis_interval_steps,
        system_poll_interval_sec=args.system_poll_sec,
        max_prompt_len=args.max_prompt_len,
        openai_model=args.openai_model,
        openai_base_url=args.openai_base_url,
        reasoning_effort=args.reasoning_effort,
        gpus=gpus,
        recovery_max_age_min=args.recovery_max_age_min,
        notify_all_analyses=args.notify_all_analyses,
        sig_figs=args.sig_figs,
        idle_timeout_min=args.idle_timeout_min,
        telegram_chat=args.telegram_chat,
        max_conversation_history_tokens=args.max_conversation_history_tokens,
        regenerate_code_analysis=args.regenerate_code_analysis,
        code_snapshot_paths=snapshot_paths,
        code_snapshot_manifest=manifest_path,
        visible_directories=visible_dirs,
    )

    def signal_handler(sig, frame):
        print("\n[monitor] Shutting down...", file=sys.stderr)
        monitor.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        monitor.run()
    except KeyboardInterrupt:
        monitor.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
