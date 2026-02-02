"""
Training Monitor with LLM Analysis
====================================

Hybrid ZMQ + SQLite architecture for monitoring ML training runs
with periodic LLM-powered analysis and Telegram/Pushover notifications.

All outputs go under: ``<root_dir>/<experiment_name>/``

Quick start::

    import training_monitor as tm

    tm.init(
        experiment_name="grpo_v1",
        config={"lr": 1e-4, "model": "gemma-3-4b"},
        openai_model="gpt-4o",
        start_monitor="auto",
    )

    for step in range(1000):
        metrics = {"loss": ..., "accuracy": ...}
        tm.log(metrics, step)

    tm.close()

The monitor can also be run standalone::

    python -m training_monitor --analysis-interval-min 5 --openai-model gpt-4o
"""

from .logger import AILogger, close, get_logger, init, log, reset
from .types import SystemMetric, TrainingMetric

__all__ = [
    "AILogger",
    "init",
    "log",
    "reset",
    "close",
    "get_logger",
    "TrainingMetric",
    "SystemMetric",
]
