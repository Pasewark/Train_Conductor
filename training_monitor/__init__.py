"""
Training Monitor with LLM Analysis
====================================

Hybrid ZMQ + SQLite architecture for monitoring ML training runs
with periodic LLM-powered analysis and Telegram/Pushover notifications.

All outputs go under: ``<root_dir>/<project>/<experiment_name>/``

Quick start::

    import training_monitor as tm

    tm.init(
        name="grpo_v1",
        project="default_project",
        config={"lr": 1e-4, "model": "gemma-3-4b"},
        openai_model="gpt-5.2",
        start_monitor="auto",
    )

    for step in range(1000):
        metrics = {"loss": ..., "accuracy": ...}
        tm.log(metrics, step)

    tm.close()

The monitor can also be run standalone::

    python -m training_monitor --analysis-interval-min 5 --openai-model gpt-5.2
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
