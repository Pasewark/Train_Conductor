"""Core data types shared across the package."""

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class TrainingMetric:
    step: int
    timestamp: float
    metrics: Dict[str, Any]


@dataclass
class SystemMetric:
    timestamp: float
    gpus: List[Dict[str, Any]]
    cpu_percent: float
    ram_percent: float
    ram_used_gb: float
    ram_total_gb: float