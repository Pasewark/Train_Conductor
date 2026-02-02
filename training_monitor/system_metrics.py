"""GPU, CPU, and RAM metrics collection via nvidia-smi and psutil."""

import subprocess
import sys
import time
from typing import List, Optional

try:
    import psutil
except ImportError:
    psutil = None

from .types import SystemMetric


def _run_nvidia_smi() -> str:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or "nvidia-smi failed")
    return r.stdout.strip()


def collect_system_metrics(gpus: Optional[List[int]] = None) -> SystemMetric:
    gpu_metrics = []

    try:
        out = _run_nvidia_smi()
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 6:
                continue

            idx_s, util_s, mem_used_s, mem_total_s, temp_s, power_s = parts

            def parse_float(s):
                try:
                    return float(s)
                except ValueError:
                    return None

            def parse_int(s):
                try:
                    return int(float(s))
                except ValueError:
                    return None

            idx = parse_int(idx_s)
            if idx is None:
                continue

            if gpus is not None and idx not in gpus:
                continue

            mem_used = parse_float(mem_used_s)
            mem_total = parse_float(mem_total_s)

            gpu_metrics.append({
                "index": idx,
                "util_percent": parse_float(util_s),
                "mem_used_gb": mem_used / 1024 if mem_used else None,
                "mem_total_gb": mem_total / 1024 if mem_total else None,
                "mem_percent": (mem_used / mem_total * 100) if (mem_used and mem_total) else None,
                "temp_c": parse_float(temp_s),
                "power_w": parse_float(power_s),
            })
    except Exception as e:
        print(f"[monitor] nvidia-smi error: {e}", file=sys.stderr)

    cpu_percent = psutil.cpu_percent() if psutil else 0.0
    ram = psutil.virtual_memory() if psutil else None

    return SystemMetric(
        timestamp=time.time(),
        gpus=gpu_metrics,
        cpu_percent=cpu_percent,
        ram_percent=ram.percent if ram else 0.0,
        ram_used_gb=ram.used / 1e9 if ram else 0.0,
        ram_total_gb=ram.total / 1e9 if ram else 0.0,
    )