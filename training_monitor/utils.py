"""Path helpers, rank/launcher helpers, and misc utilities."""

import hashlib
import json
import os
import re
import shutil
import socket
import sys
from pathlib import Path
from typing import Any, List, Optional


# =============================================================================
# Path helpers
# =============================================================================

def sanitize_experiment_name(name: str) -> str:
    """Make experiment name safe to use as a directory name."""
    name = name.strip()
    if not name:
        return "unknown"
    name = name.replace("/", "_").replace("\\", "_")
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name or "unknown"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def resolve_under(base_dir: Path, path_str: Optional[str]) -> Optional[Path]:
    """
    If path_str is:
      - None  → None
      - absolute → absolute Path
      - relative → base_dir / path_str
    """
    if path_str is None:
        return None
    p = Path(path_str)
    if p.is_absolute():
        return p
    return base_dir / p


def default_experiment_dir(root_dir: Path, experiment_name: str) -> Path:
    return root_dir / sanitize_experiment_name(experiment_name)


def safe_json_dumps(obj: Any) -> str:
    """JSON-serialize arbitrary objects (Paths, numpy types, etc.)."""
    return json.dumps(obj, default=str, sort_keys=True)


# =============================================================================
# Rank / launcher helpers
# =============================================================================

def _env_int(name: str) -> Optional[int]:
    v = os.getenv(name)
    if v is None or v == "":
        return None
    try:
        return int(v)
    except Exception:
        return None


def local_rank() -> Optional[int]:
    for k in ("LOCAL_RANK", "SLURM_LOCALID", "MPI_LOCALRANKID", "OMPI_COMM_WORLD_LOCAL_RANK"):
        r = _env_int(k)
        if r is not None:
            return r
    return None


def rank() -> Optional[int]:
    for k in ("RANK", "SLURM_PROCID", "PMI_RANK", "OMPI_COMM_WORLD_RANK"):
        r = _env_int(k)
        if r is not None:
            return r
    return None


def should_spawn_monitor(start_monitor: str | bool) -> bool:
    """
    Decide whether this process should spawn the monitor subprocess.

    start_monitor:
      - False:  never
      - True:   always (but caller still checks if already running)
      - "auto": once per node (LOCAL_RANK==0 if set, else RANK==0, else True)
    """
    if start_monitor is False:
        return False
    if start_monitor is True:
        return True
    if start_monitor == "auto":
        lr = local_rank()
        if lr is not None:
            return lr == 0
        r = rank()
        if r is not None:
            return r == 0
        return True
    return True


def port_is_listening(host: str, port: int, timeout_sec: float = 0.2) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_sec):
            return True
    except OSError:
        return False


def port_is_available(port: int, host: str = "0.0.0.0") -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, port))
        return True
    except OSError:
        return False


def find_available_port(start_port: int, host: str = "0.0.0.0", max_tries: int = 50) -> Optional[int]:
    candidate = max(start_port, 1024)
    for port in range(candidate, candidate + max_tries):
        if port_is_available(port, host=host):
            return port
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, 0))
            return sock.getsockname()[1]
    except OSError:
        return None


# =============================================================================
# Code snapshot
# =============================================================================

_SNAPSHOT_SKIP_DIRS = frozenset({
    ".git",
    "__pycache__",
    "node_modules",
    ".tox",
    ".eggs",
    ".mypy_cache",
    ".pytest_cache",
})


def hash_directory(directory: Path) -> Optional[str]:
    """
    Compute a stable SHA-256 hash of all files in *directory*.

    Files are sorted by their path relative to *directory* so the hash is
    deterministic across runs.  Common non-code directories (see
    ``_SNAPSHOT_SKIP_DIRS``) are skipped.

    Returns a hex-digest string, or ``None`` if the directory doesn't exist
    or contains no readable files.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return None

    files = sorted(
        f for f in directory.rglob("*")
        if f.is_file()
        and not any(part in _SNAPSHOT_SKIP_DIRS for part in f.relative_to(directory).parts)
    )

    if not files:
        return None

    hasher = hashlib.sha256()
    for fp in files:
        rel = str(fp.relative_to(directory))
        hasher.update(rel.encode("utf-8"))
        try:
            hasher.update(fp.read_bytes())
        except OSError:
            continue

    return hasher.hexdigest()


def create_code_snapshot(
    manifest_path: Path,
    dest_dir: Path,
    max_file_size: int = 2 * 1024 * 1024,
    max_files: int = 200,
) -> int:
    """
    Copy code files listed in a manifest into *dest_dir*.

    Manifest format (one entry per line):
      - File paths → copied directly
      - Directory paths → all files inside are copied recursively
      - Lines starting with ``#`` are comments; empty lines are skipped

    Paths in the manifest are resolved relative to the manifest file's
    parent directory.  Files larger than *max_file_size* bytes (default 2 MB)
    and common non-code directories (.git, __pycache__, node_modules, …) are
    skipped.  At most *max_files* files are copied (default 200).

    Returns the number of files actually copied.
    """
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.exists():
        return 0

    base = manifest_path.parent

    # Parse manifest
    raw_entries: List[str] = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            raw_entries.append(line)

    if not raw_entries:
        return 0

    ensure_dir(dest_dir)

    # ------------------------------------------------------------------
    # Collect (abs_source, snapshot_relative_path) pairs
    # ------------------------------------------------------------------
    all_files: List[tuple] = []  # list of (Path, Path)

    def _rel_path(src: Path) -> Path:
        """Best-effort relative path under *base*; falls back to filename."""
        try:
            return src.relative_to(base)
        except ValueError:
            return Path(src.name)

    skipped_size = 0

    for raw in raw_entries:
        p = Path(raw)
        if not p.is_absolute():
            p = base / p
        p = p.resolve()

        if not p.exists():
            continue

        if p.is_file():
            all_files.append((p, _rel_path(p)))
        elif p.is_dir():
            for child in sorted(p.rglob("*")):
                if not child.is_file():
                    continue
                # Compute path relative to the listed directory
                try:
                    child_rel_to_dir = child.relative_to(p)
                except ValueError:
                    continue
                # Skip common non-code directories
                if any(part in _SNAPSHOT_SKIP_DIRS for part in child_rel_to_dir.parts):
                    continue
                # Snapshot path preserves structure relative to base
                snapshot_rel = _rel_path(child)
                all_files.append((child, snapshot_rel))

    # ------------------------------------------------------------------
    # Deduplicate and copy
    # ------------------------------------------------------------------
    copied = 0
    seen: set = set()

    for source, rel in all_files:
        if copied >= max_files:
            remaining = len(all_files) - (all_files.index((source, rel)))
            print(
                f"[AILogger] Code snapshot: hit {max_files}-file limit, "
                f"~{remaining} entries not copied",
                file=sys.stderr,
            )
            break

        if rel in seen:
            continue
        seen.add(rel)

        try:
            size = source.stat().st_size
            if size > max_file_size:
                skipped_size += 1
                continue
            dest = dest_dir / rel
            ensure_dir(dest.parent)
            shutil.copy2(source, dest)
            copied += 1
        except Exception:
            continue

    if skipped_size:
        print(
            f"[AILogger] Code snapshot: skipped {skipped_size} file(s) "
            f"exceeding {max_file_size / (1024 * 1024):.0f} MB limit",
            file=sys.stderr,
        )

    return copied


def read_code_snapshot(
    code_dir: Path,
    max_total_chars: int = 300_000,
    max_file_chars: int = 50_000,
) -> Optional[str]:
    """
    Read all text files from a code snapshot directory and return a
    formatted string suitable for inclusion in an LLM prompt.

    Each file is rendered as::

        === relative/path/to/file.py ===
        <contents>

    Binary files (those that fail UTF-8 decoding) are skipped.
    Individual files are truncated at *max_file_chars* characters, and
    the overall output is capped at *max_total_chars*.

    Returns ``None`` if *code_dir* doesn't exist or contains no readable
    files.
    """
    code_dir = Path(code_dir)
    if not code_dir.is_dir():
        return None

    files: List[Path] = sorted(
        f for f in code_dir.rglob("*")
        if f.is_file()
        and not any(part in _SNAPSHOT_SKIP_DIRS for part in f.relative_to(code_dir).parts)
    )

    if not files:
        return None

    sections: List[str] = []
    total = 0

    for fp in files:
        rel = fp.relative_to(code_dir)
        try:
            text = fp.read_text(encoding="utf-8", errors="strict")
        except (UnicodeDecodeError, OSError):
            continue

        if len(text) > max_file_chars:
            text = text[:max_file_chars] + "\n... (truncated)"

        section = f"=== {rel} ===\n{text}\n"

        if total + len(section) > max_total_chars:
            remaining = len(files) - len(sections)
            sections.append(f"\n... ({remaining} more file(s) omitted, total size limit reached)\n")
            break

        sections.append(section)
        total += len(section)

    if not sections:
        return None

    return "\n".join(sections)