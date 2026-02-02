"""Read-only tools for inspecting a snapshotted code directory."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .utils import _SNAPSHOT_SKIP_DIRS


_SENSITIVE_FILENAMES = {
    ".env",
    ".env.local",
    "id_rsa",
    "id_dsa",
    "id_ed25519",
    "credentials.json",
}

_SENSITIVE_SUFFIXES = {
    ".pem",
    ".key",
    ".p12",
}

_LANGUAGE_BY_SUFFIX = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".json": "json",
    ".md": "markdown",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".txt": "text",
    ".html": "html",
    ".css": "css",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".swift": "swift",
    ".sh": "shell",
}

_MAX_READ_LINES = 3000
_MAX_READ_BYTES = 250_000
_MAX_SEARCH_RESULTS = 200
_MAX_MANIFEST_FILES = 5000


def _is_sensitive(path: Path) -> bool:
    name = path.name
    if name in _SENSITIVE_FILENAMES:
        return True
    if path.suffix.lower() in _SENSITIVE_SUFFIXES:
        return True
    return False


def _detect_language(path: Path) -> str:
    return _LANGUAGE_BY_SUFFIX.get(path.suffix.lower(), "unknown")


def _is_under(base_dir: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(base_dir.resolve())
        return True
    except ValueError:
        return False


def _validate_relative_path(path_str: str) -> Optional[Path]:
    if not path_str:
        return None
    p = Path(path_str)
    if p.is_absolute() or ".." in p.parts:
        return None
    return p


@dataclass
class SnapshotToolManager:
    """Tool handler for a snapshot directory."""

    base_dir: Path

    @property
    def tool_definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "snapshot_manifest",
                    "description": "Return metadata about the code snapshot: file list (truncated), sizes, languages, and ignore rules.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "max_files": {"type": "integer", "default": 500},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "snapshot_search",
                    "description": "Search within the code snapshot for a string or regex.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "mode": {
                                "type": "string",
                                "enum": ["literal", "regex"],
                                "default": "literal",
                            },
                            "case_sensitive": {"type": "boolean", "default": False},
                            "dir": {"type": "string", "default": "."},
                            "glob_include": {"type": "string"},
                            "glob_exclude": {"type": "string"},
                            "max_results": {"type": "integer", "default": 50},
                            "context_lines": {"type": "integer", "default": 1},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "snapshot_read",
                    "description": "Read a line range from a file in the code snapshot (path relative to snapshot root).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "start_line": {"type": "integer", "default": 1},
                            "end_line": {"type": "integer", "default": 200},
                            "with_line_numbers": {"type": "boolean", "default": True},
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "snapshot_read_many",
                    "description": "Read multiple file slices from the snapshot in one call.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "requests": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "path": {"type": "string"},
                                        "start_line": {"type": "integer", "default": 1},
                                        "end_line": {"type": "integer", "default": 200},
                                    },
                                    "required": ["path"],
                                },
                                "maxItems": 10,
                            },
                        },
                        "required": ["requests"],
                    },
                },
            },
        ]

    def handle(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if name == "snapshot_manifest":
            max_files = int(arguments.get("max_files", 500))
            return self.snapshot_manifest(max_files=max_files)
        if name == "snapshot_search":
            return self.snapshot_search(
                query=str(arguments.get("query", "")),
                mode=str(arguments.get("mode", "literal")),
                case_sensitive=bool(arguments.get("case_sensitive", False)),
                dir=str(arguments.get("dir", ".")),
                glob_include=arguments.get("glob_include"),
                glob_exclude=arguments.get("glob_exclude"),
                max_results=int(arguments.get("max_results", 50)),
                context_lines=int(arguments.get("context_lines", 1)),
            )
        if name == "snapshot_read":
            return self.snapshot_read(
                path=str(arguments.get("path", "")),
                start_line=int(arguments.get("start_line", 1)),
                end_line=int(arguments.get("end_line", 200)),
                with_line_numbers=bool(arguments.get("with_line_numbers", True)),
            )
        if name == "snapshot_read_many":
            requests = arguments.get("requests", [])
            return {"results": [self.snapshot_read(**req) for req in requests]}
        return {"error": f"Unknown tool: {name}"}

    def snapshot_manifest(self, max_files: int = 500) -> Dict[str, Any]:
        if not self.base_dir.is_dir():
            return {"error": "snapshot directory not found"}

        capped_max = min(max_files, _MAX_MANIFEST_FILES)
        files = []
        total_files = 0

        for fp in self._iter_files(self.base_dir):
            total_files += 1
            if len(files) >= capped_max:
                continue
            try:
                size_bytes = fp.stat().st_size
            except OSError:
                size_bytes = 0
            files.append(
                {
                    "path": str(fp.relative_to(self.base_dir)),
                    "size_bytes": size_bytes,
                    "language": _detect_language(fp),
                }
            )

        return {
            "snapshot_root": ".",
            "snapshot_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "file_count": total_files,
            "files": files,
            "ignore_rules": {
                "skip_dirs": sorted(_SNAPSHOT_SKIP_DIRS),
                "sensitive_filenames": sorted(_SENSITIVE_FILENAMES),
                "sensitive_suffixes": sorted(_SENSITIVE_SUFFIXES),
            },
        }

    def snapshot_search(
        self,
        *,
        query: str,
        mode: str = "literal",
        case_sensitive: bool = False,
        dir: str = ".",
        glob_include: Optional[str] = None,
        glob_exclude: Optional[str] = None,
        max_results: int = 50,
        context_lines: int = 1,
    ) -> Dict[str, Any]:
        if not query:
            return {"matches": [], "error": "query must be non-empty"}

        root = self._resolve_dir(dir)
        if root is None:
            return {"matches": [], "error": "invalid search dir"}

        pattern = query if mode == "regex" else re.escape(query)
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            regex = re.compile(pattern, flags)
        except re.error as exc:
            return {"matches": [], "error": f"invalid regex: {exc}"}

        max_hits = min(max_results, _MAX_SEARCH_RESULTS)
        matches: List[Dict[str, Any]] = []

        for fp in self._iter_files(root, glob_include, glob_exclude):
            try:
                text = fp.read_text(encoding="utf-8", errors="strict")
            except (UnicodeDecodeError, OSError):
                continue
            lines = text.splitlines()
            for line_idx, line in enumerate(lines):
                for match in regex.finditer(line):
                    if len(matches) >= max_hits:
                        return {"matches": matches, "truncated": True}
                    before_start = max(0, line_idx - context_lines)
                    before = "\n".join(lines[before_start:line_idx])
                    after_end = min(len(lines), line_idx + 1 + context_lines)
                    after = "\n".join(lines[line_idx + 1:after_end])
                    matches.append(
                        {
                            "path": str(fp.relative_to(self.base_dir)),
                            "line": line_idx + 1,
                            "col": match.start() + 1,
                            "snippet": line.strip(),
                            "before": before,
                            "after": after,
                        }
                    )

        return {"matches": matches}

    def snapshot_read(
        self,
        *,
        path: str,
        start_line: int = 1,
        end_line: int = 200,
        with_line_numbers: bool = True,
    ) -> Dict[str, Any]:
        rel_path = _validate_relative_path(path)
        if rel_path is None:
            return {"error": "invalid path"}

        full_path = (self.base_dir / rel_path).resolve()
        if not _is_under(self.base_dir, full_path):
            return {"error": "path escapes snapshot root"}
        if not full_path.is_file():
            return {"error": "file not found"}
        if _is_sensitive(full_path):
            return {"error": "file is blocked by sensitive file rules"}

        start = max(1, start_line)
        end = max(start, end_line)
        if end - start + 1 > _MAX_READ_LINES:
            end = start + _MAX_READ_LINES - 1

        try:
            text = full_path.read_text(encoding="utf-8", errors="strict")
        except (UnicodeDecodeError, OSError):
            return {"error": "unable to read file as utf-8"}

        lines = text.splitlines()
        if not lines:
            return {
                "path": str(rel_path),
                "start_line": start,
                "end_line": end,
                "text": "",
                "truncated": False,
            }

        slice_lines = lines[start - 1:end]
        truncated = False
        rendered_lines = []
        total_bytes = 0
        for idx, line in enumerate(slice_lines, start=start):
            if with_line_numbers:
                rendered = f"{idx:>4}: {line}"
            else:
                rendered = line
            total_bytes += len(rendered.encode("utf-8"))
            if total_bytes > _MAX_READ_BYTES:
                truncated = True
                break
            rendered_lines.append(rendered)

        if truncated:
            rendered_lines.append("... (truncated)")

        return {
            "path": str(rel_path),
            "start_line": start,
            "end_line": end,
            "text": "\n".join(rendered_lines),
            "truncated": truncated,
        }

    def _iter_files(
        self,
        root: Path,
        glob_include: Optional[str] = None,
        glob_exclude: Optional[str] = None,
    ) -> Iterable[Path]:
        if not root.is_dir():
            return []
        for fp in sorted(root.rglob("*")):
            if not fp.is_file():
                continue
            try:
                rel = fp.relative_to(self.base_dir)
            except ValueError:
                continue
            if any(part in _SNAPSHOT_SKIP_DIRS for part in rel.parts):
                continue
            if _is_sensitive(fp):
                continue
            if glob_include and not fp.match(glob_include):
                continue
            if glob_exclude and fp.match(glob_exclude):
                continue
            yield fp

    def _resolve_dir(self, dir_str: str) -> Optional[Path]:
        rel = _validate_relative_path(dir_str)
        if rel is None:
            return None
        target = (self.base_dir / rel).resolve()
        if not _is_under(self.base_dir, target):
            return None
        if not target.exists():
            return None
        return target


def json_dumps(data: Dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, default=str)
