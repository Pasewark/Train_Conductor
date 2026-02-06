"""Read-only tools for inspecting a snapshotted code directory."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
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
_MAX_READ_BYTES = 50_000
_MAX_SEARCH_RESULTS = 200
_MAX_MANIFEST_FILES = 5000
_LINECOUNT_CHUNK_BYTES = 64 * 1024
_MAX_JSONL_RESPONSE_BYTES = 50_000  # 50KB max response for JSONL tool
_MAX_JSONL_DETAIL_RESPONSE_BYTES = 200_000
_MAX_JSONL_SCHEMA_SCAN_ENTRIES = 2000
_MAX_JSONL_PREVIEW_ENTRIES = 50
_MAX_JSONL_FIELD_EXAMPLES = 5
_MAX_JSONL_GENERATION_INDEXES = 50

_JSONL_PROMPT_CANDIDATES = (
    "prompt",
    "prompts",
    "input",
    "inputs",
    "query",
    "instruction",
    "user_prompt",
)
_JSONL_GENERATIONS_CANDIDATES = (
    "generations",
    "generation",
    "responses",
    "completions",
    "outputs",
    "samples",
)
_JSONL_REWARD_CANDIDATES = (
    "reward",
    "rewards",
    "score",
    "scores",
    "return",
    "returns",
)
_JSONL_TEXT_KEYS = (
    "text",
    "content",
    "generation",
    "response",
    "completion",
    "output",
)


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


def _count_lines(path: Path) -> Optional[int]:
    count = 0
    last_byte = None
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(_LINECOUNT_CHUNK_BYTES)
                if not chunk:
                    break
                count += chunk.count(b"\n")
                last_byte = chunk[-1]
    except OSError:
        return None
    if last_byte is None:
        return 0
    if last_byte != ord("\n"):
        count += 1
    return count


def _jsonl_clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _jsonl_type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return "other"


def _jsonl_truncate_text(text: str, max_len: int) -> tuple[str, bool]:
    if len(text) <= max_len:
        return text, False
    return text[:max_len] + "...[truncated]", True


def _jsonl_safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except Exception:
        return str(value)


def _jsonl_truncate_value(
    value: Any,
    *,
    max_text_len: int,
    max_list_items: int = 4,
    max_dict_fields: int = 20,
) -> Any:
    if isinstance(value, str):
        truncated, _ = _jsonl_truncate_text(value, max_text_len)
        return truncated
    if isinstance(value, list):
        out = [
            _jsonl_truncate_value(
                item,
                max_text_len=max_text_len,
                max_list_items=max_list_items,
                max_dict_fields=max_dict_fields,
            )
            for item in value[:max_list_items]
        ]
        if len(value) > max_list_items:
            out.append(f"...[+{len(value) - max_list_items} items truncated]")
        return out
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        keys = list(value.keys())
        for key in keys[:max_dict_fields]:
            out[str(key)] = _jsonl_truncate_value(
                value[key],
                max_text_len=max_text_len,
                max_list_items=max_list_items,
                max_dict_fields=max_dict_fields,
            )
        if len(keys) > max_dict_fields:
            out["_truncated_fields"] = len(keys) - max_dict_fields
        return out
    return value


def _jsonl_guess_field(entry: Dict[str, Any], candidates: tuple[str, ...]) -> Optional[str]:
    key_lookup = {
        key.lower(): key
        for key in entry.keys()
        if isinstance(key, str)
    }
    for candidate in candidates:
        match = key_lookup.get(candidate)
        if match is not None:
            return match
    return None


def _jsonl_extract_generations(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            if isinstance(item, dict):
                text = None
                for key in _JSONL_TEXT_KEYS:
                    if key in item:
                        text = item.get(key)
                        break
                out.append(_jsonl_safe_text(text if text is not None else item))
            else:
                out.append(_jsonl_safe_text(item))
        return out
    if isinstance(value, dict):
        for key in _JSONL_TEXT_KEYS:
            if key in value:
                return [_jsonl_safe_text(value.get(key))]
    return [_jsonl_safe_text(value)]


def _jsonl_detect_fields(entry: Dict[str, Any]) -> Dict[str, Any]:
    prompt_field = _jsonl_guess_field(entry, _JSONL_PROMPT_CANDIDATES)
    generations_field = _jsonl_guess_field(entry, _JSONL_GENERATIONS_CANDIDATES)
    reward_field = _jsonl_guess_field(entry, _JSONL_REWARD_CANDIDATES)

    metadata_fields: List[str] = []
    for key in entry.keys():
        if not isinstance(key, str):
            continue
        if key in (prompt_field, generations_field, reward_field):
            continue
        metadata_fields.append(key)

    return {
        "prompt_field": prompt_field,
        "generations_field": generations_field,
        "reward_field": reward_field,
        "metadata_fields": metadata_fields[:20],
    }


def _jsonl_validate_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return "file not found"
    if _is_sensitive(path):
        return "file is blocked by sensitive file rules"
    if path.suffix.lower() != ".jsonl":
        return "file must be a .jsonl file"
    return None


def _jsonl_trim_list_response(
    result: Dict[str, Any],
    *,
    list_key: str,
    max_bytes: int,
    note: str,
) -> Dict[str, Any]:
    items = result.get(list_key)
    if not isinstance(items, list):
        return result
    encoded = json.dumps(result, default=str).encode("utf-8")
    if len(encoded) <= max_bytes:
        return result
    while items and len(json.dumps(result, default=str).encode("utf-8")) > max_bytes:
        items.pop()
    result[list_key] = items
    result["truncated"] = True
    result["note"] = note
    return result


def _jsonl_schema_from_file(
    *,
    full_path: Path,
    display_path: str,
    max_entries: int,
    max_examples: int,
) -> Dict[str, Any]:
    max_entries = _jsonl_clamp(max_entries, 1, _MAX_JSONL_SCHEMA_SCAN_ENTRIES)
    max_examples = _jsonl_clamp(max_examples, 1, _MAX_JSONL_FIELD_EXAMPLES)

    parse_errors = 0
    scanned_entries = 0
    parsed_entries = 0
    first_entry: Optional[Dict[str, Any]] = None
    field_stats: Dict[str, Dict[str, Any]] = {}

    try:
        with full_path.open("r", encoding="utf-8", errors="strict") as handle:
            for line_number, line in enumerate(handle, start=1):
                raw = line.strip()
                if not raw:
                    continue
                scanned_entries += 1
                try:
                    entry = json.loads(raw)
                except json.JSONDecodeError:
                    parse_errors += 1
                    if scanned_entries >= max_entries:
                        break
                    continue
                if not isinstance(entry, dict):
                    parse_errors += 1
                    if scanned_entries >= max_entries:
                        break
                    continue

                parsed_entries += 1
                if first_entry is None:
                    first_entry = entry

                for field, value in entry.items():
                    field_name = str(field)
                    stat = field_stats.setdefault(
                        field_name,
                        {
                            "type_counts": {},
                            "example_values": [],
                            "non_null_count": 0,
                            "numeric_count": 0,
                            "numeric_sum": 0.0,
                            "numeric_min": None,
                            "numeric_max": None,
                            "max_string_length": 0,
                            "list_min_len": None,
                            "list_max_len": None,
                        },
                    )

                    value_type = _jsonl_type_name(value)
                    stat["type_counts"][value_type] = stat["type_counts"].get(value_type, 0) + 1

                    if value is not None:
                        stat["non_null_count"] += 1

                    if len(stat["example_values"]) < max_examples:
                        stat["example_values"].append(
                            _jsonl_truncate_value(
                                value,
                                max_text_len=120,
                                max_list_items=3,
                                max_dict_fields=8,
                            )
                        )

                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        stat["numeric_count"] += 1
                        stat["numeric_sum"] += float(value)
                        stat["numeric_min"] = value if stat["numeric_min"] is None else min(stat["numeric_min"], value)
                        stat["numeric_max"] = value if stat["numeric_max"] is None else max(stat["numeric_max"], value)

                    if isinstance(value, str):
                        stat["max_string_length"] = max(stat["max_string_length"], len(value))

                    if isinstance(value, list):
                        list_len = len(value)
                        if stat["list_min_len"] is None:
                            stat["list_min_len"] = list_len
                            stat["list_max_len"] = list_len
                        else:
                            stat["list_min_len"] = min(stat["list_min_len"], list_len)
                            stat["list_max_len"] = max(stat["list_max_len"], list_len)

                if scanned_entries >= max_entries:
                    break
    except (UnicodeDecodeError, OSError) as exc:
        return {"error": f"unable to read file: {exc}"}

    fields: List[Dict[str, Any]] = []
    for field_name in sorted(field_stats.keys()):
        stat = field_stats[field_name]
        item: Dict[str, Any] = {
            "name": field_name,
            "types": stat["type_counts"],
            "non_null_count": stat["non_null_count"],
            "example_values": stat["example_values"],
        }
        if stat["numeric_count"] > 0:
            item["numeric_stats"] = {
                "count": stat["numeric_count"],
                "min": stat["numeric_min"],
                "max": stat["numeric_max"],
                "avg": stat["numeric_sum"] / stat["numeric_count"],
            }
        if stat["max_string_length"] > 0:
            item["max_string_length"] = stat["max_string_length"]
        if stat["list_min_len"] is not None:
            item["list_length_stats"] = {
                "min": stat["list_min_len"],
                "max": stat["list_max_len"],
            }
        fields.append(item)

    detected_fields = _jsonl_detect_fields(first_entry or {})
    line_count = _count_lines(full_path)
    try:
        size_bytes = full_path.stat().st_size
    except OSError:
        size_bytes = 0

    result: Dict[str, Any] = {
        "path": display_path,
        "line_count": line_count,
        "size_bytes": size_bytes,
        "scanned_entries": scanned_entries,
        "parsed_entries": parsed_entries,
        "parse_errors": parse_errors,
        "scan_truncated": scanned_entries >= max_entries,
        "detected_fields": detected_fields,
        "fields": fields,
    }
    return _jsonl_trim_list_response(
        result,
        list_key="fields",
        max_bytes=_MAX_JSONL_RESPONSE_BYTES,
        note="Response size limit reached; fewer fields returned.",
    )


def _jsonl_preview_from_file(
    *,
    full_path: Path,
    display_path: str,
    start_line: int,
    count: int,
    text_char_limit: int,
    max_generations: int,
) -> Dict[str, Any]:
    start_line = max(1, int(start_line))
    count = _jsonl_clamp(count, 1, _MAX_JSONL_PREVIEW_ENTRIES)
    text_char_limit = _jsonl_clamp(text_char_limit, 40, 2000)
    max_generations = _jsonl_clamp(max_generations, 1, 20)

    parse_errors = 0
    entries: List[Dict[str, Any]] = []
    detected_fields: Optional[Dict[str, Any]] = None

    try:
        with full_path.open("r", encoding="utf-8", errors="strict") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line_number < start_line:
                    continue
                raw = line.strip()
                if not raw:
                    continue

                try:
                    entry = json.loads(raw)
                except json.JSONDecodeError:
                    parse_errors += 1
                    continue
                if not isinstance(entry, dict):
                    parse_errors += 1
                    continue

                if detected_fields is None:
                    detected_fields = _jsonl_detect_fields(entry)

                prompt_field = detected_fields.get("prompt_field")
                generations_field = detected_fields.get("generations_field")
                reward_field = detected_fields.get("reward_field")

                prompt_text = _jsonl_safe_text(entry.get(prompt_field)) if prompt_field else ""
                prompt_preview, prompt_truncated = _jsonl_truncate_text(prompt_text, text_char_limit)

                generations = _jsonl_extract_generations(entry.get(generations_field)) if generations_field else []
                generation_previews: List[Dict[str, Any]] = []
                for idx, generation_text in enumerate(generations[:max_generations]):
                    text_preview, was_truncated = _jsonl_truncate_text(generation_text, text_char_limit)
                    generation_previews.append(
                        {
                            "index": idx,
                            "text": text_preview,
                            "truncated": was_truncated,
                        }
                    )

                excluded = {prompt_field, generations_field, reward_field}
                metadata_preview: Dict[str, Any] = {}
                for key, value in entry.items():
                    if key in excluded:
                        continue
                    metadata_preview[str(key)] = _jsonl_truncate_value(
                        value,
                        max_text_len=min(160, text_char_limit),
                        max_list_items=4,
                        max_dict_fields=12,
                    )

                entry_preview = _jsonl_truncate_value(
                    entry,
                    max_text_len=text_char_limit,
                    max_list_items=max_generations,
                    max_dict_fields=20,
                )

                preview: Dict[str, Any] = {
                    "line_number": line_number,
                    "prompt_preview": prompt_preview,
                    "prompt_truncated": prompt_truncated,
                    "generation_previews": generation_previews,
                    "metadata_preview": metadata_preview,
                    "entry_preview": entry_preview,
                }
                if reward_field:
                    preview["reward_preview"] = _jsonl_truncate_value(
                        entry.get(reward_field),
                        max_text_len=min(160, text_char_limit),
                        max_list_items=max_generations,
                        max_dict_fields=12,
                    )
                if len(generations) > max_generations:
                    preview["truncated_generation_count"] = len(generations) - max_generations

                entries.append(preview)
                if len(entries) >= count:
                    break
    except (UnicodeDecodeError, OSError) as exc:
        return {"error": f"unable to read file: {exc}"}

    result: Dict[str, Any] = {
        "path": display_path,
        "start_line": start_line,
        "requested_count": count,
        "returned_count": len(entries),
        "parse_errors": parse_errors,
        "field_selection": detected_fields or {},
        "entries": entries,
    }
    return _jsonl_trim_list_response(
        result,
        list_key="entries",
        max_bytes=_MAX_JSONL_RESPONSE_BYTES,
        note="Response size limit reached; fewer preview entries returned.",
    )


def _jsonl_read_entry_from_file(
    *,
    full_path: Path,
    display_path: str,
    line_number: int,
    generation_indexes: List[int],
) -> Dict[str, Any]:
    line_number = max(1, int(line_number))

    clean_indexes: List[int] = []
    for raw in generation_indexes[:_MAX_JSONL_GENERATION_INDEXES]:
        try:
            idx = int(raw)
        except (TypeError, ValueError):
            continue
        if idx < 0:
            continue
        if idx not in clean_indexes:
            clean_indexes.append(idx)

    target_line = None
    try:
        with full_path.open("r", encoding="utf-8", errors="strict") as handle:
            for current_line, line in enumerate(handle, start=1):
                if current_line == line_number:
                    target_line = line
                    break
    except (UnicodeDecodeError, OSError) as exc:
        return {"error": f"unable to read file: {exc}"}

    if target_line is None:
        return {"error": "line number out of range"}

    raw = target_line.strip()
    if not raw:
        return {"error": "target line is empty"}

    try:
        entry = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {"error": f"target line is not valid JSON: {exc}"}
    if not isinstance(entry, dict):
        return {"error": "target line JSON must be an object"}

    detected_fields = _jsonl_detect_fields(entry)
    prompt_field = detected_fields.get("prompt_field")
    generations_field = detected_fields.get("generations_field")
    reward_field = detected_fields.get("reward_field")

    prompt_text = _jsonl_safe_text(entry.get(prompt_field)) if prompt_field else ""
    generations = _jsonl_extract_generations(entry.get(generations_field)) if generations_field else []

    if generations:
        if clean_indexes:
            valid_indexes = [idx for idx in clean_indexes if idx < len(generations)]
            if not valid_indexes:
                return {
                    "error": "generation_indexes out of range",
                    "available_generation_count": len(generations),
                }
            selected_indexes = valid_indexes
        else:
            selected_indexes = list(range(min(3, len(generations))))
    else:
        selected_indexes = []

    selected_generations = [
        {"index": idx, "text": generations[idx]}
        for idx in selected_indexes
    ]

    excluded = {prompt_field, generations_field, reward_field}
    metadata: Dict[str, Any] = {}
    for key, value in entry.items():
        if key in excluded:
            continue
        metadata[str(key)] = value

    result: Dict[str, Any] = {
        "path": display_path,
        "line_number": line_number,
        "field_selection": detected_fields,
        "prompt": prompt_text,
        "selected_generations": selected_generations,
        "reward": entry.get(reward_field) if reward_field else None,
        "metadata": metadata,
        "available_generation_count": len(generations),
    }
    if not clean_indexes and len(generations) > len(selected_indexes):
        result["note"] = (
            "No generation_indexes provided; returned the first few generations. "
            "Provide generation_indexes to inspect specific generations."
        )

    encoded_size = len(json.dumps(result, default=str).encode("utf-8"))
    if encoded_size > _MAX_JSONL_DETAIL_RESPONSE_BYTES:
        return {
            "error": "response too large",
            "path": display_path,
            "line_number": line_number,
            "available_generation_count": len(generations),
            "estimated_response_bytes": encoded_size,
            "hint": "Request fewer generation indexes to keep response size manageable.",
        }

    return result


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
                    "name": "snapshot_line_count",
                    "description": "Return the line count for a file in the code snapshot (path relative to snapshot root).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
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
            {
                "type": "function",
                "function": {
                    "name": "snapshot_jsonl_schema",
                    "description": "Inspect JSONL structure safely: field names, types, examples, and detected rollout keys.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "max_entries": {"type": "integer", "default": 200},
                            "max_examples": {"type": "integer", "default": 3},
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "snapshot_jsonl_preview",
                    "description": "Preview JSONL rollout entries with truncated long text fields (prompt/generations) and compact metadata.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "start_line": {"type": "integer", "default": 1},
                            "count": {"type": "integer", "default": 5},
                            "text_char_limit": {"type": "integer", "default": 240},
                            "max_generations": {"type": "integer", "default": 3},
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "snapshot_jsonl_read_entry",
                    "description": "Read one JSONL entry by line number, returning full prompt and selected generations without text truncation.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "line_number": {"type": "integer"},
                            "generation_indexes": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "default": [],
                            },
                        },
                        "required": ["path", "line_number"],
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
        if name == "snapshot_line_count":
            return self.snapshot_line_count(path=str(arguments.get("path", "")))
        if name == "snapshot_read_many":
            requests = arguments.get("requests", [])
            return {"results": [self.snapshot_read(**req) for req in requests]}
        if name == "snapshot_jsonl_schema":
            return self.snapshot_jsonl_schema(
                path=str(arguments.get("path", "")),
                max_entries=_safe_int(arguments.get("max_entries", 200), 200),
                max_examples=_safe_int(arguments.get("max_examples", 3), 3),
            )
        if name == "snapshot_jsonl_preview":
            return self.snapshot_jsonl_preview(
                path=str(arguments.get("path", "")),
                start_line=_safe_int(arguments.get("start_line", 1), 1),
                count=_safe_int(arguments.get("count", 5), 5),
                text_char_limit=_safe_int(arguments.get("text_char_limit", 240), 240),
                max_generations=_safe_int(arguments.get("max_generations", 3), 3),
            )
        if name == "snapshot_jsonl_read_entry":
            raw_indexes = arguments.get("generation_indexes", [])
            if not isinstance(raw_indexes, list):
                raw_indexes = []
            parsed_indexes: List[int] = []
            for value in raw_indexes:
                try:
                    parsed_indexes.append(int(value))
                except (TypeError, ValueError):
                    continue
            return self.snapshot_jsonl_read_entry(
                path=str(arguments.get("path", "")),
                line_number=_safe_int(arguments.get("line_number", 1), 1),
                generation_indexes=parsed_indexes,
            )
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

    def snapshot_line_count(self, *, path: str) -> Dict[str, Any]:
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

        try:
            size_bytes = full_path.stat().st_size
        except OSError:
            size_bytes = 0

        line_count = _count_lines(full_path)
        if line_count is None:
            return {"error": "unable to read file"}

        return {
            "path": str(rel_path),
            "line_count": line_count,
            "size_bytes": size_bytes,
            "language": _detect_language(full_path),
        }

    def snapshot_jsonl_schema(
        self,
        *,
        path: str,
        max_entries: int = 200,
        max_examples: int = 3,
    ) -> Dict[str, Any]:
        rel_path = _validate_relative_path(path)
        if rel_path is None:
            return {"error": "invalid path"}
        full_path = (self.base_dir / rel_path).resolve()
        if not _is_under(self.base_dir, full_path):
            return {"error": "path escapes snapshot root"}
        file_error = _jsonl_validate_file(full_path)
        if file_error is not None:
            return {"error": file_error}
        return _jsonl_schema_from_file(
            full_path=full_path,
            display_path=str(rel_path),
            max_entries=max_entries,
            max_examples=max_examples,
        )

    def snapshot_jsonl_preview(
        self,
        *,
        path: str,
        start_line: int = 1,
        count: int = 5,
        text_char_limit: int = 240,
        max_generations: int = 3,
    ) -> Dict[str, Any]:
        rel_path = _validate_relative_path(path)
        if rel_path is None:
            return {"error": "invalid path"}
        full_path = (self.base_dir / rel_path).resolve()
        if not _is_under(self.base_dir, full_path):
            return {"error": "path escapes snapshot root"}
        file_error = _jsonl_validate_file(full_path)
        if file_error is not None:
            return {"error": file_error}
        return _jsonl_preview_from_file(
            full_path=full_path,
            display_path=str(rel_path),
            start_line=start_line,
            count=count,
            text_char_limit=text_char_limit,
            max_generations=max_generations,
        )

    def snapshot_jsonl_read_entry(
        self,
        *,
        path: str,
        line_number: int,
        generation_indexes: List[int],
    ) -> Dict[str, Any]:
        rel_path = _validate_relative_path(path)
        if rel_path is None:
            return {"error": "invalid path"}
        full_path = (self.base_dir / rel_path).resolve()
        if not _is_under(self.base_dir, full_path):
            return {"error": "path escapes snapshot root"}
        file_error = _jsonl_validate_file(full_path)
        if file_error is not None:
            return {"error": file_error}
        return _jsonl_read_entry_from_file(
            full_path=full_path,
            display_path=str(rel_path),
            line_number=line_number,
            generation_indexes=generation_indexes,
        )

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


@dataclass
class MultiRootSnapshotToolManager:
    """Tool handler for multiple visible roots (read-only)."""

    roots: Dict[str, Path]
    root_labels: Dict[str, str] = field(default_factory=dict)
    default_root: Optional[str] = None

    def __post_init__(self) -> None:
        # Normalize and drop invalid roots early.
        normalized: Dict[str, Path] = {}
        for name, path in self.roots.items():
            p = Path(path).expanduser().resolve()
            if p.is_dir():
                normalized[name] = p
        self.roots = normalized
        if self.default_root is None and "code" in self.roots:
            self.default_root = "code"

    @property
    def tool_definitions(self) -> List[Dict[str, Any]]:
        return SnapshotToolManager(base_dir=Path(".")).tool_definitions

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
        if name == "snapshot_line_count":
            return self.snapshot_line_count(path=str(arguments.get("path", "")))
        if name == "snapshot_read_many":
            requests = arguments.get("requests", [])
            return {"results": [self.snapshot_read(**req) for req in requests]}
        if name == "snapshot_jsonl_schema":
            return self.snapshot_jsonl_schema(
                path=str(arguments.get("path", "")),
                max_entries=_safe_int(arguments.get("max_entries", 200), 200),
                max_examples=_safe_int(arguments.get("max_examples", 3), 3),
            )
        if name == "snapshot_jsonl_preview":
            return self.snapshot_jsonl_preview(
                path=str(arguments.get("path", "")),
                start_line=_safe_int(arguments.get("start_line", 1), 1),
                count=_safe_int(arguments.get("count", 5), 5),
                text_char_limit=_safe_int(arguments.get("text_char_limit", 240), 240),
                max_generations=_safe_int(arguments.get("max_generations", 3), 3),
            )
        if name == "snapshot_jsonl_read_entry":
            raw_indexes = arguments.get("generation_indexes", [])
            if not isinstance(raw_indexes, list):
                raw_indexes = []
            parsed_indexes: List[int] = []
            for value in raw_indexes:
                try:
                    parsed_indexes.append(int(value))
                except (TypeError, ValueError):
                    continue
            return self.snapshot_jsonl_read_entry(
                path=str(arguments.get("path", "")),
                line_number=_safe_int(arguments.get("line_number", 1), 1),
                generation_indexes=parsed_indexes,
            )
        return {"error": f"Unknown tool: {name}"}

    def snapshot_manifest(self, max_files: int = 500) -> Dict[str, Any]:
        capped_max = min(max_files, _MAX_MANIFEST_FILES)
        files = []
        total_files = 0

        for root_name, root_dir in sorted(self.roots.items()):
            for fp in self._iter_files(root_dir):
                total_files += 1
                if len(files) >= capped_max:
                    continue
                try:
                    size_bytes = fp.stat().st_size
                except OSError:
                    size_bytes = 0
                rel = fp.relative_to(root_dir)
                files.append(
                    {
                        "path": f"{root_name}/{rel}",
                        "size_bytes": size_bytes,
                        "language": _detect_language(fp),
                    }
                )

        roots_meta = [
            {"name": name, "label": self.root_labels.get(name, "")}
            for name in sorted(self.roots.keys())
        ]

        return {
            "snapshot_root": ".",
            "snapshot_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "root_count": len(self.roots),
            "roots": roots_meta,
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

        search_targets = self._resolve_search_targets(dir)
        if search_targets is None:
            return {"matches": [], "error": "invalid search dir"}

        pattern = query if mode == "regex" else re.escape(query)
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            regex = re.compile(pattern, flags)
        except re.error as exc:
            return {"matches": [], "error": f"invalid regex: {exc}"}

        max_hits = min(max_results, _MAX_SEARCH_RESULTS)
        matches: List[Dict[str, Any]] = []

        for root_name, root_dir, search_root in search_targets:
            for fp in self._iter_files(search_root, glob_include, glob_exclude, root_dir=root_dir):
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
                        rel = fp.relative_to(root_dir)
                        matches.append(
                            {
                                "path": f"{root_name}/{rel}",
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
        root_name, rel_path = self._split_root_path(path)
        if root_name is None or rel_path is None:
            return {"error": "invalid path"}

        root_dir = self.roots.get(root_name)
        if root_dir is None:
            return {"error": "unknown root"}

        full_path = (root_dir / rel_path).resolve()
        if not _is_under(root_dir, full_path):
            return {"error": "path escapes root"}
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
                "path": f"{root_name}/{rel_path}",
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
            "path": f"{root_name}/{rel_path}",
            "start_line": start,
            "end_line": end,
            "text": "\n".join(rendered_lines),
            "truncated": truncated,
        }

    def snapshot_line_count(self, *, path: str) -> Dict[str, Any]:
        root_name, rel_path = self._split_root_path(path)
        if root_name is None or rel_path is None:
            return {"error": "invalid path"}

        root_dir = self.roots.get(root_name)
        if root_dir is None:
            return {"error": "unknown root"}

        full_path = (root_dir / rel_path).resolve()
        if not _is_under(root_dir, full_path):
            return {"error": "path escapes root"}
        if not full_path.is_file():
            return {"error": "file not found"}
        if _is_sensitive(full_path):
            return {"error": "file is blocked by sensitive file rules"}

        try:
            size_bytes = full_path.stat().st_size
        except OSError:
            size_bytes = 0

        line_count = _count_lines(full_path)
        if line_count is None:
            return {"error": "unable to read file"}

        return {
            "path": f"{root_name}/{rel_path}",
            "line_count": line_count,
            "size_bytes": size_bytes,
            "language": _detect_language(full_path),
        }

    def snapshot_jsonl_schema(
        self,
        *,
        path: str,
        max_entries: int = 200,
        max_examples: int = 3,
    ) -> Dict[str, Any]:
        root_name, rel_path = self._split_root_path(path)
        if root_name is None or rel_path is None:
            return {"error": "invalid path"}
        root_dir = self.roots.get(root_name)
        if root_dir is None:
            return {"error": "unknown root"}
        full_path = (root_dir / rel_path).resolve()
        if not _is_under(root_dir, full_path):
            return {"error": "path escapes root"}
        file_error = _jsonl_validate_file(full_path)
        if file_error is not None:
            return {"error": file_error}
        return _jsonl_schema_from_file(
            full_path=full_path,
            display_path=f"{root_name}/{rel_path}",
            max_entries=max_entries,
            max_examples=max_examples,
        )

    def snapshot_jsonl_preview(
        self,
        *,
        path: str,
        start_line: int = 1,
        count: int = 5,
        text_char_limit: int = 240,
        max_generations: int = 3,
    ) -> Dict[str, Any]:
        root_name, rel_path = self._split_root_path(path)
        if root_name is None or rel_path is None:
            return {"error": "invalid path"}
        root_dir = self.roots.get(root_name)
        if root_dir is None:
            return {"error": "unknown root"}
        full_path = (root_dir / rel_path).resolve()
        if not _is_under(root_dir, full_path):
            return {"error": "path escapes root"}
        file_error = _jsonl_validate_file(full_path)
        if file_error is not None:
            return {"error": file_error}
        return _jsonl_preview_from_file(
            full_path=full_path,
            display_path=f"{root_name}/{rel_path}",
            start_line=start_line,
            count=count,
            text_char_limit=text_char_limit,
            max_generations=max_generations,
        )

    def snapshot_jsonl_read_entry(
        self,
        *,
        path: str,
        line_number: int,
        generation_indexes: List[int],
    ) -> Dict[str, Any]:
        root_name, rel_path = self._split_root_path(path)
        if root_name is None or rel_path is None:
            return {"error": "invalid path"}
        root_dir = self.roots.get(root_name)
        if root_dir is None:
            return {"error": "unknown root"}
        full_path = (root_dir / rel_path).resolve()
        if not _is_under(root_dir, full_path):
            return {"error": "path escapes root"}
        file_error = _jsonl_validate_file(full_path)
        if file_error is not None:
            return {"error": file_error}
        return _jsonl_read_entry_from_file(
            full_path=full_path,
            display_path=f"{root_name}/{rel_path}",
            line_number=line_number,
            generation_indexes=generation_indexes,
        )

    def _split_root_path(self, path_str: str) -> tuple[Optional[str], Optional[Path]]:
        if not path_str:
            return None, None
        p = Path(path_str)
        if p.is_absolute() or ".." in p.parts:
            return None, None
        parts = p.parts
        if not parts:
            return None, None
        root_name = parts[0]
        if root_name in self.roots:
            rel = Path(*parts[1:]) if len(parts) > 1 else Path(".")
            return root_name, rel
        if self.default_root is not None:
            return self.default_root, p
        return None, None

    def _resolve_search_targets(
        self,
        dir_str: str,
    ) -> Optional[List[tuple[str, Path, Path]]]:
        if not dir_str or dir_str == ".":
            targets = []
            for name, root in sorted(self.roots.items()):
                targets.append((name, root, root))
            return targets

        root_name, rel = self._split_root_path(dir_str)
        if root_name is None or rel is None:
            return None

        root_dir = self.roots[root_name]
        search_root = (root_dir / rel).resolve()
        if not _is_under(root_dir, search_root):
            return None
        if not search_root.exists():
            return None
        return [(root_name, root_dir, search_root)]

    def _iter_files(
        self,
        root: Path,
        glob_include: Optional[str] = None,
        glob_exclude: Optional[str] = None,
        root_dir: Optional[Path] = None,
    ) -> Iterable[Path]:
        if not root.is_dir():
            return []
        root_dir = root_dir or root
        for fp in sorted(root.rglob("*")):
            if not fp.is_file():
                continue
            try:
                rel = fp.relative_to(root_dir)
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
