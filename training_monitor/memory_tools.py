"""Memory tools for writing and reading model-generated memories."""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from .utils import ensure_dir


MEMORY_CATEGORIES = (
    "bugs",
    "issues",
    "suggestions",
    "anomalies",
    "messages_from_user",
    "other",
)


def _sanitize_text(value: Any) -> str:
    text = str(value or "").strip()
    return " ".join(text.split())


@dataclass
class MemoryToolManager:
    """Read/write tool handler for memory entries (restricted to a directory)."""

    memory_dir: Path
    get_experiment: Optional[Callable[[], str]] = None
    lock: threading.RLock = field(default_factory=threading.RLock)

    @property
    def tool_names(self) -> List[str]:
        return ["memory_write", "memory_read", "memory_list", "memory_search"]

    @property
    def tool_definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "memory_write",
                    "description": (
                        "Write a new memory entry (short summary + longer description) "
                        "into the memories store. Category is required."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "enum": list(MEMORY_CATEGORIES),
                            },
                            "summary": {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": ["category", "summary", "description"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "memory_read",
                    "description": (
                        "Read recent memory entries (including full descriptions) "
                        "from a category."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "enum": list(MEMORY_CATEGORIES),
                            },
                            "limit": {"type": "integer", "default": 5},
                        },
                        "required": ["category"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "memory_list",
                    "description": (
                        "List recent memory summaries for a category (summaries only)."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "enum": list(MEMORY_CATEGORIES),
                            },
                            "limit": {"type": "integer", "default": 50},
                        },
                        "required": ["category"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "memory_search",
                    "description": (
                        "Search memories by text query across summaries and descriptions. "
                        "If category is omitted, searches all categories."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "category": {
                                "type": "string",
                                "enum": list(MEMORY_CATEGORIES),
                            },
                            "limit": {"type": "integer", "default": 10},
                            "case_sensitive": {"type": "boolean", "default": False},
                        },
                        "required": ["query"],
                    },
                },
            },
        ]

    def handle(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if name == "memory_write":
            return self.memory_write(
                category=str(arguments.get("category", "")),
                summary=arguments.get("summary", ""),
                description=arguments.get("description", ""),
            )
        if name == "memory_read":
            return self.memory_read(
                category=str(arguments.get("category", "")),
                limit=int(arguments.get("limit", 5)),
            )
        if name == "memory_list":
            return self.memory_list(
                category=str(arguments.get("category", "")),
                limit=int(arguments.get("limit", 50)),
            )
        if name == "memory_search":
            category = arguments.get("category")
            category_value = str(category) if category is not None else None
            return self.memory_search(
                query=str(arguments.get("query", "")),
                category=category_value,
                limit=int(arguments.get("limit", 10)),
                case_sensitive=bool(arguments.get("case_sensitive", False)),
            )
        return {"error": f"Unknown tool: {name}"}

    def memory_write(self, category: str, summary: Any, description: Any) -> Dict[str, Any]:
        if category not in MEMORY_CATEGORIES:
            return {"error": f"Invalid category: {category}"}

        summary_text = _sanitize_text(summary)
        description_text = _sanitize_text(description)
        if not summary_text or not description_text:
            return {"error": "summary and description must be non-empty"}

        experiment = self.get_experiment() if self.get_experiment else "unknown"
        entry = {
            "id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "created_at": time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime()),
            "experiment": experiment,
            "category": category,
            "summary": summary_text,
            "description": description_text,
        }

        ensure_dir(self.memory_dir)
        path = self._category_path(category)
        try:
            with self.lock:
                entries = self._load_entries(category)
                entries.append(entry)
                path.write_text(json.dumps(entries, indent=2, sort_keys=True), encoding="utf-8")
        except Exception as e:
            return {"error": f"Failed to write memory: {e}"}

        return {
            "ok": True,
            "entry": {
                "id": entry["id"],
                "category": category,
                "summary": summary_text,
                "created_at": entry["created_at"],
            },
        }

    def memory_read(self, category: str, limit: int = 5) -> Dict[str, Any]:
        if category not in MEMORY_CATEGORIES:
            return {"error": f"Invalid category: {category}"}
        limit = max(1, min(int(limit), 100))
        entries = self._load_entries(category)
        if not entries:
            return {"entries": []}
        return {"entries": entries[-limit:]}

    def memory_list(self, category: str, limit: int = 50) -> Dict[str, Any]:
        if category not in MEMORY_CATEGORIES:
            return {"error": f"Invalid category: {category}"}
        limit = max(1, min(int(limit), 200))
        entries = self._load_entries(category)
        if not entries:
            return {"summaries": []}
        summaries = [
            {
                "id": e.get("id"),
                "created_at": e.get("created_at"),
                "summary": e.get("summary"),
            }
            for e in entries[-limit:]
        ]
        return {"summaries": summaries}

    def memory_search(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 10,
        case_sensitive: bool = False,
    ) -> Dict[str, Any]:
        q = _sanitize_text(query)
        if not q:
            return {"error": "query must be non-empty"}

        categories: List[str] = []
        if category is None:
            categories = list(MEMORY_CATEGORIES)
        else:
            if category not in MEMORY_CATEGORIES:
                return {"error": f"Invalid category: {category}"}
            categories = [category]

        limit = max(1, min(int(limit), 200))
        results: List[Dict[str, Any]] = []

        q_cmp = q if case_sensitive else q.lower()

        for cat in categories:
            entries = self._load_entries(cat)
            for entry in reversed(entries):
                summary = str(entry.get("summary", "") or "")
                description = str(entry.get("description", "") or "")
                haystack = f"{summary}\n{description}"
                hay_cmp = haystack if case_sensitive else haystack.lower()
                if q_cmp in hay_cmp:
                    results.append({
                        "id": entry.get("id"),
                        "created_at": entry.get("created_at"),
                        "category": cat,
                        "summary": summary,
                        "description": description,
                    })
                    if len(results) >= limit:
                        return {"results": results}

        return {"results": results}

    def recent_summaries(self, limit: int = 50) -> Dict[str, List[str]]:
        output: Dict[str, List[str]] = {}
        for category in MEMORY_CATEGORIES:
            entries = self._load_entries(category)
            output[category] = [
                e.get("summary", "")
                for e in entries[-limit:]
                if isinstance(e.get("summary"), str) and e.get("summary")
            ]
        return output

    def _category_path(self, category: str) -> Path:
        return self.memory_dir / f"{category}.json"

    def _load_entries(self, category: str) -> List[Dict[str, Any]]:
        path = self._category_path(category)
        if not path.exists():
            return []
        entries: List[Dict[str, Any]] = []
        try:
            with self.lock:
                raw = path.read_text(encoding="utf-8").strip()
        except Exception:
            return []
        if not raw:
            return []
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    entries.append(item)
        return entries


@dataclass
class CombinedToolManager:
    """Combine a base tool manager with additional tool managers."""

    base_manager: Optional[Any] = None
    extra_managers: Iterable[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._extra_managers = list(self.extra_managers)
        self._extra_tool_map: Dict[str, Any] = {}
        for mgr in self._extra_managers:
            for name in getattr(mgr, "tool_names", []):
                self._extra_tool_map[name] = mgr

    @property
    def tool_definitions(self) -> List[Dict[str, Any]]:
        defs: List[Dict[str, Any]] = []
        if self.base_manager is not None:
            defs.extend(self.base_manager.tool_definitions)
        for mgr in self._extra_managers:
            defs.extend(mgr.tool_definitions)
        return defs

    def handle(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        mgr = self._extra_tool_map.get(name)
        if mgr is not None:
            return mgr.handle(name, arguments)
        if self.base_manager is not None:
            return self.base_manager.handle(name, arguments)
        return {"error": f"Unknown tool: {name}"}
