"""LLM-based training analysis: prompt construction and API calls."""

import json
import math
import os
import sys
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

try:
    import requests
except ImportError:
    requests = None

from .types import TrainingMetric, SystemMetric
from .utils import safe_json_dumps


class LLMAnalyzer:
    SYSTEM_PROMPT = """\
You are going to see data and code from a training run that is in progress. \
Your job is to analyze metrics and provide insights.

## Workflow

You have tools to inspect a snapshot of the training code that is running. Use them BEFORE writing \
your analysis — do not skip this step.

Available tools:
- snapshot_manifest: high-level file listing
- snapshot_search: find definitions/usages (e.g. search for loss functions, \
learning rate schedules, metric names, data loading logic)
- snapshot_read: read focused line ranges from a file
- snapshot_line_count: return the line count for a file
- snapshot_read_many: read multiple file slices at once
- snapshot_jsonl_schema: inspect JSONL field structure and types
- snapshot_jsonl_preview: preview rollout JSONL rows with truncated long text
- snapshot_jsonl_read_entry: read one JSONL row with full selected generations
- memory_write: write a memory (summary + long description) to persistent storage
- memory_read: read recent memories (including long descriptions) by category
- memory_list: list recent memory summaries by category
- memory_search: search memories by text query

On each analysis, use the tools to investigate at least one aspect of the code \
that is relevant to the current metrics. Only after investigating, write your analysis.

## Memory

You can store durable findings (bugs, issues, suggestions, anomalies, messages_from_user, or other)
using memory_write. Each memory has a short summary and a longer description.
Memory categories: bugs, issues, suggestions, anomalies, messages_from_user, other.
Use memory_read to view the full descriptions later.
When you discover a durable insight that should persist across analyses, write a memory.

The prompt includes recent memory summaries; use memory_read if you need the full descriptions.
Use memory_search to find relevant memories by keyword or phrase.
Do not repeat suggestions from the memories in your analysis. Assume the user knows the contents of the memories.

## Analysis checklist

Consider:
1. Loss trajectory: Is it decreasing? Any spikes or plateaus?
2. Gradient health: Are gradient norms stable? Signs of exploding/vanishing gradients (inf/nan)?
3. Learning rate: Is the schedule appropriate for the current phase?
4. GPU utilization: Are GPUs being used efficiently? Any bottlenecks? Any opportunities to improve utilization?
5. Memory usage: Any signs of memory leaks? Any opportunities for more efficiency?
6. Training speed: Are things running as expected? Any opportunities to run faster?
7. Is anything weird, does anything look wrong?
8. Are there any metrics that could be added to improve training monitoring?
9. Any other code improvements or modifications that could be helpful?

## Response format

- Start with a 1-2 sentence summary of training health
- List any concerns or anomalies (be specific with numbers, list 0-3)
- If something requires immediate attention, start your response with [ALERT]
- End with any recommendations (list 0-3)
- When you reference code, cite the file and line numbers you inspected

Be concise and focus on actionable insights. Don't repeat raw data unnecessarily.
Keep in mind the user will not be changing code or hyperparameters mid-training run. Any recommended code changes will be implemented in later runs.

## Conversation history

A conversation history may be provided containing your previous periodic \
analyses as well as any user questions and your responses from interactive \
Telegram chat. Focus on what has CHANGED since since previous analyses. \
Don't repeat observations you've already made unless they're still relevant. \
Highlight new developments, trends, or resolved issues. \
Do not repeat insights from past analyses. \
Do not alert on information that previous analyses already alerted on. \
If the user has asked questions via chat, consider their concerns in your analysis.

A summary of the training code will also be provided for quick reference, \
but always use the tools to check specifics rather than relying solely on the summary."""

    CODE_SUMMARY_PROMPT = """\
You are summarizing a codebase for an ML training monitoring system.
This summary will be included in future training analysis prompts so the \
analyst understands what the training code does, what model is being trained, \
and how the training loop works.

Respond in the following format (and nothing else):

## High-Level Summary
[2-10 sentences describing: what is being trained, the model architecture, \
training approach/algorithm, dataset details, loss functions, key hyperparameter \
strategies, and any other notable aspects of the code.]

## File Summaries
### <filepath>
[1-5 sentences per file describing its purpose and key contents.]

(Repeat for each file. Skip files that are trivial or boilerplate.)"""

    METRIC_DESCRIPTIONS_PROMPT = """\
You are analyzing ML training code to identify and describe the metrics \
that will be logged during training.

Look at the code and identify all metrics that are passed to the training \
monitor/logger (e.g. via tm.log(), logger.log(), or similar calls). For each \
metric key, provide a brief but specific natural-language description (2 sentences at most) of what \
it measures.

Dig through the code to find all the metrics that are logged. Also read through the code to make sure your descriptions are accurate.

Respond ONLY with a JSON object mapping metric names (exactly as they appear \
in the code as dictionary keys) to short descriptions. Do not include any \
other text, markdown formatting, or code blocks.

Example response:
{"loss": "cross-entropy loss", "accuracy": "prediction accuracy on batch", "lr": "learning rate", "grad_norm": "gradient L2 norm"}

If you cannot identify any logged metrics, respond with an empty object: {}"""

    MISSING_METRIC_DESCRIPTIONS_PROMPT = """\
You are analyzing ML training code to describe specific metrics that have \
been observed in a live training run but do not yet have descriptions.

The following metric keys have been observed in the training data but lack descriptions:
{missing_metrics}

The following metrics already have descriptions (for context):
{existing_descriptions}

Look at the code to understand what each missing metric measures. For each \
missing metric key listed above, provide a brief but specific natural-language \
description (2 sentences at most).

Respond ONLY with a JSON object mapping the missing metric names (exactly as \
listed above) to short descriptions. Do not include metrics that already have \
descriptions. Do not include any other text, markdown formatting, or code blocks.

If you cannot determine what a metric measures from the code, provide your \
best guess based on the metric name and the surrounding code context.

Example response:
{{"new_metric1": "description of metric 1", "new_metric2": "description of metric 2"}}

If none of the missing metrics can be identified, respond with an empty object: {{}}"""

    def __init__(
        self,
        model: str = "gpt-5.2",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: int = 8192,
        reasoning_effort: Optional[str] = None,
        sig_figs: int = 5,
    ):
        self.model = model
        self.base_url = base_url or "https://api.openai.com/v1"
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.sig_figs = sig_figs

        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY env var.")
        
        print(self.SYSTEM_PROMPT)

    @staticmethod
    def _is_reasoning_model_name(model: str) -> bool:
        return any(x in model.lower() for x in ["gpt-5", "o1", "o3"])

    @property
    def is_reasoning_model(self) -> bool:
        return self._is_reasoning_model_name(self.model)

    def _estimate_tokens(self, text: str) -> int:
        return len(text) // 4

    def _build_api_payload(
        self,
        input_items: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        model_override: Optional[str] = None,
        reasoning_effort_override: Optional[str] = None,
        instructions: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build an OpenAI Responses API payload."""
        max_tok = max_tokens or self.max_tokens
        model = model_override or self.model
        payload: Dict[str, Any] = {"model": model, "input": input_items}
        payload["max_output_tokens"] = max_tok
        if instructions:
            payload["instructions"] = instructions

        is_reasoning = self._is_reasoning_model_name(model)
        effort = reasoning_effort_override if reasoning_effort_override is not None else self.reasoning_effort
        if is_reasoning and effort is not None and str(effort).strip():
            payload["reasoning"] = {"effort": str(effort).strip()}

        if not is_reasoning:
            payload["temperature"] = 0.3

        return payload

    def _content_to_text(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                text = item.get("text")
                if text:
                    parts.append(str(text))
            return "\n".join(p for p in parts if p)
        if isinstance(content, dict):
            text = content.get("text")
            if text:
                return str(text)
        return str(content)

    def _content_to_input_parts(self, content: Any, default_type: str = "input_text") -> List[Dict[str, Any]]:
        if content is None:
            return []
        if isinstance(content, list):
            # Normalize existing content parts and coerce text parts to the default type.
            normalized: List[Dict[str, Any]] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                part = dict(item)
                part_type = part.get("type")
                if part_type in ("input_text", "output_text"):
                    part["type"] = default_type
                elif not part_type and "text" in part:
                    part["type"] = default_type
                normalized.append(part)
            return normalized
        if isinstance(content, dict):
            if "type" in content:
                part = dict(content)
                if part.get("type") in ("input_text", "output_text"):
                    part["type"] = default_type
                return [part]
            text = content.get("text")
            if text:
                return [{"type": default_type, "text": str(text)}]
        if isinstance(content, str):
            return [{"type": default_type, "text": content}]
        return [{"type": default_type, "text": str(content)}]

    def _messages_to_input_items(
        self,
        messages: List[Dict[str, Any]],
    ) -> tuple[Optional[str], List[Dict[str, Any]]]:
        instructions_parts: List[str] = []
        input_items: List[Dict[str, Any]] = []
        for message in messages:
            role = message.get("role") or "user"
            content = message.get("content")
            if role == "system":
                text = self._content_to_text(content).strip()
                if text:
                    instructions_parts.append(text)
                continue
            default_type = "output_text" if role == "assistant" else "input_text"
            input_items.append({
                "type": "message",
                "role": role,
                "content": self._content_to_input_parts(content, default_type=default_type),
            })
        instructions = "\n\n".join(instructions_parts).strip() if instructions_parts else None
        return instructions, input_items

    def _post_responses(self, payload: Dict[str, Any], timeout: int = 1020) -> Dict[str, Any]:
        """Send a Responses API request and return the raw response JSON."""
        response = requests.post(
            f"{self.base_url}/responses",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout,
        )
        try:
            response.raise_for_status()
        except Exception:
            self._log_responses_error(payload, response)
            raise
        return response.json()

    def _summarize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "model": payload.get("model"),
            "max_output_tokens": payload.get("max_output_tokens"),
            "has_instructions": bool(payload.get("instructions")),
        }
        instructions = payload.get("instructions") or ""
        summary["instructions_len"] = len(str(instructions))

        input_items = payload.get("input") or []
        summary["input_count"] = len(input_items)
        item_summaries: List[Dict[str, Any]] = []
        for item in input_items[:10]:
            item_type = item.get("type")
            role = item.get("role")
            content = item.get("content") or []
            if isinstance(content, list):
                content_types = [c.get("type") for c in content if isinstance(c, dict)]
                content_lengths = [
                    len(str(c.get("text", ""))) for c in content if isinstance(c, dict) and "text" in c
                ]
            else:
                content_types = [type(content).__name__]
                content_lengths = []
            item_summaries.append({
                "type": item_type,
                "role": role,
                "content_types": content_types,
                "content_text_lens": content_lengths[:3],
            })
        summary["inputs_preview"] = item_summaries

        tools = payload.get("tools") or []
        summary["tool_count"] = len(tools)
        tool_names: List[str] = []
        for tool in tools[:10]:
            if tool.get("type") == "function":
                tool_names.append(tool.get("name") or tool.get("function", {}).get("name") or "")
        summary["tool_names_preview"] = tool_names
        return summary

    def _log_responses_error(self, payload: Dict[str, Any], response: Any) -> None:
        try:
            summary = self._summarize_payload(payload)
            print("[analyzer] Responses API error", file=sys.stderr)
            print(f"[analyzer] Status: {getattr(response, 'status_code', 'unknown')}", file=sys.stderr)
            print(f"[analyzer] Payload summary: {summary}", file=sys.stderr)
            body = ""
            try:
                body = response.text  # type: ignore[assignment]
            except Exception:
                body = ""
            if body:
                body = body.strip()
                if len(body) > 2000:
                    body = body[:2000] + "\n...[truncated]"
                print(f"[analyzer] Response body: {body}", file=sys.stderr)
        except Exception:
            pass

    def _extract_response_text(self, response: Dict[str, Any]) -> str:
        """Extract concatenated assistant text from a Responses API response."""
        output_text = response.get("output_text")
        if isinstance(output_text, str) and output_text:
            return output_text

        texts: List[str] = []
        for item in response.get("output", []) or []:
            item_type = item.get("type")
            if item_type == "message" and item.get("role") == "assistant":
                for part in item.get("content", []) or []:
                    if part.get("type") in ("output_text", "text"):
                        text = part.get("text")
                        if text:
                            texts.append(text)
            elif item_type in ("output_text", "text"):
                text = item.get("text")
                if text:
                    texts.append(text)

        return "\n".join(texts).strip()

    def _normalize_tools_for_responses(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        def _ensure_required_for_object(schema: Dict[str, Any]) -> None:
            props = schema.get("properties")
            if not isinstance(props, dict):
                schema["properties"] = {}
                schema["required"] = []
                return
            prop_keys = list(props.keys())
            schema["required"] = prop_keys

        def _ensure_schema_rules(schema: Any) -> None:
            if not isinstance(schema, dict):
                return
            schema_type = schema.get("type")
            if schema_type == "object":
                if "additionalProperties" not in schema:
                    schema["additionalProperties"] = False
                _ensure_required_for_object(schema)
                props = schema.get("properties")
                if isinstance(props, dict):
                    for value in props.values():
                        _ensure_schema_rules(value)
            elif schema_type == "array":
                _ensure_schema_rules(schema.get("items"))

        normalized: List[Dict[str, Any]] = []
        for tool in tools:
            if tool.get("type") != "function":
                normalized.append(tool)
                continue
            if "function" in tool:
                fn = tool.get("function", {})
                params = dict(fn.get("parameters", {}) or {})
                _ensure_schema_rules(params)
                normalized.append({
                    "type": "function",
                    "name": fn.get("name"),
                    "description": fn.get("description"),
                    "parameters": params,
                    "strict": fn.get("strict", True),
                })
            else:
                normalized_tool = dict(tool)
                params = dict(normalized_tool.get("parameters", {}) or {})
                _ensure_schema_rules(params)
                normalized_tool["parameters"] = params
                normalized_tool.setdefault("strict", True)
                normalized.append(normalized_tool)
        return normalized

    def _call_api_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_handler: Callable[[str, Dict[str, Any]], Dict[str, Any]],
        timeout: int = 1020,
        max_rounds: int = 120,
        max_tokens: Optional[int] = None,
        openai_model_override: Optional[str] = None,
        reasoning_effort_override: Optional[str] = None,
        on_tool_call: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> str:
        instructions, input_items = self._messages_to_input_items(messages)
        tools_payload = self._normalize_tools_for_responses(tools)

        for _ in range(max_rounds):
            payload = self._build_api_payload(
                input_items,
                max_tokens=max_tokens,
                model_override=openai_model_override,
                reasoning_effort_override=reasoning_effort_override,
                instructions=instructions,
            )
            payload["tools"] = tools_payload
            payload["tool_choice"] = "auto"

            result = self._post_responses(payload, timeout=timeout)
            output_items = result.get("output", []) or []
            if output_items:
                input_items.extend(output_items)

            function_calls = [
                item for item in output_items
                if item.get("type") in ("function_call", "tool_call")
            ]

            if not function_calls:
                return self._extract_response_text(result)

            for tool_call in function_calls:
                name = tool_call.get("name", "")
                raw_args = tool_call.get("arguments", "{}")
                args: Dict[str, Any]
                if isinstance(raw_args, dict):
                    args = raw_args
                else:
                    try:
                        args = json.loads(raw_args) if raw_args else {}
                    except json.JSONDecodeError:
                        args = {}

                if on_tool_call is not None:
                    on_tool_call(name, args)
                else:
                    # Fallback logging when no external callback is provided.
                    args_preview = safe_json_dumps(args)
                    if len(args_preview) > 400:
                        args_preview = args_preview[:400] + "...[truncated]"
                    print(
                        f"[analyzer] Tool call: {name} args={args_preview}",
                        file=sys.stderr,
                    )
                result_payload = tool_handler(name, args)
                call_id = tool_call.get("call_id") or tool_call.get("id")
                if not call_id:
                    continue
                input_items.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": safe_json_dumps(result_payload),
                })

        return "Analysis failed: tool loop did not resolve in time."

    def _call_api_for_messages(
        self,
        messages: List[Dict[str, Any]],
        timeout: int = 1020,
        max_tokens: Optional[int] = None,
        tool_manager: Optional[Any] = None,
        openai_model_override: Optional[str] = None,
        reasoning_effort_override: Optional[str] = None,
        on_tool_call: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> str:
        if tool_manager is not None:
            return self._call_api_with_tools(
                messages,
                tool_manager.tool_definitions,
                tool_manager.handle,
                timeout=timeout,
                max_tokens=max_tokens,
                openai_model_override=openai_model_override,
                reasoning_effort_override=reasoning_effort_override,
                on_tool_call=on_tool_call,
            )
        instructions, input_items = self._messages_to_input_items(messages)
        payload = self._build_api_payload(
            input_items,
            max_tokens=max_tokens,
            model_override=openai_model_override,
            reasoning_effort_override=reasoning_effort_override,
            instructions=instructions,
        )
        result = self._post_responses(payload, timeout=timeout)
        return self._extract_response_text(result)

    # ------------------------------------------------------------------ #
    #  Code summary generation
    # ------------------------------------------------------------------ #

    def generate_code_summary(
        self,
        tool_manager: Any,
        code_diff: Optional[str] = None,
        on_tool_call: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> str:
        """
        Use snapshot tools to explore the code and return a structured
        summary (high-level + per-file).

        The LLM uses snapshot_manifest, snapshot_read, snapshot_search, etc.
        to inspect the code rather than receiving it as raw text.
        If code_diff is provided, it is included as context (not as code).
        """
        user_content = (
            "Please explore the code snapshot using the available tools. "
            "Start by calling snapshot_manifest to see what files are available, "
            "then read the key files to understand the codebase. "
            "After investigating, produce the summary."
        )
        if code_diff:
            user_content += (
                "\n\nNote: The following is a generated diff (NOT part of the code). "
                "Use it only as context about what changed:\n\n"
                + code_diff
            )
        messages = [
            {"role": "system", "content": self.CODE_SUMMARY_PROMPT},
            {"role": "user", "content": user_content},
        ]
        return self._call_api_with_tools(
            messages,
            tool_manager.tool_definitions,
            tool_manager.handle,
            timeout=1080,
            max_tokens=10000,
            on_tool_call=on_tool_call,
        )

    # ------------------------------------------------------------------ #
    #  Metric descriptions generation
    # ------------------------------------------------------------------ #

    def generate_metric_descriptions(
        self,
        tool_manager: Any,
        code_summary: Optional[str] = None,
        code_diff: Optional[str] = None,
        max_retries: int = 3,
        on_tool_call: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Dict[str, str]:
        """
        Use snapshot tools to explore the code and return a dict mapping
        metric names to short natural-language descriptions.

        Retries up to *max_retries* times on parse failures.
        Returns an empty dict if all attempts fail.
        If code_diff is provided, it is included as context (not as code).
        """
        user_content = (
            "Please explore the code snapshot using the available tools. "
            "Start by calling snapshot_manifest to see what files exist. "
            "Then search for logging calls (e.g. tm.log, logger.log, .log() calls) "
            "and read the surrounding code to identify all logged metrics "
            "and understand what each one measures. "
            "After investigating, produce the JSON object."
        )
        if code_diff:
            user_content += (
                "\n\nNote: The following is a generated diff (NOT part of the code). "
                "Use it only as context about what changed:\n\n"
                + code_diff
            )
        if code_summary:
            user_content += (
                "\n\nCode summary (for quick context — still use the tools to verify specifics):\n"
                + code_summary
            )

        for attempt in range(1, max_retries + 1):
            messages = [
                {"role": "system", "content": self.METRIC_DESCRIPTIONS_PROMPT},
                {"role": "user", "content": user_content},
            ]
            try:
                raw = self._call_api_with_tools(
                    messages,
                    tool_manager.tool_definitions,
                    tool_manager.handle,
                    timeout=1020,
                    max_tokens=10024,
                    on_tool_call=on_tool_call,
                )
                print(f"[analyzer] Metric descriptions response (attempt {attempt}/{max_retries}):\n{raw}")

                # Strip markdown fences if present
                cleaned = raw.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
                descriptions = json.loads(cleaned)
                if not isinstance(descriptions, dict):
                    print(f"[analyzer] Attempt {attempt}/{max_retries}: response parsed but was not a dict, got {type(descriptions).__name__}")
                    continue
                # Validate: all keys and values should be strings
                result = {
                    str(k): str(v)
                    for k, v in descriptions.items()
                    if isinstance(k, str) and isinstance(v, str)
                }
                if not result:
                    print(f"[analyzer] Attempt {attempt}/{max_retries}: parsed dict was empty after validation")
                    continue
                return result
            except json.JSONDecodeError as e:
                print(f"[analyzer] Attempt {attempt}/{max_retries}: JSON parse error: {e}")
            except Exception as e:
                print(f"[analyzer] Attempt {attempt}/{max_retries}: error: {e}")

        print(f"[analyzer] Failed to parse metric descriptions after {max_retries} attempts")
        return {}

    # ------------------------------------------------------------------ #
    #  Missing metric descriptions generation
    # ------------------------------------------------------------------ #

    def generate_missing_metric_descriptions(
        self,
        missing_metrics: List[str],
        existing_descriptions: Dict[str, str],
        tool_manager: Any,
        code_summary: Optional[str] = None,
        max_retries: int = 3,
        on_tool_call: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Dict[str, str]:
        """
        Use snapshot tools to explore the code and generate descriptions
        for specific metric keys that are missing from the existing set.

        Returns a dict of {metric_name: description} for newly described
        metrics.  Returns an empty dict if all attempts fail or no
        descriptions could be generated.
        """
        if not missing_metrics:
            return {}

        existing_str = (
            json.dumps(existing_descriptions, indent=2, sort_keys=True)
            if existing_descriptions
            else "{}"
        )
        missing_str = ", ".join(f'"{m}"' for m in sorted(missing_metrics))

        system_content = self.MISSING_METRIC_DESCRIPTIONS_PROMPT.format(
            missing_metrics=missing_str,
            existing_descriptions=existing_str,
        )

        user_content = (
            "Please explore the code snapshot using the available tools. "
            "Search for the missing metric names in the code to understand "
            "what they measure. After investigating, produce the JSON object "
            "with descriptions for the missing metrics only."
        )
        if code_summary:
            user_content += (
                "\n\nCode summary (for quick context — still use the tools to verify specifics):\n"
                + code_summary
            )

        for attempt in range(1, max_retries + 1):
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ]
            try:
                raw = self._call_api_with_tools(
                    messages,
                    tool_manager.tool_definitions,
                    tool_manager.handle,
                    timeout=1020,
                    max_tokens=10024,
                    on_tool_call=on_tool_call,
                )
                print(
                    f"[analyzer] Missing metric descriptions response "
                    f"(attempt {attempt}/{max_retries}):\n{raw}"
                )

                # Strip markdown fences if present
                cleaned = raw.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()

                descriptions = json.loads(cleaned)
                if not isinstance(descriptions, dict):
                    print(
                        f"[analyzer] Attempt {attempt}/{max_retries}: "
                        f"response was not a dict, got {type(descriptions).__name__}"
                    )
                    continue

                result = {
                    str(k): str(v)
                    for k, v in descriptions.items()
                    if isinstance(k, str) and isinstance(v, str)
                }
                # Return even if empty — that means the model couldn't find them
                return result

            except json.JSONDecodeError as e:
                print(f"[analyzer] Attempt {attempt}/{max_retries}: JSON parse error: {e}")
            except Exception as e:
                print(f"[analyzer] Attempt {attempt}/{max_retries}: error: {e}")

        print(
            f"[analyzer] Failed to generate missing metric descriptions "
            f"after {max_retries} attempts"
        )
        return {}

    # ------------------------------------------------------------------ #
    #  Conversation history formatting
    # ------------------------------------------------------------------ #

    @staticmethod
    def _format_conversation_message(msg: Dict[str, Any]) -> str:
        """Format a single conversation history entry for inclusion in a prompt."""
        timestamp = datetime.fromtimestamp(msg["timestamp"]).strftime('%H:%M:%S')
        msg_type = msg.get("type", "")

        if msg_type == "analysis":
            label = f"[{timestamp}] [MODEL - Periodic Analysis]"
        elif msg_type == "chat_response":
            label = f"[{timestamp}] [MODEL - Response to User]"
        elif msg_type == "user_message":
            label = f"[{timestamp}] [USER - Chat Message]"
        else:
            role = msg.get("role", "unknown").upper()
            label = f"[{timestamp}] [{role}]"

        return f"{label}:\n{msg['text']}"

    def _select_conversation_messages(
        self,
        conversation_history: List[Dict[str, Any]],
        max_tokens: int = 5000,
    ) -> List[str]:
        """
        Select the most recent conversation messages that fit within the
        token budget. Messages are not truncated — if a message doesn't fit,
        it and all older messages are excluded.

        Returns a list of formatted message strings in chronological order.
        """
        header = "\n## Conversation History (previous analyses and user interactions, oldest first)\n"
        footer = "\n\n## END OF CONVERSATION HISTORY"
        overhead = self._estimate_tokens(header) + self._estimate_tokens(footer)
        budget = max(0, max_tokens - overhead)

        selected: list[str] = []
        tokens_used = 0

        for msg in reversed(conversation_history):
            formatted = self._format_conversation_message(msg)
            msg_tokens = self._estimate_tokens(formatted)
            if tokens_used + msg_tokens > budget:
                break
            selected.append(formatted)
            tokens_used += msg_tokens

        selected.reverse()  # back to chronological order
        return selected

    # ------------------------------------------------------------------ #
    #  Metric merging
    # ------------------------------------------------------------------ #

    @staticmethod
    def _merge_metrics_by_step(metrics: List[TrainingMetric]) -> List[TrainingMetric]:
        """Merge multiple TrainingMetric entries for the same step into one.

        When multiple ``tm.log()`` calls happen for the same step with
        different metric keys, this combines them into a single entry per
        step.  Later values overwrite earlier ones for the same key, but
        ``None`` does not overwrite a real value.
        """
        by_step: dict[int, TrainingMetric] = {}
        for m in sorted(metrics, key=lambda x: (x.step, x.timestamp)):
            if m.step in by_step:
                existing = by_step[m.step]
                merged = dict(existing.metrics)
                for k, v in m.metrics.items():
                    if v is not None or k not in merged:
                        merged[k] = v
                by_step[m.step] = TrainingMetric(
                    step=m.step,
                    timestamp=max(existing.timestamp, m.timestamp),
                    metrics=merged,
                )
            else:
                by_step[m.step] = TrainingMetric(
                    step=m.step,
                    timestamp=m.timestamp,
                    metrics=dict(m.metrics),
                )
        return sorted(by_step.values(), key=lambda x: x.step)

    # ------------------------------------------------------------------ #
    #  Analysis prompt building
    # ------------------------------------------------------------------ #

    def build_prompt(
        self,
        training_metrics: List[TrainingMetric],
        system_metrics: List[SystemMetric],
        max_prompt_len: int = 8000,
        experiment_name: str = "unknown",
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        max_conversation_history_tokens: int = 5000,
        config: Optional[Dict[str, Any]] = None,
        code_summary: Optional[str] = None,
        metric_descriptions: Optional[Dict[str, str]] = None,
        memory_summaries: Optional[Dict[str, List[str]]] = None,
    ) -> str:
        # Merge metrics logged in multiple calls for the same step
        if training_metrics:
            training_metrics = self._merge_metrics_by_step(training_metrics)

        sections = []

        header = f"""\
Training Analysis Request
========================
Experiment: {experiment_name}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Training steps collected: {len(training_metrics)}
System samples collected: {len(system_metrics)}
"""
        sections.append(header)

        if code_summary:
            sections.append("\n## Summary of Visible Training Code\n" + code_summary)

        config_for_prompt = config
        training_command = None
        if isinstance(config, dict):
            config_for_prompt = dict(config)
            training_command = config_for_prompt.pop("training_command", None)

        if training_command:
            sections.append("\n## Command Used to Start Training\n" + str(training_command))

        if config_for_prompt:
            try:
                cfg_text = safe_json_dumps(config_for_prompt)
            except Exception:
                cfg_text = str(config_for_prompt)
            if len(cfg_text) > 6000:
                cfg_text = cfg_text[:6000] + "\n... (truncated)"
            sections.append("\n## Run Config\n" + cfg_text)

        if memory_summaries is not None:
            mem_section_lines = ["\n## Memory Summaries (most recent 50 per category)"]
            for category in ["bugs", "issues", "suggestions", "anomalies", "messages_from_user", "other"]:
                summaries = memory_summaries.get(category, [])
                mem_section_lines.append(f"\n### {category.capitalize()}")
                if summaries:
                    for summary in summaries[-50:]:
                        mem_section_lines.append(f"- {summary}")
                else:
                    mem_section_lines.append("- None")
            sections.append("\n".join(mem_section_lines))

        # Conversation history (replaces old previous_analyses section)
        if conversation_history:
            selected = self._select_conversation_messages(
                conversation_history,
                max_tokens=max_conversation_history_tokens,
            )
            if selected:
                conv_section = (
                    "\n## Conversation History "
                    "(previous analyses and user interactions, oldest first)\n\n"
                )
                conv_section += "\n\n".join(selected)
                conv_section += "\n\n## END OF CONVERSATION HISTORY"
                sections.append(conv_section)

        if training_metrics:
            sorted_metrics = sorted(training_metrics, key=lambda x: x.step)
            first, last = sorted_metrics[0], sorted_metrics[-1]

            all_keys: set[str] = set()
            for m in sorted_metrics:
                all_keys.update(m.metrics.keys())

            # Separate section for metric descriptions (static per-run)
            if metric_descriptions:
                desc_lines = ["\n## Metric Descriptions"]
                for key in sorted(all_keys):
                    if key in metric_descriptions:
                        desc_lines.append(f"- {key}: {metric_descriptions[key]}")
                if len(desc_lines) > 1:
                    sections.append("\n".join(desc_lines))

            summary_lines = ["\n## Training Metrics Summary"]
            summary_lines.append(f"Step range: {first.step} -> {last.step}")
            summary_lines.append(f"Time span: {(last.timestamp - first.timestamp) / 60:.1f} minutes")

            for key in sorted(all_keys):
                values = [m.metrics.get(key) for m in sorted_metrics if key in m.metrics]
                numeric_values = [v for v in values if isinstance(v, (int, float))]
                if numeric_values:
                    first_val, last_val = numeric_values[0], numeric_values[-1]
                    min_val, max_val = min(numeric_values), max(numeric_values)
                    avg_val = sum(numeric_values) / len(numeric_values)
                    delta = last_val - first_val

                    summary_lines.append(
                        f"  {key}: {first_val:.6g} -> {last_val:.6g} "
                        f"(Δ={delta:+.6g}, min={min_val:.6g}, max={max_val:.6g}, avg={avg_val:.6g})"
                    )
            sections.append("\n".join(summary_lines))

        if system_metrics:
            sys_lines = ["\n## System Metrics Summary"]

            if system_metrics[0].gpus:
                num_gpus = len(system_metrics[0].gpus)
                for gpu_idx in range(num_gpus):
                    utils = [s.gpus[gpu_idx]["util_percent"] for s in system_metrics if len(s.gpus) > gpu_idx]
                    mems = [s.gpus[gpu_idx]["mem_percent"] for s in system_metrics if len(s.gpus) > gpu_idx]
                    temps = [s.gpus[gpu_idx]["temp_c"] for s in system_metrics if len(s.gpus) > gpu_idx and s.gpus[gpu_idx]["temp_c"]]
                    powers = [s.gpus[gpu_idx]["power_w"] for s in system_metrics if len(s.gpus) > gpu_idx and s.gpus[gpu_idx]["power_w"]]

                    sys_lines.append(f"  GPU {gpu_idx}:")
                    if utils:
                        sys_lines.append(f"    Utilization: avg={sum(utils)/len(utils):.1f}%, min={min(utils):.1f}%, max={max(utils):.1f}%")
                    if mems:
                        sys_lines.append(f"    Memory: avg={sum(mems)/len(mems):.1f}%, max={max(mems):.1f}%")
                    if temps:
                        sys_lines.append(f"    Temperature: avg={sum(temps)/len(temps):.1f}°C, max={max(temps):.1f}°C")
                    if powers:
                        sys_lines.append(f"    Power: avg={sum(powers)/len(powers):.1f}W, max={max(powers):.1f}W")

            cpu_vals = [s.cpu_percent for s in system_metrics]
            ram_vals = [s.ram_percent for s in system_metrics]
            sys_lines.append(f"  CPU: avg={sum(cpu_vals)/len(cpu_vals):.1f}%, max={max(cpu_vals):.1f}%")
            sys_lines.append(f"  RAM: avg={sum(ram_vals)/len(ram_vals):.1f}%, max={max(ram_vals):.1f}%")
            sections.append("\n".join(sys_lines))

        # Raw metrics (fill remaining token budget)
        current_len = sum(self._estimate_tokens(s) for s in sections)
        remaining = max_prompt_len - current_len - 500

        if remaining > 500 and training_metrics:
            raw_lines = ["\n## Recent Raw Metrics (by metric, oldest to newest)"]
            sorted_metrics = sorted(training_metrics, key=lambda x: x.step)

            all_keys = set()
            for m in sorted_metrics:
                all_keys.update(m.metrics.keys())

            def fmt_val(v):
                if v is None:
                    return "None"
                if isinstance(v, float):
                    if not math.isfinite(v):
                        return str(v)
                    if v == 0:
                        return "0"
                    return f"{v:.{self.sig_figs}g}"
                return str(v)

            steps = [m.step for m in sorted_metrics]
            raw_lines.append(f"steps: {', '.join(str(s) for s in steps)}")

            for key in sorted(all_keys):
                values = [m.metrics.get(key) for m in sorted_metrics]
                formatted = [fmt_val(v) for v in values]
                line = f"{key}: {', '.join(formatted)}"
                line_tokens = self._estimate_tokens(line)
                if remaining - line_tokens < 0:
                    raw_lines.append("... (truncated)")
                    break
                raw_lines.append(line)
                remaining -= line_tokens

            sections.append("\n".join(raw_lines))

        return "\n".join(sections)

    def analyze(
        self,
        training_metrics: List[TrainingMetric],
        system_metrics: List[SystemMetric],
        max_prompt_len: int = 8000,
        experiment_name: str = "unknown",
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        max_conversation_history_tokens: int = 5000,
        config: Optional[Dict[str, Any]] = None,
        code_summary: Optional[str] = None,
        metric_descriptions: Optional[Dict[str, str]] = None,
        memory_summaries: Optional[Dict[str, List[str]]] = None,
        tool_manager: Optional[Any] = None,
        openai_model_override: Optional[str] = None,
        reasoning_effort_override: Optional[str] = None,
        on_tool_call: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> tuple[str, bool]:
        prompt = self.build_prompt(
            training_metrics, system_metrics, max_prompt_len, experiment_name,
            conversation_history=conversation_history,
            max_conversation_history_tokens=max_conversation_history_tokens,
            config=config,
            code_summary=code_summary,
            metric_descriptions=metric_descriptions,
            memory_summaries=memory_summaries,
        )

        print('prompt' + '=' * 40 + prompt + '=' * 40)  # keep for debugging

        try:
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            if tool_manager is not None:
                text = self._call_api_with_tools(
                    messages,
                    tool_manager.tool_definitions,
                    tool_manager.handle,
                    openai_model_override=openai_model_override,
                    reasoning_effort_override=reasoning_effort_override,
                    on_tool_call=on_tool_call,
                )
            else:
                instructions, input_items = self._messages_to_input_items(messages)
                payload = self._build_api_payload(
                    input_items,
                    model_override=openai_model_override,
                    reasoning_effort_override=reasoning_effort_override,
                    instructions=instructions,
                )
                result = self._post_responses(payload)
                text = self._extract_response_text(result)
            should_alert = text.strip().startswith("[ALERT]")
            return text, should_alert

        except Exception as e:
            return f"Analysis failed: {e}", False
