"""
kagentic/schema.py
----------------
Pydantic schema returned by the LLM at every ReAct step.

Key trick (same as complemon/agent/schemas.py):
  - Passing schema=AgentReActStep to llm.prompt() makes kbench check
    has_nested_models() on the schema's JSON representation.
  - If True ($defs present) -> kbench uses TEXT-based JSON instructions
    (appends "Output JSON using this schema: ...") -- works on ANY model.
  - If False -> kbench tries client.beta.chat.completions.parse, which
    the Kaggle model proxy cannot convert -> 400 "failed to convert config".

Solution: nest ToolCall inside AgentReActStep so $defs is always present,
exactly like complemon's AgentResponse + ToolCall design.
"""
import json as _json
import re as _re
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, Field, field_validator

try:
    from json_repair import loads as _json_repair_loads
except ImportError:
    _json_repair_loads = None  # type: ignore[assignment]


class ToolCall(BaseModel):
    """A single tool invocation -- name + JSON-serialized argument string."""
    name: str = Field(
        description=(
            "Name of the tool to call. "
            "Use 'final_answer' when you have a complete answer for the user."
        )
    )
    arguments: Union[Dict[str, Any], str] = Field(
        default="{}",
        description=(
            "Arguments to pass to the tool as a JSON object (native dict, no escaping needed). "
            'Example: {"query": "hello world"}. '
            'For \'final_answer\' without structured output, use: {"answer": "<your complete answer>"}. '
            'For \'final_answer\' with structured output, spread all schema fields here directly: '
            '{"answer": "short answer", "explanation": "reasoning here"}.'
        ),
    )

    @field_validator("arguments", mode="before")
    @classmethod
    def _coerce_arguments(cls, v: Any) -> Any:
        """Normalize arguments to a compact JSON string for internal use.

        Accepts either:
        - A native dict/list (preferred, no escaping needed) — serialized to JSON string.
        - A JSON-encoded string — passed through as-is (json_repair handles malformed cases).

        This means the LLM can freely output arguments as a real JSON object
        (e.g. ``{"query": "hello", "topk": 3}``) without any backslash escaping,
        saving tokens and eliminating escape-related parse failures.
        """
        if isinstance(v, (dict, list)):
            return _json.dumps(v)
        return v



class AgentReActStep(BaseModel):
    """
    One step in the ReAct loop.
    The LLM must ALWAYS respond with this exact JSON structure.
    """
    thought: Optional[str] = Field(
        default=None,
        description=(
            "Your internal reasoning / chain-of-thought before acting. "
            "Be concise. This is NOT shown to the user."
        ),
    )
    action: ToolCall = Field(
        description="The tool call to execute this step."
    )

    # ------------------------------------------------------------------
    # Robust JSON parsing (called by kbench on every raw LLM response)
    # ------------------------------------------------------------------
    @classmethod
    def model_validate_json(
        cls,
        json_data: "Union[str, bytes, bytearray]",
        **kwargs: Any,
    ) -> "AgentReActStep":
        """Parse the LLM's raw response with a json_repair fallback waterfall.

        kbench calls this method on every raw response string it receives from
        the LLM.  Overriding it here lets us handle the two most common
        failure modes transparently, without touching kbench or agent.py:

        Strategy 1 — strict Pydantic ``model_validate_json`` (fast path).
            Works for any model that outputs well-formed JSON.

        Strategy 2 — ``json_repair.loads()`` + ``model_validate``.
            Fixes the most common real-world failure: a long ``thought`` value
            that contains literal newlines (``\\n``) or other control chars
            (\\u0000-\\u001F) that are valid in JSON *values* only when escaped
            as ``\\\\n``.  json_repair re-escapes them automatically.
            Also handles single-quoted dicts, trailing commas, etc.

        Strategy 3 — regex JSON-block extraction + json_repair.
            When the LLM outputs plain prose (e.g. forgetting the required
            schema entirely), we scan for the outermost ``{...}`` block that
            contains an ``"action"`` key and try to parse that slice.
        """
        raw = json_data if isinstance(json_data, str) else json_data.decode()

        # Strategy 1: strict (fast path — most models most of the time)
        try:
            return super().model_validate_json(raw, **kwargs)
        except Exception:
            pass

        # Strategy 2: json_repair for malformed-but-close JSON
        if _json_repair_loads is not None:
            try:
                data = _json_repair_loads(raw)
                if isinstance(data, dict) and "action" in data:
                    return cls.model_validate(data)
            except Exception:
                pass

        # Strategy 3: extract JSON block from plain-text response
        try:
            # Find the first '{' that is followed (somewhere) by an "action" key.
            match = _re.search(r'\{', raw)
            if match:
                candidate = raw[match.start():]
                if _json_repair_loads is not None:
                    data = _json_repair_loads(candidate)
                else:
                    data = _json.loads(candidate)
                if isinstance(data, dict) and "action" in data:
                    return cls.model_validate(data)
        except Exception:
            pass

        # All strategies exhausted — re-raise via the original path so the
        # caller (kbench / _safe_prompt retry loop) sees a clean exception.
        return super().model_validate_json(raw, **kwargs)

    # ------------------------------------------------------------------
    # kbench UI rendering
    # ------------------------------------------------------------------
    def _repr_markdown_(self) -> str:
        """Pretty markdown shown in the Kaggle Benchmark UI chat feed.

        kbench's ``render_message_content`` checks for this method first.
        Returning structured markdown here avoids the raw JSON text dump
        that would otherwise appear when the LLM response message is
        rendered in the panel.
        """
        lines = []
        if self.thought:
            lines.append(f"💭 **Thought:** {self.thought}")
        lines.append(
            f"🎯 **Action:** `{self.action.name}` &nbsp;·&nbsp; `{self.action.arguments}`"
        )
        return "\n\n".join(lines)

    def get_payload(self) -> str:
        """Serialize back to JSON for the LLM's next turn.

        kbench's ``Message.payload`` property calls ``get_payload()`` if
        present, so the LLM always receives the original JSON rather than
        the markdown repr.
        """
        return self.model_dump_json()
