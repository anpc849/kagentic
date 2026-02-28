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
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, Field, field_validator


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
