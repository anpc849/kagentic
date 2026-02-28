"""
kagentic/tools/final_answer.py
-----------------------------
The FinalAnswerTool is the agent's exit signal.

When the LLM calls final_answer(answer="..."), the ReAct loop terminates
and the answer is returned to the caller.  This tool is automatically added
to every CodeAgent — the user never needs to register it manually.

Structured-output support
==========================
When CodeAgent is constructed with ``response_format=SomeModel``, it passes
that class to ``FinalAnswerTool(response_format=SomeModel)``.  Two things
happen at construction time:

1. ``setup_for_response_format()`` rewrites the tool's *description* and
   the ``answer`` parameter description to include a compact schema hint.
   Because tool descriptions are re-emitted on every LLM turn, this keeps
   the schema in the model's active attention window even for long contexts.

2. ``parse_answer(raw)`` is called by ``_execute_step`` when the LLM fires
   ``final_answer``.  It runs a multi-strategy parse waterfall to tolerate
   common LLM formatting quirks (single-quote dicts, plain strings, etc.).
   On failure it raises ``ValueError`` — the caller converts this into an
   ``is_final=False`` ``StepResult`` so the existing observation-feedback
   loop retries automatically without any duplicated code.
"""
from __future__ import annotations

import ast
import json
from typing import Any, Optional

from kagentic.tools.base import Tool
from kagentic.types import ToolInput

try:
    from json_repair import loads as json_repair_loads
except ImportError:
    json_repair_loads = None  # type: ignore[assignment]


# Default descriptions (used when no response_format is set)
_DEFAULT_DESCRIPTION = (
    "Use this tool when you have a complete, well-formed answer for the user. "
    "Calling this tool ends the agent loop and returns the answer immediately. "
    "The 'answer' argument should be a clear, complete response."
)
_DEFAULT_ANSWER_DESCRIPTION = "The final answer to return to the user."


class FinalAnswerTool(Tool):
    name = "final_answer"
    description = _DEFAULT_DESCRIPTION
    inputs = {
        "answer": ToolInput(
            type="string",
            description=_DEFAULT_ANSWER_DESCRIPTION,
            required=True,
        )
    }
    output_type = "string"

    def __init__(self, response_format: Optional[Any] = None) -> None:
        self.response_format = response_format
        if response_format is not None:
            self.setup_for_response_format(response_format)

    def forward(self, answer: str) -> str:  # noqa: D401
        """Pass-through — the agent loop intercepts this before forward() runs."""
        return answer

    # ------------------------------------------------------------------
    # Structured-output helpers
    # ------------------------------------------------------------------

    def setup_for_response_format(self, model_cls: Any) -> None:
        """
        Patch this tool's description and ``answer`` parameter description to
        embed a compact schema hint for ``model_cls``.

        Called once at agent construction time.  Because tool descriptions are
        re-serialised into the system prompt on *every* LLM call, the schema
        reminder stays visible even when the top of the system prompt has
        scrolled out of the model's attention window.
        """
        schema_hint = self._build_schema_hint(model_cls)

        self.description = (
            f"Use this tool when you have a complete answer for the user. "
            f"⚠️  STRUCTURED OUTPUT REQUIRED: spread ALL {model_cls.__name__} fields "
            f"directly into action.arguments as a plain JSON object: {schema_hint}. "
            f"Do NOT wrap them in a string or nest under an 'answer' key."
        )
        self.inputs["answer"].description = (
            f"Pass ALL {model_cls.__name__} fields as direct JSON object keys. "
            f"Required fields: {schema_hint}. "
            f"Example: spread fields directly — "
            f"do NOT use a nested 'answer' string key."
        )


    def parse_answer(self, raw: str) -> Any:
        """
        Parse ``raw`` (the LLM's ``answer`` string) into ``self.response_format``.

        If ``response_format`` is None, returns ``raw`` unchanged.

        Parse waterfall (stops at first success):
          1. Pydantic v2 ``model_validate_json``  — strict JSON
          2. ``json.loads`` + ``model_validate``  — strict JSON, dict path
          3. ``ast.literal_eval`` + ``model_validate`` — Python single-quote dicts
          4. ``json_repair`` + ``model_validate`` — last resort

        Raises:
            ValueError: if all strategies fail — caller should convert this
                        into an is_final=False StepResult to trigger a retry.
        """
        if self.response_format is None:
            return raw

        model_cls = self.response_format
        last_exc: Exception = RuntimeError("No parse strategy attempted.")

        def _validate(data: dict) -> Any:
            if hasattr(model_cls, "model_validate"):
                return model_cls.model_validate(data)
            if hasattr(model_cls, "parse_obj"):
                return model_cls.parse_obj(data)
            return model_cls(**data)

        # Strategy 1: Pydantic v2 model_validate_json (strict JSON)
        if hasattr(model_cls, "model_validate_json"):
            try:
                return model_cls.model_validate_json(raw)
            except Exception as exc:
                last_exc = exc

        # Strategy 2: json.loads → model_validate
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return _validate(data)
        except Exception as exc:
            last_exc = exc

        # Strategy 3: ast.literal_eval for Python-style single-quote dicts
        # e.g.  "{'city': 'Tokyo', 'temperature_c': 14}"
        try:
            data = ast.literal_eval(raw)
            if isinstance(data, dict):
                return _validate(data)
        except Exception as exc:
            last_exc = exc

        # Strategy 4: json_repair as last resort
        if json_repair_loads is not None:
            try:
                data = json_repair_loads(raw)
                if isinstance(data, dict):
                    return _validate(data)
            except Exception as exc:
                last_exc = exc

        raise ValueError(
            f"[kagentic] response_format={model_cls.__name__!r}: "
            f"all parse strategies failed.\n"
            f"Raw answer: {raw!r}\n"
            f"Last error: {last_exc}"
        ) from last_exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_schema_hint(model_cls: Any) -> str:
        """Return a compact one-line schema description, e.g. ``{city: string, ...}``."""
        try:
            if hasattr(model_cls, "model_json_schema"):
                schema = model_cls.model_json_schema()
            elif hasattr(model_cls, "schema"):
                schema = model_cls.schema()
            else:
                return model_cls.__name__

            props = schema.get("properties", {})
            summary = ", ".join(
                f"{k}: {v.get('type', '?')}" for k, v in props.items()
            )
            return f"{{{summary}}}" if summary else model_cls.__name__
        except Exception:
            return model_cls.__name__
