"""
kagentic/prompts.py
-----------------
Prompt templates for the kagentic ReAct loop.

The system prompt injects:
  - The ReAct THOUGHT → ACTION → OBSERVATION pattern
  - The exact AgentReActStep JSON schema the LLM must follow
  - Descriptions of every registered tool

Keeping this in one place makes it easy to tune for different LLMs.
"""
from __future__ import annotations

import json
from typing import Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from kagentic.tools.base import Tool


# ---------------------------------------------------------------------------
# System prompt skeleton
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert AI assistant that solves tasks step by step using a ReAct loop.

## How you must respond
Every response MUST be a valid JSON object matching this schema:
{{
  "thought": "<optional: your internal reasoning (not shown to user)>",
  "action": {{
    "name": "<name of the tool to call>",
    "arguments": "<JSON-encoded arguments string, e.g. {{\\"query\\": \\"hello\\"}}>" 
  }}
}}

## Rules
1. Think step-by-step in "thought" before choosing a tool.
2. Call exactly ONE tool per response.
3. When you have a complete answer, use action.name = "final_answer" and action.arguments = {{"answer": "<your answer>"}}.
4. NEVER output plain text outside of the JSON structure.
5. Use the tool results (provided as "Observation:") to decide your next step.
6. When a **Structured Output Schema** section is present below, the value of
   action.arguments.answer MUST be a JSON-encoded object matching that schema exactly.

## Available tools
{tool_descriptions}

## Important
- Only use one of the tool names listed above.
- action.arguments must be a valid JSON string.
- If a tool call fails, read the error carefully and try a corrected call.
"""

# Shown inside the tool descriptions block
_TOOL_BLOCK_TEMPLATE = """\
### {name}
{description}
Parameters:
{params}
"""

_PARAM_LINE = "  - {name} ({type}, {required_str}): {description}"

# ---------------------------------------------------------------------------
# Task prompt (first user message that starts the ReAct loop)
# ---------------------------------------------------------------------------
TASK_PROMPT_TEMPLATE = """\
Here is the task you must solve:

{task}

Begin by thinking about the task, then choose the right tool to start solving it.
Remember: respond ONLY with a JSON object matching the required schema.
"""


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------
def _format_tool(tool: "Tool") -> str:
    """Render a single tool into its prose description block."""
    param_lines = []
    for param_name, tool_input in tool.inputs.items():
        param_lines.append(
            _PARAM_LINE.format(
                name=param_name,
                type=tool_input.type,
                required_str="required" if tool_input.required else "optional",
                description=tool_input.description,
            )
        )
    return _TOOL_BLOCK_TEMPLATE.format(
        name=tool.name,
        description=tool.description,
        params="\n".join(param_lines) if param_lines else "  (no parameters)",
    )


def build_system_prompt(
    tools: List["Tool"],
    additional_instructions: str = "",
    response_format: Optional[Any] = None,
) -> str:
    """
    Build the full system prompt with all tool descriptions injected.

    Args:
        tools: All tools available to the agent (including FinalAnswerTool).
        additional_instructions: Extra instructions appended to the system prompt
            (e.g. domain-specific rules, output format requirements).
        response_format: Optional Pydantic BaseModel subclass. When provided,
            the model's JSON schema is embedded in the system prompt so the LLM
            knows to produce a matching JSON object inside ``final_answer.answer``.

    Returns:
        A complete system-instruction string ready to pass to kbench.chats.new().
    """
    tool_descriptions = "\n".join(_format_tool(t) for t in tools)
    prompt = _SYSTEM_PROMPT_TEMPLATE.format(tool_descriptions=tool_descriptions)
    if additional_instructions and additional_instructions.strip():
        prompt += f"\n## Additional Instructions\n{additional_instructions.strip()}\n"
    if response_format is not None:
        prompt += _build_response_format_section(response_format)
    return prompt


def _build_response_format_section(model_cls: Any) -> str:
    """
    Build a section that tells the LLM the exact JSON schema it must output
    inside ``final_answer.answer`` when ``response_format`` is set.
    """
    # Prefer Pydantic v2 model_json_schema(), fall back to v1 schema()
    if hasattr(model_cls, "model_json_schema"):
        schema = model_cls.model_json_schema()
    elif hasattr(model_cls, "schema"):
        schema = model_cls.schema()
    else:
        # Dataclass or plain class: best-effort via field introspection
        import dataclasses
        if dataclasses.is_dataclass(model_cls):
            schema = {
                "type": "object",
                "properties": {
                    f.name: {"type": "string", "description": f.name}
                    for f in dataclasses.fields(model_cls)
                },
            }
        else:
            schema = {"type": "object", "description": str(model_cls)}

    schema_str = json.dumps(schema, indent=2)
    return (
        f"\n## Structured Output Schema\n"
        f"Your final answer MUST be a JSON object that strictly follows this schema.\n"
        f"When calling `final_answer`, set `answer` to a JSON-encoded string of this object.\n\n"
        f"```json\n{schema_str}\n```\n"
        f"\nDo NOT include any text outside this JSON object in the `answer` value.\n"
    )


def build_task_prompt(task: str) -> str:
    """Wrap a user task string as the first turn message."""
    return TASK_PROMPT_TEMPLATE.format(task=task)
