"""
kagents/prompts.py
-----------------
Prompt templates for the kagents ReAct loop.

The system prompt injects:
  - The ReAct THOUGHT → ACTION → OBSERVATION pattern
  - The exact AgentReActStep JSON schema the LLM must follow
  - Descriptions of every registered tool

Keeping this in one place makes it easy to tune for different LLMs.
"""
from __future__ import annotations

import json
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from kagents.tools.base import Tool


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


def build_system_prompt(tools: List["Tool"], additional_instructions: str = "") -> str:
    """
    Build the full system prompt with all tool descriptions injected.

    Args:
        tools: All tools available to the agent (including FinalAnswerTool).
        additional_instructions: Extra instructions appended to the system prompt
            (e.g. domain-specific rules, output format requirements).

    Returns:
        A complete system-instruction string ready to pass to kbench.chats.new().
    """
    tool_descriptions = "\n".join(_format_tool(t) for t in tools)
    prompt = _SYSTEM_PROMPT_TEMPLATE.format(tool_descriptions=tool_descriptions)
    if additional_instructions and additional_instructions.strip():
        prompt += f"\n## Additional Instructions\n{additional_instructions.strip()}\n"
    return prompt


def build_task_prompt(task: str) -> str:
    """Wrap a user task string as the first turn message."""
    return TASK_PROMPT_TEMPLATE.format(task=task)
