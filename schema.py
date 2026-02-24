"""
kagents/schema.py
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
from pydantic import BaseModel, Field
from typing import Optional


class ToolCall(BaseModel):
    """A single tool invocation -- name + JSON-serialized argument string."""
    name: str = Field(
        description=(
            "Name of the tool to call. "
            "Use 'final_answer' when you have a complete answer for the user."
        )
    )
    arguments: str = Field(
        default="{}",
        description=(
            "Arguments to pass to the tool, as a JSON-encoded string. "
            "Example: {\"query\": \"hello world\"}. "
            "For 'final_answer', use: {\"answer\": \"<your complete answer>\"}."
        ),
    )


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
