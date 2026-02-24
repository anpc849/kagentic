"""
kagents/tools/base.py
--------------------
Abstract base class for all kagents tools.

To create a custom tool, subclass Tool and:
  1. Set class attributes: name, description, inputs, output_type
  2. Implement forward(**kwargs) -> str

Example:
    class SearchTool(Tool):
        name = "web_search"
        description = "Searches the web for information."
        inputs = {
            "query": ToolInput(type="string", description="Search query")
        }
        output_type = "string"

        def forward(self, query: str) -> str:
            ...
"""
from __future__ import annotations

from typing import Dict

from kagents.types import ToolInput


class Tool:
    """
    Base class for all kagents tools.

    Class-level attributes (set these in subclasses):
        name        – unique snake_case identifier used by the LLM
        description – plain-English description shown in the system prompt
        inputs      – mapping of param_name → ToolInput describing each arg
        output_type – plain string describing the return type (e.g. "string")
    """

    name: str = ""
    description: str = ""
    inputs: Dict[str, ToolInput] = {}
    output_type: str = "string"

    def __init__(self, **kwargs):
        """Allow subclasses to accept constructor kwargs (e.g. vector_store)."""
        for key, value in kwargs.items():
            setattr(self, key, value)

    def forward(self, **kwargs) -> str:
        """
        Execute the tool.  Subclasses MUST override this method.
        All kwargs correspond to the keys declared in self.inputs.
        """
        raise NotImplementedError(
            f"Tool '{self.name}' must implement forward(**kwargs)."
        )

    def __call__(self, **kwargs) -> str:
        """Convenience: call the tool like a function."""
        return self.forward(**kwargs)

    def to_json_schema(self) -> dict:
        """
        Render this tool as an OpenAI-style function-call JSON schema.
        Used by build_system_prompt() to describe available tools to the LLM.
        """
        properties = {}
        required = []
        for param_name, tool_input in self.inputs.items():
            properties[param_name] = {
                "type": tool_input.type,
                "description": tool_input.description,
            }
            if tool_input.required:
                required.append(param_name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def __repr__(self) -> str:
        return f"Tool(name='{self.name}')"
