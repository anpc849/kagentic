"""
kagentic/types.py
---------------
Core data types for the kagentic framework.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Document:
    """
    A document with text content and optional metadata.
    Matches the LangChain-style Document interface so existing
    loaders / splitters work with kagentic out of the box.
    """
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.page_content[:80].replace("\n", " ")
        return f"Document(content='{preview}...', metadata={self.metadata})"


@dataclass
class ToolInput:
    """
    Descriptor for a single tool parameter — shown to the LLM in the system prompt.
    """
    type: str                        # e.g. "string", "integer", "boolean"
    description: str
    required: bool = True


@dataclass
class StepResult:
    """
    Result of one ReAct loop iteration.

    Fields:
        tool_name:  Name of the tool that was called.
        output:     Raw string output from the tool.
        is_final:   True when tool_name == "final_answer".
        parsed:     Populated by FinalAnswerTool.parse_answer() when
                    response_format is set; None otherwise.
    """
    tool_name: str
    output: str
    is_final: bool
    parsed: Any = None
