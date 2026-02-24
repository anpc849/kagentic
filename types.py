"""
kagents/types.py
---------------
Core data types for the kagents framework.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Document:
    """
    A document with text content and optional metadata.
    Matches the LangChain-style Document interface so existing
    loaders / splitters work with kagents out of the box.
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
    """
    tool_name: str
    output: str
    is_final: bool                   # True when tool_name == "final_answer"
