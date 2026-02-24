"""
kagents/__init__.py
------------------
Public API surface for the kagents package.

Quick start:

    from kagents import CodeAgent, Tool, Document

    class MyTool(Tool):
        name = "my_tool"
        description = "Does something useful."
        inputs = {"query": ToolInput(type="string", description="Input query")}
        output_type = "string"

        def forward(self, query: str) -> str:
            return f"Result for: {query}"

    agent = CodeAgent(tools=[MyTool()], model=kbench.llm, max_steps=5)
    answer = agent.run("Do something useful.")
"""

from kagents.agent import CodeAgent
from kagents.memory import AgentMemory
from kagents.prompts import build_system_prompt, build_task_prompt
from kagents.schema import AgentReActStep
from kagents.tools.base import Tool
from kagents.tools.final_answer import FinalAnswerTool
from kagents.tools.python_runner import PythonCodeRunnerTool
from kagents.tools.web_browse import WebBrowseTool
from kagents.tools.web_search import WebSearchTool
from kagents.types import Document, StepResult, ToolInput

__all__ = [
    # Agent
    "CodeAgent",
    # Types
    "Document",
    "ToolInput",
    "StepResult",
    # Schema
    "AgentReActStep",
    # Memory
    "AgentMemory",
    # Tools
    "Tool",
    "FinalAnswerTool",
    "PythonCodeRunnerTool",
    "WebBrowseTool",
    "WebSearchTool",
    # Prompts
    "build_system_prompt",
    "build_task_prompt",
]
