"""
kagentic/__init__.py
------------------
Public API surface for the kagentic package.

Quick start:

    from kagentic import CodeAgent, Tool, Document

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

from kagentic.agent import CodeAgent
from kagentic.memory import AgentMemory
from kagentic.prompts import build_system_prompt, build_task_prompt
from kagentic.schema import AgentReActStep
from kagentic.tools.agent_tool import AgentTool
from kagentic.tools.base import Tool
from kagentic.tools.final_answer import FinalAnswerTool
from kagentic.tools.python_runner import PythonCodeRunnerTool
from kagentic.tools.web_browse import WebBrowseTool
from kagentic.tools.web_search import WebSearchTool
from kagentic.types import Document, StepResult, ToolInput

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
    "AgentTool",
    "Tool",
    "FinalAnswerTool",
    "PythonCodeRunnerTool",
    "WebBrowseTool",
    "WebSearchTool",
    # Prompts
    "build_system_prompt",
    "build_task_prompt",
]

