"""
kagents/tools/final_answer.py
-----------------------------
The FinalAnswerTool is the agent's exit signal.

When the LLM calls final_answer(answer="..."), the ReAct loop terminates
and the answer string is returned to the caller. This tool is automatically
added to every CodeAgent — the user never needs to register it manually.
"""
from kagents.tools.base import Tool
from kagents.types import ToolInput


class FinalAnswerTool(Tool):
    name = "final_answer"
    description = (
        "Use this tool when you have a complete, well-formed answer for the user. "
        "Calling this tool ends the agent loop and returns the answer immediately. "
        "The 'answer' argument should be a clear, complete response."
    )
    inputs = {
        "answer": ToolInput(
            type="string",
            description="The final answer to return to the user.",
            required=True,
        )
    }
    output_type = "string"

    def forward(self, answer: str) -> str:  # noqa: D401
        """Pass-through — the agent loop intercepts this before forward() runs."""
        return answer
