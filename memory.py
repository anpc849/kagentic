"""
kagents/memory.py
----------------
AgentMemory is NOT responsible for tracking conversation history —
kbench.chats.new() does that automatically inside a single context.

AgentMemory's two jobs:
  1. Step logging  — keep a human-readable trace for debugging
  2. Summarization — when the context grows too large, compress the entire
     orchestrator chat into a short summary string and signal the agent to
     open a fresh kbench.chats.new() seeded with that summary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from kagents.schema import AgentReActStep
    from kagents.types import StepResult


@dataclass
class StepLog:
    step: int
    thought: Optional[str]
    tool_name: str
    tool_arguments: dict
    observation: str


class AgentMemory:
    """
    Lightweight step-level memory for a single agent run.

    Usage inside the ReAct loop:
        memory = AgentMemory(compress_threshold=10)
        ...
        memory.log_step(step_idx, agent_step, result)
        if memory.should_compress():
            summary = memory.summarize(model)
            # caller opens a new kbench.chats.new seeded with summary
    """

    def __init__(self, compress_threshold: int = 0):
        """
        Args:
            compress_threshold: After this many steps, should_compress() returns
                True. Set to 0 (default) to disable auto-compression.
        """
        self.compress_threshold = compress_threshold
        self._logs: List[StepLog] = []

    # ------------------------------------------------------------------
    # Step logging
    # ------------------------------------------------------------------
    def log_step(self, step: int, agent_step: "AgentReActStep", result: "StepResult") -> None:
        """Record one completed ReAct step."""
        self._logs.append(
            StepLog(
                step=step,
                thought=agent_step.thought,
                tool_name=agent_step.action.name,
                tool_arguments={"arguments": agent_step.action.arguments},
                observation=result.output,
            )
        )

    @property
    def step_count(self) -> int:
        return len(self._logs)

    @property
    def logs(self) -> List[StepLog]:
        return list(self._logs)

    # ------------------------------------------------------------------
    # Context compression
    # ------------------------------------------------------------------
    def should_compress(self) -> bool:
        """Return True when it's time to compress the orchestrator chat."""
        if self.compress_threshold <= 0:
            return False
        return self.step_count > 0 and self.step_count % self.compress_threshold == 0

    def build_summary_prompt(self) -> str:
        """
        Build a prompt asking the LLM to summarize the work done so far.
        The caller sends this inside the *existing* orchestrator chat before
        closing it so the summary captures full context.
        """
        log_text = "\n".join(
            f"Step {log.step}: [{log.tool_name}] → {log.observation[:300]}"
            for log in self._logs
        )
        return (
            "Please write a concise summary of all the work done so far in this session. "
            "Include: what tasks were completed, key findings, files written, and any "
            "important context the next session should know about.\n\n"
            f"Step log:\n{log_text}"
        )

    def format_summary_as_context(self, summary: str) -> str:
        """Wrap summary for use as the first user message in a new chat."""
        return (
            "=== CONTEXT FROM PREVIOUS SESSION ===\n"
            f"{summary}\n"
            "=== END CONTEXT ===\n\n"
            "Continue the task from where we left off."
        )

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"AgentMemory(steps={self.step_count}, compress_threshold={self.compress_threshold})"
