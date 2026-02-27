"""
kagentic/memory.py
----------------
AgentMemory is NOT responsible for tracking conversation history —
kbench.chats.new() does that automatically inside a single context.

AgentMemory's two jobs:
  1. Step counting  — track how many steps have been executed so
     should_compress() can fire at the right time.
  2. Summarization — when the step count crosses the threshold, the agent
     sends a plain summary request to the LLM (the chat itself already
     carries full context, so no step log is needed) and opens a fresh
     kbench.chats.new() seeded with that summary.
"""
from __future__ import annotations


class AgentMemory:
    """
    Lightweight step counter + compression trigger for a single agent run.

    Usage inside the ReAct loop::

        memory = AgentMemory(compress_threshold=10)
        ...
        memory.increment()
        if memory.should_compress():
            # ask LLM to summarize (the chat already has full context)
            # caller opens a new kbench.chats.new seeded with the summary
    """

    SUMMARY_PROMPT = (
        "Please write a concise summary of all the work done so far in this session. "
        "Include: what tasks were completed, key findings, files written, and any "
        "important context the next session should know about."
    )

    def __init__(self, compress_threshold: int = 0):
        """
        Args:
            compress_threshold: After this many steps, should_compress() returns
                True. Set to 0 (default) to disable auto-compression.
        """
        self.compress_threshold = compress_threshold
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Step counting
    # ------------------------------------------------------------------
    def increment(self) -> None:
        """Record that one ReAct step completed."""
        self._step_count += 1

    @property
    def step_count(self) -> int:
        return self._step_count

    def reset(self) -> None:
        """Reset the step counter after a compression cycle.

        Must be called by the agent after it captures the summary and before
        opening the next kbench.chats.new() window.  Without this reset,
        should_compress() would fire again on the very first step of the new
        window (because the old count still satisfies the modulo check).
        """
        self._step_count = 0

    # ------------------------------------------------------------------
    # Context compression
    # ------------------------------------------------------------------
    def should_compress(self) -> bool:
        """Return True when it's time to compress the orchestrator chat."""
        if self.compress_threshold <= 0:
            return False
        return self._step_count > 0 and self._step_count % self.compress_threshold == 0

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