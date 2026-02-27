"""
kagentic/tools/agent_tool.py
---------------------------
AgentTool: wraps a CodeAgent as a Tool so a Manager CodeAgent can delegate
sub-tasks to Worker agents via the standard tool-calling interface.

Persistent-chat design
======================
Each AgentTool owns a single ``chats.Chat`` object (``self._worker_chat``)
that is created once and lives for the lifetime of the tool.

Every call to ``forward(task)`` re-enters that *same* Chat via
``contexts.enter(chat=self._worker_chat)``.  Because the Chat object is
mutable and accumulates messages, the worker retains full conversation
history across multiple Manager->Worker round-trips.

This means:
  - Call 1: Manager assigns initial task -> worker reasons, returns result
  - Call 2: Manager asks to refine -> worker sees its own previous reasoning
    and the original task, then refines accordingly

Isolation
=========
The Manager's ``_execute_step`` wraps every ``tool.forward()`` in a
``kbench.chats.new()`` sub-chat.  For ``AgentTool``, the inner
``contexts.enter(chat=self._worker_chat)`` immediately overrides that
throwaway chat, so the Manager's orchestrator chat stays clean and the
worker's persistent chat is unaffected.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from kaggle_benchmarks import actors as kbench_actors
from kaggle_benchmarks import chats, contexts

from kagentic.prompts import build_system_prompt, build_task_prompt
from kagentic.tools.base import Tool
from kagentic.types import ToolInput

if TYPE_CHECKING:
    from kagentic.agent import CodeAgent


class AgentTool(Tool):
    """
    Wraps a ``CodeAgent`` as a callable ``Tool`` for use in a Manager agent.

    The wrapped agent (worker) maintains a **persistent chat history** across
    all calls, so the Manager can refine, follow-up, or send additional tasks
    without the worker losing context.

    Args:
        agent:       The ``CodeAgent`` instance to wrap.
        name:        Tool name shown to the Manager LLM.
                     Defaults to ``agent.name``.
        description: Tool description shown to the Manager LLM.
                     Defaults to ``agent.description`` (or a sensible fallback).

    Usage::

        search_worker = CodeAgent(
            name="search_agent",
            description="Searches the web and summarises findings.",
            tools=[WebSearchTool()],
            model=kbench.llm,
        )

        manager = CodeAgent(
            tools=[],
            model=kbench.llm,
            managed_agents=[search_worker],
        )

        result = manager.run("Research the top 3 papers on RAG.")
    """

    output_type = "string"

    def __init__(
        self,
        agent: "CodeAgent",
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        self._agent = agent

        # Tool identity — shown to the Manager LLM in the system prompt
        self.name = name or agent.name or "worker_agent"
        _desc = description or agent.description or (
            f"A specialised worker agent named '{self.name}'. "
            "Delegate sub-tasks to it by providing a clear task description."
        )
        self.description = (
            f"{_desc}\n"
            "When you want to refine or follow-up, call this tool again with "
            "the updated instruction — the agent remembers all prior context."
        )
        self.inputs = {
            "task": ToolInput(
                type="string",
                description=(
                    f"The task or instruction to send to '{self.name}'. "
                    "Be specific and self-contained."
                ),
                required=True,
            )
        }

        # ------------------------------------------------------------------ #
        # Persistent chat: created ONCE, re-entered on every forward() call. #
        # The Chat object accumulates messages across all Manager->Worker     #
        # round-trips via contexts.enter(chat=self._worker_chat).             #
        # ------------------------------------------------------------------ #
        self._worker_chat: chats.Chat = chats.Chat(
            name=f"kagentic_worker_{self.name}"
        )
        self._initialized: bool = False

        # Pre-build the worker's system prompt once (tools don't change)
        self._system_prompt: str = build_system_prompt(
            agent.tools,
            additional_instructions=agent.additional_instructions,
            response_format=agent.response_format,
        )

    # ---------------------------------------------------------------------- #
    # Tool interface                                                           #
    # ---------------------------------------------------------------------- #
    def forward(self, task: str) -> str:
        """
        Send ``task`` to the worker agent and return its final answer string.

        On the first call the worker chat is seeded with the system prompt and
        appended to the current active chat so the Kaggle Benchmark UI renders
        it as a collapsible sub-thread (mirroring what ``chats.new()`` does).
        Subsequent calls re-enter the same Chat, preserving full history so
        the worker remembers all previous task/result exchanges.
        """
        # ── UI visibility ──────────────────────────────────────────────────
        # `chats.new()` makes a sub-chat visible in the UI by calling
        #   ctx.parent.chat.append(ctx.chat)
        # which fires the `new_message` event → PanelUI.new_message renders
        # the Chat as a ChatStep inside the parent feed.
        #
        # Our `contexts.enter(chat=worker_chat)` skips that step, so we do
        # it manually on the FIRST call only (subsequent calls re-enter the
        # same chat object that is already in the hierarchy).
        if not self._initialized:
            current_parent = chats.get_current_chat()
            current_parent.append(self._worker_chat)

        with contexts.enter(chat=self._worker_chat):
            if not self._initialized:
                # Seed the worker chat with system instructions on first call.
                kbench_actors.system.send(self._system_prompt)
                self._initialized = True

            # Send this task as the next user message, then run the worker's
            # ReAct loop.  _inner_loop() operates on the currently-active chat
            # (self._worker_chat) — no new kbench.chats.new() is opened.
            task_prompt = build_task_prompt(task)
            result = self._agent._inner_loop(
                task_prompt=task_prompt,
                max_steps=self._agent.max_steps,
                seed_context=None,
            )

        if result.is_final:
            return str(result.parsed if result.parsed is not None else result.output)
        return f"[{self.name}] Reached max steps without a final answer."

    def __repr__(self) -> str:
        return f"AgentTool(name='{self.name}', agent={self._agent!r})"
