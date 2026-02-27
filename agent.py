"""
kagentic/agent.py
---------------
CodeAgent: the main ReAct orchestrator.

Architecture
============
The agent opens a single kbench.chats.new("kagentic_orchestrator") context.
kbench automatically tracks conversation history inside that context, so we
never need to pass a message list manually.

Loop per step:
  1. LLM produces a JSON-structured AgentReActStep (thought + tool_name + args)
  2. If tool_name == "final_answer" → parse answer (with retry on format failure).
  3. Otherwise: run the tool inside an isolated kbench.chats.new("tool_<name>")
     sub-chat (so the orchestrator never sees raw tool internals).
  4. Inject the tool result back into orchestrator history via tool_actor.send().
  5. Call llm.respond(schema=AgentReActStep) to get the next step.

Context compression (optional)
==============================
If compress_threshold > 0, every N steps AgentMemory.should_compress() fires:
  - We ask the LLM to summarize everything so far (still inside current chat).
  - We close the chat and reopen a fresh one seeded with the summary.
  - This keeps orchestrator tokens bounded for long-running agents.

Structured output (response_format)
====================================
All structured-output logic lives in FinalAnswerTool (tools/final_answer.py):
  - Schema hints injected into the tool description on every LLM turn.
  - Multi-strategy parse waterfall (JSON / Python dict / json_repair).
  - On parse failure, _execute_step returns is_final=False so the existing
    observation-feedback loop retries — no duplicated code.
"""
from __future__ import annotations

import json
try:
    from json_repair import loads as json_loads
except ImportError:
    import json
    json_loads = json.loads  # fallback if json_repair not installed

import json as _json
from typing import Any, Dict, List, Optional, Type, Union

try:
    from pydantic import BaseModel
except ImportError:
    BaseModel = None  # type: ignore[assignment,misc]

import kaggle_benchmarks as kbench
from kaggle_benchmarks import actors as kbench_actors

from kagentic.memory import AgentMemory
from kagentic.prompts import build_system_prompt, build_task_prompt
from kagentic.schema import AgentReActStep
from kagentic.tools.base import Tool
from kagentic.tools.final_answer import FinalAnswerTool
from kagentic.types import StepResult

# Imported lazily inside __init__ to avoid circular imports
# (AgentTool → agent.py → AgentTool)


class CodeAgent:
    """
    A ReAct-style code agent that runs inside the Kaggle Benchmarks framework.

    Args:
        tools:             List of Tool instances to make available to the LLM.
                           FinalAnswerTool is added automatically if not present.
        model:             The kbench LLM object (e.g. kbench.llm, kbench.llms["vendor/model"]).
        name:              Short snake_case identifier for this agent.  Used as the
                           tool name when this agent is registered as a worker inside
                           a Manager ``CodeAgent``.
        description:       Plain-English description of what this agent does.  Shown
                           to the Manager LLM as the tool description.
        managed_agents:    List of ``CodeAgent`` instances to manage as workers.
                           Each is automatically wrapped in an ``AgentTool`` and
                           added to this agent's tool list, so the Manager LLM can
                           delegate sub-tasks to them by name.
        max_steps:         Maximum number of ReAct iterations before giving up.
        verbosity_level:   0 = silent, 1 = step summaries, 2 = full thoughts.
        stream_outputs:    Not used by kbench LLMs (kept for API compatibility).
        compress_threshold: Compress orchestrator context every N steps (0 = off).
        additional_instructions: Extra instructions appended to the system prompt.
        response_format:   Optional Pydantic BaseModel subclass. When set, the
                           agent instructs the LLM to output a JSON object matching
                           the model's schema inside ``final_answer.answer``, then
                           automatically parses and validates that JSON before
                           returning the typed model instance from ``run()``.
                           If ``None`` (default), ``run()`` returns a plain ``str``.

    Usage (plain string)::

        agent = CodeAgent(tools=[my_tool], model=kbench.llm, max_steps=10)
        answer = agent.run("What is the capital of France?")  # str

    Usage (structured output)::

        class ContactInfo(BaseModel):
            name: str
            email: str

        agent = CodeAgent(tools=[...], model=kbench.llm, response_format=ContactInfo)
        contact = agent.run("Find contact info for Alice.")  # ContactInfo instance

    Usage (manager with workers)::

        search_agent = CodeAgent(
            name="search_agent",
            description="Searches the web for information.",
            tools=[WebSearchTool()],
            model=kbench.llm,
        )
        manager = CodeAgent(
            tools=[],
            model=kbench.llm,
            managed_agents=[search_agent],
        )
        result = manager.run("Research the latest advances in RAG.")
    """

    def __init__(
        self,
        tools: List[Tool],
        model: Any,
        name: str = "agent",
        description: str = "",
        managed_agents: Optional[List["CodeAgent"]] = None,
        max_steps: int = 10,
        verbosity_level: int = 1,
        stream_outputs: bool = False,
        compress_threshold: int = 0,
        additional_instructions: str = "",
        response_format: Optional[Any] = None,
        return_full_result: bool = False,
    ):
        self.name = name
        self.description = description
        self.model = model
        self.max_steps = max_steps
        self.verbosity = verbosity_level
        self.stream_outputs = stream_outputs
        self.response_format = response_format
        self.additional_instructions = additional_instructions
        self.return_full_result = return_full_result

        # Wrap managed worker agents as AgentTools and merge into the tool list.
        # Import here to avoid a circular import (agent_tool imports agent).
        from kagentic.tools.agent_tool import AgentTool

        agent_tools: List[Tool] = [
            AgentTool(worker) for worker in (managed_agents or [])
        ]
        # Caller's explicit tools come first; worker AgentTools appended after.
        tools = list(tools) + agent_tools

        # Always include FinalAnswerTool — pass response_format so it
        # self-configures (patches descriptions + owns parse_answer logic).
        tool_names = {t.name for t in tools}
        if "final_answer" not in tool_names:
            tools = list(tools) + [FinalAnswerTool(response_format=response_format)]
        self.tools = tools
        self._tool_map: Dict[str, Tool] = {t.name: t for t in tools}

        self.memory = AgentMemory(compress_threshold=compress_threshold)

        # A dedicated actor for injecting tool observations back into chat history.
        # Initialized once here (not inside the loop) with role="assistant" so the
        # LLM sees it as a peer turn rather than a user prompt.
        self.tool_actor = kbench_actors.Actor(
            name="Tool",
            role="assistant",
            avatar="🔧",
        )

        # Populated by _inner_loop at every step. Always initialized here so
        # workers called via AgentTool (which bypasses run()) have it ready.
        self._step_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, task: str) -> Any:
        """
        Execute the ReAct loop for the given task and return the final answer.

        Args:
            task: Natural-language task description.

        Returns:
            - If ``response_format`` is None: a ``str`` answer.
            - If ``response_format`` is a Pydantic model: a validated instance of
              that model, parsed from the JSON that the LLM placed in
              ``final_answer.answer``.

        Raises:
            ValueError: If ``response_format`` is set but the LLM's answer could
                not be parsed/validated into the requested model.
        """
        self._log(f"\n{'='*60}")
        self._log(f"🤖 kagentic starting — model: {getattr(self.model, 'name', str(self.model))}")
        self._log(f"📋 Task: {task[:120]}{'...' if len(task) > 120 else ''}")
        if self.response_format is not None:
            self._log(f"📐 response_format: {self.response_format.__name__}")
        self._log(f"{'='*60}\n")

        system_prompt = build_system_prompt(
            self.tools,
            self.additional_instructions,
            response_format=self.response_format,
        )
        task_prompt = build_task_prompt(task)

        # Reset step history at the start of each run() so repeated calls
        # on the same agent instance don't accumulate across tasks.
        self._step_history = []

        # Run the full loop, supporting context compression restarts
        remaining_steps = self.max_steps
        total_steps = 0          # cumulative steps across all compression windows
        chat_index = 1           # incremented on each new kbench.chats.new() window
        seed_context: Optional[str] = None  # set after compression

        while remaining_steps > 0:
            result = self._run_loop(
                system_prompt=system_prompt,
                task_prompt=task_prompt,
                max_steps=remaining_steps,
                seed_context=seed_context,
                chat_index=chat_index,
            )
            if result.is_final:
                total_steps += self.memory.step_count
                self._log(f"\n✅ Final answer after {total_steps} steps.")
                if self.return_full_result:
                    return self._step_history
                return result.parsed      # str or Pydantic instance

            # Loop returned early due to compression
            if not result.is_final and result.tool_name == "compress":
                steps_consumed = self.memory.step_count  # record BEFORE reset
                total_steps += steps_consumed
                self.memory.reset()          # reset counter for the fresh window
                remaining_steps -= steps_consumed
                chat_index += 1              # new indexed chat name next iteration
                seed_context = result.output
                self._log(f"\n🔄 Context compressed. Continuing with {remaining_steps} steps left.")
            else:
                break  # max steps exhausted

        self._log(f"\n⚠️  Max steps ({self.max_steps}) reached without a final answer.")
        return "[kagentic] Max steps reached without a final answer."


    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------
    def _run_loop(
        self,
        system_prompt: str,
        task_prompt: str,
        max_steps: int,
        seed_context: Optional[str],
        chat_index: int = 1,
        use_existing_context: bool = False,
    ) -> StepResult:
        """
        Open one kbench.chats.new() context and run up to max_steps iterations.

        Args:
            use_existing_context: When ``True`` (set by ``AgentTool``), the caller
                has already established the correct chat context via
                ``contexts.enter(chat=worker_chat)``.  We skip opening a new
                ``kbench.chats.new()`` and instead send the first message directly
                into the active context.  When ``False`` (default, normal agent
                usage) we open a fresh named chat as usual.

        Returns a StepResult where:
          - is_final=True  → agent produced a final_answer
          - is_final=False → context was compressed; output contains the summary
        """
        if use_existing_context:
            # Called by AgentTool — we're already inside contexts.enter(worker_chat).
            # The system prompt was already sent on first init; just run the loop.
            return self._inner_loop(
                task_prompt=task_prompt,
                max_steps=max_steps,
                seed_context=seed_context,
            )
        else:
            chat_name = f"kagentic_orchestrator_{chat_index}"
            with kbench.chats.new(name=chat_name, system_instructions=system_prompt):
                return self._inner_loop(
                    task_prompt=task_prompt,
                    max_steps=max_steps,
                    seed_context=seed_context,
                )

    def _inner_loop(
        self,
        task_prompt: str,
        max_steps: int,
        seed_context: Optional[str],
    ) -> StepResult:
        """
        The actual ReAct step loop. Expects the correct chat context to already
        be active (either from kbench.chats.new or contexts.enter).

        Always accumulates every StepResult into ``result.steps`` on the final
        returned StepResult, so ``run()`` can surface it when
        ``return_full_result=True``.
        """
        # First user message: either the task or a compressed-context continuation
        first_message = (
            self.memory.format_summary_as_context(seed_context) + "\n\n" + task_prompt
            if seed_context
            else task_prompt
        )

        # Turn 1: prompt() sends the message AND waits for the LLM response
        step = self._safe_prompt(first_message)
        if step is None:
            return StepResult(tool_name="error", output="LLM failed to respond.", is_final=True)

        for i in range(max_steps):
            self._log_step(i, step)

            # Execute the step (intercepts final_answer before calling tool)
            result = self._execute_step(step)
            self.memory.increment()  # kbench chat already tracks full context

            # Record this step as a plain dict in the shared run history.
            self._step_history.append({
                "step_idx":  len(self._step_history),
                "thought":   step.thought,
                "tool_name": result.tool_name,
                "output":    result.output,
                "is_final":  result.is_final,
                "parsed":    result.parsed,
            })

            if result.is_final:
                return result

            # Check compression AFTER incrementing so count is current.
            # Fires every compress_threshold steps exactly.
            if self.memory.should_compress():
                # The chat already holds full context — just ask for a summary.
                kbench.user.send(AgentMemory.SUMMARY_PROMPT)
                summary = self.model.respond()   # plain text summary, no schema
                return StepResult(
                    tool_name="compress",
                    output=str(summary),
                    is_final=False,
                )

            # Feed observation back into orchestrator history via tool_actor
            observation_msg = f"Observation from '{result.tool_name}':\n{result.output}"
            self.tool_actor.send(observation_msg)

            # Get the next LLM step (responds to the full accumulated history)
            step = self._safe_respond()
            if step is None:
                return StepResult(tool_name="error", output="LLM failed mid-loop.", is_final=True)

        return StepResult(tool_name="max_steps", output="", is_final=False)

    # ------------------------------------------------------------------
    # Step execution
    # ------------------------------------------------------------------
    def _parse_args(self, raw: str) -> dict:
        """Parse tool_arguments JSON string using json_repair for robustness."""
        try:
            result = json_loads(raw)
            if not isinstance(result, dict):
                return {}
            return result
        except Exception:
            return {}

    def _execute_step(self, step: AgentReActStep) -> StepResult:
        """
        Execute one ReAct step.

        final_answer handling
        ---------------------
        We delegate to FinalAnswerTool.parse_answer() which owns all
        structured-output logic.  On a parse failure it raises ValueError;
        we convert that into is_final=False with a correction message so
        the *existing* tool_actor observation loop retries automatically —
        no duplicated feedback code needed.

        All other tools run inside their own isolated kbench.chats.new()
        sub-chat so the orchestrator never accumulates raw tool internals.
        """
        tool_name = step.action.name
        args = self._parse_args(step.action.arguments)

        if tool_name == "final_answer":
            raw_answer = args.get("answer", step.action.arguments)
            final_tool = self._tool_map["final_answer"]
            try:
                parsed = final_tool.parse_answer(str(raw_answer))
                return StepResult(
                    tool_name="final_answer",
                    output=str(raw_answer),
                    is_final=True,
                    parsed=parsed,
                )
            except ValueError as parse_err:
                # Parse failed → send correction back as an observation so the
                # LLM retries.  Re-uses the existing tool_actor feedback path.
                
                # Fetch the compact schema hint we generated earlier
                schema_hint = final_tool._build_schema_hint(self.response_format)
                
                correction = (
                    f"Your final_answer was rejected because the 'answer' value "
                    f"did not match the required JSON schema.\n"
                    f"Error: {parse_err}\n"
                    f"Please call final_answer again with a properly JSON-encoded "
                    f"object matching this schema: {schema_hint}\n"
                    f"Example: {{\"answer\": \"{schema_hint}\"}}"
                )
                self._log(f"  ⚠️  response_format parse failed — requesting retry.")
                return StepResult(
                    tool_name="final_answer",
                    output=correction,
                    is_final=False,
                )

        tool = self._tool_map.get(tool_name)
        if tool is None:
            error_msg = (
                f"Unknown tool '{tool_name}'. "
                f"Available tools: {list(self._tool_map.keys())}."
            )
            self._log(f"  ⚠️  {error_msg}")
            return StepResult(tool_name=tool_name, output=error_msg, is_final=False)

        try:
            self._log(f"  🔧 Calling tool: {tool_name}({step.action.arguments})")

            # AgentTool (worker agent) manages its own persistent chat context
            # via contexts.enter(chat=worker_chat) inside forward() — no outer
            # wrapper needed. This avoids the double-nesting:
            #   kagentic_tool_search_agent
            #     └─ kagentic_worker_search_agent   ← unwanted
            # and gives a clean single chat:
            #   kagentic_worker_search_agent
            from kagentic.tools.agent_tool import AgentTool
            if isinstance(tool, AgentTool):
                output = tool.forward(**args)
            else:
                # Regular tool: run in an isolated throwaway sub-chat so the
                # orchestrator never accumulates raw tool internals.
                with kbench.chats.new(name=f"kagentic_tool_{tool_name}"):
                    output = tool.forward(**args)

            output_str = str(output)
            self._log(f"  📤 Tool result: {output_str[:200]}{'...' if len(output_str) > 200 else ''}")
            return StepResult(tool_name=tool_name, output=output_str, is_final=False)

        except Exception as e:
            error_msg = f"Tool '{tool_name}' raised {type(e).__name__}: {e}"
            self._log(f"  ❌ {error_msg}")
            return StepResult(tool_name=tool_name, output=error_msg, is_final=False)

    # ------------------------------------------------------------------
    # LLM call wrappers (with retry)
    # ------------------------------------------------------------------
    def _safe_prompt(self, message: str, retries: int = 3) -> Optional[AgentReActStep]:
        """llm.prompt() the first message; retry on failure.
        llm.prompt() returns T directly (already calls .content internally).
        """
        for attempt in range(retries):
            try:
                result = self.model.prompt(message, schema=AgentReActStep)
                # llm.prompt() returns T, but guard defensively
                if isinstance(result, AgentReActStep):
                    return result
                if hasattr(result, "content"):
                    return result.content
                return result
            except Exception as e:
                self._log(f"  ⚠️  llm.prompt() attempt {attempt+1}/{retries} failed: {e}")
        return None

    def _safe_respond(self, retries: int = 3) -> Optional[AgentReActStep]:
        """llm.respond() for subsequent turns; retry on failure.
        IMPORTANT: llm.respond() returns Message[T], not T directly.
        We must extract .content to get the parsed AgentReActStep.
        (Same issue as complemon/main.py: 'response if isinstance(response, AgentResponse) else response.content')
        """
        for attempt in range(retries):
            try:
                result = self.model.respond(schema=AgentReActStep)
                # llm.respond() returns Message[T] — extract the parsed content
                if isinstance(result, AgentReActStep):
                    return result
                if hasattr(result, "content"):
                    return result.content
                return result
            except Exception as e:
                self._log(f"  ⚠️  llm.respond() attempt {attempt+1}/{retries} failed: {e}")
        return None

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    @property
    def _tag(self) -> str:
        """Short bracketed prefix identifying this agent in log output."""
        return f"[{self.name}]"

    def _log(self, msg: str) -> None:
        if self.verbosity >= 1:
            print(f"{self._tag} {msg}")

    def _log_step(self, i: int, step: AgentReActStep) -> None:
        if self.verbosity >= 1:
            print(f"\n{self._tag} --- Step {i+1} ---")
            if self.verbosity >= 2 and step.thought:
                print(f"{self._tag}   💭 Thought: {step.thought}")
            print(f"{self._tag}   🎯 Action:  {step.action.name}({step.action.arguments})")