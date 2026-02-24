"""
kagents/agent.py
---------------
CodeAgent: the main ReAct orchestrator.

Architecture
============
The agent opens a single kbench.chats.new("kagents_orchestrator") context.
kbench automatically tracks conversation history inside that context, so we
never need to pass a message list manually.

Loop per step:
  1. LLM produces a JSON-structured AgentReActStep (thought + tool_name + args)
  2. If tool_name == "final_answer" → return the answer, done.
  3. Otherwise: run the tool inside an isolated kbench.chats.new("tool_<name>")
     sub-chat (so the orchestrator never sees raw tool internals).
  4. Inject the tool result back into orchestrator history via kbench.user.send().
  5. Call llm.respond(schema=AgentReActStep) to get the next step.

Context compression (optional)
==============================
If compress_threshold > 0, every N steps AgentMemory.should_compress() fires:
  - We ask the LLM to summarize everything so far (still inside current chat).
  - We close the chat and reopen a fresh one seeded with the summary.
  - This keeps orchestrator tokens bounded for long-running agents.
"""
from __future__ import annotations

import json
try:
    from json_repair import loads as json_loads
except ImportError:
    import json
    json_loads = json.loads  # fallback if json_repair not installed

from typing import Any, Dict, List, Optional

import kaggle_benchmarks as kbench
from kaggle_benchmarks import actors as kbench_actors

from kagents.memory import AgentMemory
from kagents.prompts import build_system_prompt, build_task_prompt
from kagents.schema import AgentReActStep
from kagents.tools.base import Tool
from kagents.tools.final_answer import FinalAnswerTool
from kagents.types import StepResult


class CodeAgent:
    """
    A ReAct-style code agent that runs inside the Kaggle Benchmarks framework.

    Args:
        tools:             List of Tool instances to make available to the LLM.
                           FinalAnswerTool is added automatically if not present.
        model:             The kbench LLM object (e.g. kbench.llm, kbench.llms["vendor/model"]).
        max_steps:         Maximum number of ReAct iterations before giving up.
        verbosity_level:   0 = silent, 1 = step summaries, 2 = full thoughts.
        stream_outputs:    Not used by kbench LLMs (kept for API compatibility).
        compress_threshold: Compress orchestrator context every N steps (0 = off).

    Usage::

        agent = CodeAgent(tools=[my_tool], model=kbench.llm, max_steps=10)
        answer = agent.run("What is the capital of France?")
    """

    def __init__(
        self,
        tools: List[Tool],
        model: Any,
        max_steps: int = 10,
        verbosity_level: int = 1,
        stream_outputs: bool = False,
        compress_threshold: int = 0,
        additional_instructions: str = "",
    ):
        self.model = model
        self.max_steps = max_steps
        self.verbosity = verbosity_level
        self.stream_outputs = stream_outputs

        # Always include FinalAnswerTool
        tool_names = {t.name for t in tools}
        if "final_answer" not in tool_names:
            tools = list(tools) + [FinalAnswerTool()]
        self.tools = tools
        self._tool_map: Dict[str, Tool] = {t.name: t for t in tools}

        self.memory = AgentMemory(compress_threshold=compress_threshold)
        self.additional_instructions = additional_instructions

        # A dedicated actor for injecting tool observations back into chat history.
        # Initialized once here (not inside the loop) with role="assistant" so the
        # LLM sees it as a peer turn rather than a user prompt.
        self.tool_actor = kbench_actors.Actor(
            name="Tool",
            role="assistant",
            avatar="🔧",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, task: str) -> str:
        """
        Execute the ReAct loop for the given task and return the final answer.

        Args:
            task: Natural-language task description.

        Returns:
            The string answer produced by the final_answer tool, or an error message.
        """
        self._log(f"\n{'='*60}")
        self._log(f"🤖 kagents starting — model: {getattr(self.model, 'name', str(self.model))}")
        self._log(f"📋 Task: {task[:120]}{'...' if len(task) > 120 else ''}")
        self._log(f"{'='*60}\n")

        system_prompt = build_system_prompt(self.tools, self.additional_instructions)
        task_prompt = build_task_prompt(task)

        # Run the full loop, supporting context compression restarts
        remaining_steps = self.max_steps
        seed_context: Optional[str] = None  # set after compression

        while remaining_steps > 0:
            result = self._run_loop(
                system_prompt=system_prompt,
                task_prompt=task_prompt,
                max_steps=remaining_steps,
                seed_context=seed_context,
            )
            if result.is_final:
                self._log(f"\n✅ Final answer after {self.memory.step_count} steps.")
                return result.output

            # If we're here, the loop returned early due to compression
            if self.memory.should_compress():
                remaining_steps -= self.memory.step_count
                seed_context = result.output  # output carries the summary
                self._log(f"\n🔄 Context compressed. Continuing with {remaining_steps} steps left.")
            else:
                break  # max steps exhausted

        self._log(f"\n⚠️  Max steps ({self.max_steps}) reached without a final answer.")
        return "[kagents] Max steps reached without a final answer."

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------
    def _run_loop(
        self,
        system_prompt: str,
        task_prompt: str,
        max_steps: int,
        seed_context: Optional[str],
    ) -> StepResult:
        """
        Open one kbench.chats.new() context and run up to max_steps iterations.

        Returns a StepResult where:
          - is_final=True  → agent produced a final_answer
          - is_final=False → context was compressed; output contains the summary
        """
        with kbench.chats.new(name="kagents_orchestrator", system_instructions=system_prompt):

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

                # Check for compression trigger BEFORE executing the step
                if self.memory.should_compress():
                    summary_prompt = self.memory.build_summary_prompt()
                    # Ask the LLM to summarize (still inside current chat = full context)
                    kbench.user.send(summary_prompt)
                    summary = self.model.respond()   # plain text summary, no schema
                    return StepResult(
                        tool_name="compress",
                        output=str(summary),
                        is_final=False,
                    )

                # Execute the step (intercepts final_answer before calling tool)
                result = self._execute_step(step)
                self.memory.log_step(i, step, result)

                if result.is_final:
                    return result

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

        final_answer is intercepted here -- it never actually calls the tool's
        forward() method, we just extract the answer string directly.

        All other tools run inside their own isolated kbench.chats.new() sub-chat
        so the orchestrator never accumulates raw tool internals.
        """
        tool_name = step.action.name
        args = self._parse_args(step.action.arguments)

        if tool_name == "final_answer":
            answer = args.get("answer", step.action.arguments)  # fallback to raw string
            return StepResult(tool_name="final_answer", output=str(answer), is_final=True)

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
            # Isolated sub-chat: orchestrator only sees the return value, not internals
            with kbench.chats.new(name=f"kagents_tool_{tool_name}"):
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
    def _log(self, msg: str) -> None:
        if self.verbosity >= 1:
            print(msg)

    def _log_step(self, i: int, step: AgentReActStep) -> None:
        if self.verbosity >= 1:
            print(f"\n--- Step {i+1} ---")
            if self.verbosity >= 2 and step.thought:
                print(f"  💭 Thought: {step.thought}")
            print(f"  🎯 Action:  {step.action.name}({step.action.arguments})")
