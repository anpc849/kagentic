"""
kagents/examples/python_runner_example.py
-----------------------------------------
Demonstrates kagents's built-in PythonCodeRunnerTool.

The agent receives a question that requires computation and is given the
`python_interpreter` tool so it can write and execute Python snippets to
find the answer.

Run in a Kaggle Benchmarks notebook:
    %run kagents/examples/python_runner_example.py

Or directly:
    python kagents/examples/python_runner_example.py
"""

import kaggle_benchmarks as kbench
from kagents import CodeAgent, PythonCodeRunnerTool


# ---------------------------------------------------------------------------
# Helper – run the agent on a single question
# ---------------------------------------------------------------------------
def run_python_agent(llm, question: str) -> str:
    """
    Creates a CodeAgent equipped with PythonCodeRunnerTool and runs it on
    the given question. Returns the final answer string.
    """
    agent = CodeAgent(
        tools=[PythonCodeRunnerTool()],
        model=llm,
        max_steps=6,
        verbosity_level=2,
        additional_instructions=(
            "You can write and execute Python code to solve problems. "
            "Use print() inside your code snippets so the output is visible. "
            "Each code snippet runs in an isolated environment — do not rely "
            "on variables defined in previous snippets."
        ),
    )

    print(f"\nQuestion: {question}\n")

    answer = agent.run(question)

    print("\n" + "=" * 60)
    print("FINAL ANSWER:")
    print(answer)
    print("=" * 60)

    return answer


# ---------------------------------------------------------------------------
# @kbench.task entry point
# ---------------------------------------------------------------------------
@kbench.task(name="kagents_python_runner")
def python_runner_task(llm, question):
    """
    Kaggle Benchmarks task that uses the PythonCodeRunnerTool to answer a
    computational question.

    Assertions check that:
      1. The agent produced a non-empty answer.
      2. The answer contains the correct numeric result (4 926).
    """
    answer = run_python_agent(llm, question)

    kbench.assertions.assert_true(
        len(str(answer).strip()) > 0,
        expectation="Agent must return a non-empty answer.",
    )

    kbench.assertions.assert_true(
        "4926" in str(answer).replace(",", "").replace(" ", ""),
        expectation=(
            "Answer should contain 4926, the sum of all integers from 1 to 99 "
            "that are divisible by 3 or 5."
        ),
    )


# ---------------------------------------------------------------------------
# Direct entry point (local testing)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    python_runner_task.run(
        llm=kbench.llm,
        question=(
            "What is the sum of all integers from 1 to 99 (inclusive) "
            "that are divisible by 3 or by 5? "
            "Write Python code to compute the answer."
        ),
    )
