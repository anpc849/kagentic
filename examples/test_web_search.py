"""
kagents/test_web_search.py
--------------------------
Test the WebSearchTool with a real agent run inside a @kbench.task.

Run in Kaggle notebook:
    # Cell 1 – install deps
    !pip install -q requests beautifulsoup4

    # Cell 2 – run
    %run kagents/test_web_search.py
"""

import kaggle_benchmarks as kbench
from kagents import CodeAgent, WebBrowseTool, WebSearchTool


# ---------------------------------------------------------------------------
# Run function (usable standalone or inside a @kbench.task)
# ---------------------------------------------------------------------------
def run_test(llm, question):
    agent = CodeAgent(
        tools=[WebSearchTool(), WebBrowseTool()],
        model=llm,
        max_steps=5,
        verbosity_level=2,
        additional_instructions=(
            "Use web_search first to find relevant links, then web_browse to "
            "read the full content of a specific URL if needed. Keep answers concise."
        ),
    )

    print(f"Question: {question}\n")

    answer = agent.run(question)

    print("\n" + "=" * 60)
    print("FINAL ANSWER:")
    print(answer)
    print("=" * 60)

    return answer


# ---------------------------------------------------------------------------
# @kbench.task entry point
# ---------------------------------------------------------------------------
@kbench.task(name="kagents_web_search_test")
def test_web_search_agent(llm, question):
    answer = run_test(llm, question)
    kbench.assertions.assert_true(
        len(answer) > 10,
        expectation="Agent should return a non-empty answer about the Python version.",
    )


if __name__ == "__main__":
    test_web_search_agent.run(llm=kbench.llm, question="What is the latest stable version of Python as of today?")
