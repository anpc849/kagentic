"""
kagents/simple_test.py
---------------------
Simple smoke test using a mock WeatherTool.
Run inside a Kaggle Benchmarks notebook:

    import kaggle_benchmarks as kbench

    @kbench.task(name="kagents_weather_test")
    def test_weather_agent(llm):
        from kagents.simple_test import run_test
        run_test(llm)

    test_weather_agent.run(llm=kbench.llm)
"""

import kaggle_benchmarks as kbench
from kagents import CodeAgent, Tool, ToolInput


# ---------------------------------------------------------------------------
# Mock weather tool
# ---------------------------------------------------------------------------
MOCK_WEATHER_DB = {
    "hanoi":     {"condition": "Sunny", "temperature_c": 28},
    "tokyo":     {"condition": "Cloudy", "temperature_c": 14},
    "london":    {"condition": "Rainy",  "temperature_c": 10},
    "new york":  {"condition": "Windy",  "temperature_c": 5},
    "sydney":    {"condition": "Sunny",  "temperature_c": 31},
}


class WeatherTool(Tool):
    name = "get_weather"
    description = (
        "Returns the current weather for a given city. "
        "Use this to look up temperature and conditions."
    )
    inputs = {
        "city": ToolInput(
            type="string",
            description="Name of the city to get weather for (e.g. 'Tokyo', 'London').",
            required=True,
        )
    }
    output_type = "string"

    def forward(self, city: str) -> str:
        key = city.lower().strip()
        if key in MOCK_WEATHER_DB:
            data = MOCK_WEATHER_DB[key]
            return (
                f"Weather in {city.title()}: {data['condition']}, "
                f"{data['temperature_c']}°C"
            )
        return f"Sorry, no weather data available for '{city}'."


# ---------------------------------------------------------------------------
# Run function
# ---------------------------------------------------------------------------
def run_test(llm, question):
    """
    Creates a CodeAgent with WeatherTool and asks a weather question.
    """
    agent = CodeAgent(
        tools=[WeatherTool()],
        model=llm,
        max_steps=5,
        verbosity_level=2,
    )

    print(f"Question: {question}\n")

    answer = agent.run(question)

    print("\n" + "="*60)
    print("FINAL ANSWER:")
    print(answer)
    print("="*60)

    return answer


# ---------------------------------------------------------------------------
# Kaggle task entry point
# ---------------------------------------------------------------------------
@kbench.task(name="kagents_weather_test")
def test_weather_agent(llm, question):
    answer = run_test(llm, question)
    # Basic sanity: answer should mention both cities or temperature
    kbench.assertions.assert_true(
        len(answer) > 10,
        expectation="Agent should return a non-empty answer about the weather.",
    )


if __name__ == "__main__":
    test_weather_agent.run(llm=kbench.llm, question="What is the weather like in Tokyo and London? Which one is warmer?")
