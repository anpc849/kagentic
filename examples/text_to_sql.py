"""
kagents/text_to_sql.py
---------------------
Text-to-SQL example using kagents (converted from smolagents).

Install:
    !pip install -q sqlalchemy

Run in Kaggle notebook:
    %run kagents/text_to_sql.py
"""

from sqlalchemy import (
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    insert,
    inspect,
    text,
)

import kaggle_benchmarks as kbench
from kagents import CodeAgent, Tool, ToolInput

# ---------------------------------------------------------------------------
# Setup in-memory SQLite DB
# ---------------------------------------------------------------------------
engine = create_engine("sqlite:///:memory:")
metadata_obj = MetaData()

receipts = Table(
    "receipts",
    metadata_obj,
    Column("receipt_id", Integer, primary_key=True),
    Column("customer_name", String(16), primary_key=True),
    Column("price", Float),
    Column("tip", Float),
)
metadata_obj.create_all(engine)

rows = [
    {"receipt_id": 1, "customer_name": "Alan Payne",      "price": 12.06, "tip": 1.20},
    {"receipt_id": 2, "customer_name": "Alex Mason",       "price": 23.86, "tip": 0.24},
    {"receipt_id": 3, "customer_name": "Woodrow Wilson",   "price": 53.43, "tip": 5.43},
    {"receipt_id": 4, "customer_name": "Margaret James",   "price": 21.11, "tip": 1.00},
]
for row in rows:
    with engine.begin() as conn:
        conn.execute(insert(receipts).values(**row))

# Build table description for the prompt
inspector = inspect(engine)
columns_info = [(col["name"], col["type"]) for col in inspector.get_columns("receipts")]
table_description = "Columns:\n" + "\n".join(
    f"  - {name}: {col_type}" for name, col_type in columns_info
)
print(table_description)


# ---------------------------------------------------------------------------
# SQL tool (kagents-style)
# ---------------------------------------------------------------------------
class SQLEngineTool(Tool):
    name = "sql_engine"
    description = (
        "Executes a SQL query against the 'receipts' table and returns the result. "
        f"Table schema:\n{table_description}"
    )
    inputs = {
        "query": ToolInput(
            type="string",
            description="A valid SQL SELECT query to run against the 'receipts' table.",
            required=True,
        )
    }
    output_type = "string"

    def forward(self, query: str) -> str:
        output = ""
        with engine.connect() as con:
            rows = con.execute(text(query))
            for row in rows:
                output += "\n" + str(row)
        return output.strip() or "(no rows returned)"


# ---------------------------------------------------------------------------
# @kbench.task entry point
# ---------------------------------------------------------------------------
@kbench.task(name="kagents_text_to_sql")
def text_to_sql_task(llm, question):
    agent = CodeAgent(
        tools=[SQLEngineTool()],
        model=llm,
        max_steps=5,
        verbosity_level=2,
        additional_instructions=(
            "You have access to an in-memory SQLite database. "
            "Use sql_engine to query it. Write correct SQL only."
        ),
    )

    print(f"\nQuestion: {question}\n")

    answer = agent.run(question)

    print("\n" + "=" * 60)
    print("FINAL ANSWER:")
    print(answer)
    print("=" * 60)

    kbench.assertions.assert_true(
        "Woodrow Wilson" in answer,
        expectation="Answer should contain the name Woodrow Wilson."
    )


if __name__ == "__main__":
    text_to_sql_task.run(llm=kbench.llm, question="Can you give me the name of the client who got the most expensive receipt?")