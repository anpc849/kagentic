"""
kagents/tools/python_runner.py
------------------------------
Built-in tool that lets the agent execute Python code snippets.
Relies on kbench.tools.python.script_runner which is available in every
Kaggle Benchmarks notebook environment.

The tool runs the code in an isolated subprocess and returns stdout + stderr.
"""
from __future__ import annotations

from kagents.tools.base import Tool
from kagents.types import ToolInput


class PythonCodeRunnerTool(Tool):
    name = "python_interpreter"
    description = (
        "Executes a Python code snippet and returns its stdout output (and stderr if any errors). "
        "Use this to perform calculations, data manipulation, or any task requiring code execution. "
        "The code runs in an isolated environment — do not rely on state from previous calls."
    )
    inputs = {
        "code": ToolInput(
            type="string",
            description=(
                "Valid Python source code to execute. "
                "Use print() to produce output. The tool returns everything printed to stdout."
            ),
            required=True,
        )
    }
    output_type = "string"

    def forward(self, code: str) -> str:
        """Run code and return combined stdout/stderr."""
        try:
            import kaggle_benchmarks as kbench  # only available in Kaggle env

            result = kbench.tools.python.script_runner.run_code(code)
            output_parts = []

            if result.stdout and result.stdout.strip():
                output_parts.append(f"[stdout]\n{result.stdout.strip()}")

            if result.stderr and result.stderr.strip():
                output_parts.append(f"[stderr]\n{result.stderr.strip()}")

            if result.exit_code != 0:
                output_parts.append(f"[exit code: {result.exit_code}]")

            return "\n".join(output_parts) if output_parts else "(no output)"

        except ImportError:
            # Fallback for local testing outside Kaggle
            import subprocess
            import sys

            proc = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=30,
            )
            parts = []
            if proc.stdout.strip():
                parts.append(f"[stdout]\n{proc.stdout.strip()}")
            if proc.stderr.strip():
                parts.append(f"[stderr]\n{proc.stderr.strip()}")
            return "\n".join(parts) if parts else "(no output)"

        except Exception as e:
            return f"[error] Code execution failed: {type(e).__name__}: {e}"
