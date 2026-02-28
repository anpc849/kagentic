import subprocess
import os

from kagentic.tools.base import Tool
from kagentic.types import ToolInput


class ShellExecutionTool(Tool):
    """
    Tool that allows the agent to execute shell commands.
    This is extremely useful when the agent needs to run tests, search files with grep, or execute python scripts.
    """
    name = "shell_execution"
    description = "Execute a shell command in the current environment or a specified directory. Returns the standard output and standard error."
    inputs = {
        "command": ToolInput(
            type="string",
            description="The shell command to execute.",
            required=True
        ),
        "cwd": ToolInput(
            type="string",
            description="The current working directory to execute the command in. If not provided, uses the current directory.",
            required=False
        ),
        "timeout": ToolInput(
            type="integer",
            description="Maximum time in seconds to wait for the command to finish. Defaults to 300 (5 minutes).",
            required=False
        )
    }
    output_type = "string"

    def forward(self, command: str, cwd: str = None, timeout: int = 300) -> str:
        try:
            working_dir = cwd if cwd else os.getcwd()
            # We use shell=True to allow arbitrary bash-like commands
            result = subprocess.run(
                command, 
                shell=True, 
                cwd=working_dir, 
                text=True, 
                capture_output=True, 
                timeout=timeout
            )
            
            output = ""
            if result.stdout:
                output += f"STDOUT:\n{result.stdout}\n"
            if result.stderr:
                output += f"STDERR:\n{result.stderr}\n"
                
            if result.returncode != 0:
                output = f"Command failed with exit code {result.returncode}.\n{output}"
            else:
                output = f"Command succeeded.\n{output}"
                
            if not output.strip():
                output = "Command executed successfully with no output."
                
            return output
            
        except subprocess.TimeoutExpired:
            return f"Command execution timed out after {timeout} seconds."
        except Exception as e:
            return f"Error executing command: {str(e)}"
