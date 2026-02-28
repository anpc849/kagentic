import os
import re
from kagentic.tools.base import Tool
from kagentic.types import ToolInput


class RegexSearchTool(Tool):
    """
    Fast regex-based string search in files.
    This helps the agent locate classes, variables, or bugs without needing a full language server or indexing setup.
    """
    name = "regex_search"
    description = "Searches for a regular expression pattern in a specific file or all files in a directory. Returns the matching lines and their line numbers."
    inputs = {
        "pattern": ToolInput(
            type="string",
            description="The regular expression pattern to search for (e.g. 'def process_data', 'class User').",
            required=True
        ),
        "path": ToolInput(
            type="string",
            description="The file path or directory path to search within.",
            required=True
        ),
        "case_sensitive": ToolInput(
            type="boolean",
            description="Whether the search should be case sensitive. Defaults to False.",
            required=False
        )
    }
    output_type = "string"

    def forward(self, pattern: str, path: str, case_sensitive: bool = False) -> str:
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return f"Error compiling regular expression: {str(e)}"

        if not os.path.exists(path):
            return f"Error: Path '{path}' does not exist."

        results = []
        max_results = 50

        def search_file(filepath):
            if len(results) >= max_results:
                return
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f, 1):
                        if regex.search(line):
                            results.append(f"{filepath}:{i}: {line.strip()}")
                            if len(results) >= max_results:
                                results.append(f"... Truncating at {max_results} results ...")
                                break
            except UnicodeDecodeError:
                pass  # Skip binary files
            except Exception as e:
                results.append(f"Error reading {filepath}: {str(e)}")

        if os.path.isfile(path):
            search_file(path)
        else:
            for root, _, files in os.walk(path):
                # Optionally filter out .git, __pycache__, node_modules etc if needed later
                if any(x in root for x in [".git", "__pycache__", "node_modules"]):
                    continue
                for file in files:
                    if file.endswith('.py') or file.endswith('.txt') or file.endswith('.md'):
                        search_file(os.path.join(root, file))
                    if len(results) >= max_results:
                        break
                if len(results) >= max_results:
                    break

        if not results:
            return f"No matches found for pattern '{pattern}' in '{path}'."
            
        return "\n".join(results)
