import os
from kagentic.tools.base import Tool
from kagentic.types import ToolInput


class SearchAndReplaceTool(Tool):
    """
    Very precise tool to replace text in an existing file.
    This works better than full-file rewriting for LLMs because it requires them to write less code,
    reducing hallucinations and dropped lines.
    Requires exactly matching the old text (including indentation and whitespace).
    """
    name = "search_and_replace"
    description = "Use this tool to replace a specific snippet of text in a file with new text. You MUST provide the exact old text (including whitespace) to replace. If it appears multiple times, this tool will replace the first occurrence, so provide enough context lines to make it unique."
    inputs = {
        "file_path": ToolInput(
            type="string",
            description="The path to the file you want to edit.",
            required=True
        ),
        "old_string": ToolInput(
            type="string",
            description="The exact text to replace. THIS MUST EXACTLY MATCH THE FILE CONTENTS (including all whitespace, newlines, and indentation) or the tool will fail.",
            required=True
        ),
        "new_string": ToolInput(
            type="string",
            description="The new text to insert in place of the old_string. To delete code, just pass an empty string.",
            required=True
        )
    }
    output_type = "string"

    def forward(self, file_path: str, old_string: str, new_string: str) -> str:
        if not os.path.exists(file_path):
            return f"Error: File '{file_path}' does not exist."
        if not os.path.isfile(file_path):
            return f"Error: '{file_path}' is not a file."
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            return f"Error: File '{file_path}' appears to be a binary file or uses an unsupported encoding."
        except Exception as e:
            return f"Error opening file: {str(e)}"
            
        if old_string not in content:
            return "Error: The `old_string` was not found in the file. Make sure you matched the indentation, newlines, and whitespace EXACTLY as they appear in the file."
            
        occurrences = content.count(old_string)
        if occurrences > 1:
            # We enforce replacing only the first occurrence to avoid destroying similar blocks of code by accident.
            # But we warn the agent so it knows.
            content = content.replace(old_string, new_string, 1)
            msg = f"Replaced 1 occurrence of `old_string` successfully (Warning: `old_string` appeared {occurrences} times in the file; only the first was modified. If you meant to modify a different occurrence, add more context to `old_string` to make it unique.)"
        else:
            content = content.replace(old_string, new_string)
            msg = "Successfully replaced the `old_string` with the `new_string`."
            
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            return f"Error writing to file: {str(e)}"
            
        return msg
