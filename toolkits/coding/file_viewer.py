import os
from kagentic.tools.base import Tool
from kagentic.types import ToolInput


class FileViewerTool(Tool):
    """
    Tool that allows the agent to read the contents of a file.
    Crucially, it supports start_line and end_line, allowing the agent to read massive files
    in chunks without blowing up the context window (a necessity for SWE-bench repositories).
    """
    name = "file_viewer"
    description = "Read the contents of a file. You can optionally specify a start_line and end_line (1-indexed, inclusive) to only read a specific portion of the file. This is strongly recommended for large files to avoid exceeding your context window."
    inputs = {
        "file_path": ToolInput(
            type="string",
            description="The absolute path to the file to read, or relative to the current working directory.",
            required=True
        ),
        "start_line": ToolInput(
            type="integer",
            description="The 1-indexed line number to start reading from (inclusive). Defaults to 1.",
            required=False
        ),
        "end_line": ToolInput(
            type="integer",
            description="The 1-indexed line number to stop reading at (inclusive). Defaults to the end of the file.",
            required=False
        )
    }
    output_type = "string"

    def forward(self, file_path: str, start_line: int = 1, end_line: int = None) -> str:
        if not os.path.exists(file_path):
            return f"Error: File '{file_path}' does not exist."
        if not os.path.isfile(file_path):
            return f"Error: '{file_path}' is not a file."
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
             return f"Error: File '{file_path}' appears to be a binary file or uses an unsupported encoding."
        except Exception as e:
            return f"Error opening file: {str(e)}"
            
        total_lines = len(lines)
        if total_lines == 0:
            return f"File '{file_path}' is empty."
            
        start_idx = max(0, start_line - 1)
        
        if end_line is None:
            end_idx = total_lines
        else:
            end_idx = min(total_lines, end_line)
            
        if start_idx >= total_lines:
            return f"Error: start_line ({start_line}) is beyond the end of the file ({total_lines} lines)."
            
        if start_idx >= end_idx:
            return f"Error: start_line ({start_line}) must be less than end_line ({end_line})."
            
        content = "".join(lines[start_idx:end_idx])
        
        header = f"--- {os.path.basename(file_path)} (Lines {start_idx + 1}-{end_idx} of {total_lines}) ---\n"
        return header + content
