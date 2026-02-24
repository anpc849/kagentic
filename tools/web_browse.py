"""
kagents/tools/web_browse.py
--------------------------
WebBrowseTool: fetch and read the text content of any URL.

Complements WebSearchTool — use web_search to find relevant links,
then web_browse to read the full content of a specific page.

Dependencies: requests, beautifulsoup4
    !pip install -q requests beautifulsoup4
"""
from __future__ import annotations

import textwrap
import urllib.parse

from kagents.tools.base import Tool
from kagents.types import ToolInput

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}


class WebBrowseTool(Tool):
    """
    Fetches the text content of a URL and returns it in a readable format.
    Use this after web_search to read the full content of a specific page.

    Dependencies: requests, beautifulsoup4
    Install with: !pip install -q requests beautifulsoup4
    """

    name = "web_browse"
    description = (
        "Fetches a URL and returns its main text content. "
        "Use this to read the full content of a specific web page "
        "after finding its URL via web_search."
    )
    inputs = {
        "url": ToolInput(
            type="string",
            description="The full URL of the page to fetch (must start with http:// or https://).",
            required=True,
        ),
        "max_chars": ToolInput(
            type="integer",
            description=(
                "Maximum number of characters to return from the page (default 4000). "
                "Increase if you need more content."
            ),
            required=False,
        ),
    }
    output_type = "string"

    def forward(self, url: str, max_chars: int = 4000) -> str:
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            return (
                "[WebBrowseTool] Missing dependencies. "
                "Run: !pip install -q requests beautifulsoup4"
            )

        url = url.strip()
        if not url.startswith(("http://", "https://")):
            return f"[WebBrowseTool] Invalid URL: '{url}'. Must start with http:// or https://"

        max_chars = max(500, min(int(max_chars), 20000))

        try:
            resp = requests.get(url, headers=_HEADERS, timeout=15)
            resp.raise_for_status()
        except Exception as e:
            return f"[WebBrowseTool] Failed to fetch '{url}': {e}"

        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(resp.text, "html.parser")

            # Remove noise
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
                tag.decompose()

            # Try to get the main content area first
            main = (
                soup.find("main")
                or soup.find("article")
                or soup.find(id="content")
                or soup.find(id="main")
                or soup.body
            )
            text = (main or soup).get_text(separator="\n", strip=True)

            # Collapse excessive blank lines
            lines = [l for l in text.splitlines() if l.strip()]
            text = "\n".join(lines)

            truncated = textwrap.shorten(text, width=max_chars, placeholder="\n… [truncated]")

            domain = urllib.parse.urlparse(url).netloc
            return f"[Page: {domain}]\nURL: {url}\n\n{truncated}"

        except Exception as e:
            return f"[WebBrowseTool] Failed to parse page content: {e}"
