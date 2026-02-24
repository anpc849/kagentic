"""
kagents/tools/web_search.py
--------------------------
WebSearchTool: search the web and return result snippets + URLs.

Primary:  ddgs package (no API key, handles bot-detection properly)
Fallback: DuckDuckGo Lite HTML endpoint (requests + BeautifulSoup)

Install:
    !pip install -q ddgs requests beautifulsoup4
"""
from __future__ import annotations

import urllib.parse
from typing import List

from kagents.tools.base import Tool
from kagents.types import ToolInput

_LITE_URL = "https://lite.duckduckgo.com/lite/"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


class WebSearchTool(Tool):
    """
    Searches the web using DuckDuckGo (no API key required) and returns
    the top result titles, URLs, and snippets.

    Use WebBrowseTool to read the full content of a specific URL.

    Dependencies: duckduckgo-search (primary), requests + beautifulsoup4 (fallback)
    Install with: !pip install -q duckduckgo-search requests beautifulsoup4
    """

    name = "web_search"
    description = (
        "Searches the web for a query and returns the top result titles, URLs, and snippets. "
        "Use this to find relevant links, then use web_browse to read a specific page in full."
    )
    inputs = {
        "query": ToolInput(
            type="string",
            description="The search query to look up on the web.",
            required=True,
        ),
        "num_results": ToolInput(
            type="integer",
            description="Number of top results to return (default 5, max 10).",
            required=False,
        ),
    }
    output_type = "string"

    def forward(self, query: str, num_results: int = 5) -> str:
        num_results = max(1, min(int(num_results), 10))

        # --- Primary: duckduckgo-search package ---
        result = _search_with_ddgs(query, num_results)
        if result is not None:
            return result

        # --- Fallback: DuckDuckGo Lite HTML scraping ---
        result = _search_with_lite(query, num_results)
        if result is not None:
            return result

        return (
            "[WebSearchTool] All search methods failed. "
            "Make sure at least one of these is installed:\n"
            "  !pip install -q duckduckgo-search\n"
            "  !pip install -q requests beautifulsoup4"
        )


# ---------------------------------------------------------------------------
# Primary: duckduckgo-search package
# ---------------------------------------------------------------------------

def _search_with_ddgs(query: str, num_results: int) -> str | None:
    """Use the duckduckgo_search package (most reliable, no bot detection)."""
    try:
        from ddgs import DDGS

        results: List[str] = []
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=num_results))

        if not hits:
            return f"[WebSearchTool] No results found for query: '{query}'"

        for i, hit in enumerate(hits):
            title = hit.get("title", "No title")
            url = hit.get("href", "")
            snippet = hit.get("body", "No snippet")
            results.append(f"[{i+1}] {title}\n    URL: {url}\n    {snippet}")

        return f"Search results for '{query}':\n\n" + "\n\n".join(results)

    except ImportError:
        return None  # package not installed, try fallback
    except Exception as e:
        return None  # network or other error, try fallback


# ---------------------------------------------------------------------------
# Fallback: DuckDuckGo Lite HTML scraping
# ---------------------------------------------------------------------------

def _search_with_lite(query: str, num_results: int) -> str | None:
    """Scrape DuckDuckGo Lite (simpler HTML, less bot detection than main endpoint)."""
    try:
        import requests
        from bs4 import BeautifulSoup

        resp = requests.post(
            _LITE_URL,
            data={"q": query, "s": "0", "o": "json", "dc": "", "v": "l", "api": "d.js"},
            headers=_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # DDG Lite results: <a class="result-link"> + adjacent <td class="result-snippet">
        links = soup.select("a.result-link")[:num_results]
        snippets = soup.select("td.result-snippet")

        if not links:
            return None  # might still be blocked, signal fallback failure

        results: List[str] = []
        for i, link in enumerate(links):
            title = link.get_text(strip=True)
            url = link.get("href", "")
            snippet = snippets[i].get_text(strip=True) if i < len(snippets) else ""
            results.append(f"[{i+1}] {title}\n    URL: {url}\n    {snippet}")

        return f"Search results for '{query}':\n\n" + "\n\n".join(results)

    except ImportError:
        return None
    except Exception:
        return None
