"""
tools/web.py

Web search without any paid API key. This is implemented as a small
provider interface (WebSearchProvider) so the underlying source can be
swapped later without touching the agent or tool_manager code.

Default provider: DuckDuckGoHTMLProvider, which scrapes DuckDuckGo's
plain HTML endpoint (https://html.duckduckgo.com/html/). This endpoint
does not require an API key and is commonly used for lightweight,
key-free search. It IS a scrape of a third-party HTML page, so:

    - it can break if DuckDuckGo changes their markup
    - it should be used moderately (this is not a high-volume search API)
    - network access must be enabled for this to work at all; if it is
      not, the tool returns a clear error rather than fake results

If you have another free/local search backend (e.g. a self-hosted
SearXNG instance, which is fully open-source and free), you can add a
new provider class implementing `.search(query, max_results)` and wire
it up in agent.py / config without changing anything else.
"""

from dataclasses import dataclass
from typing import List
from urllib.parse import urlencode

import requests


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str

    def to_dict(self):
        return {"title": self.title, "url": self.url, "snippet": self.snippet}


class WebSearchProvider:
    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        raise NotImplementedError


class DuckDuckGoHTMLProvider(WebSearchProvider):
    """Free, key-free web search via DuckDuckGo's HTML endpoint."""

    ENDPOINT = "https://html.duckduckgo.com/html/"
    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) local-agent/1.0 "
        "(open-source local AI agent; no paid API used)"
    )

    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        try:
            from bs4 import BeautifulSoup
        except ImportError as e:
            raise RuntimeError(
                "beautifulsoup4 is required for web search. Install it with: "
                "pip install beautifulsoup4"
            ) from e

        params = {"q": query}
        try:
            resp = requests.post(
                self.ENDPOINT,
                data=params,
                headers={"User-Agent": self.USER_AGENT},
                timeout=15,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"Web search request failed: {e}") from e

        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for result_div in soup.select(".result")[: max_results * 2]:
            link = result_div.select_one(".result__a")
            snippet_el = result_div.select_one(".result__snippet")
            if not link:
                continue
            url = link.get("href", "")
            title = link.get_text(strip=True)
            snippet = snippet_el.get_text(strip=True) if snippet_el else ""
            if title and url:
                results.append(SearchResult(title=title, url=url, snippet=snippet))
            if len(results) >= max_results:
                break
        return results


class NullProvider(WebSearchProvider):
    """Fallback used when no network / no working provider is available.
    Never fabricates results -- returns an empty list and lets the tool
    manager surface a clear error to the model instead of fake data."""

    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        return []


def get_default_provider() -> WebSearchProvider:
    return DuckDuckGoHTMLProvider()


def web_search(query: str, max_results: int = 5, provider: WebSearchProvider = None) -> dict:
    provider = provider or get_default_provider()
    try:
        results = provider.search(query, max_results=max_results)
    except RuntimeError as e:
        return {"success": False, "error": str(e), "query": query}

    if not results:
        return {
            "success": False,
            "error": "No results returned (search backend may be unreachable, or blocked network access).",
            "query": query,
        }
    return {"success": True, "query": query, "results": [r.to_dict() for r in results]}
