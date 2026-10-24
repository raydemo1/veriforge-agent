"""Lightweight web search and fetch tools."""
from __future__ import annotations

from ..tool_result import ToolResult


def web_search(query: str, max_results: int = 5) -> ToolResult:
    """Search the web using DuckDuckGo and return text results.
    Uses DDG's lite HTML endpoint — no API key needed, works in any container.
    """
    import html as html_mod
    import re
    import urllib.parse
    import urllib.request

    try:
        encoded = urllib.parse.urlencode({"q": query})
        url = f"https://lite.duckduckgo.com/lite/?{encoded}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=15)
        raw = resp.read().decode("utf-8", errors="replace")

        # Extract result links (DDG lite uses rel="nofollow" for result links)
        links = re.findall(
            r'<a[^>]*rel="nofollow"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            raw, re.DOTALL
        )

        # Extract snippets (text in <td> cells that aren't links/navigation)
        cells = re.findall(r'<td[^>]*>(.*?)</td>', raw, re.DOTALL)
        snippets = []
        for cell in cells:
            text = re.sub(r'<[^>]+>', '', cell).strip()
            if len(text) > 50 and not text.startswith('http'):
                snippets.append(text)

        results = []
        for i, (href, title) in enumerate(links):
            if i >= max_results:
                break
            title = html_mod.unescape(re.sub(r'<[^>]+>', '', title).strip())
            # Decode DDG redirect URL
            real_url = href
            m = re.search(r'uddg=([^&]+)', href)
            if m:
                real_url = urllib.parse.unquote(m.group(1))
            snippet = snippets[i] if i < len(snippets) else ""
            results.append(f"{i+1}. {title}\n   {real_url}\n   {snippet[:200]}\n")

        if results:
            return ToolResult(
                tool="web_search",
                status="success",
                output=f"Search results for: {query}\n\n" + "\n".join(results),
                metadata={"query": query, "result_count": len(results), "status_source": "native"},
            )

        return ToolResult(
            tool="web_search",
            status="success",
            output=f"No results found for: {query}",
            metadata={"query": query, "result_count": 0, "status_source": "native"},
        )

    except Exception as e:
        return ToolResult(
            tool="web_search",
            status="failed",
            output=f"[error] Web search failed: {e}",
            error=f"Web search failed: {e}",
            metadata={"query": query, "status_source": "exception"},
        )


def web_fetch(url: str) -> ToolResult:
    """Fetch the content of a web page and return as text."""
    import re
    import urllib.request

    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        })
        resp = urllib.request.urlopen(req, timeout=15)
        html = resp.read().decode("utf-8", errors="replace")

        # Strip HTML tags, keep text
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return ToolResult(
            tool="web_fetch",
            status="success",
            output=text or "(empty page)",
            metadata={"url": url, "status_source": "native"},
        )

    except Exception as e:
        return ToolResult(
            tool="web_fetch",
            status="failed",
            output=f"[error] Web fetch failed: {e}",
            error=f"Web fetch failed: {e}",
            metadata={"url": url, "status_source": "exception"},
        )
