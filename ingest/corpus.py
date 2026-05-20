"""Fetch the source corpus as plain-text Wikipedia articles.

Wikipedia is used because it is freely licensed (CC BY-SA), reproducible, and
the MediaWiki API returns clean plain text. To point the assistant at a
different knowledge domain, swap the titles in ARTICLE_TITLES and rerun the
ingestion: nothing else in the pipeline is topic-specific.
"""

from __future__ import annotations

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "rag-assistant/0.1 (portfolio project)"
TIMEOUT = httpx.Timeout(30.0, connect=10.0)

# Curated knowledge base: climate, air quality and environmental science.
ARTICLE_TITLES: list[str] = [
    "Air pollution",
    "Particulates",
    "Smog",
    "Air quality index",
    "Indoor air quality",
    "Greenhouse gas",
    "Climate change",
    "Carbon dioxide in Earth's atmosphere",
    "Ozone",
    "Tropospheric ozone",
    "Ozone depletion",
    "Nitrogen dioxide",
    "Sulfur dioxide",
    "Carbon monoxide",
    "Acid rain",
    "Renewable energy",
    "Solar power",
    "Wind power",
    "Fossil fuel",
    "Emission standard",
    "Paris Agreement",
    "Kyoto Protocol",
    "Carbon footprint",
    "Carbon offset",
    "Deforestation",
    "Ocean acidification",
    "Sea level rise",
    "Effects of climate change",
    "Global warming potential",
    "Air pollution in Poland",
]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _fetch_article(client: httpx.Client, title: str) -> dict | None:
    """Return one article as plain text, or None if the page does not exist."""
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts|info",
        "inprop": "url",
        "explaintext": 1,
        "redirects": 1,
        "titles": title,
    }
    resp = client.get(WIKI_API, params=params, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    pages = resp.json().get("query", {}).get("pages", {})

    for page in pages.values():
        if "missing" in page:
            return None
        extract = (page.get("extract") or "").strip()
        if not extract:
            return None
        return {
            "doc_id": str(page.get("pageid")),
            "title": page.get("title", title),
            "source_url": page.get("fullurl", ""),
            "text": extract,
        }
    return None


def fetch_corpus(titles: list[str] | None = None) -> list[dict]:
    """Fetch every article, skipping any that cannot be resolved."""
    titles = titles or ARTICLE_TITLES
    documents: list[dict] = []

    with httpx.Client(timeout=TIMEOUT) as client:
        for title in titles:
            try:
                doc = _fetch_article(client, title)
            except Exception as exc:  # noqa: BLE001 - log and continue
                print(f"  error fetching '{title}': {exc}")
                continue
            if doc is None:
                print(f"  skipped (not found): {title}")
                continue
            documents.append(doc)
            print(f"  fetched: {doc['title']} ({len(doc['text']):,} chars)")

    return documents
