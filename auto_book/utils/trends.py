"""Fetch trending related keywords from Google Trends for a topic."""

from __future__ import annotations

import time

from auto_book.utils.logger import get_logger


def fetch_trending_keywords(topic: str, max_keywords: int = 8) -> list[str]:
    """Return top related search keywords for a topic via Google Trends.

    Returns an empty list on any error — never blocks the run.
    """
    try:
        from pytrends.request import TrendReq

        pytrends = TrendReq(hl="en-US", tz=0, timeout=(10, 25), retries=2, backoff_factor=0.5)
        pytrends.build_payload([topic[:100]], timeframe="today 12-m")
        related = pytrends.related_queries()

        keywords: list[str] = []
        data = related.get(topic[:100], {})
        for key in ("top", "rising"):
            df = data.get(key)
            if df is not None and not df.empty and "query" in df.columns:
                keywords.extend(df["query"].tolist())
            if len(keywords) >= max_keywords:
                break

        keywords = list(dict.fromkeys(keywords))[:max_keywords]
        get_logger().info("Google Trends: fetched %s keyword(s) for %r", len(keywords), topic[:60])
        return keywords

    except Exception as exc:
        get_logger().warning("Google Trends fetch failed (non-fatal): %s", exc)
        return []
