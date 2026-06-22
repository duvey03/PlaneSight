"""Minimal STAC API client (dependency-free).

Queries a STAC API (default: AWS Earth Search) using only the Python standard
library - no pystac-client, no requests - to keep v1 dependency-free (D11).
Supports POST /search with bbox / datetime / query-extension filters and follows
``next`` pagination links.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

DEFAULT_TIMEOUT = 60

# A urlopen-compatible callable: (request, timeout) -> context-manager response.
Opener = Callable[..., Any]


class StacError(RuntimeError):
    """Raised when a STAC request fails or returns an unexpected payload."""


def _post_json(
    url: str,
    body: Dict[str, Any],
    timeout: int = DEFAULT_TIMEOUT,
    opener: Optional[Opener] = None,
) -> Dict[str, Any]:
    """POST a JSON body and return the parsed JSON response."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    do_open = opener or urllib.request.urlopen
    try:
        with do_open(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (OSError, ValueError) as exc:  # network error or bad JSON
        raise StacError(f"STAC request to {url} failed: {exc}") from exc


def _expand_datetime_token(token: str, end_of_day: bool) -> str:
    """Expand one date-only token to a full RFC3339 timestamp; pass others through."""
    token = token.strip()
    if token in ("", "..") or "T" in token:
        return token  # open end, or already a full timestamp
    return f"{token}T23:59:59Z" if end_of_day else f"{token}T00:00:00Z"


def _normalize_datetime(value: str) -> str:
    """Make a STAC datetime forgiving of date-only input.

    Earth Search requires full RFC3339 timestamps (date-only values return HTTP
    400). This expands ``YYYY-MM-DD`` parts to start/end-of-day UTC. Accepts a
    single value or a ``start/end`` interval; ``..`` open ends are preserved. A
    bare single date becomes a whole-day interval.
    """
    if "/" in value:
        start, _, stop = value.partition("/")
        return f"{_expand_datetime_token(start, False)}/{_expand_datetime_token(stop, True)}"
    if "T" in value:
        return value
    return f"{_expand_datetime_token(value, False)}/{_expand_datetime_token(value, True)}"


def _next_page(
    payload: Dict[str, Any], search_url: str
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Return (url, body) for the next page, or (None, None) if there is none."""
    for link in payload.get("links", []):
        if link.get("rel") == "next":
            url = link.get("href", search_url)
            method = (link.get("method") or "POST").upper()
            body = link.get("body") if method == "POST" else None
            return url, body
    return None, None


def search_items(
    catalog_url: str,
    collections: Sequence[str] | str,
    bbox: Sequence[float],
    datetime: Optional[str] = None,
    query: Optional[Dict[str, Any]] = None,
    limit: int = 100,
    max_items: int = 1000,
    opener: Optional[Opener] = None,
) -> List[Dict[str, Any]]:
    """Search a STAC API and return a list of item (Feature) dicts.

    Args:
        catalog_url: base STAC API URL (without the trailing ``/search``).
        collections: one collection id or a sequence of ids.
        bbox: (minx, miny, maxx, maxy) in WGS84 lon/lat.
        datetime: optional RFC3339 interval, e.g. ``"2023-01-01/2023-12-31"``.
        query: optional STAC query-extension dict, e.g.
            ``{"eo:cloud_cover": {"lt": 10}}`` (honoured by Earth Search).
        limit: page size requested from the API.
        max_items: hard cap on total items returned (safety valve).
        opener: optional urlopen-compatible callable, for testing.
    """
    if isinstance(collections, str):
        collections = [collections]
    search_url = catalog_url.rstrip("/") + "/search"
    body: Dict[str, Any] = {
        "collections": list(collections),
        "bbox": list(bbox),
        "limit": limit,
    }
    if datetime:
        body["datetime"] = _normalize_datetime(datetime)
    if query:
        body["query"] = query

    items: List[Dict[str, Any]] = []
    next_url: Optional[str] = search_url
    next_body: Optional[Dict[str, Any]] = body
    while next_url and next_body is not None and len(items) < max_items:
        payload = _post_json(next_url, next_body, opener=opener)
        items.extend(payload.get("features", []))
        next_url, next_body = _next_page(payload, search_url)
    return items[:max_items]


def asset_href(item: Dict[str, Any], key: str, prefer_s3: bool = False) -> str:
    """Return the href of asset ``key`` for a STAC item.

    Args:
        item: a STAC item dict.
        key: the asset key (e.g. "data", "red", "swir22").
        prefer_s3: if True, return the s3:// alternate href when present.
    """
    assets = item.get("assets", {})
    if key not in assets:
        raise KeyError(
            f"Asset '{key}' not in item '{item.get('id')}'. "
            f"Available: {sorted(assets)}"
        )
    asset = assets[key]
    if prefer_s3:
        s3 = asset.get("alternate", {}).get("s3", {}).get("href")
        if s3:
            return s3
    return asset["href"]


def least_cloudy(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Return the item with the lowest eo:cloud_cover (ties broken by order)."""
    if not items:
        raise StacError("no items to choose from")
    return min(
        items, key=lambda i: i.get("properties", {}).get("eo:cloud_cover", 100.0)
    )
