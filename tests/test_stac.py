"""Tests for the dependency-free STAC client and fetch helpers (no network)."""

import json

import pytest

from planesight.core.data import (
    StacError,
    asset_href,
    clearest_months,
    least_cloudy,
    monthly_cloud_stats,
    pick_scene,
    search_clear_sentinel2,
    search_items,
    to_vsicurl,
)
from planesight.core.data.sources import EARTH_SEARCH_V1
from planesight.core.data.stac import _normalize_datetime, item_month


def _scene(id_, month, cloud):
    """Build a minimal STAC item for selection tests."""
    return {
        "id": id_,
        "properties": {"datetime": f"2023-{month:02d}-15T06:00:00Z", "eo:cloud_cover": cloud},
        "assets": {"red": {"href": f"https://x/{id_}.tif"}},
    }


class _FakeResp:
    """Minimal context-manager response wrapping a JSON payload."""

    def __init__(self, payload):
        self._b = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _opener_from_pages(pages, captured=None):
    """Return a urlopen-compatible callable yielding pages in order."""
    state = {"i": 0}

    def opener(req, timeout=None):
        if captured is not None:
            captured.append(json.loads(req.data.decode("utf-8")))
        page = pages[min(state["i"], len(pages) - 1)]
        state["i"] += 1
        return _FakeResp(page)

    return opener


def test_search_single_page_and_request_body():
    captured = []
    pages = [{"features": [{"id": "a"}, {"id": "b"}], "links": []}]
    items = search_items(
        EARTH_SEARCH_V1,
        "cop-dem-glo-30",
        bbox=[62.0, 25.0, 62.5, 25.5],
        datetime="2023-01-01/2023-12-31",
        query={"eo:cloud_cover": {"lt": 10}},
        opener=_opener_from_pages(pages, captured),
    )
    assert [i["id"] for i in items] == ["a", "b"]
    body = captured[0]
    assert body["collections"] == ["cop-dem-glo-30"]
    assert body["bbox"] == [62.0, 25.0, 62.5, 25.5]
    assert body["datetime"] == "2023-01-01T00:00:00Z/2023-12-31T23:59:59Z"
    assert body["query"] == {"eo:cloud_cover": {"lt": 10}}


def test_search_follows_pagination():
    search_url = EARTH_SEARCH_V1 + "/search"
    pages = [
        {
            "features": [{"id": "a"}, {"id": "b"}],
            "links": [{"rel": "next", "method": "POST", "href": search_url, "body": {}}],
        },
        {"features": [{"id": "c"}], "links": []},
    ]
    items = search_items(
        EARTH_SEARCH_V1, ["sentinel-2-l2a"], bbox=[0, 0, 1, 1],
        opener=_opener_from_pages(pages),
    )
    assert [i["id"] for i in items] == ["a", "b", "c"]


def test_search_respects_max_items():
    search_url = EARTH_SEARCH_V1 + "/search"
    page = {
        "features": [{"id": "a"}, {"id": "b"}],
        "links": [{"rel": "next", "method": "POST", "href": search_url, "body": {}}],
    }
    items = search_items(
        EARTH_SEARCH_V1, "cop-dem-glo-30", bbox=[0, 0, 1, 1], max_items=3,
        opener=_opener_from_pages([page]),  # infinite next link
    )
    assert len(items) == 3


def test_search_wraps_network_error():
    def boom(req, timeout=None):
        raise OSError("connection refused")

    with pytest.raises(StacError):
        search_items(EARTH_SEARCH_V1, "cop-dem-glo-30", bbox=[0, 0, 1, 1], opener=boom)


def test_asset_href_and_s3_alternate():
    item = {
        "id": "x",
        "assets": {
            "data": {
                "href": "https://example.com/dem.tif",
                "alternate": {"s3": {"href": "s3://bucket/dem.tif"}},
            }
        },
    }
    assert asset_href(item, "data") == "https://example.com/dem.tif"
    assert asset_href(item, "data", prefer_s3=True) == "s3://bucket/dem.tif"
    with pytest.raises(KeyError):
        asset_href(item, "missing")


def test_least_cloudy_picks_minimum():
    items = [
        {"id": "hi", "properties": {"eo:cloud_cover": 80.0}},
        {"id": "lo", "properties": {"eo:cloud_cover": 5.0}},
    ]
    assert least_cloudy(items)["id"] == "lo"
    with pytest.raises(StacError):
        least_cloudy([])


def test_normalize_datetime():
    # date-only interval -> full RFC3339 start/end of day (Earth Search requires this)
    assert (
        _normalize_datetime("2023-11-01/2024-03-31")
        == "2023-11-01T00:00:00Z/2024-03-31T23:59:59Z"
    )
    # bare single date -> whole-day interval
    assert (
        _normalize_datetime("2023-11-01")
        == "2023-11-01T00:00:00Z/2023-11-01T23:59:59Z"
    )
    # already-full timestamps and open ends are preserved
    assert (
        _normalize_datetime("2023-11-01T12:00:00Z/..")
        == "2023-11-01T12:00:00Z/.."
    )
    assert _normalize_datetime("2023-11-01T06:30:00Z") == "2023-11-01T06:30:00Z"


def test_datetime_normalized_in_request_body():
    captured = []
    pages = [{"features": [], "links": []}]
    search_items(
        EARTH_SEARCH_V1, "sentinel-2-l2a", bbox=[0, 0, 1, 1],
        datetime="2023-11-01/2024-03-31",
        opener=_opener_from_pages(pages, captured),
    )
    assert captured[0]["datetime"] == "2023-11-01T00:00:00Z/2024-03-31T23:59:59Z"


def test_clearest_months_finds_dry_season():
    # Dec/Jan have several near-clear scenes; Jul is cloudy -> dry season = Dec, Jan
    items = [
        _scene("a", 12, 2.0), _scene("b", 12, 4.0), _scene("c", 1, 3.0),
        _scene("d", 7, 60.0), _scene("e", 7, 70.0), _scene("f", 1, 1.0),
    ]
    dry = clearest_months(items, top_n=2, max_cloud=5.0)
    assert dry == [1, 12] or dry == [12, 1]  # both have 2 clear scenes; order by count then month
    stats = monthly_cloud_stats(items)
    assert stats[7]["n"] == 2 and stats[7]["min"] == 60.0


def test_pick_scene_prefers_months_then_cloud():
    items = [
        _scene("dry-clearish", 12, 4.0),
        _scene("wet-clearest", 7, 0.5),  # lower cloud but wrong season
        _scene("dry-clearest", 12, 1.0),
    ]
    # prefer December: should pick the clearest December scene, not the clearer July one
    assert pick_scene(items, prefer_months=[12])["id"] == "dry-clearest"
    # with no month preference, pick the globally clearest
    assert pick_scene(items)["id"] == "wet-clearest"
    assert item_month(items[0]) == 12


def test_search_clear_sentinel2_escalates_when_too_cloudy():
    clear_items = [_scene("ok", 12, 8.0)]

    def opener(req, timeout=None):
        thr = json.loads(req.data.decode())["query"]["eo:cloud_cover"]["lt"]
        # nothing under 5%, but scenes appear once the cap reaches >=10%
        return _FakeResp({"features": clear_items if thr >= 10 else [], "links": []})

    items, used = search_clear_sentinel2([0, 0, 1, 1], cloud_max=5.0, opener=opener)
    assert used == 10.0 and [i["id"] for i in items] == ["ok"]


def test_search_clear_sentinel2_no_escalation_returns_empty():
    def opener(req, timeout=None):
        return _FakeResp({"features": [], "links": []})

    items, used = search_clear_sentinel2(
        [0, 0, 1, 1], cloud_max=5.0, escalate=False, opener=opener
    )
    assert items == [] and used == 5.0


def test_to_vsicurl_mapping():
    assert to_vsicurl("https://x/y.tif") == "/vsicurl/https://x/y.tif"
    assert to_vsicurl("s3://b/k.tif") == "/vsis3/b/k.tif"
    assert to_vsicurl("/vsicurl/https://x/y.tif") == "/vsicurl/https://x/y.tif"
    assert to_vsicurl("/local/path.tif") == "/local/path.tif"
