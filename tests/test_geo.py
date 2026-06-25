"""Tests for per-AOI UTM CRS selection (core/geo.py)."""

from planesight.core.geo import utm_epsg


def test_nepal_aoi_picks_utm_44n():
    # the AOI used throughout the project -> EPSG:32644 (UTM 44N), matching the
    # headless runs (scripts/detect_attitudes_nepal.py uses EPSG=32644).
    assert utm_epsg([82.40, 27.70, 82.50, 27.80]) == 32644


def test_southern_hemisphere_uses_327xx():
    # a southern AOL -> 327xx band.
    code = utm_epsg([-59.0, -30.0, -58.0, -29.0])
    assert 32701 <= code <= 32760
    assert code == 32721                       # zone 21 S


def test_zone_boundaries():
    assert utm_epsg([0.0, 10.0, 0.0, 10.0]) == 32631      # 0 deg -> zone 31 N
    assert utm_epsg([-180.0, 10.0, -180.0, 10.0]) == 32601  # antimeridian -> zone 1
    assert utm_epsg([5.9, 1.0, 5.9, 1.0]) == 32631         # still zone 31 (0-6E)
    assert utm_epsg([6.1, 1.0, 6.1, 1.0]) == 32632         # zone 32 (6-12E)


def test_equator_counts_as_northern():
    # lat == 0 resolves to the northern band (326xx) by the >= 0 rule.
    assert utm_epsg([10.0, 0.0, 12.0, 0.0]) // 100 == 326
