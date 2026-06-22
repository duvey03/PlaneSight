"""Data aggregation: AOI -> STAC -> DEM + Sentinel-2.

The STAC search layer (``stac``) is pure standard library and fully testable
without GDAL/QGIS. The retrieval layer (``fetch``) uses GDAL (bundled with QGIS,
decision D11), imported lazily so this package still imports in CI.

Default source is AWS Earth Search (genuinely anonymous, D13); see
ARCHITECTURE.md S7. CRS reprojection (P0c) and cloud compositing (P0d) build on
top of this fetch layer.
"""

from __future__ import annotations

from . import sources
from .fetch import fetch_clip, fetch_dem, fetch_sentinel2_band, to_vsicurl
from .stac import StacError, asset_href, least_cloudy, search_items

__all__ = [
    "sources",
    "StacError",
    "search_items",
    "asset_href",
    "least_cloudy",
    "to_vsicurl",
    "fetch_clip",
    "fetch_dem",
    "fetch_sentinel2_band",
]
