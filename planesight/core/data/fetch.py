"""Raster retrieval: clip/mosaic STAC COG assets to an AOI using GDAL.

GDAL is bundled with QGIS (decision D11) and is imported lazily so that
``planesight.core`` stays importable without GDAL present (e.g. in CI). This
module fetches in each asset's native CRS; reprojection to a metric analysis CRS
is handled separately (P0c), and robust cloud compositing is P0d - here we take
the single least-cloudy scene as a first pass.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from .sources import COP_DEM_ASSET, COP_DEM_GLO_30, EARTH_SEARCH_V1, SENTINEL2_L2A
from .stac import StacError, asset_href, least_cloudy, search_items

BBox = Sequence[float]  # (minx, miny, maxx, maxy) in WGS84 lon/lat


def to_vsicurl(href: str) -> str:
    """Map an http(s)/s3 asset href to a GDAL virtual-filesystem path."""
    if href.startswith(("/vsicurl/", "/vsis3/")):
        return href
    if href.startswith("s3://"):
        return "/vsis3/" + href[len("s3://") :]
    if href.startswith(("http://", "https://")):
        return "/vsicurl/" + href
    return href  # assume a local path


def fetch_clip(
    hrefs: Sequence[str],
    bbox: BBox,
    out_path: str,
    resampling: str = "bilinear",
    bbox_srs: str = "EPSG:4326",
) -> str:
    """Mosaic + clip one or more COG hrefs to ``bbox`` into ``out_path``.

    Args:
        hrefs: asset href(s) (http/s3); multiple inputs are mosaicked.
        bbox: clip bounds, expressed in ``bbox_srs``.
        out_path: output GeoTIFF path.
        resampling: GDAL resampling algorithm (e.g. "bilinear", "cubic").
        bbox_srs: CRS of ``bbox`` (default WGS84 lon/lat).

    Returns:
        ``out_path``.
    """
    from osgeo import gdal  # lazy: bundled with QGIS, intentionally absent in CI

    gdal.UseExceptions()
    # Anonymous access to public open-data buckets (e.g. s3://copernicus-dem-30m,
    # whose DEM assets are s3:// only). Required for /vsis3/ without credentials.
    gdal.SetConfigOption("AWS_NO_SIGN_REQUEST", "YES")
    gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    sources = [to_vsicurl(h) for h in hrefs]
    warp_opts = gdal.WarpOptions(
        format="GTiff",
        outputBounds=list(bbox),
        outputBoundsSRS=bbox_srs,
        resampleAlg=resampling,
        multithread=True,
        creationOptions=["COMPRESS=DEFLATE", "TILED=YES"],
    )
    gdal.Warp(out_path, sources, options=warp_opts)
    return out_path


def fetch_dem(
    bbox: BBox,
    out_path: str,
    catalog_url: str = EARTH_SEARCH_V1,
    opener=None,
) -> str:
    """Fetch and clip the Copernicus GLO-30 DEM over ``bbox`` to ``out_path``.

    Mosaics all DEM tiles intersecting the AOI.
    """
    items = search_items(catalog_url, COP_DEM_GLO_30, bbox, opener=opener)
    hrefs: List[str] = [asset_href(it, COP_DEM_ASSET) for it in items]
    if not hrefs:
        raise StacError(f"no Copernicus GLO-30 tiles found for bbox {tuple(bbox)}")
    return fetch_clip(hrefs, bbox, out_path, resampling="bilinear")


def fetch_sentinel2_band(
    bbox: BBox,
    band: str,
    out_path: str,
    datetime: Optional[str] = None,
    max_cloud: Optional[float] = None,
    catalog_url: str = EARTH_SEARCH_V1,
    opener=None,
) -> str:
    """Fetch one Sentinel-2 L2A band over ``bbox``, using the least-cloudy scene.

    Note: this is a first-pass single-scene fetch. Multi-scene cloud masking and
    temporal compositing are implemented in P0d (issue planesight-bcn).

    Args:
        band: a key from ``sources.S2_BANDS`` (e.g. "red", "swir22", "scl").
        datetime: optional RFC3339 interval to constrain the search window.
        max_cloud: optional max scene cloud cover percent (query-extension).
    """
    query = {"eo:cloud_cover": {"lt": max_cloud}} if max_cloud is not None else None
    items = search_items(
        catalog_url, SENTINEL2_L2A, bbox, datetime=datetime, query=query, opener=opener
    )
    if not items:
        raise StacError(f"no Sentinel-2 L2A scenes found for bbox {tuple(bbox)}")
    href = asset_href(least_cloudy(items), band)
    return fetch_clip([href], bbox, out_path, resampling="bilinear")
