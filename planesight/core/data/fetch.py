"""Raster retrieval: clip/mosaic STAC COG assets to an AOI using GDAL.

GDAL is bundled with QGIS (decision D11) and is imported lazily so that
``planesight.core`` stays importable without GDAL present (e.g. in CI). This
module fetches in each asset's native CRS; reprojection to a metric analysis CRS
is handled separately (P0c), and robust cloud compositing is P0d - here we take
the single least-cloudy scene as a first pass.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

from .sources import (
    CLOUD_CEILING,
    COP_DEM_ASSET,
    COP_DEM_GLO_30,
    DEFAULT_MAX_CLOUD,
    EARTH_SEARCH_V1,
    SENTINEL2_L2A,
)
from .stac import StacError, asset_href, clearest_months, item_month, pick_scene, search_items

logger = logging.getLogger("planesight.core.data")

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


def _escalation_thresholds(cloud_max: float, ceiling: float) -> List[float]:
    """Cloud thresholds to try in turn: start strict, double up to the ceiling."""
    thresholds = [cloud_max]
    t = cloud_max
    while t < ceiling:
        t = min(t * 2, ceiling)
        thresholds.append(t)
    return thresholds


def search_clear_sentinel2(
    bbox: BBox,
    cloud_max: float = DEFAULT_MAX_CLOUD,
    datetime: Optional[str] = None,
    escalate: bool = True,
    cloud_ceiling: float = CLOUD_CEILING,
    catalog_url: str = EARTH_SEARCH_V1,
    opener=None,
) -> Tuple[List[Dict], float]:
    """Find near-clear Sentinel-2 L2A scenes, escalating the cloud cap if needed.

    Tries scenes with cloud cover < ``cloud_max`` first. If none exist in the
    window and ``escalate`` is True, the cap is raised stepwise (doubling) up to
    ``cloud_ceiling``, logging a warning. Returns ``(items, cloud_used)``.
    """
    thresholds = _escalation_thresholds(cloud_max, cloud_ceiling) if escalate else [cloud_max]
    for thr in thresholds:
        items = search_items(
            catalog_url,
            SENTINEL2_L2A,
            bbox,
            datetime=datetime,
            query={"eo:cloud_cover": {"lt": thr}},
            opener=opener,
        )
        if items:
            if thr > cloud_max:
                logger.warning(
                    "No Sentinel-2 scene under %.0f%% cloud for bbox %s; "
                    "relaxed to <%.0f%% (%d scenes).",
                    cloud_max, tuple(bbox), thr, len(items),
                )
            return items, thr
    return [], thresholds[-1]


def fetch_sentinel2_band(
    bbox: BBox,
    band: str,
    out_path: str,
    datetime: Optional[str] = None,
    cloud_max: float = DEFAULT_MAX_CLOUD,
    months: Optional[Sequence[int]] = None,
    auto_season: bool = True,
    escalate: bool = True,
    catalog_url: str = EARTH_SEARCH_V1,
    opener=None,
) -> str:
    """Fetch one Sentinel-2 L2A band over ``bbox`` from a near-clear, dry-season scene.

    Selection prefers (in order): the dry-season months, then the lowest cloud
    cover. The dry season is taken from ``months`` if given, else - when
    ``auto_season`` is set - derived empirically from the AOI's own clear-scene
    histogram (the months with the most near-clear scenes). Full multi-scene
    cloud masking and temporal compositing remain P0d (issue planesight-bcn).

    Args:
        band: a key from ``sources.S2_BANDS`` (e.g. "red", "swir22", "scl").
        datetime: optional RFC3339 interval; a multi-year window improves
            dry-season detection.
        cloud_max: maximum scene cloud cover percent (default 5%).
        months: explicit preferred (dry-season) months, 1-12; overrides auto.
        auto_season: when ``months`` is None, infer the dry season from the data.
        escalate: relax the cloud cap for cloudy AOIs rather than returning none.
    """
    items, _ = search_clear_sentinel2(
        bbox, cloud_max=cloud_max, datetime=datetime, escalate=escalate,
        catalog_url=catalog_url, opener=opener,
    )
    if not items:
        raise StacError(
            f"no Sentinel-2 L2A scene under {cloud_max}% cloud for bbox {tuple(bbox)}"
        )
    prefer = list(months) if months else (clearest_months(items) if auto_season else None)
    scene = pick_scene(items, prefer_months=prefer)
    logger.info(
        "Sentinel-2 scene %s (cloud %.1f%%, month %d) selected for bbox %s",
        scene.get("id"), scene.get("properties", {}).get("eo:cloud_cover", -1),
        item_month(scene), tuple(bbox),
    )
    href = asset_href(scene, band)
    return fetch_clip([href], bbox, out_path, resampling="bilinear")
