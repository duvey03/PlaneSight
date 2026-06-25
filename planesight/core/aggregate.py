"""Data-aggregator orchestration (GUI functionality 1): AOI -> styled raster layers.

Given a lon/lat AOI, fetch the Copernicus GLO-30 DEM (and optionally Sentinel-2),
reproject to the AOI's local UTM zone, compute the interpretation derivatives
(hillshade, slope, curvature, ...), and write them as GeoTIFFs ready to load into
QGIS. GDAL lives at the I/O edge (lazy-imported, bundled with QGIS - D11); the
derivative math is the tested core/derivatives engine. This is the headless body
that M1's AggregateTask (QgsTask) wraps off the UI thread.
"""

from __future__ import annotations

import os
import tempfile
from typing import Callable, NamedTuple, Optional

import numpy as np

from planesight.core.data import align_to_grid, fetch_dem, fetch_sentinel2_band
from planesight.core.derivatives import build_terrain_stack
from planesight.core.geo import utm_epsg

# Terrain layers most useful for interpreting geologic boundaries (hillshade first).
DEFAULT_TERRAIN = ("hillshade", "slope", "profile_curvature")
DERIV_NODATA = -9999.0


class LayerSpec(NamedTuple):
    """A written raster output and how the GUI should label/style it."""

    path: str
    name: str
    kind: str          # 'dem' | 'hillshade' | 'slope' | 'profile_curvature' | ... | 's2_rgb'


def _emit(progress: Optional[Callable[[float, str], None]], frac: float, msg: str):
    if progress is not None:
        progress(float(frac), msg)


def _read_dem(path):
    """Read a DEM GeoTIFF -> (array with NaN nodata, geotransform, projection)."""
    from osgeo import gdal

    gdal.UseExceptions()
    ds = gdal.Open(path)
    arr = ds.ReadAsArray().astype(float)
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    nodata = ds.GetRasterBand(1).GetNoDataValue()
    ds = None
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    return arr, gt, proj


def _write_gtiff(path, array, gt, proj, nodata=DERIV_NODATA):
    """Write a single-band float GeoTIFF, replacing non-finite cells with nodata."""
    from osgeo import gdal

    arr = np.asarray(array, dtype=np.float32)
    arr = np.where(np.isfinite(arr), arr, nodata).astype(np.float32)
    h, w = arr.shape
    ds = gdal.GetDriverByName("GTiff").Create(
        path, w, h, 1, gdal.GDT_Float32, options=["COMPRESS=DEFLATE"]
    )
    ds.SetGeoTransform(gt)
    ds.SetProjection(proj)
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(float(nodata))
    band.WriteArray(arr)
    band.FlushCache()
    ds = None
    return path


def _stretch_u8(a, lo=2.0, hi=98.0):
    """Percentile contrast-stretch a band to uint8 [0, 255] (nodata -> 0)."""
    a = np.asarray(a, dtype=float)
    finite = np.isfinite(a)
    if not finite.any():
        return np.zeros(a.shape, dtype=np.uint8)
    plo, phi = np.percentile(a[finite], [lo, hi])
    if phi <= plo:
        phi = plo + 1.0
    scaled = np.clip((a - plo) / (phi - plo), 0.0, 1.0) * 255.0
    return np.where(finite, scaled, 0.0).astype(np.uint8)


def _write_rgb(path, bands_u8, gt, proj):
    """Write a 3-band (R, G, B) uint8 GeoTIFF."""
    from osgeo import gdal

    h, w = bands_u8[0].shape
    ds = gdal.GetDriverByName("GTiff").Create(
        path, w, h, 3, gdal.GDT_Byte, options=["COMPRESS=DEFLATE", "PHOTOMETRIC=RGB"]
    )
    ds.SetGeoTransform(gt)
    ds.SetProjection(proj)
    for i, band in enumerate(bands_u8, start=1):
        ds.GetRasterBand(i).WriteArray(band)
    ds = None
    return path


def _sentinel2_rgb(bbox, dem_path, out_dir, *, datetime, cloud_max, tmp):
    """Fetch S2 red/green/blue, align to the DEM grid, stretch -> RGB GeoTIFF."""
    rgb = []
    for band in ("red", "green", "blue"):
        raw = os.path.join(tmp, f"s2_{band}.tif")
        fetch_sentinel2_band(bbox, band, raw, datetime=datetime, cloud_max=cloud_max)
        aligned = os.path.join(tmp, f"s2_{band}_aligned.tif")
        align_to_grid(raw, aligned, dem_path)
        rgb.append(aligned)

    from osgeo import gdal

    chans = []
    for p in rgb:
        ds = gdal.Open(p)
        chans.append(_stretch_u8(ds.ReadAsArray().astype(float)))
        ds = None
    grid_ds = gdal.Open(dem_path)
    gt, proj = grid_ds.GetGeoTransform(), grid_ds.GetProjection()
    grid_ds = None
    out = os.path.join(out_dir, "s2_truecolor.tif")
    return _write_rgb(out, chans, gt, proj)


def aggregate_terrain(
    bbox,
    out_dir,
    *,
    res: float = 30.0,
    terrain_bands=DEFAULT_TERRAIN,
    include_s2: bool = True,
    s2_datetime: str = "2023-11-01/2024-03-31",
    cloud_max: float = 5.0,
    progress: Optional[Callable[[float, str], None]] = None,
) -> list[LayerSpec]:
    """Fetch + derive an AOI's interpretation layers, written as GeoTIFFs in out_dir.

    Args:
        bbox: ``[west, south, east, north]`` lon/lat (EPSG:4326).
        out_dir: directory to write the output GeoTIFFs into (created if missing).
        res: ground sample distance (m) for the UTM-reprojected grid.
        terrain_bands: derivative bands to compute (see core.derivatives TERRAIN_BANDS).
        include_s2: also fetch a Sentinel-2 true-color composite (best-effort).
        s2_datetime, cloud_max: Sentinel-2 scene-selection window and cloud ceiling.
        progress: optional ``callback(fraction, message)`` for QgsTask reporting.

    Returns:
        A list of LayerSpec (DEM first, then derivatives, then S2 if available).
        Failures fetching S2 are logged via the progress message and skipped, not
        raised - the terrain layers are the guaranteed product.
    """
    from osgeo import gdal

    gdal.UseExceptions()
    os.makedirs(out_dir, exist_ok=True)
    tmp = tempfile.mkdtemp()
    crs = f"EPSG:{utm_epsg(bbox)}"

    _emit(progress, 0.05, f"Fetching GLO-30 DEM ({crs}) ...")
    dem4326 = os.path.join(tmp, "dem_4326.tif")
    fetch_dem(bbox, dem4326)
    dem_path = os.path.join(out_dir, "dem.tif")
    gdal.Warp(dem_path, dem4326, dstSRS=crs, xRes=res, yRes=res, resampleAlg="bilinear")
    dem, gt, proj = _read_dem(dem_path)
    specs = [LayerSpec(dem_path, "PlaneSight DEM", "dem")]

    _emit(progress, 0.4, "Computing terrain derivatives ...")
    _, terr = build_terrain_stack(dem, res, names=terrain_bands)
    n = len(terrain_bands)
    for i, band in enumerate(terrain_bands):
        path = os.path.join(out_dir, f"{band}.tif")
        _write_gtiff(path, terr[band], gt, proj)
        specs.append(LayerSpec(path, f"PlaneSight {band}", band))
        _emit(progress, 0.4 + 0.3 * (i + 1) / n, f"Wrote {band}")

    if include_s2:
        _emit(progress, 0.75, "Fetching Sentinel-2 true-color ...")
        try:
            s2 = _sentinel2_rgb(
                bbox, dem_path, out_dir, datetime=s2_datetime,
                cloud_max=cloud_max, tmp=tmp,
            )
            specs.append(LayerSpec(s2, "PlaneSight Sentinel-2 (true color)", "s2_rgb"))
        except Exception as exc:  # noqa: BLE001 - S2 is best-effort; terrain is the product
            _emit(progress, 0.95, f"Sentinel-2 skipped ({type(exc).__name__}: {exc})")

    _emit(progress, 1.0, f"Done - {len(specs)} layers")
    return specs
