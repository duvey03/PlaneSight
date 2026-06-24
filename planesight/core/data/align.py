"""Co-register rasters onto a common analysis grid (GDAL I/O edge).

The detector and the scorer need every band - DEM derivatives and Sentinel-2
alike - on one shared grid (same CRS, extent, pixel size, and dimensions) so they
stack pixel-for-pixel. The DEM (already reprojected to the per-AOI metric CRS) is
the natural reference grid; S2 bands are warped onto it here.

GDAL is imported lazily (decision D11) so ``planesight.core`` still imports in CI
without GDAL; the actual warping is exercised in the GDAL test job. Resampling
choice matters: continuous bands use bilinear/cubic, while a categorical raster
(e.g. the S2 SCL cloud mask) must use nearest to avoid inventing class values.
"""

from __future__ import annotations

from typing import Optional


def read_grid(ref_path: str) -> dict:
    """Read a reference raster's grid definition.

    Returns a dict with ``projection`` (WKT), ``geotransform`` (6-tuple),
    ``width``, ``height``, and ``bounds`` (minx, miny, maxx, maxy) - everything
    needed to warp another raster onto an identical grid.
    """
    from osgeo import gdal  # lazy: bundled with QGIS, intentionally absent in CI

    gdal.UseExceptions()
    ds = gdal.Open(ref_path)
    if ds is None:
        raise ValueError(f"could not open reference raster {ref_path!r}")
    gt = ds.GetGeoTransform()
    w, h = ds.RasterXSize, ds.RasterYSize
    minx, maxy = gt[0], gt[3]
    maxx = minx + w * gt[1] + h * gt[2]
    miny = maxy + w * gt[4] + h * gt[5]
    grid = {
        "projection": ds.GetProjection(),
        "geotransform": gt,
        "width": w,
        "height": h,
        "bounds": (minx, miny, maxx, maxy),
    }
    ds = None
    return grid


def align_to_grid(
    src_path,
    out_path: str,
    ref_path: str,
    resampling: str = "bilinear",
    src_nodata: Optional[float] = None,
) -> str:
    """Warp ``src_path`` onto ``ref_path``'s exact grid, writing ``out_path``.

    The output matches the reference CRS, extent, pixel size, and dimensions
    pixel-for-pixel, so it can be stacked with the reference and other aligned
    bands. Use ``resampling="near"`` for categorical inputs (e.g. SCL).

    ``src_path`` may be a single path or a list of paths; a list is mosaicked
    onto the grid (e.g. several same-date Sentinel-2 tiles covering one AOI).

    Returns ``out_path``.
    """
    from osgeo import gdal  # lazy

    gdal.UseExceptions()
    grid = read_grid(ref_path)
    minx, miny, maxx, maxy = grid["bounds"]
    warp_opts = gdal.WarpOptions(
        format="GTiff",
        dstSRS=grid["projection"],
        outputBounds=(minx, miny, maxx, maxy),
        width=grid["width"],
        height=grid["height"],
        resampleAlg=resampling,
        srcNodata=src_nodata,
        multithread=True,
        creationOptions=["COMPRESS=DEFLATE", "TILED=YES"],
    )
    gdal.Warp(out_path, src_path, options=warp_opts)
    return out_path
