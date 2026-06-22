"""GDAL-dependent integration tests for the raster fetch/clip layer.

Auto-skipped where ``osgeo.gdal`` is unavailable (e.g. the pure-Python CI job and
this WSL dev shell without GDAL). Runs in the GDAL CI job and in a local
micromamba/QGIS environment. Uses only local rasters - no network.
"""

import numpy as np
import pytest

gdal = pytest.importorskip("osgeo.gdal", reason="GDAL (osgeo) not installed")
gdal.UseExceptions()

from planesight.core.data import fetch_clip, to_vsicurl  # noqa: E402


def _make_geotiff(path, bbox, nx=40, ny=40):
    """Write a small float32 GeoTIFF over bbox (minx, miny, maxx, maxy), EPSG:4326."""
    from osgeo import osr

    minx, miny, maxx, maxy = bbox
    ds = gdal.GetDriverByName("GTiff").Create(str(path), nx, ny, 1, gdal.GDT_Float32)
    ds.SetGeoTransform((minx, (maxx - minx) / nx, 0, maxy, 0, -(maxy - miny) / ny))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).WriteArray(np.arange(nx * ny, dtype="float32").reshape(ny, nx))
    ds.FlushCache()
    ds = None


def test_fetch_clip_clips_local_geotiff(tmp_path):
    src = tmp_path / "src.tif"
    _make_geotiff(src, (10.0, 50.0, 11.0, 51.0))  # 1 degree square source
    out = tmp_path / "out.tif"

    fetch_clip([str(src)], (10.25, 50.25, 10.75, 50.75), str(out))  # central 0.5 deg

    assert out.exists()
    ds = gdal.Open(str(out))
    gt = ds.GetGeoTransform()
    width_deg = ds.RasterXSize * gt[1]
    height_deg = abs(ds.RasterYSize * gt[5])
    assert 0.4 < width_deg < 0.6  # clipped to ~0.5 deg, smaller than the source
    assert 0.4 < height_deg < 0.6
    ds = None


def test_fetch_clip_mosaics_two_tiles(tmp_path):
    left = tmp_path / "left.tif"
    right = tmp_path / "right.tif"
    _make_geotiff(left, (10.0, 50.0, 11.0, 51.0))
    _make_geotiff(right, (11.0, 50.0, 12.0, 51.0))
    out = tmp_path / "mosaic.tif"

    fetch_clip([str(left), str(right)], (10.5, 50.0, 11.5, 51.0), str(out))

    ds = gdal.Open(str(out))
    width_deg = ds.RasterXSize * ds.GetGeoTransform()[1]
    assert 0.9 < width_deg < 1.1  # spans across the seam of both tiles
    ds = None


def test_to_vsicurl_passthrough_local():
    assert to_vsicurl("/tmp/x.tif") == "/tmp/x.tif"
