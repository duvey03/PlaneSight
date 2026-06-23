"""GDAL-dependent tests for grid co-registration (align onto a reference grid).

Auto-skipped where ``osgeo.gdal`` is unavailable (pure-Python CI job, bare WSL
shell); runs in the GDAL CI job / micromamba. Uses only local rasters - no network.
"""

import numpy as np
import pytest

gdal = pytest.importorskip("osgeo.gdal", reason="GDAL (osgeo) not installed")
gdal.UseExceptions()

from planesight.core.data import align_to_grid, read_grid  # noqa: E402


def _make_geotiff(path, bbox, nx, ny, epsg=4326, fill=None):
    from osgeo import osr

    minx, miny, maxx, maxy = bbox
    ds = gdal.GetDriverByName("GTiff").Create(str(path), nx, ny, 1, gdal.GDT_Float32)
    ds.SetGeoTransform((minx, (maxx - minx) / nx, 0, maxy, 0, -(maxy - miny) / ny))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    ds.SetProjection(srs.ExportToWkt())
    arr = (np.arange(nx * ny, dtype="float32").reshape(ny, nx) if fill is None
           else np.full((ny, nx), fill, dtype="float32"))
    ds.GetRasterBand(1).WriteArray(arr)
    ds.FlushCache()
    ds = None


def test_read_grid_reports_bounds_and_size(tmp_path):
    ref = tmp_path / "ref.tif"
    _make_geotiff(ref, (10.0, 50.0, 11.0, 51.0), nx=50, ny=40)
    grid = read_grid(str(ref))
    assert grid["width"] == 50 and grid["height"] == 40
    assert np.allclose(grid["bounds"], (10.0, 50.0, 11.0, 51.0))


def test_align_matches_reference_grid_exactly(tmp_path):
    # Reference: 50x40 over a 1-degree box. Source: a coarser, differently-sized
    # raster over a larger, offset extent. After alignment it must match the ref
    # grid pixel-for-pixel.
    ref = tmp_path / "ref.tif"
    src = tmp_path / "src.tif"
    out = tmp_path / "aligned.tif"
    _make_geotiff(ref, (10.0, 50.0, 11.0, 51.0), nx=50, ny=40)
    _make_geotiff(src, (9.5, 49.5, 11.5, 51.5), nx=20, ny=20)

    align_to_grid(str(src), str(out), str(ref))

    rg, og = read_grid(str(ref)), read_grid(str(out))
    assert og["width"] == rg["width"] and og["height"] == rg["height"]
    assert np.allclose(og["geotransform"], rg["geotransform"])
    assert np.allclose(og["bounds"], rg["bounds"])
    # output is finite where the source covered the reference (full overlap here)
    arr = gdal.Open(str(out)).ReadAsArray()
    assert np.isfinite(arr).all()


def test_align_bad_reference_raises(tmp_path):
    with pytest.raises(Exception):  # noqa: B017 - gdal raises its own error type
        read_grid(str(tmp_path / "does_not_exist.tif"))
