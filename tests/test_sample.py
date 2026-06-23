"""Tests for DEM sampling primitives (pure numpy, no GDAL)."""

import numpy as np
import pytest

from planesight.core.attitude import (
    densify_line,
    fit_plane,
    sample_bilinear,
    sample_trace,
)


def test_densify_spacing_and_endpoints():
    line = [(0.0, 0.0), (1000.0, 0.0)]
    dense = densify_line(line, spacing=100.0)
    assert len(dense) == 11  # 0..1000 every 100
    assert np.allclose(dense[0], [0, 0]) and np.allclose(dense[-1], [1000, 0])
    steps = np.diff(dense[:, 0])
    assert np.allclose(steps, 100.0)


def test_densify_requires_two_points():
    with pytest.raises(ValueError):
        densify_line([(0, 0)], spacing=10)


# A north-up geotransform: top-left at (1000, 5000), 10 m pixels.
GT = (1000.0, 10.0, 0.0, 5000.0, 0.0, -10.0)


def _linear_dem(a, b, c, nrows=50, ncols=60):
    """Raster of z = a*x + b*y + c sampled at pixel centres of GT."""
    j, i = np.meshgrid(np.arange(ncols), np.arange(nrows))
    x = GT[0] + (j + 0.5) * GT[1]
    y = GT[3] + (i + 0.5) * GT[5]
    return (a * x + b * y + c).astype(float)


def test_bilinear_is_exact_on_a_linear_field():
    a, b, c = 0.03, -0.05, 120.0
    arr = _linear_dem(a, b, c)
    # query arbitrary interior world points
    qx = np.array([1100.0, 1234.5, 1450.0])
    qy = np.array([4950.0, 4880.0, 4700.0])
    got = sample_bilinear(arr, GT, np.column_stack([qx, qy]))
    expected = a * qx + b * qy + c
    assert np.allclose(got, expected, atol=1e-6)


def test_bilinear_out_of_bounds_is_nan():
    arr = _linear_dem(0.01, 0.01, 0.0)
    got = sample_bilinear(arr, GT, [(10.0, 10.0), (99999.0, 0.0)])
    assert np.all(np.isnan(got))


def test_sample_trace_then_fit_recovers_known_attitude():
    # Build a DEM that IS a plane of known dip/dip-direction, sample a trace
    # across it, and confirm the full sample->fit pipeline recovers the attitude.
    import math

    dip, dipdir = 22.0, 90.0  # dipping due East
    # plane z = -tan(dip) * (downslope distance). For east dip, z decreases with x.
    slope = math.tan(math.radians(dip))
    a = -slope  # dz/dx (east is downhill)
    arr = _linear_dem(a, 0.0, 3000.0)
    # a sinuous trace across the raster (in world coords within bounds)
    t = np.linspace(0, 540, 30)
    trace = np.column_stack([1050 + t, 4950 - 0.0 * t - 200 * np.sin(t / 120.0)])
    pts3d = sample_trace(trace, arr, GT, spacing=10.0)
    assert len(pts3d) > 20
    att = fit_plane(pts3d)
    assert abs(att.dip - dip) < 0.5
    assert min(abs(att.dip_direction - dipdir), 360 - abs(att.dip_direction - dipdir)) < 1.0
