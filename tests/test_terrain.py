"""Tests for DEM terrain derivatives (pure numpy/scipy, no GDAL).

Each derivative is checked against an analytic surface of known geometry - the
same exact-on-synthetic discipline used for the plane-fit and sampler.
"""

import math

import numpy as np
import pytest

from planesight.core.derivatives import terrain

PX = 10.0  # metres


def _plane(a, b, c=100.0, nrows=40, ncols=50, px=PX):
    """z = a*X + b*Y + c on a north-up grid (row 0 = north, X east, Y north)."""
    j, i = np.meshgrid(np.arange(ncols), np.arange(nrows))
    x = j * px
    y = (nrows - 1 - i) * px
    return (a * x + b * y + c).astype(float)


def _interior(arr, m=2):
    return arr[m:-m, m:-m]


def test_gradients_exact_on_a_plane():
    a, b = 0.3, -0.2
    fx, fy = terrain.gradients(_plane(a, b), PX)
    assert np.allclose(_interior(fx), a, atol=1e-9)
    assert np.allclose(_interior(fy), b, atol=1e-9)


def test_slope_exact_on_a_plane():
    a, b = 0.1, 0.0
    expected = math.degrees(math.atan(math.hypot(a, b)))
    s = terrain.slope(_plane(a, b), PX)
    assert np.allclose(_interior(s), expected, atol=1e-6)


def test_slope_zero_on_flat():
    flat = np.full((20, 20), 42.0)
    assert np.allclose(terrain.slope(flat, PX), 0.0)


def test_aspect_points_downhill():
    # z decreases toward east (a < 0) -> water flows east -> aspect ~ 90 deg.
    asp = terrain.aspect(_plane(-0.2, 0.0), PX)
    assert np.allclose(_interior(asp), 90.0, atol=1e-6)
    # z increasing toward north (b > 0) -> downhill is south -> aspect ~ 180.
    asp2 = terrain.aspect(_plane(0.0, 0.2), PX)
    assert np.allclose(_interior(asp2), 180.0, atol=1e-6)


def test_hillshade_flat_equals_sin_altitude():
    flat = np.full((15, 15), 5.0)
    hs = terrain.hillshade(flat, PX, altitude=30.0)
    assert np.allclose(hs, math.sin(math.radians(30.0)), atol=1e-9)


def test_hillshade_in_unit_range():
    rng = np.random.default_rng(0)
    dem = np.cumsum(rng.normal(size=(30, 30)), axis=0) * 5.0
    hs = terrain.hillshade(dem, PX)
    assert hs.min() >= 0.0 and hs.max() <= 1.0


def test_multi_hillshade_is_mean_of_components():
    rng = np.random.default_rng(1)
    dem = np.cumsum(rng.normal(size=(25, 25)), axis=1) * 4.0
    azimuths = (315.0, 45.0)
    mh = terrain.multi_hillshade(dem, PX, azimuths=azimuths)
    manual = sum(terrain.hillshade(dem, PX, azimuth=a) for a in azimuths) / 2.0
    assert np.allclose(mh, manual)


def test_tpi_zero_on_planar_slope():
    # A linear surface has no local relief: TPI ~ 0 away from the padded edge.
    tp = terrain.tpi(_plane(0.2, -0.1), radius=3)
    assert np.allclose(_interior(tp, m=4), 0.0, atol=1e-9)


def test_tpi_positive_on_a_bump():
    dem = np.zeros((31, 31))
    dem[15, 15] = 100.0  # an isolated peak
    tp = terrain.tpi(dem, radius=5)
    assert tp[15, 15] > 0.0  # peak sits above its neighbourhood mean


def test_curvature_zero_on_a_plane():
    dem = _plane(0.3, -0.2)
    for kind in ("total", "plan", "profile"):
        cur = terrain.curvature(dem, PX, kind=kind)
        assert np.allclose(_interior(cur), 0.0, atol=1e-9)


def test_total_curvature_matches_paraboloid():
    # z = k*(X^2 + Y^2): Laplacian = 4k, so total curvature = -2*(d+e) = -4k.
    k = 1e-3
    j, i = np.meshgrid(np.arange(40), np.arange(40))
    x = j * PX
    y = (40 - 1 - i) * PX
    dem = k * (x * x + y * y)
    cur = terrain.curvature(dem, PX, kind="total")
    assert np.allclose(_interior(cur), -4.0 * k, atol=1e-9)


def test_nan_propagates_locally_but_edges_survive():
    dem = _plane(0.1, 0.1)
    dem[20, 25] = np.nan
    s = terrain.slope(dem, PX)
    assert np.isnan(s[20, 25])
    assert np.isnan(s[19, 25])  # neighbour touched by the nodata cell
    assert np.isfinite(s[0, 0])  # replicated border, not NaN


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        terrain.gradients(np.zeros((5, 5)), px=0.0)
    with pytest.raises(ValueError):
        terrain.tpi(np.zeros((5, 5)), radius=0)
    with pytest.raises(ValueError):
        terrain.curvature(np.zeros((5, 5)), PX, kind="bogus")
    with pytest.raises(ValueError):
        terrain.gradients(np.zeros((5, 5, 5)), PX)
