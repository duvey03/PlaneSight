"""Synthetic validation of the strike/dip plane-fit engine (the D14 gate).

Builds traces on planes of known attitude and verifies exact recovery, the
dip-direction downslope sign, and that the two quality metrics behave: that
conditioning catches collinear/straight traces (even at high relief) and that
planarity catches folds. Pure numpy - no DEM, no QGIS.
"""

import dataclasses
import math

import numpy as np
import pytest

from planesight.core.attitude import Attitude, fit_plane

# --- synthetic-geometry helpers ---------------------------------------------

def plane_normal(dip_deg, dipdir_deg):
    """Upward unit normal of a plane with the given dip and dip-direction."""
    d, a = math.radians(dip_deg), math.radians(dipdir_deg)
    return np.array([math.sin(d) * math.sin(a), math.sin(d) * math.cos(a), math.cos(d)])


def inplane_axes(normal):
    """Return (strike_vector, dip_vector): horizontal strike axis and down-dip axis."""
    up = np.array([0.0, 0.0, 1.0])
    strike = np.cross(normal, up)
    if np.linalg.norm(strike) < 1e-9:  # horizontal plane
        strike = np.array([1.0, 0.0, 0.0])
    strike = strike / np.linalg.norm(strike)
    dip = np.cross(normal, strike)
    return strike, dip / np.linalg.norm(dip)


def sinuous_trace(dip_deg, dipdir_deg, centroid=(1000.0, 2000.0, 300.0), n=40):
    """A 2D-spanning ('V-ing') trace lying exactly on the given plane."""
    nrm = plane_normal(dip_deg, dipdir_deg)
    s_vec, d_vec = inplane_axes(nrm)
    t = np.linspace(-500.0, 500.0, n)
    along = t                       # along strike
    across = 80.0 * np.sin(t / 120.0)  # sinuous excursion -> spans 2D
    return np.array(centroid) + np.outer(along, s_vec) + np.outer(across, d_vec)


def straight_trace(dip_deg, dipdir_deg, axis="dip", n=40):
    """A collinear (straight-in-map-view) trace on the plane.

    axis="dip" follows the down-dip line (high relief); axis="strike" the
    horizontal strike line (no relief).
    """
    nrm = plane_normal(dip_deg, dipdir_deg)
    s_vec, d_vec = inplane_axes(nrm)
    vec = d_vec if axis == "dip" else s_vec
    t = np.linspace(-500.0, 500.0, n)
    return np.array([300.0, 400.0, 200.0]) + np.outer(t, vec)


def _ang_diff(a, b):
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


# --- recovery ---------------------------------------------------------------

@pytest.mark.parametrize("dip", [10.0, 30.0, 55.0, 80.0])
@pytest.mark.parametrize("dipdir", [0.0, 45.0, 90.0, 135.0, 200.0, 300.0])
def test_exact_recovery_of_known_attitude(dip, dipdir):
    att = fit_plane(sinuous_trace(dip, dipdir))
    assert isinstance(att, Attitude)
    assert abs(att.dip - dip) < 1e-3
    assert _ang_diff(att.dip_direction, dipdir) < 1e-3
    assert _ang_diff(att.strike, (dipdir - 90.0) % 360.0) < 1e-3
    # a clean planar trace: well-conditioned and essentially planar
    assert att.conditioning > 1e-3
    assert att.planarity < 1e-6
    assert att.residual_rms < 1e-6


def test_dip_direction_points_downslope():
    # Plane dipping due East: the easternmost (max-x) point must be the lowest.
    pts = sinuous_trace(dip_deg=35.0, dipdir_deg=90.0)
    east_idx = int(np.argmax(pts[:, 0]))
    assert pts[east_idx, 2] == pytest.approx(pts[:, 2].min(), abs=1e-6)
    att = fit_plane(pts)
    assert _ang_diff(att.dip_direction, 90.0) < 1e-3
    assert _ang_diff(att.strike, 0.0) < 1e-3  # N-S strike for an E-dipping plane


# --- degeneracy: the review's key corrections -------------------------------

def test_conditioning_flags_straight_trace_even_with_high_relief():
    # A straight down-dip trace: large relief, but collinear in map view.
    pts = straight_trace(dip_deg=60.0, dipdir_deg=120.0, axis="dip")
    att = fit_plane(pts)
    assert att.relief > 100.0          # genuinely steep / high relief
    assert att.conditioning < 1e-9     # ...yet the fit is unconstrained
    # contrast: a sinuous trace on the same plane is well-conditioned
    good = fit_plane(sinuous_trace(60.0, 120.0))
    assert good.conditioning > att.conditioning


def test_conditioning_flags_flat_straight_trace():
    pts = straight_trace(dip_deg=40.0, dipdir_deg=10.0, axis="strike")
    att = fit_plane(pts)
    assert att.relief < 1e-6           # horizontal strike line: no relief
    assert att.conditioning < 1e-9


def test_planarity_flags_folded_trace():
    # Start from a planar trace, then bend it off-plane (systematic curvature).
    nrm = plane_normal(30.0, 90.0)
    s_vec, d_vec = inplane_axes(nrm)
    t = np.linspace(-500.0, 500.0, 60)
    pts = np.outer(t, s_vec) + np.outer(40.0 * np.sin(t / 120.0), d_vec)
    folded = pts + np.outer((t / 300.0) ** 2 * 50.0, nrm)  # push off the plane
    planar_att = fit_plane(pts)
    folded_att = fit_plane(folded)
    assert planar_att.planarity < 1e-6
    assert folded_att.planarity > 0.01
    assert folded_att.conditioning > 1e-3  # still spans 2D - only planarity drops


# --- noise robustness & residuals -------------------------------------------

def test_noise_robust_recovery_and_residual():
    rng = np.random.default_rng(0)
    sigma = 2.0
    pts = sinuous_trace(45.0, 90.0, n=120)
    noisy = pts + rng.normal(0.0, sigma, pts.shape)
    att = fit_plane(noisy)
    assert abs(att.dip - 45.0) < 3.0
    assert _ang_diff(att.dip_direction, 90.0) < 3.0
    assert 0.5 * sigma < att.residual_rms < 2.0 * sigma


# --- input validation -------------------------------------------------------

def test_too_few_points_raises():
    with pytest.raises(ValueError):
        fit_plane([(0, 0, 0), (1, 1, 1)])


def test_wrong_shape_raises():
    with pytest.raises(ValueError):
        fit_plane([(0, 0), (1, 1), (2, 2)])


def test_attitude_is_frozen_dataclass():
    att = fit_plane(sinuous_trace(30.0, 90.0))
    with pytest.raises(dataclasses.FrozenInstanceError):
        att.dip = 10.0  # frozen
