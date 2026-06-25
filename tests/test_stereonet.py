"""Unit tests for the stereonet / structural-analysis math (planesight-8et).

Validates the geological CONVENTION (RHR strike, downward pole, lower-hemisphere
equal-area) against plane_fit, a TEXTBOOK Fisher example (PmagPy), and fold-axis
recovery from synthetic bedding rotated about a KNOWN axis - plus graceful
degenerate handling. Pure numpy; no DEM, no QGIS.
"""

import math

import numpy as np
import pytest

from planesight.core.attitude.plane_fit import _fit_core, fit_plane
from planesight.core.structural import (
    axial_mean,
    equal_area_xy,
    fisher_mean,
    fold_axis,
    great_circle,
    line_to_xyz,
    orientation_tensor,
    pole_to_strike_dip,
    principal_orientations,
    rose_bins,
    strike_dip_to_pole,
    xyz_to_line,
)

# --- helpers -----------------------------------------------------------------


def _ang_diff(a, b):
    """Smallest absolute difference between two azimuths (degrees)."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _axis_angle(u, v):
    """Acute angle (deg) between two axes (sign-insensitive)."""
    u = np.asarray(u, float) / np.linalg.norm(u)
    v = np.asarray(v, float) / np.linalg.norm(v)
    return math.degrees(math.acos(min(1.0, abs(float(u @ v)))))


def _rodrigues(v, axis, angle_deg):
    """Rotate vector(s) v about a unit axis by angle (Rodrigues' formula).

    Preserves input shape: a ``(3,)`` vector returns ``(3,)``; ``(N, 3)`` -> ``(N, 3)``.
    """
    a = axis / np.linalg.norm(axis)
    th = math.radians(angle_deg)
    v = np.asarray(v, float)
    dot = np.tensordot(v, a, axes=([-1], [0]))[..., None]  # (...,1)
    return (v * math.cos(th)
            + np.cross(a, v) * math.sin(th)
            + a * dot * (1 - math.cos(th)))


# --- trend/plunge <-> xyz ----------------------------------------------------


def test_line_to_xyz_cardinals():
    # straight down -> nadir
    assert np.allclose(line_to_xyz(0.0, 90.0), [0, 0, -1])
    # horizontal North / East (plunge 0) on the rim plane
    assert np.allclose(line_to_xyz(0.0, 0.0), [0, 1, 0])     # North
    assert np.allclose(line_to_xyz(90.0, 0.0), [1, 0, 0])    # East
    # downward line has z <= 0
    assert line_to_xyz(45.0, 30.0)[2] < 0


def test_trend_plunge_roundtrip():
    for trend in range(0, 360, 17):
        for plunge in range(0, 91, 13):
            v = line_to_xyz(trend, plunge)
            t2, p2 = xyz_to_line(v, lower_hemisphere=True)
            assert abs(p2 - plunge) < 1e-9
            if plunge < 90 - 1e-9:  # trend undefined when vertical
                assert _ang_diff(t2, trend) < 1e-7


def test_xyz_to_line_treats_upward_as_axis():
    # an upward vector is flipped to its downward antipode
    up = line_to_xyz(40.0, 50.0) * -1.0       # now points up
    t, p = xyz_to_line(up, lower_hemisphere=True)
    assert abs(p - 50.0) < 1e-9
    assert _ang_diff(t, 40.0) < 1e-7


# --- strike/dip <-> pole, and agreement with plane_fit -----------------------


def test_strike_dip_pole_roundtrip():
    for strike in range(0, 360, 23):
        for dip in range(1, 90, 11):  # avoid dip 0 (strike undefined)
            pole = strike_dip_to_pole(strike, dip)
            assert pole[2] <= 1e-12  # downward (lower hemisphere)
            s2, d2 = pole_to_strike_dip(pole)
            assert abs(d2 - dip) < 1e-9
            assert _ang_diff(s2, strike) < 1e-7


def test_pole_matches_plane_fit_normal():
    # Our downward pole must be exactly the NEGATED plane_fit upward normal,
    # i.e. the same plane orientation - guards against a mirrored convention.
    rng = np.random.default_rng(0)
    for _ in range(20):
        dip = float(rng.uniform(5, 85))
        dipdir = float(rng.uniform(0, 360))
        strike = (dipdir - 90.0) % 360.0
        # synthetic planar points -> plane_fit upward normal
        a = math.radians(dipdir)
        d = math.radians(dip)
        normal_up = np.array([math.sin(d) * math.sin(a),
                              math.sin(d) * math.cos(a), math.cos(d)])
        # build 3 non-collinear in-plane points
        u = np.cross(normal_up, [0, 0, 1.0])
        u /= np.linalg.norm(u)
        w = np.cross(normal_up, u)
        pts = np.array([[0, 0, 0], 3 * u, 2 * u + 4 * w], float) + [10, 20, 5]
        _, _, _, n_fit, _, _ = _fit_core(pts)
        pole = strike_dip_to_pole(strike, dip)
        assert np.allclose(pole, -n_fit, atol=1e-9)


def test_pole_points_opposite_to_dip_direction():
    # Plane dipping due EAST (strike 0, RHR -> dips east, dip_direction 90):
    # its pole must plunge toward the WEST (trend 270), NOT east. This is the
    # geological check a naive round-trip would pass even if mirrored.
    pole = strike_dip_to_pole(0.0, 30.0)
    trend, plunge = xyz_to_line(pole, lower_hemisphere=True)
    assert _ang_diff(trend, 270.0) < 1e-7
    assert abs(plunge - (90.0 - 30.0)) < 1e-7


# --- equal-area projection ---------------------------------------------------


def test_equal_area_centre_and_rim_exact():
    # vertical line -> disk centre exactly
    for trend in (0.0, 137.0, 311.0):
        xy = equal_area_xy(trend, 90.0)
        assert np.array_equal(xy, np.array([0.0, 0.0]))
    # horizontal line -> rim exactly (r == 1)
    for trend in (0.0, 45.0, 200.0):
        xy = equal_area_xy(trend, 0.0)
        assert math.isclose(float(np.hypot(*xy)), 1.0, rel_tol=0, abs_tol=1e-12)


def test_equal_area_is_area_true():
    # Equal-area: equal solid-angle annuli map to equal disk area. Check the
    # 0-45 deg plunge band and 45-90 band have radii partitioning area equally.
    r45 = float(np.hypot(*equal_area_xy(0.0, 45.0)))
    # solid angle from nadir to colatitude th is proportional to (1 - cos th);
    # plunge 45 is the half-area colatitude only for area measure -> r^2 = 0.5*?
    # Direct identity check instead: r^2 == 1 - sin(plunge).
    for plunge in (10.0, 33.0, 71.0):
        r2 = float(np.hypot(*equal_area_xy(0.0, plunge))) ** 2
        assert math.isclose(r2, 1.0 - math.sin(math.radians(plunge)), abs_tol=1e-12)
    assert 0 < r45 < 1


def test_equal_area_north_is_up():
    # A horizontal line trending North projects to +y; East to +x.
    assert np.allclose(equal_area_xy(0.0, 0.0), [0.0, 1.0], atol=1e-12)
    assert np.allclose(equal_area_xy(90.0, 0.0), [1.0, 0.0], atol=1e-12)


def test_great_circle_geometry():
    gc = great_circle(0.0, 40.0, n=181)
    assert gc.shape == (181, 2)
    # all points within the unit disk
    assert np.all(np.hypot(gc[:, 0], gc[:, 1]) <= 1.0 + 1e-9)
    # endpoints are the strike line on the rim (strike 0 -> N-S diameter ends)
    assert math.isclose(float(np.hypot(*gc[0])), 1.0, abs_tol=1e-9)
    assert math.isclose(float(np.hypot(*gc[-1])), 1.0, abs_tol=1e-9)
    # the midpoint is the down-dip line: plunge == dip, toward dip_direction (E)
    mid = gc[90]
    # recover plunge from radius: r^2 = 1 - sin(plunge)
    r_mid = float(np.hypot(*mid))
    plunge_mid = math.degrees(math.asin(1.0 - r_mid ** 2))
    assert abs(plunge_mid - 40.0) < 1e-6
    assert mid[0] > 0  # down-dip points East


# --- Fisher statistics -------------------------------------------------------


def test_fisher_textbook_pmagpy():
    """Reproduce the PmagPy ``fisher_mean`` worked example.

    Source: PmagPy documentation, "Fisher mean, a95" example
    (https://pmagpy.github.io/PmagPy-docs). Input dec/inc lists give
    dec=136.31, inc=21.35, n=4, r=3.98121, k=159.69, alpha95=7.29, csd=6.41.
    """
    decs = [140.0, 127.0, 142.0, 136.0]
    incs = [21.0, 23.0, 19.0, 22.0]
    # dec == trend, inc == plunge (downward) in our frame
    vecs = np.array([line_to_xyz(d, i) for d, i in zip(decs, incs)])
    f = fisher_mean(vecs)
    assert f.n == 4
    assert _ang_diff(f.trend, 136.30838974272072) < 1e-2
    assert abs(f.plunge - 21.347784026899987) < 1e-2
    assert abs(f.r - 3.9812138971889026) < 1e-4
    assert abs(f.kappa - 159.69251473636305) < 1e-1
    assert abs(f.alpha95 - 7.292891411309177) < 1e-2
    assert abs(f.csd - 6.4097743211340896) < 1e-2


def test_fisher_recovers_known_direction():
    rng = np.random.default_rng(42)
    true = line_to_xyz(57.0, 34.0)
    # tight cluster around `true`
    vecs = true + 0.02 * rng.standard_normal((200, 3))
    f = fisher_mean(vecs)
    assert _axis_angle(f.mean_vec, true) < 1.0
    assert f.kappa > 100


def test_fisher_alpha95_shrinks_with_n_and_concentration():
    rng = np.random.default_rng(1)
    true = line_to_xyz(120.0, 40.0)

    def a95(n, spread):
        v = true + spread * rng.standard_normal((n, 3))
        return fisher_mean(v).alpha95

    # more samples -> smaller cone
    assert a95(400, 0.1) < a95(25, 0.1)
    # tighter concentration -> smaller cone (same n)
    assert a95(100, 0.05) < a95(100, 0.3)


def test_fisher_degenerate():
    nan_single = fisher_mean(line_to_xyz(10.0, 20.0)[None, :])
    assert nan_single.n == 1
    assert math.isnan(nan_single.kappa)
    assert not math.isnan(nan_single.trend)  # mean is the vector itself
    # antipodal pair -> fully dispersed, no mean direction (no crash)
    v = np.array([line_to_xyz(0.0, 10.0), -line_to_xyz(0.0, 10.0)])
    f = fisher_mean(v)
    assert math.isnan(f.trend)
    # empty / all-NaN input
    assert fisher_mean(np.full((3, 3), np.nan)).n == 0


# --- orientation tensor / eigen-analysis -------------------------------------


def test_orientation_tensor_cluster_vs_girdle():
    rng = np.random.default_rng(7)
    # cluster: all vectors near one direction -> S1 ~ 1, K > 1
    clust = line_to_xyz(30.0, 50.0) + 0.05 * rng.standard_normal((300, 3))
    pc = principal_orientations(clust)
    assert math.isclose(float(pc.eigenvalues.sum()), 1.0, abs_tol=1e-9)
    assert pc.eigenvalues[0] > 0.95
    assert pc.woodcock_k > 1.0
    # girdle: vectors spread in a plane -> S1 ~ S2 >> S3, K < 1
    th = np.linspace(0, 2 * np.pi, 200)
    girdle = np.stack([np.cos(th), np.sin(th), 0.01 * rng.standard_normal(200)], 1)
    pg = principal_orientations(girdle)
    assert pg.eigenvalues[2] < 0.05
    assert pg.woodcock_k < 1.0


def test_orientation_tensor_symmetry_and_trace():
    v = line_to_xyz([10.0, 80.0, 200.0], [20.0, 40.0, 60.0])
    t = orientation_tensor(v)
    assert np.allclose(t, t.T)
    assert math.isclose(np.trace(t), 3.0)  # 3 unit vectors
    tn = orientation_tensor(v, normalize=True)
    assert math.isclose(np.trace(tn), 1.0)


# --- axial mean --------------------------------------------------------------


def test_axial_mean_sign_invariant():
    rng = np.random.default_rng(3)
    true = line_to_xyz(75.0, 25.0)
    vecs = true + 0.04 * rng.standard_normal((150, 3))
    am1 = axial_mean(vecs)
    # flip half the vectors to their antipodes: axial mean must be UNCHANGED
    flipped = vecs.copy()
    flipped[::2] *= -1.0
    am2 = axial_mean(flipped)
    assert _axis_angle(am1.vector, am2.vector) < 1e-6
    assert _axis_angle(am1.vector, true) < 2.0
    assert am1.concentration > 0.9


def test_axial_mean_vs_fisher_on_mixed_hemisphere():
    # A directed Fisher mean is WRONG on mixed-hemisphere axial data; the axial
    # mean is right. Build an axis cluster, flip half -> Fisher resultant
    # partially cancels, axial mean does not.
    rng = np.random.default_rng(9)
    true = line_to_xyz(10.0, 5.0)  # shallow -> antipodes nearly cancel in Fisher
    vecs = true + 0.03 * rng.standard_normal((100, 3))
    vecs[::2] *= -1.0
    am = axial_mean(vecs)
    assert _axis_angle(am.vector, true) < 3.0


# --- fold axis ---------------------------------------------------------------


def test_fold_axis_recovers_known_axis():
    # Synthesize cylindrically folded bedding: poles lie perpendicular to a
    # KNOWN fold axis. Rotate a starting pole about the axis -> a girdle whose
    # pole is the fold axis. Recover it.
    axis = line_to_xyz(75.0, 15.0)            # known beta: trend 75, plunge 15
    axis /= np.linalg.norm(axis)
    # a starting bedding pole perpendicular to the axis
    p0 = np.cross(axis, [0, 0, 1.0])
    p0 /= np.linalg.norm(p0)
    angles = np.linspace(-70, 70, 25)         # limbs of an open fold
    poles = np.array([_rodrigues(p0, axis, a) for a in angles])
    strikes_dips = [pole_to_strike_dip(p) for p in poles]
    strikes = [sd[0] for sd in strikes_dips]
    dips = [sd[1] for sd in strikes_dips]
    fa = fold_axis(strikes, dips)
    beta = line_to_xyz(fa.trend, fa.plunge)
    assert _axis_angle(beta, axis) < 1.0
    assert fa.classification == "girdle"
    assert fa.is_girdle


def test_fold_axis_flags_cluster_not_fold():
    # Bedding all near-identical -> poles are a CLUSTER, not a girdle. The
    # shape statistic must refuse to call this a fold.
    rng = np.random.default_rng(11)
    strikes = 40.0 + 2.0 * rng.standard_normal(30)
    dips = 30.0 + 2.0 * rng.standard_normal(30)
    fa = fold_axis(strikes, dips)
    assert fa.classification == "cluster"
    assert not fa.is_girdle


def test_fold_axis_too_few_points():
    fa = fold_axis([10.0, 20.0], [30.0, 40.0])
    assert fa.n == 2
    assert not fa.is_girdle
    assert math.isnan(fa.trend)


# --- rose diagram ------------------------------------------------------------


def test_rose_bins_bidirectional_symmetry():
    strikes = [5.0, 7.0, 95.0, 182.0, 350.0]
    counts, edges = rose_bins(strikes, bin_deg=10.0)
    assert counts.shape == (36,)
    assert edges.shape == (37,)
    assert counts.sum() == 2 * len(strikes)  # each strike counted both ways
    half = 18
    assert np.array_equal(counts[:half], counts[half:])  # diametric symmetry


def test_rose_bins_validation_and_empty():
    with pytest.raises(ValueError):
        rose_bins([10.0], bin_deg=7.0)  # 7 does not divide 360
    counts, edges = rose_bins([], bin_deg=10.0)
    assert counts.sum() == 0
    counts2, _ = rose_bins([np.nan, np.nan], bin_deg=10.0)
    assert counts2.sum() == 0


# --- integration with plane_fit Attitude -------------------------------------


def test_attitude_pole_pipeline():
    # An Attitude produced by fit_plane should flow through our pole math and
    # round-trip back to the same strike/dip.
    normal_up = np.array([math.sin(math.radians(35)) * math.sin(math.radians(120)),
                          math.sin(math.radians(35)) * math.cos(math.radians(120)),
                          math.cos(math.radians(35))])
    u = np.cross(normal_up, [0, 0, 1.0])
    u /= np.linalg.norm(u)
    w = np.cross(normal_up, u)
    t = np.linspace(-1, 1, 12)
    pts = (np.outer(t, u) + np.outer(0.3 * np.sin(3 * t), w)) * 100 + [5, 6, 7]
    att = fit_plane(pts)
    pole = strike_dip_to_pole(att.strike, att.dip)
    s2, d2 = pole_to_strike_dip(pole)
    assert _ang_diff(s2, att.strike) < 1e-6
    assert abs(d2 - att.dip) < 1e-6
