"""Unit tests for the local-variability + morphology helpers (planesight-5ug).

Pure numpy - exercises the circular (mod-180) strike statistics, the windowed
neighbour-deviation computation, the rule-of-V's morphology classifier, and the
polyline length. No DEM, no GDAL.
"""

import math

import numpy as np
import pytest

from planesight.core.attitude import (
    CONTOUR_PARALLEL,
    STRAIGHT,
    V,
    circular_mean_strike,
    circular_resultant_length,
    classify_morphology,
    polyline_length,
    strike_difference,
    windowed_deviations,
)

# --- strike_difference (undirected, mod 180) --------------------------------

def test_strike_difference_wraps_across_zero():
    # 175 and 5 are 10 deg apart, NOT 170 (strike is a line, not a ray).
    assert strike_difference(175.0, 5.0) == pytest.approx(10.0)
    assert strike_difference(5.0, 175.0) == pytest.approx(10.0)


def test_strike_difference_caps_at_90():
    # perpendicular strikes are the maximum possible separation.
    assert strike_difference(0.0, 90.0) == pytest.approx(90.0)
    assert strike_difference(10.0, 100.0) == pytest.approx(90.0)


def test_strike_difference_identity_and_180_periodicity():
    assert strike_difference(40.0, 40.0) == pytest.approx(0.0)
    assert strike_difference(40.0, 220.0) == pytest.approx(0.0)  # 220 == 40 mod 180


def test_strike_difference_broadcasts():
    out = strike_difference(np.array([10.0, 170.0]), np.array([20.0, 0.0]))
    np.testing.assert_allclose(out, [10.0, 10.0])


# --- circular_mean_strike (doubled-angle) -----------------------------------

def test_circular_mean_strike_wraps_correctly():
    # mean of 170 and 10 should be 0/180, not 90.
    m = circular_mean_strike([170.0, 10.0])
    assert strike_difference(m, 0.0) == pytest.approx(0.0, abs=1e-6)


def test_circular_mean_strike_plain_cluster():
    assert circular_mean_strike([40.0, 50.0, 60.0]) == pytest.approx(50.0, abs=1e-6)


def test_circular_mean_strike_empty_is_nan():
    assert math.isnan(circular_mean_strike([]))


def test_circular_mean_strike_opposed_is_nan():
    # 0 and 90 doubled are antipodal -> zero resultant -> undefined.
    assert math.isnan(circular_mean_strike([0.0, 90.0]))


# --- circular_resultant_length ----------------------------------------------

def test_resultant_length_identical_is_one():
    assert circular_resultant_length([30.0, 30.0, 30.0]) == pytest.approx(1.0)


def test_resultant_length_opposed_is_zero():
    assert circular_resultant_length([0.0, 90.0]) == pytest.approx(0.0, abs=1e-12)


# --- windowed_deviations ----------------------------------------------------

def test_windowed_deviation_flags_local_outlier():
    # four near-coincident points: three strike ~10, one strike 80 (the outlier).
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    strikes = np.array([10.0, 12.0, 8.0, 80.0])
    dips = np.array([30.0, 31.0, 29.0, 75.0])
    sdev, ddev, nbr = windowed_deviations(xy, strikes, dips, radius=10.0)
    # all points see the other three as neighbours
    assert (nbr == 3).all()
    # the outlier has by far the largest strike + dip deviation
    assert np.argmax(sdev) == 3
    assert np.argmax(ddev) == 3
    # neighbour mean strike of point 3 ~10 -> deviation ~70
    expected = strike_difference(80.0, circular_mean_strike([10.0, 12.0, 8.0]))
    assert sdev[3] == pytest.approx(expected)


def test_windowed_deviation_respects_radius():
    # two tight clusters far apart; with a small radius each point sees only its
    # own cluster, so deviations stay tiny despite the clusters differing.
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [1000.0, 0.0], [1001.0, 0.0]])
    strikes = np.array([10.0, 12.0, 80.0, 82.0])
    dips = np.array([20.0, 21.0, 70.0, 71.0])
    sdev, ddev, nbr = windowed_deviations(xy, strikes, dips, radius=10.0, min_neighbors=1)
    assert (nbr == 1).all()
    assert np.nanmax(sdev) < 5.0


def test_windowed_deviation_min_neighbors_gives_nan():
    xy = np.array([[0.0, 0.0], [1000.0, 0.0]])
    strikes = np.array([10.0, 80.0])
    dips = np.array([20.0, 70.0])
    sdev, ddev, nbr = windowed_deviations(xy, strikes, dips, radius=10.0, min_neighbors=2)
    assert (nbr == 0).all()
    assert np.isnan(sdev).all() and np.isnan(ddev).all()


def test_windowed_deviation_strike_uses_circular_mean():
    # neighbours straddle 0/180; a point at 0 should read ~0 deviation, not ~90.
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    strikes = np.array([0.0, 178.0, 2.0])
    dips = np.array([45.0, 45.0, 45.0])
    sdev, _, _ = windowed_deviations(xy, strikes, dips, radius=10.0)
    assert sdev[0] == pytest.approx(0.0, abs=1.0)


def test_windowed_deviation_validates_shapes():
    with pytest.raises(ValueError):
        windowed_deviations(np.zeros((3, 3)), [1, 2, 3], [1, 2, 3], radius=1.0)
    with pytest.raises(ValueError):
        windowed_deviations(np.zeros((3, 2)), [1, 2], [1, 2, 3], radius=1.0)
    with pytest.raises(ValueError):
        windowed_deviations(np.zeros((3, 2)), [1, 2, 3], [1, 2, 3], radius=0.0)


# --- classify_morphology ----------------------------------------------------

def test_classify_morphology_classes():
    assert classify_morphology(85.0) == STRAIGHT
    assert classify_morphology(5.0) == CONTOUR_PARALLEL
    assert classify_morphology(45.0) == V


def test_classify_morphology_boundaries_inclusive():
    # boundaries belong to the extreme classes.
    assert classify_morphology(75.0) == STRAIGHT
    assert classify_morphology(20.0) == CONTOUR_PARALLEL
    assert classify_morphology(74.9) == V
    assert classify_morphology(20.1) == V


def test_classify_morphology_custom_thresholds():
    assert classify_morphology(60.0, vertical_dip=55.0) == STRAIGHT
    assert classify_morphology(30.0, horizontal_dip=35.0) == CONTOUR_PARALLEL


def test_classify_morphology_nan_is_none():
    assert classify_morphology(float("nan")) is None
    assert classify_morphology(None) is None


# --- polyline_length --------------------------------------------------------

def test_polyline_length_2d():
    pts = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]])
    assert polyline_length(pts) == pytest.approx(7.0)


def test_polyline_length_ignores_z():
    # length is map-view: the z column must not change it.
    flat = np.array([[0.0, 0.0], [3.0, 4.0]])
    with_z = np.array([[0.0, 0.0, 100.0], [3.0, 4.0, 900.0]])
    assert polyline_length(with_z) == pytest.approx(polyline_length(flat))
    assert polyline_length(with_z) == pytest.approx(5.0)


def test_polyline_length_degenerate_is_zero():
    assert polyline_length(np.array([[1.0, 1.0]])) == 0.0
    assert polyline_length(np.zeros((0, 2))) == 0.0
