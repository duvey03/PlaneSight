"""Tests for positive-unlabeled detection scoring (pure numpy/scipy)."""

import numpy as np
import pytest

from planesight.core.detect import (
    detect_at_budget,
    disk,
    recall_at_budget,
    recall_curve,
)


def test_disk_shapes_and_membership():
    assert disk(0).shape == (1, 1) and disk(0)[0, 0]
    d = disk(2)
    assert d.shape == (5, 5)
    assert d[2, 2] and d[0, 2] and not d[0, 0]  # centre + axis in, corner out
    with pytest.raises(ValueError):
        disk(-1)


def test_detect_at_budget_flags_expected_fraction():
    resp = np.linspace(0.0, 1.0, 100).reshape(10, 10)
    det = detect_at_budget(resp, budget=0.1)
    # ~10% of 100 valid pixels fire (top decile)
    assert 9 <= det.sum() <= 12
    # the highest-response pixels are the ones flagged
    assert det.flat[-1] and not det.flat[0]


def test_detect_at_budget_validates_budget():
    resp = np.ones((4, 4))
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            detect_at_budget(resp, budget=bad)


def test_detect_at_budget_all_invalid_returns_empty():
    resp = np.full((5, 5), np.nan)
    assert detect_at_budget(resp, budget=0.5).sum() == 0


def test_perfect_response_recovers_all_truth():
    # Response peaks exactly on the truth pixels -> recall 1.0 at a tiny budget.
    resp = np.zeros((20, 20))
    truth = np.zeros((20, 20), dtype=bool)
    truth[5, :] = True  # a horizontal contact line (20 px)
    resp[truth] = 1.0
    r = recall_at_budget(resp, truth, budget=0.05, tolerance_px=0)
    assert r == 1.0


def test_tolerance_buffer_recovers_near_misses():
    # Response fires one pixel off the truth line; tolerance should bridge it.
    resp = np.zeros((20, 20))
    truth = np.zeros((20, 20), dtype=bool)
    truth[10, :] = True
    resp[11, :] = 1.0  # detected one row below the true line (20 of 400 px = 5%)
    assert recall_at_budget(resp, truth, budget=0.05, tolerance_px=0) == 0.0
    assert recall_at_budget(resp, truth, budget=0.05, tolerance_px=1) == 1.0


def test_uninformative_response_recovers_little_at_small_budget():
    rng = np.random.default_rng(0)
    resp = rng.random((50, 50))  # noise, unrelated to truth
    truth = np.zeros((50, 50), dtype=bool)
    truth[25, :] = True
    r = recall_at_budget(resp, truth, budget=0.05, tolerance_px=1)
    # random response at a 5% budget should recover only a small fraction
    assert r < 0.5


def test_recall_is_nan_without_truth():
    resp = np.random.default_rng(1).random((10, 10))
    truth = np.zeros((10, 10), dtype=bool)
    assert np.isnan(recall_at_budget(resp, truth, budget=0.1))


def test_truth_outside_valid_mask_is_excluded():
    resp = np.zeros((10, 10))
    truth = np.zeros((10, 10), dtype=bool)
    truth[0, :] = True
    valid = np.ones((10, 10), dtype=bool)
    valid[0, :] = False  # the only truth row is masked out -> no scorable truth
    assert np.isnan(recall_at_budget(resp, truth, budget=0.2, valid_mask=valid))


def test_recall_is_monotonic_nondecreasing_in_budget():
    rng = np.random.default_rng(2)
    resp = rng.random((40, 40))
    truth = np.zeros((40, 40), dtype=bool)
    truth[20, 5:35] = True
    curve = recall_curve(resp, truth, budgets=(0.01, 0.05, 0.1, 0.3, 0.6),
                         tolerance_px=1)
    recalls = [r for _, r in curve]
    assert all(b <= a + 1e-9 for b, a in zip(recalls, recalls[1:]))
    assert recalls[-1] >= recalls[0]


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        recall_at_budget(np.zeros((4, 4)), np.zeros((4, 5), dtype=bool))
