"""Tests for label-free linearity scoring of detection masks (pure numpy/scipy)."""

import numpy as np

from planesight.core.detect import linearity_at_budget, linearity_metrics


def test_empty_mask_scores_zero():
    m = linearity_metrics(np.zeros((30, 30), dtype=bool))
    assert m["linearity"] == 0.0 and m["n_components"] == 0


def test_straight_line_is_maximally_linear():
    mask = np.zeros((40, 40), dtype=bool)
    mask[20, 5:35] = True  # one long thin horizontal line
    m = linearity_metrics(mask)
    assert m["linearity"] > 0.95      # one variance axis, the other ~0
    assert m["n_components"] == 1


def test_diagonal_line_stays_connected_and_linear():
    mask = np.zeros((40, 40), dtype=bool)
    for i in range(5, 35):
        mask[i, i] = True  # 8-connectivity must keep this one component
    m = linearity_metrics(mask)
    assert m["n_components"] == 1
    assert m["linearity"] > 0.95


def test_blob_is_not_linear():
    y, x = np.mgrid[0:40, 0:40]
    blob = ((x - 20) ** 2 + (y - 20) ** 2) <= 7 ** 2  # filled disk
    m = linearity_metrics(blob)
    assert m["n_components"] == 1
    assert m["linearity"] < 0.3       # isotropic -> low elongation


def test_line_beats_blob():
    line = np.zeros((40, 40), dtype=bool)
    line[20, 2:38] = True
    y, x = np.mgrid[0:40, 0:40]
    blob = ((x - 20) ** 2 + (y - 20) ** 2) <= 6 ** 2
    assert linearity_metrics(line)["linearity"] > linearity_metrics(blob)["linearity"] + 0.5


def test_scattered_speckle_scores_low():
    rng = np.random.default_rng(0)
    mask = rng.random((60, 60)) < 0.03  # sparse single-pixel noise
    m = linearity_metrics(mask, min_pixels=5)
    assert m["linearity"] < 0.1        # tiny components -> not lines
    assert m["n_components"] > 10      # fragmented


def test_min_pixels_filters_short_segments():
    mask = np.zeros((20, 20), dtype=bool)
    mask[10, 5:8] = True  # a 3-pixel segment
    assert linearity_metrics(mask, min_pixels=5)["linearity"] == 0.0
    assert linearity_metrics(mask, min_pixels=2)["linearity"] > 0.9


def test_linearity_at_budget_prefers_a_linear_response():
    # A response peaked along a line yields linear detections; random noise does not.
    line = np.zeros((50, 50))
    line[25, :] = 1.0
    rng = np.random.default_rng(1)
    noise = rng.random((50, 50))
    assert linearity_at_budget(line, budget=0.03) > linearity_at_budget(noise, budget=0.03) + 0.5
