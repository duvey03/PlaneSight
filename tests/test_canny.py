"""Tests for the Canny edge front-end (pure numpy/scipy)."""

import numpy as np

from planesight.core.detect import (
    canny,
    gaussian_gradient,
    hysteresis,
    non_max_suppression,
)


def _step_edge(n=40, col=20):
    img = np.zeros((n, n))
    img[:, col:] = 1.0
    return img


def test_gradient_points_across_a_step_edge():
    gx, gy = gaussian_gradient(_step_edge(), sigma=1.0)
    # a vertical edge: gradient is in x (cols), ~none in y near the edge
    assert np.abs(gx[20, 20]) > 0.1
    assert np.abs(gy[20, 20]) < 1e-3


def test_nms_thins_a_blurred_edge_to_one_pixel():
    img = _step_edge()
    gx, gy = gaussian_gradient(img, sigma=2.0)
    mag = np.hypot(gx, gy)
    nms = non_max_suppression(mag, gx, gy)
    # along a row, the thick gradient ridge collapses to ~1 nonzero column
    row = nms[20, 10:30]
    assert (row > 1e-6).sum() <= 2
    assert mag[20, 10:30].max() > 0  # there was a thick ridge to begin with


def test_hysteresis_links_weak_to_strong():
    nms = np.zeros((10, 10))
    nms[5, 2] = 1.0        # strong
    nms[5, 3:7] = 0.4      # weak, connected to the strong pixel
    nms[0, 0] = 0.4        # weak, isolated -> dropped
    edges = hysteresis(nms, low=0.3, high=0.9)
    assert edges[5, 2] and edges[5, 5]      # strong + connected weak kept
    assert not edges[0, 0]                   # isolated weak dropped


def test_canny_detects_a_clean_edge():
    edges = canny(_step_edge(), sigma=1.5)
    assert edges.any()
    # edges land near the step (column ~20), thin
    cols = np.flatnonzero(edges.any(axis=0))
    assert cols.min() >= 17 and cols.max() <= 23
    assert edges.sum(axis=1).max() <= 2  # ~1-px thick per row


def test_canny_blank_image_returns_no_edges():
    assert not canny(np.full((20, 20), 0.5)).any()


def test_canny_respects_valid_mask_and_nan():
    img = _step_edge()
    img[0:5, :] = np.nan          # invalid band of rows
    valid = np.isfinite(img)
    edges = canny(img, sigma=1.0, valid_mask=valid)
    assert not edges[0:5, :].any()  # no edges in the invalid region


def test_canny_validates_quantiles():
    import pytest
    with pytest.raises(ValueError):
        canny(_step_edge(), low_quantile=0.9, high_quantile=0.5)
