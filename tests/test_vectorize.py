"""Tests for the response->polylines back-end (pure numpy/scipy)."""

import numpy as np

from planesight.core.detect import (
    extract_polylines,
    pixels_to_world,
    simplify,
    thin,
    trace_skeleton,
)


def test_thin_reduces_thick_bar_to_one_pixel():
    bar = np.zeros((20, 30), dtype=bool)
    bar[9:12, 5:25] = True  # 3-px-thick horizontal bar
    skel = thin(bar)
    assert skel.sum() < bar.sum()
    # interior columns should be a single pixel thick after thinning
    interior = skel[:, 8:22].sum(axis=0)
    assert np.all(interior == 1)


def test_thin_preserves_connectivity_of_a_line():
    line = np.zeros((10, 20), dtype=bool)
    line[5, 2:18] = True  # already 1-px
    skel = thin(line)
    assert skel[5, 2:18].all()  # an existing thin line survives intact


def test_trace_single_line_gives_one_polyline():
    line = np.zeros((10, 20), dtype=bool)
    line[5, 2:18] = True
    paths = trace_skeleton(line)
    assert len(paths) == 1
    assert len(paths[0]) == 16  # pixels at cols 2..17
    # ordered end-to-end
    cols = [c for _, c in paths[0]]
    assert cols[0] in (2, 17) and cols[-1] in (2, 17)


def test_trace_cross_splits_at_junction():
    img = np.zeros((21, 21), dtype=bool)
    img[10, :] = True
    img[:, 10] = True  # a plus sign: one degree-4 junction, four arms
    paths = trace_skeleton(thin(img))
    assert len(paths) >= 4


def test_trace_drops_short_paths():
    img = np.zeros((10, 10), dtype=bool)
    img[5, 5] = True
    img[5, 6] = True  # a 2-px nub
    assert trace_skeleton(img, min_length=3) == []


def test_simplify_collapses_collinear_points():
    pts = [(0, i) for i in range(11)]  # 11 collinear points
    s = simplify(pts, tolerance=0.5)
    assert len(s) == 2  # just the endpoints


def test_simplify_keeps_a_corner():
    bend = [(0, 0), (0, 3), (0, 6), (3, 6), (6, 6)]  # an L-shape
    s = simplify(bend, tolerance=0.5)
    assert len(s) == 3  # both ends + the corner


def test_simplify_short_input_unchanged():
    pts = [(0, 0), (1, 1)]
    assert np.allclose(simplify(pts, 1.0), pts)


def test_pixels_to_world_uses_pixel_centres():
    gt = (1000.0, 10.0, 0.0, 5000.0, 0.0, -10.0)  # north-up, 10 m pixels
    world = pixels_to_world([(0, 0), (1, 2)], gt)
    assert np.allclose(world[0], [1005.0, 4995.0])   # centre of pixel (row0,col0)
    assert np.allclose(world[1], [1025.0, 4985.0])   # centre of pixel (row1,col2)


def test_extract_polylines_recovers_a_line():
    resp = np.zeros((30, 40))
    resp[15, :] = 1.0  # a strong horizontal ridge
    pls = extract_polylines(resp, budget=0.05, min_length=5)
    assert len(pls) >= 1
    # spans most of the width and lies on row 15 (-> y/pixel-row constant)
    main = max(pls, key=lambda t: np.ptp(t[:, 0]))  # widest x-extent
    xs = main[:, 0]
    assert xs.max() - xs.min() > 25


def test_extract_polylines_world_coords_when_transform_given():
    resp = np.zeros((20, 20))
    resp[10, :] = 1.0
    gt = (500.0, 30.0, 0.0, 8000.0, 0.0, -30.0)
    pls = extract_polylines(resp, budget=0.1, min_length=5, transform=gt)
    assert len(pls) >= 1
    # row 10 -> world y = 8000 + (10.5)*(-30) = 7685
    main = max(pls, key=lambda t: np.ptp(t[:, 0]))  # widest x-extent
    assert np.allclose(main[:, 1], 7685.0)
