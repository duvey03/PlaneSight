"""Tests for the response->polylines back-end (pure numpy/scipy)."""

import numpy as np

from planesight.core.detect import (
    close_gaps,
    extract_polylines,
    link_polylines,
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


# --- link_polylines: the four continuity risk cases ------------------------
# Polylines are (row, col) here; link_polylines is coordinate-agnostic. Defaults
# are max_gap_px=5.0, max_angle_deg=20.0 unless a test overrides them.


def test_link_colinear_gap_fragments_merge():
    # Two horizontal fragments on the same row, separated by a 3px gap.
    a = np.array([(0, 0), (0, 10)], dtype=float)
    b = np.array([(0, 13), (0, 23)], dtype=float)
    out = link_polylines([a, b])
    assert len(out) == 1  # bridged into a single trace
    merged = out[0]
    # spans the full extent end to end (col 0 .. 23)
    assert merged[:, 1].min() == 0 and merged[:, 1].max() == 23
    assert np.all(merged[:, 0] == 0)  # stayed on the row


def test_link_undirected_orientation_mod180():
    # a ends heading +col, b's near endpoint heads -col: raw tangents are 180deg
    # apart but the SAME line (mod 180) -> must still link.
    a = np.array([(0, 0), (0, 10)], dtype=float)
    b = np.array([(0, 23), (0, 13)], dtype=float)  # ordered so near end is b[-1]
    out = link_polylines([a, b])
    assert len(out) == 1


def test_link_parallel_offset_layers_do_not_merge():
    # Adjacent bedding layers: parallel (both horizontal) but offset by 3 rows.
    # Tangents are collinear, so the tangent-only check would wrongly merge;
    # the gap vector (3 rows, 2 cols ~= 56deg) runs ACROSS the layers and fails
    # the gap-collinearity guard.
    a = np.array([(0, 0), (0, 10)], dtype=float)
    b = np.array([(3, 12), (3, 22)], dtype=float)
    out = link_polylines([a, b])
    assert len(out) == 2  # stayed separate


def test_link_over_large_gap_does_not_merge():
    # Perfectly colinear but the gap (10px) exceeds max_gap_px (5).
    a = np.array([(0, 0), (0, 10)], dtype=float)
    b = np.array([(0, 20), (0, 30)], dtype=float)
    out = link_polylines([a, b], max_gap_px=5.0)
    assert len(out) == 2


def test_link_junction_angle_mismatch_does_not_merge():
    # Endpoints are 1.4px apart (a true junction) but b heads perpendicular to a.
    a = np.array([(0, 0), (0, 10)], dtype=float)       # horizontal
    b = np.array([(1, 11), (8, 11)], dtype=float)      # near-vertical
    out = link_polylines([a, b])
    assert len(out) == 2  # junction, not a continuation


def test_link_chains_more_than_two_fragments():
    # Three colinear fragments with small gaps collapse to one.
    a = np.array([(0, 0), (0, 8)], dtype=float)
    b = np.array([(0, 11), (0, 19)], dtype=float)
    c = np.array([(0, 22), (0, 30)], dtype=float)
    out = link_polylines([a, b, c])
    assert len(out) == 1
    assert out[0][:, 1].max() == 30


def test_link_passes_through_short_and_empty():
    assert link_polylines([]) == []
    nub = np.array([(5, 5)], dtype=float)  # single vertex, no orientation
    out = link_polylines([nub])
    assert len(out) == 1


def test_close_gaps_bridges_small_break():
    line = np.zeros((9, 20), dtype=bool)
    line[4, 2:9] = True
    line[4, 11:18] = True  # a 2px break at cols 9,10
    bridged = close_gaps(line, size=1)
    assert bridged[4, 9] and bridged[4, 10]  # gap closed
    assert bridged[4, 2:18].all()
