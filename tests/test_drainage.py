"""Tests for the D8 drainage / flow-accumulation network (pure numpy)."""

import numpy as np

from planesight.core.detect.drainage import (
    channel_network,
    channel_proximity,
    flow_accumulation,
    flow_azimuth,
    flow_directions,
    is_drainage,
    trace_drainage_fraction,
)


def _tilted_south(h=20, w=10, slope=1.0):
    """z decreases toward +row (south), so every cell drains south."""
    rr, _ = np.indices((h, w))
    return (h - 1 - rr).astype(float) * slope


def test_flow_directions_all_point_downhill():
    down = flow_directions(_tilted_south())
    w = 10
    # interior cell drains to the cell one row south (same column)
    r, c = 5, 4
    assert down[r, c] == (r + 1) * w + c


def test_accumulation_increases_downstream():
    acc = flow_accumulation(_tilted_south(h=20, w=8))
    col = acc[:, 4]
    # strictly increasing top (upstream) to bottom (downstream)
    assert np.all(np.diff(col) > 0)
    assert acc[-1, 4] >= 19  # bottom cell drains its whole column


def test_flow_azimuth_points_south_on_south_tilt():
    az = flow_azimuth(_tilted_south())
    assert np.allclose(az[5, 4], 180.0)  # draining due south


def test_v_valley_concentrates_a_channel_on_the_axis():
    # A V-shaped valley along column cx, tilted so water runs down the axis.
    h, w, cx = 40, 21, 10
    rr, cc = np.indices((h, w))
    dem = np.abs(cc - cx) * 1.0 + (h - 1 - rr) * 0.2  # walls + downstream tilt
    acc = flow_accumulation(dem)
    # the axis accumulates far more than the walls at the same (downstream) row
    assert acc[35, cx] > acc[35, cx + 5] * 5
    mask, az = channel_network(dem, min_accum_cells=h)
    assert mask[35, cx]                 # axis is a channel near the bottom
    assert not mask[35, cx + 7]         # the wall is not
    assert np.isfinite(az[35, cx])


def test_nan_cells_are_sinks_and_excluded():
    dem = _tilted_south(h=10, w=6)
    dem[5, 3] = np.nan
    down = flow_directions(dem)
    assert down[5, 3] == -1                      # NaN cell has no flow
    acc = flow_accumulation(dem)
    assert acc[5, 3] == 0.0                       # and contributes nothing


def test_downsample_preserves_grid_shape():
    dem = _tilted_south(h=40, w=24)
    mask, az = channel_network(dem, min_accum_cells=5, downsample=4)
    assert mask.shape == dem.shape and az.shape == dem.shape


def test_trace_along_channel_flagged_crossing_kept():
    # A vertical channel (col=10) flowing south (az 180). A trace running ALONG it
    # is drainage; a trace CROSSING it is not - even though both overlap.
    h, w, cx = 40, 21, 10
    channel = np.zeros((h, w), dtype=bool)
    channel[:, cx] = True
    flow_az = np.where(channel, 180.0, np.nan)
    buf, near_az = channel_proximity(channel, flow_az, buffer_px=2)

    along = np.array([[cx, 5], [cx, 35]], dtype=float)        # (col,row) vertical
    crossing = np.array([[2, 20], [18, 20]], dtype=float)      # horizontal

    _, a_along = trace_drainage_fraction(along, buf, near_az)
    _, a_cross = trace_drainage_fraction(crossing, buf, near_az)
    assert a_along > 0.8           # runs with the flow -> drainage
    assert a_cross < 0.2           # crosses the flow -> not drainage
    assert is_drainage(along, buf, near_az)
    assert not is_drainage(crossing, buf, near_az)


def test_trace_away_from_channels_has_no_overlap():
    channel = np.zeros((30, 30), dtype=bool)
    channel[:, 5] = True
    buf, near_az = channel_proximity(channel, np.where(channel, 90.0, np.nan))
    far = np.array([[25, 2], [25, 28]], dtype=float)  # nowhere near col 5
    overlap, aligned = trace_drainage_fraction(far, buf, near_az)
    assert overlap < 0.1 and aligned == 0.0


def test_total_accumulation_conserved():
    # every finite cell counts itself, so the total accumulation entering all sinks
    # equals the number of finite cells (single-flow-direction conservation).
    dem = _tilted_south(h=15, w=9)
    acc = flow_accumulation(dem)
    down = flow_directions(dem).ravel()
    sinks = down < 0
    assert np.isclose(acc.ravel()[sinks].sum(), np.isfinite(dem).sum())
