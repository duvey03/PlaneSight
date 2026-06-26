"""T0 tests for the live-wire tracing core (core/trace): cost surface + least-cost path.

The gate for planesight-fe7 T0: on a synthetic cost grid the solver recovers the known
cheapest path (follows a cheap channel through an expensive field, even when that channel
detours away from the straight line), the interactive cost_to_all + backtrace trick
agrees with the one-shot solve, windowing doesn't change the optimum, and the cost
surface is cheap where curvature is high. Pure numpy/scipy - no GDAL/QGIS.
"""

import numpy as np

from planesight.core.trace import (
    backtrace,
    build_cost_surface,
    cost_to_all,
    curvature_magnitude,
    least_cost_path,
)

_CHEAP = 0.1
_EXPENSIVE = 10.0


def _diagonal_channel(n=40):
    """Expensive field with a one-pixel-wide cheap channel down the main diagonal."""
    cost = np.full((n, n), _EXPENSIVE)
    cost[np.arange(n), np.arange(n)] = _CHEAP
    return cost


def _l_channel(n=40, bend=30, lo=5):
    """Expensive field with an L-shaped cheap channel: row ``lo`` then column ``bend``.

    The cheapest route from (lo, lo) to (bend, bend) is the L - it detours far from the
    straight diagonal, so a path that follows it proves the solver tracks the signal.
    """
    cost = np.full((n, n), _EXPENSIVE)
    cost[lo, lo:bend + 1] = _CHEAP          # horizontal leg
    cost[lo:bend + 1, bend] = _CHEAP        # vertical leg
    return cost


def test_least_cost_path_follows_diagonal_channel():
    cost = _diagonal_channel(40)
    path = least_cost_path(cost, (2, 2), (37, 37))
    assert path.shape[0] >= 2
    assert tuple(path[0]) == (2, 2) and tuple(path[-1]) == (37, 37)
    # every step stays on the cheap diagonal (r == c), never straying into the field
    assert np.all(path[:, 0] == path[:, 1])
    assert np.all(cost[path[:, 0], path[:, 1]] == _CHEAP)


def test_least_cost_path_detours_along_l_channel():
    # the optimum is the L, NOT the straight diagonal between the endpoints
    cost = _l_channel()
    path = least_cost_path(cost, (5, 5), (30, 30))
    assert tuple(path[0]) == (5, 5) and tuple(path[-1]) == (30, 30)
    # the wire never enters the expensive field - it rides the cheap corridor
    assert cost[path[:, 0], path[:, 1]].max() == _CHEAP
    # and it genuinely detours far from the straight diagonal (where r == c): the L bend
    # drives |row - col| up to ~25, so the path is nowhere near the naive straight line
    assert np.abs(path[:, 0] - path[:, 1]).max() >= 20


def test_cost_to_all_backtrace_matches_one_shot():
    # the interactive trick (solve once from the anchor, backtrace per cursor move)
    # must reproduce the one-shot least_cost_path exactly
    cost = _l_channel()
    field = cost_to_all(cost, (5, 5))
    assert np.array_equal(backtrace(field, (30, 30)), least_cost_path(cost, (5, 5), (30, 30)))


def test_windowing_preserves_optimum():
    # bounding the search to a window around the endpoints must not change the recovered
    # path when that window still contains the optimal corridor
    cost = _l_channel()
    full = least_cost_path(cost, (5, 5), (30, 30), margin=None)
    windowed = least_cost_path(cost, (5, 5), (30, 30), margin=4)
    assert np.array_equal(full, windowed)


def test_backtrace_empty_for_target_outside_window():
    cost = _diagonal_channel(40)
    field = cost_to_all(cost, (2, 2), margin=3)     # small window around the seed
    assert backtrace(field, (37, 37)).shape == (0, 2)   # target far outside -> no path


# --- cost surface (cost.py) ---


def test_build_cost_surface_is_cheap_where_curvature_is_high():
    curv = np.zeros((20, 20))
    curv[:, 10] = 5.0                              # a high-curvature stripe (a "contact")
    cost = build_cost_surface(curv, floor=0.1, w_edge=1.0)
    assert cost.min() > 0.0                        # Dijkstra needs strictly positive
    assert cost[:, 10].mean() < cost[:, 0].mean()  # the stripe is cheaper than flat ground
    assert abs(cost[:, 10].mean() - 0.1) < 1e-6    # peak curvature -> ~floor


def test_build_cost_surface_pushes_nodata_high():
    curv = np.ones((10, 10))
    valid = np.ones((10, 10), dtype=bool)
    valid[0, 0] = False
    cost = build_cost_surface(curv, valid=valid, floor=0.1, w_edge=1.0)
    assert cost[0, 0] > cost[valid].max() + 1.0    # the wire avoids nodata


def test_curvature_magnitude_peaks_on_a_step():
    # a sigmoid escarpment: curvature magnitude must spike at the break, not on the flats
    _, cc = np.indices((30, 30))
    dem = 100.0 / (1.0 + np.exp(-(cc - 15) / 1.5))
    mag = curvature_magnitude(dem, px=30.0)
    assert mag[15, 13:18].max() > mag[15, :5].max() * 5
