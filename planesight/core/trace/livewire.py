"""Live-wire least-cost path over a cost surface (assisted tracing core, T0).

The path-finding half of the live-wire / intelligent-scissors tracer (planesight-fe7):
given a cost grid where contact-like pixels are cheap, find the cheapest 8-connected
path between two clicks. ``cost_to_all`` runs a single-source Dijkstra from an anchor and
returns the distance + predecessor field; ``backtrace`` then recovers the wire to ANY
target (the moving cursor) by a cheap predecessor walk - so the live preview redraws in
O(path length) and the expensive solve happens only when a new anchor is committed
(exactly the seismic/Photoshop trick). Pure numpy/scipy - no GDAL/Qt. World-coord
conversion and the QgsMapTool live in the gui layer.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

_SQRT2 = float(np.sqrt(2.0))
# 8-connected neighbour offsets (dr, dc, step length); step weights diagonals by sqrt(2).
_OFFSETS = (
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, _SQRT2), (-1, 1, _SQRT2), (1, -1, _SQRT2), (1, 1, _SQRT2),
)
_NO_PRED = -9999          # scipy's "no predecessor" sentinel


class LiveWireField(NamedTuple):
    """A solved single-source field over a (possibly windowed) cost grid.

    ``dist``/``predecessors`` are flat (h*w,) arrays in the window's local indexing;
    ``origin`` is the window's (row, col) offset into the full grid and ``shape`` its
    (h, w). Pass to :func:`backtrace` with a full-grid target to recover the wire.
    """

    dist: np.ndarray
    predecessors: np.ndarray
    origin: tuple
    shape: tuple


def _build_graph(cost: np.ndarray) -> csr_matrix:
    """8-connected directed graph; edge a->b weight = mean(cost_a, cost_b) * step.

    Averaging the two endpoint costs makes the wire cheap to enter AND leave a contact
    pixel; the geometric step weights diagonal moves by sqrt(2) so the path length is
    metric-consistent. Costs must be non-negative (Dijkstra requirement).
    """
    h, w = cost.shape
    c = cost.ravel().astype(float)
    idx = np.arange(h * w).reshape(h, w)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    for dr, dc, step in _OFFSETS:
        # source block = pixels whose (r+dr, c+dc) neighbour is in-bounds.
        sr0, sc0 = max(0, -dr), max(0, -dc)
        sr1, sc1 = h - max(0, dr), w - max(0, dc)
        if sr0 >= sr1 or sc0 >= sc1:
            continue
        src = idx[sr0:sr1, sc0:sc1].ravel()
        dst = idx[sr0 + dr:sr1 + dr, sc0 + dc:sc1 + dc].ravel()
        rows.append(src)
        cols.append(dst)
        data.append(0.5 * (c[src] + c[dst]) * step)
    n = h * w
    return csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    )


def _clip(value, lo, hi):
    return max(lo, min(hi, value))


def _bbox_bounds(shape, points, margin):
    """Window covering all ``points`` (r, c) expanded by ``margin`` px, clipped to grid."""
    h, w = shape
    rr = [p[0] for p in points]
    cc = [p[1] for p in points]
    if margin is None:
        return 0, h, 0, w
    r0 = _clip(min(rr) - margin, 0, h)
    r1 = _clip(max(rr) + margin + 1, 0, h)
    c0 = _clip(min(cc) - margin, 0, w)
    c1 = _clip(max(cc) + margin + 1, 0, w)
    return r0, r1, c0, c1


def _solve(cost: np.ndarray, seed_rc, bounds) -> LiveWireField:
    r0, r1, c0, c1 = bounds
    sub = cost[r0:r1, c0:c1]
    h, w = sub.shape
    sr, sc = seed_rc[0] - r0, seed_rc[1] - c0
    if not (0 <= sr < h and 0 <= sc < w):
        raise ValueError(f"seed {seed_rc} falls outside the search window {bounds}")
    s_idx = sr * w + sc
    dist, pred = dijkstra(
        _build_graph(sub), directed=True, indices=s_idx, return_predecessors=True
    )
    return LiveWireField(dist, pred, (r0, c0), (h, w))


def cost_to_all(cost: np.ndarray, seed_rc, *, margin: int | None = None) -> LiveWireField:
    """Single-source Dijkstra from ``seed_rc`` over the cost grid (the commit-time solve).

    With ``margin`` set, the search is bounded to a square of that half-width around the
    seed (keeps the live solve interactive on large AOIs); ``None`` solves the full grid.
    Call once per committed anchor, then :func:`backtrace` to the cursor on every move.
    """
    bounds = _bbox_bounds(cost.shape, [seed_rc], margin)
    return _solve(cost, seed_rc, bounds)


def backtrace(field: LiveWireField, target_rc) -> np.ndarray:
    """Walk predecessors from ``target_rc`` back to the seed -> (M, 2) full-grid path.

    Returns rows of (row, col) from seed to target. Empty (0, 2) array if the target is
    outside the window or unreachable (infinite cost).
    """
    r0, c0 = field.origin
    h, w = field.shape
    tr, tc = target_rc[0] - r0, target_rc[1] - c0
    if not (0 <= tr < h and 0 <= tc < w):
        return np.empty((0, 2), dtype=int)
    t = tr * w + tc
    if not np.isfinite(field.dist[t]):
        return np.empty((0, 2), dtype=int)
    path: list[tuple] = []
    while t != _NO_PRED and t >= 0:
        path.append((t // w + r0, t % w + c0))
        t = int(field.predecessors[t])
    return np.array(path[::-1], dtype=int)


def least_cost_path(cost: np.ndarray, seed_rc, target_rc, *,
                    margin: int | None = None) -> np.ndarray:
    """Cheapest 8-connected path from ``seed_rc`` to ``target_rc`` -> (M, 2) (row, col).

    Convenience one-shot (solve + backtrace) for known endpoints; the search is windowed
    to the seed/target bounding box expanded by ``margin`` px (``None`` = full grid).
    Empty (0, 2) array if no finite path exists. For interactive tracing use
    :func:`cost_to_all` once per anchor + :func:`backtrace` per cursor move instead.
    """
    bounds = _bbox_bounds(cost.shape, [seed_rc, target_rc], margin)
    field = _solve(cost, seed_rc, bounds)
    return backtrace(field, target_rc)
