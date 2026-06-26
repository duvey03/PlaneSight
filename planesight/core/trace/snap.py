"""Snap-on-click: pull each anchor onto the nearest strong-contact pixel (T1).

When the user clicks near a contact, snap the anchor to the actual contact pixel so the
wire starts/ends on the signal (spec S6). Mirrors the snap-to-edge logic in
``debug/ml_snap_labels.py``: treat the top fraction of a contact-strength field as the
edge set, then a distance transform gives the nearest edge for any click. The edge set +
distance transform are precomputed once per AOI (:func:`build_snap_field`) so each click
is O(1). A click with no edge within the radius returns ``None`` (a concealed contact ->
the caller keeps the raw point / free-draws). Pure numpy/scipy; no GDAL/Qt.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.ndimage import distance_transform_edt


class SnapField(NamedTuple):
    """Precomputed snap state for one AOI: the edge mask + nearest-edge lookup."""

    edges: np.ndarray   # bool (H, W): the strong-contact pixels
    dist: np.ndarray    # float (H, W): distance to the nearest edge pixel
    iy: np.ndarray      # int (H, W): row of the nearest edge pixel
    ix: np.ndarray      # int (H, W): col of the nearest edge pixel


def edge_mask(strength: np.ndarray, *, valid: np.ndarray | None = None,
              budget: float = 0.07) -> np.ndarray:
    """Top-``budget`` fraction (by value) of ``strength`` over valid pixels -> bool edges.

    ``strength`` is any contact-strength field (curvature magnitude, detector/ML response,
    or ``1 - cost``); higher = more contact-like.
    """
    s = np.nan_to_num(np.asarray(strength, dtype=float))
    pool = s[valid] if valid is not None else s.ravel()
    if pool.size == 0:
        return np.zeros(s.shape, dtype=bool)
    k = max(1, int(round(budget * pool.size)))
    thr = np.partition(pool, pool.size - k)[pool.size - k]
    mask = s >= thr
    if valid is not None:
        mask &= valid
    return mask


def build_snap_field(strength: np.ndarray, *, valid: np.ndarray | None = None,
                     budget: float = 0.07) -> SnapField:
    """Precompute the edge set and nearest-edge distance transform once per AOI."""
    edges = edge_mask(strength, valid=valid, budget=budget)
    if edges.any():
        dist, idx = distance_transform_edt(~edges, return_indices=True)
        iy, ix = idx[0], idx[1]
    else:
        dist = np.full(edges.shape, np.inf)
        iy = np.zeros(edges.shape, dtype=int)
        ix = np.zeros(edges.shape, dtype=int)
    return SnapField(edges, dist, iy, ix)


def snap_point(field: SnapField, rc, radius: float):
    """Snap a click ``rc`` to the nearest edge within ``radius`` px.

    Returns the snapped ``(row, col)``, or ``None`` if the click is out of bounds or no
    edge pixel lies within ``radius`` (a concealed contact - keep the raw click).
    """
    r, c = int(round(rc[0])), int(round(rc[1]))
    h, w = field.edges.shape
    if not (0 <= r < h and 0 <= c < w):
        return None
    if field.dist[r, c] > radius:
        return None
    return (int(field.iy[r, c]), int(field.ix[r, c]))
