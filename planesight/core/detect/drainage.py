"""Drainage / flow-accumulation channel network from a DEM (pure numpy/scipy).

The geomorphic pre-filter the architecture calls for (S9.1): an edge detector on
topography lights up the drainage network - incised valleys are the densest, most
continuous slope/curvature breaks - which is geomorphic, not geological. Drainage is
a CONNECTIVITY property: a cell is "drainage" because it routes accumulated water,
which a local curvature test cannot see (hence the dead-end of the concavity filter).
This computes a D8 flow network (each cell drains to its steepest-descent neighbour)
and exposes a channel mask + per-cell flow azimuth, so detected traces that FOLLOW
the drainage - overlapping it AND running along the flow direction - can be flagged.

D11: numpy/scipy only. Boundary cells and pits are treated as sinks (accumulation
piles there; the channels leading to them still read high). For speed on large DEMs,
compute on a downsampled copy via ``channel_network(..., downsample=F)`` - drainage
masks fine at 60-120 m.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt

from ..attitude.sample import densify_line
from .score import disk

__all__ = [
    "flow_directions",
    "flow_accumulation",
    "flow_azimuth",
    "channel_network",
    "channel_proximity",
    "trace_drainage_fraction",
    "is_drainage",
]

_SQRT2 = float(np.sqrt(2.0))
# (drow, dcol, distance) for the 8 neighbours
_OFFSETS = [(-1, -1, _SQRT2), (-1, 0, 1.0), (-1, 1, _SQRT2), (0, -1, 1.0),
            (0, 1, 1.0), (1, -1, _SQRT2), (1, 0, 1.0), (1, 1, _SQRT2)]


def flow_directions(dem):
    """D8 steepest-descent neighbour of every cell, as a flat downstream index.

    Returns an int array (h, w) whose value is the flattened index of the cell each
    cell drains to, or -1 for a sink (no lower neighbour / NaN / off-grid).
    """
    a = np.asarray(dem, dtype=float)
    if a.ndim != 2:
        raise ValueError("dem must be 2D")
    h, w = a.shape
    finite = np.isfinite(a)
    p = np.pad(a, 1, constant_values=np.inf)  # off-grid neighbours are never lower
    best = np.zeros((h, w))                    # best positive descent slope so far
    down = np.full((h, w), -1, dtype=np.int64)
    rr, cc = np.indices((h, w))
    for dr, dc, dist in _OFFSETS:
        neigh = p[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        slope = (a - neigh) / dist
        better = finite & np.isfinite(neigh) & (slope > best)
        best = np.where(better, slope, best)
        down = np.where(better, (rr + dr) * w + (cc + dc), down)
    down[~finite] = -1
    return down


def flow_accumulation(dem):
    """Number of cells draining through each cell (each cell counts itself).

    Processes cells from highest to lowest elevation so every upstream contributor
    is summed before its downstream cell - exact for D8 single-flow-direction.
    """
    a = np.asarray(dem, dtype=float)
    down = flow_directions(a).ravel()
    flat = a.ravel()
    finite = np.isfinite(flat)
    acc = finite.astype(float)
    order = np.argsort(np.where(finite, flat, -np.inf))[::-1]  # high -> low
    n = int(finite.sum())
    for k in range(n):
        idx = order[k]
        nd = down[idx]
        if nd >= 0:
            acc[nd] += acc[idx]
    return acc.reshape(a.shape)


def flow_azimuth(dem):
    """Azimuth (degrees clockwise from North) of each cell's D8 flow; NaN at sinks."""
    a = np.asarray(dem, dtype=float)
    down = flow_directions(a)
    h, w = a.shape
    rr, cc = np.indices((h, w))
    valid = down >= 0
    safe = np.where(valid, down, 0)
    nr, nc = safe // w, safe % w
    east = (nc - cc).astype(float)        # +col = East
    north = (rr - nr).astype(float)       # -row = North
    az = np.degrees(np.arctan2(east, north)) % 360.0
    return np.where(valid, az, np.nan)


def channel_network(dem, min_accum_cells: int = 200, downsample: int = 1):
    """Channel mask (high flow accumulation) + per-cell flow azimuth, at full grid.

    Args:
        dem: 2D elevation array.
        min_accum_cells: accumulation threshold (in *downsampled* cells) above which
            a cell is a channel.
        downsample: compute flow on ``dem[::downsample, ::downsample]`` for speed,
            then nearest-upsample the mask/azimuth back to the full grid.

    Returns:
        ``(channel_mask, flow_az)`` both shape (h, w): boolean channels and the flow
        azimuth (deg, NaN off-channel/at sinks).
    """
    a = np.asarray(dem, dtype=float)
    if downsample < 1:
        raise ValueError("downsample must be >= 1")
    sub = a[::downsample, ::downsample] if downsample > 1 else a
    acc = flow_accumulation(sub)
    az = flow_azimuth(sub)
    mask = acc >= float(min_accum_cells)
    if downsample > 1:
        h, w = a.shape
        ri = np.minimum(np.arange(h) // downsample, sub.shape[0] - 1)
        ci = np.minimum(np.arange(w) // downsample, sub.shape[1] - 1)
        mask = mask[np.ix_(ri, ci)]
        az = az[np.ix_(ri, ci)]
    return mask, az


def channel_proximity(channel_mask, flow_az, buffer_px: int = 2):
    """Precompute, once per grid: a buffered channel mask and, for every cell, the
    flow azimuth of the *nearest* channel cell (so a trace within the buffer can be
    tested for flow alignment even when it is a pixel or two off the exact channel).

    Returns ``(channel_buffer, nearest_flow_az)``.
    """
    mask = np.asarray(channel_mask, dtype=bool)
    buf = binary_dilation(mask, structure=disk(buffer_px))
    if mask.any():
        idx = distance_transform_edt(~mask, return_distances=False, return_indices=True)
        nearest_az = np.asarray(flow_az)[tuple(idx)]
    else:
        nearest_az = np.full(mask.shape, np.nan)
    return buf, nearest_az


def trace_drainage_fraction(poly_xy, channel_buffer, nearest_flow_az,
                            angle_tol_deg: float = 30.0):
    """Fraction of a trace that overlaps the channel buffer, and the fraction that
    BOTH overlaps AND runs along the flow direction (|trace - flow| within tol, mod
    180 since traces are undirected).

    Args:
        poly_xy: (n, 2) vertices as (x=col, y=row) - the detector's pixel output.
        channel_buffer, nearest_flow_az: from ``channel_proximity``.

    Returns:
        ``(overlap_fraction, aligned_fraction)`` in [0, 1] by densified length.
    """
    pts = np.asarray(poly_xy, dtype=float)
    if pts.shape[0] < 2:
        return 0.0, 0.0
    d = densify_line(pts, spacing=1.0)
    if d.shape[0] < 2:
        return 0.0, 0.0
    step = np.gradient(d, axis=0)                       # (m, 2): dcol(E), drow(S)
    trace_az = np.degrees(np.arctan2(step[:, 0], -step[:, 1])) % 360.0
    h, w = channel_buffer.shape
    cols = np.clip(np.round(d[:, 0]).astype(int), 0, w - 1)
    rows = np.clip(np.round(d[:, 1]).astype(int), 0, h - 1)
    on = channel_buffer[rows, cols]
    fa = nearest_flow_az[rows, cols]
    diff = np.abs(((trace_az - fa + 90.0) % 180.0) - 90.0)  # undirected angular gap
    aligned = on & np.isfinite(fa) & (diff < angle_tol_deg)
    return float(on.mean()), float(aligned.mean())


def is_drainage(poly_xy, channel_buffer, nearest_flow_az,
                min_aligned_fraction: float = 0.5, angle_tol_deg: float = 30.0):
    """True if a trace runs along the drainage for at least ``min_aligned_fraction``
    of its length (overlap + flow alignment) - i.e. it is a channel, not a contact
    that merely crosses one."""
    _, aligned = trace_drainage_fraction(poly_xy, channel_buffer, nearest_flow_az,
                                         angle_tol_deg=angle_tol_deg)
    return aligned >= min_aligned_fraction
