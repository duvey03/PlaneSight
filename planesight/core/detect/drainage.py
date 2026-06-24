"""Drainage / flow-accumulation channel network from a DEM (pure numpy/scipy).

The geomorphic pre-filter the architecture calls for (S9.1): an edge detector on
topography lights up the drainage network - incised valleys are the densest, most
continuous slope/curvature breaks - which is geomorphic, not geological. Drainage is
a CONNECTIVITY property: a cell is "drainage" because it routes accumulated water,
which a local curvature test cannot see (hence the dead-end of the concavity filter).
This computes a D8 flow network (each cell drains to its steepest-descent neighbour)
and exposes a channel mask + per-cell flow azimuth, so detected traces that FOLLOW
the drainage - overlapping it AND running along the flow direction - can be flagged.

D11: numpy/scipy only. Before routing, the DEM is depression-filled (priority-flood
+ epsilon, ``fill_depressions``) so closed pits and flats spill through instead of
becoming false sinks that fragment channels; on low-relief terrain this is what keeps
the network continuous. For speed on large DEMs, compute on a downsampled copy via
``channel_network(..., downsample=F)`` - drainage masks fine at 60-120 m. Downsampling
uses a NaN-aware BLOCK MEAN (``block_mean``), not stride subsampling, so a narrow
channel that a stride grid would step over still lowers its block and survives.
"""

from __future__ import annotations

import heapq

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt

from ..attitude.sample import densify_line
from .score import disk

__all__ = [
    "flow_directions",
    "flow_accumulation",
    "flow_azimuth",
    "fill_depressions",
    "block_mean",
    "flow_network",
    "channel_network",
    "channel_proximity",
    "trace_drainage_fraction",
    "is_drainage",
    "flag_drainage",
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


def fill_depressions(dem, epsilon: float = 1e-4):
    """Priority-flood depression filling with an epsilon gradient (Barnes 2014).

    Raises every closed pit and flat to just above its lowest spill point so that
    each finite cell has a strictly-downhill path to the grid edge (or to a NaN
    barrier). Without this, pits and flats become D8 sinks where accumulation stops
    and channels fragment - the dominant failure mode on low-relief terrain.

    The ``epsilon`` per-step lift (default 1e-4, tiny vs GLO-30's ~m vertical noise)
    breaks flats so flow continues across filled regions; terrain already above the
    spill level is left unchanged. NaN cells act as barriers and stay NaN.
    """
    a = np.asarray(dem, dtype=float)
    if a.ndim != 2:
        raise ValueError("dem must be 2D")
    h, w = a.shape
    finite = np.isfinite(a)
    filled = a.copy()
    closed = ~finite                              # NaN are barriers, never filled
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    nan_adj = (binary_dilation(~finite) & finite) if (~finite).any() \
        else np.zeros((h, w), dtype=bool)
    seed = finite & (border | nan_adj)            # outlets: edge + cells touching NaN
    heap = []
    for r, c in zip(*np.where(seed)):
        heapq.heappush(heap, (float(a[r, c]), int(r), int(c)))
        closed[r, c] = True
    nbrs = ((-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1))
    while heap:
        e, r, c = heapq.heappop(heap)
        for dr, dc in nbrs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not closed[nr, nc]:
                ne = a[nr, nc]
                if ne <= e + epsilon:             # pit/flat: lift to spill + epsilon
                    ne = e + epsilon
                filled[nr, nc] = ne
                closed[nr, nc] = True
                heapq.heappush(heap, (ne, nr, nc))
    return filled


def block_mean(dem, factor: int):
    """NaN-aware block-mean downsample by an integer ``factor`` (cropping any
    remainder rows/cols). Each output cell is the mean of the finite values in its
    ``factor x factor`` block (NaN only where the whole block is NaN) - unlike stride
    subsampling, a narrow low channel still pulls its block down and is not skipped.
    """
    a = np.asarray(dem, dtype=float)
    if factor < 1:
        raise ValueError("factor must be >= 1")
    if factor == 1:
        return a
    h, w = a.shape
    big_h, big_w = h // factor, w // factor
    if big_h == 0 or big_w == 0:
        return a
    crop = a[:big_h * factor, :big_w * factor].reshape(big_h, factor, big_w, factor)
    fin = np.isfinite(crop)
    n = fin.sum(axis=(1, 3))
    s = np.where(fin, crop, 0.0).sum(axis=(1, 3))
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def flow_network(dem, downsample: int = 1, fill: bool = True, epsilon: float = 1e-4):
    """Flow accumulation + flow azimuth at the FULL grid, with hardening applied.

    Block-mean downsamples (``downsample`` > 1), depression-fills (``fill``), routes
    D8, then nearest-upsamples the accumulation and azimuth back to the input shape.
    Returns ``(acc, az)`` both shape (h, w) - accumulation in *downsampled* cells.
    """
    a = np.asarray(dem, dtype=float)
    if downsample < 1:
        raise ValueError("downsample must be >= 1")
    sub = block_mean(a, downsample) if downsample > 1 else a
    if fill:
        sub = fill_depressions(sub, epsilon=epsilon)
    acc = flow_accumulation(sub)
    az = flow_azimuth(sub)
    if downsample > 1:
        h, w = a.shape
        ri = np.minimum(np.arange(h) // downsample, sub.shape[0] - 1)
        ci = np.minimum(np.arange(w) // downsample, sub.shape[1] - 1)
        acc = acc[np.ix_(ri, ci)]
        az = az[np.ix_(ri, ci)]
    return acc, az


def channel_network(dem, min_accum_cells: int = 200, downsample: int = 1,
                    fill: bool = True):
    """Channel mask (high flow accumulation) + per-cell flow azimuth, at full grid.

    Args:
        dem: 2D elevation array.
        min_accum_cells: accumulation threshold (in *downsampled* cells) above which
            a cell is a channel.
        downsample: block-mean downsample factor for speed (then nearest-upsampled).
        fill: depression-fill before routing (recommended; see ``fill_depressions``).

    Returns:
        ``(channel_mask, flow_az)`` both shape (h, w): boolean channels and the flow
        azimuth (deg, NaN off-channel/at sinks).
    """
    acc, az = flow_network(dem, downsample=downsample, fill=fill)
    return acc >= float(min_accum_cells), az


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


def flag_drainage(traces, dem, *, downsample: int = 3, min_accum_cells: int = 15,
                  buffer_px: int = 2, min_aligned_fraction: float = 0.5,
                  angle_tol_deg: float = 30.0, valid_mask=None):
    """Classify each detected trace as drainage (runs along a channel) or not.

    The standalone post-detection step the pipeline calls AFTER any detector - the
    detector stays terrain-agnostic (band-in, polylines-out) and swappable, while this
    builds the hardened flow network once and scores every trace against it.

    This is a review-FLAG, not a filter: NO trace is dropped. Returns a list aligned
    with ``traces`` of ``(is_drainage: bool, aligned_fraction: float)``. The
    ``aligned_fraction`` (0..1, how much of the trace runs along the flow) doubles as
    the review-queue ranking score - rank flagged traces by it (most creek-like first).

    Args:
        traces: detector output as (n, 2) pixel ``(col, row)`` vertex arrays - i.e.
            ``detect(stack, transform=None)``; world coords cannot index the grid.
        dem: 2D elevation array on the SAME grid the traces were detected on.
        downsample, min_accum_cells, buffer_px, min_aligned_fraction, angle_tol_deg:
            flow + alignment parameters. Defaults are the Nepal-calibrated operating
            point (planesight-5p3 verification + j8t hardening: filled network, 31%
            removal on the flat part of the false-negative curve).
        valid_mask: optional finite-DEM mask AND-ed with the channel network.

    Returns:
        ``list[tuple[bool, float]]`` - one ``(is_drainage, score)`` per input trace.
    """
    acc, az = flow_network(dem, downsample=downsample, fill=True)
    channel = acc >= float(min_accum_cells)
    if valid_mask is not None:
        channel = channel & np.asarray(valid_mask, dtype=bool)
    buf, near = channel_proximity(channel, az, buffer_px=buffer_px)
    out = []
    for poly in traces:
        _, aligned = trace_drainage_fraction(poly, buf, near,
                                             angle_tol_deg=angle_tol_deg)
        out.append((bool(aligned >= min_aligned_fraction), float(aligned)))
    return out
