"""Turn a detection response map into vector trace polylines (pure numpy/scipy).

The detector-agnostic back half of the pipeline (ARCHITECTURE.md S8.1): whatever
front-end produces the "contact-ness" response (gradient, structure tensor, Canny,
later an ML probability map), this stage converts it to ordered polylines ready for
the strike/dip engine:

    response -> threshold (budget) -> thin to 1-px -> trace into ordered
    polylines -> simplify gently -> (optionally) reproject to world coords

scikit-image is NOT available (decision D11 - numpy/scipy/GDAL only), so the
thinning (Zhang-Suen) and skeleton tracing are implemented here. Simplification is
DELIBERATELY GENTLE: Douglas-Peucker with a sub-pixel tolerance removes raster
jitter but preserves the trace sinuosity (the valley V's) that the plane fit relies
on (S6.3) - over-simplifying would straighten away the strike/dip signal.
"""

from __future__ import annotations

import numpy as np

from .score import detect_at_budget

__all__ = [
    "thin",
    "trace_skeleton",
    "simplify",
    "pixels_to_world",
    "polylines_from_mask",
    "extract_polylines",
]


def _neighbors8(img):
    """The eight 3x3 neighbours (Zhang-Suen order P2..P9) as 0/1 arrays."""
    p = np.pad(img, 1)
    return (
        p[:-2, 1:-1],  # P2 N
        p[:-2, 2:],    # P3 NE
        p[1:-1, 2:],   # P4 E
        p[2:, 2:],     # P5 SE
        p[2:, 1:-1],   # P6 S
        p[2:, :-2],    # P7 SW
        p[1:-1, :-2],  # P8 W
        p[:-2, :-2],   # P9 NW
    )


def _transitions(neigh):
    """Count of 0->1 transitions around the ordered neighbour ring (A(P1))."""
    seq = list(neigh) + [neigh[0]]
    a = np.zeros(neigh[0].shape, dtype=int)
    for cur, nxt in zip(seq[:-1], seq[1:]):
        a += (cur == 0) & (nxt == 1)
    return a


def thin(mask):
    """Zhang-Suen morphological thinning to a 1-pixel-wide skeleton.

    Iterates the two Zhang-Suen sub-passes until no pixels change. Pure numpy,
    vectorised over the whole image per pass. Returns a boolean skeleton.
    """
    img = np.asarray(mask, dtype=np.uint8)
    if img.ndim != 2:
        raise ValueError("mask must be 2D")
    img = (img > 0).astype(np.uint8)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            p2, p3, p4, p5, p6, p7, p8, p9 = _neighbors8(img)
            b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            a = _transitions((p2, p3, p4, p5, p6, p7, p8, p9))
            if step == 0:
                c1 = (p2 * p4 * p6) == 0
                c2 = (p4 * p6 * p8) == 0
            else:
                c1 = (p2 * p4 * p8) == 0
                c2 = (p2 * p6 * p8) == 0
            cond = (img == 1) & (b >= 2) & (b <= 6) & (a == 1) & c1 & c2
            if cond.any():
                img[cond] = 0
                changed = True
    return img.astype(bool)


_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def trace_skeleton(skel, min_length: int = 2):
    """Link skeleton pixels into ordered polylines of (row, col) vertices.

    Splits the skeleton at endpoints (degree 1) and junctions (degree >= 3) and
    walks each degree-2 chain between them; isolated loops are walked from an
    arbitrary start. Returns a list of vertex lists (pixel coordinates), dropping
    any with fewer than ``min_length`` vertices.
    """
    skel = np.asarray(skel, dtype=bool)
    ptset = {(int(r), int(c)) for r, c in zip(*np.nonzero(skel))}
    if not ptset:
        return []
    pts = sorted(ptset)  # deterministic iteration order (reproducible results)
    adj = {p: [(p[0] + dr, p[1] + dc) for dr, dc in _OFFSETS
               if (p[0] + dr, p[1] + dc) in ptset] for p in pts}
    deg = {p: len(adj[p]) for p in pts}
    used = set()

    def walk(start, nxt):
        path = [start]
        prev, cur = start, nxt
        while True:
            path.append(cur)
            used.add(frozenset((prev, cur)))
            if deg[cur] != 2:
                break
            nxts = [n for n in adj[cur] if n != prev]
            if not nxts:
                break
            nxt2 = nxts[0]
            if frozenset((cur, nxt2)) in used:  # closed loop
                path.append(nxt2)
                break
            prev, cur = cur, nxt2
        return path

    polylines = []
    for node in [p for p in pts if deg[p] != 2]:
        for nb in adj[node]:
            if frozenset((node, nb)) not in used:
                polylines.append(walk(node, nb))
    for p in pts:  # any remaining pure loops (all degree 2)
        for nb in adj[p]:
            if frozenset((p, nb)) not in used:
                polylines.append(walk(p, nb))
    return [pl for pl in polylines if len(pl) >= min_length]


def simplify(points, tolerance: float = 1.0):
    """Douglas-Peucker simplification of a polyline (keep tolerance sub-pixel).

    Removes vertices that lie within ``tolerance`` of the chord, preserving
    corners and curvature. Coordinate-agnostic (works in pixel or world space).
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return pts
    start, end = pts[0], pts[-1]
    chord = end - start
    length = float(np.hypot(*chord))
    if length == 0:  # closed/degenerate: distance from the start point
        dists = np.hypot(pts[:, 0] - start[0], pts[:, 1] - start[1])
    else:
        v = pts - start
        cross = chord[0] * v[:, 1] - chord[1] * v[:, 0]  # 2D cross z-component
        dists = np.abs(cross) / length
    idx = int(np.argmax(dists))
    if dists[idx] > tolerance:
        left = simplify(pts[: idx + 1], tolerance)
        right = simplify(pts[idx:], tolerance)
        return np.vstack([left[:-1], right])
    return np.vstack([start, end])


def pixels_to_world(rc_points, transform):
    """Map (row, col) pixel vertices to (x, y) world coords via a GDAL geotransform.

    Uses pixel centres (the +0.5 convention), consistent with ``attitude.sample``.
    """
    pts = np.asarray(rc_points, dtype=float)
    gt0, gt1, gt2, gt3, gt4, gt5 = (float(t) for t in transform)
    col = pts[:, 1] + 0.5
    row = pts[:, 0] + 0.5
    x = gt0 + col * gt1 + row * gt2
    y = gt3 + col * gt4 + row * gt5
    return np.column_stack([x, y])


def polylines_from_mask(mask, min_length: int = 5, simplify_tol: float = 1.0,
                        transform=None):
    """Binary detection mask -> simplified trace polylines.

    Thins, traces, simplifies, and drops polylines shorter than ``min_length``
    vertices. Takes an already-binarised mask (e.g. Canny edges), so it skips the
    budget threshold. Returns a list of ``(n, 2)`` arrays: world ``(x, y)`` if
    ``transform`` is given, else pixel ``(col, row)``.
    """
    skel = thin(mask)
    out = []
    for path in trace_skeleton(skel, min_length=2):
        if len(path) < min_length:
            continue
        rc = simplify(path, tolerance=simplify_tol)  # (row, col)
        if transform is not None:
            out.append(pixels_to_world(rc, transform))
        else:
            out.append(rc[:, ::-1])  # -> (col, row) = (x, y) in pixel space
    return out


def extract_polylines(response, budget: float = 0.05, valid_mask=None,
                      min_length: int = 5, simplify_tol: float = 1.0,
                      transform=None):
    """Full back-end for a continuous response: threshold at ``budget`` (top-k
    strong pixels), then ``polylines_from_mask``. See that function for the rest.
    """
    det = detect_at_budget(response, budget, valid_mask=valid_mask)
    return polylines_from_mask(det, min_length=min_length,
                               simplify_tol=simplify_tol, transform=transform)
