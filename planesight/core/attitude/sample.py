"""Sample a DEM along a trace to produce 3D points for plane fitting.

Pure numpy - no GDAL. A caller reads the DEM into an array + affine geotransform
(GDAL convention) and passes them here; this keeps the sampling logic
CI-testable. Both the trace and the DEM must be in the SAME metric CRS (so x, y,
z are all in metres) before fitting (ARCHITECTURE.md S6).
"""

from __future__ import annotations

import numpy as np


def densify_line(points, spacing: float):
    """Insert vertices along a polyline so consecutive points are ~``spacing`` apart.

    Args:
        points: (n, d) vertices (d = 2 or 3); horizontal distance uses the first
            two columns.
        spacing: target spacing in CRS units (e.g. metres).

    Returns:
        (m, d) densified vertices including the originals, in order.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2:
        raise ValueError("need at least 2 vertices of shape (n, d)")
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    out = [pts[0]]
    for i in range(1, len(pts)):
        a, b = pts[i - 1], pts[i]
        seg_len = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        n = max(1, int(np.ceil(seg_len / spacing)))
        for k in range(1, n + 1):
            out.append(a + (b - a) * (k / n))
    return np.asarray(out)


def sample_bilinear(array, transform, xy, nodata=None):
    """Bilinearly sample a 2D raster at world coordinates.

    Args:
        array: 2D raster (rows, cols).
        transform: 6-tuple GDAL geotransform
            (x0, px_w, row_rot, y0, col_rot, px_h).
        xy: (m, 2) world coordinates in the raster's CRS.
        nodata: optional value treated as missing (-> NaN).

    Returns:
        (m,) sampled values; NaN where out of bounds or touching nodata.
    """
    arr = np.asarray(array, dtype=float)
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    pts = np.asarray(xy, dtype=float)
    gt0, gt1, gt2, gt3, gt4, gt5 = (float(t) for t in transform)
    det = gt1 * gt5 - gt2 * gt4
    if det == 0:
        raise ValueError("degenerate geotransform")

    dx = pts[:, 0] - gt0
    dy = pts[:, 1] - gt3
    # world -> fractional pixel (corner-based), then shift to pixel-centre index space
    col = (gt5 * dx - gt2 * dy) / det - 0.5
    row = (-gt4 * dx + gt1 * dy) / det - 0.5

    nrows, ncols = arr.shape
    r0 = np.floor(row).astype(int)
    c0 = np.floor(col).astype(int)
    r1, c1 = r0 + 1, c0 + 1
    wr, wc = row - r0, col - c0
    valid = (r0 >= 0) & (c0 >= 0) & (r1 < nrows) & (c1 < ncols)

    rr0 = np.clip(r0, 0, nrows - 1)
    rr1 = np.clip(r1, 0, nrows - 1)
    cc0 = np.clip(c0, 0, ncols - 1)
    cc1 = np.clip(c1, 0, ncols - 1)
    top = arr[rr0, cc0] * (1 - wc) + arr[rr0, cc1] * wc
    bot = arr[rr1, cc0] * (1 - wc) + arr[rr1, cc1] * wc
    vals = top * (1 - wr) + bot * wr

    out = np.full(pts.shape[0], np.nan)
    out[valid] = vals[valid]
    return out


def sample_trace(points, array, transform, spacing: float, nodata=None):
    """Densify a 2D trace and sample the DEM to yield clean 3D (x, y, z) points.

    Rows that fall outside the raster or on nodata are dropped. Returns an
    (m, 3) array (possibly empty).
    """
    dense = densify_line(points, spacing)[:, :2]
    z = sample_bilinear(array, transform, dense, nodata=nodata)
    good = np.isfinite(z)
    return np.column_stack([dense[good], z[good]])
