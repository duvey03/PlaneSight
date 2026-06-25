"""Local attitude variability + trace-morphology priors (pure numpy).

Data-driven inputs for the downstream skeptic/smoothness checks (planesight-5ug):

  - Strike is *circular and undirected* (mod 180 deg): a strike of 5 deg and
    175 deg differ by 10 deg, not 170. All strike means/deviations therefore use
    DOUBLED-ANGLE circular statistics (ARCHITECTURE.md S6; the doubled-angle
    vector mean in scripts/detect_attitudes_nepal.py), never plain arithmetic.
  - "Local variability" = how much a measurement's strike/dip departs from the
    central tendency of its spatial neighbours within a radius. Its high
    percentiles set the empirical "implausible local outlier" bar.
  - "Morphology" = the rule-of-V's class of a hand trace from its fitted dip:
    near-vertical strata cut STRAIGHT across topography, near-horizontal bedding
    runs CONTOUR-PARALLEL, and dipping beds make the diagnostic V (ARCHITECTURE.md
    S6.3). These are plausibility priors, not a complete census (the hand traces
    are positive-unlabeled).

Pure (numpy only, D11) so the Stop hook's pytest suite gates it; the GDAL driver
that feeds it real attitudes lives in scripts/attitude_rules.py.
"""

from __future__ import annotations

import numpy as np

# Morphology labels (rule of V's).
STRAIGHT = "straight"            # near-vertical strata -> straight map trace
CONTOUR_PARALLEL = "contour-parallel"  # near-horizontal bedding -> hugs contours
V = "V"                          # dipping beds -> V's across topography


def strike_difference(a, b):
    """Undirected angular difference between two strikes, in [0, 90] degrees.

    Strike is defined mod 180 (a line, not a ray), so 175 and 5 differ by 10.
    Accepts scalars or broadcasting array-likes.
    """
    d = np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) % 180.0
    return np.minimum(d, 180.0 - d)


def circular_mean_strike(strikes):
    """Mean strike (deg, in [0, 180)) via the doubled-angle vector mean.

    Doubling maps the mod-180 strike onto a full circle so antipodal strikes
    (e.g. 1 and 179) average correctly. Returns NaN for an empty input or when
    the resultant vanishes (perfectly opposed strikes -> mean undefined).
    """
    s = np.asarray(strikes, dtype=float)
    if s.size == 0:
        return float("nan")
    ang = np.radians(2.0 * s)
    c, sn = np.cos(ang).mean(), np.sin(ang).mean()
    if np.hypot(c, sn) < 1e-12:  # opposed strikes -> resultant vanishes -> undefined
        return float("nan")
    return float((np.degrees(np.arctan2(sn, c)) / 2.0) % 180.0)


def circular_resultant_length(strikes):
    """Resultant length R in [0, 1] of the doubled-angle strikes.

    R near 1 = tightly clustered strikes; R near 0 = dispersed/opposed. The
    circular standard deviation is sqrt(-2 ln R) (radians on the doubled angle).
    """
    s = np.asarray(strikes, dtype=float)
    if s.size == 0:
        return float("nan")
    ang = np.radians(2.0 * s)
    return float(np.hypot(np.cos(ang).mean(), np.sin(ang).mean()))


def windowed_deviations(xy, strikes, dips, radius, min_neighbors=2):
    """Per-point local strike/dip deviation from spatial neighbours within ``radius``.

    For each measurement i, the neighbours are all OTHER points within ``radius``
    (Euclidean, in the xy CRS units). The deviation is how far point i departs
    from its neighbours' central tendency:

      - strike: undirected difference (mod 180) from the neighbours' doubled-angle
        circular mean strike;
      - dip: absolute difference from the neighbours' median dip.

    Points with fewer than ``min_neighbors`` neighbours get NaN (no local context
    to judge them against). Returns ``(strike_dev, dip_dev, n_neighbors)`` arrays
    aligned with the inputs; take ``np.nanpercentile`` over them for the
    distribution.

    O(n^2) in the number of points (fine for the few-thousand hand attitudes here).
    """
    xy = np.asarray(xy, dtype=float)
    strikes = np.asarray(strikes, dtype=float)
    dips = np.asarray(dips, dtype=float)
    n = xy.shape[0]
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must have shape (n, 2)")
    if not (n == strikes.shape[0] == dips.shape[0]):
        raise ValueError("xy, strikes, dips must be the same length")
    if radius <= 0:
        raise ValueError("radius must be positive")

    strike_dev = np.full(n, np.nan)
    dip_dev = np.full(n, np.nan)
    n_neighbors = np.zeros(n, dtype=int)
    r2 = float(radius) ** 2
    for i in range(n):
        d2 = ((xy - xy[i]) ** 2).sum(axis=1)
        mask = (d2 <= r2)
        mask[i] = False
        k = int(mask.sum())
        n_neighbors[i] = k
        if k < min_neighbors:
            continue
        strike_dev[i] = strike_difference(strikes[i], circular_mean_strike(strikes[mask]))
        dip_dev[i] = abs(dips[i] - float(np.median(dips[mask])))
    return strike_dev, dip_dev, n_neighbors


def classify_morphology(dip, vertical_dip=75.0, horizontal_dip=20.0):
    """Rule-of-V's morphology class from a fitted dip (degrees).

    - dip >= ``vertical_dip``   -> ``STRAIGHT`` (near-vertical strata; the contact
      cuts straight across topography, no V).
    - dip <= ``horizontal_dip`` -> ``CONTOUR_PARALLEL`` (near-horizontal bedding;
      the contact hugs topographic contours).
    - otherwise                 -> ``V`` (dipping beds; diagnostic V across valleys).

    NaN dip (degenerate fit) -> ``None`` (unclassifiable). The dip is the single
    geometric discriminator of the three classes (ARCHITECTURE.md S6.3): it should
    come from a well-conditioned fit, so the driver gates on conditioning first.
    """
    if dip is None or not np.isfinite(dip):
        return None
    if dip >= vertical_dip:
        return STRAIGHT
    if dip <= horizontal_dip:
        return CONTOUR_PARALLEL
    return V


def polyline_length(points):
    """Total map-view (xy) length of a polyline, in CRS units.

    Uses the first two columns, so it works on 2D or 3D vertex arrays. Returns
    0.0 for a degenerate (<2 vertex) line.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return 0.0
    seg = np.diff(pts[:, :2], axis=0)
    return float(np.hypot(seg[:, 0], seg[:, 1]).sum())
