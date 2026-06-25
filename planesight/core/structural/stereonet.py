"""Stereonet / structural-analysis math (pure numpy, dependency-free).

The "Analyze" capability set for GUI milestone M3 (planesight-8et):

  1. strike/dip <-> pole (downward unit normal) and trend/plunge <-> xyz.
  2. lower-hemisphere EQUAL-AREA (Schmidt) projection of a line, plus a
     great-circle generator for a plane.
  3. Fisher (1953) statistics: mean vector, R, kappa, alpha95 cone.
  4. orientation tensor + sorted eigen-decomposition (principal orientations).
  5. axial mean (poles are AXES, +/- equivalent) via the principal eigenvector.
  6. fold_axis: poles to folded bedding lie on a GIRDLE; the girdle pole (the
     MINIMUM-eigenvalue eigenvector) is the fold axis (beta), reported with a
     girdle-vs-cluster shape statistic (Woodcock K).
  7. rose_bins: bidirectional strike-azimuth histogram.

CONVENTIONS (must match plane_fit.py / ARCHITECTURE.md S6 exactly)
------------------------------------------------------------------
Coordinate frame is right-handed ``(x=East, y=North, z=Up)``.

- **Azimuth / trend**: degrees clockwise from North. A horizontal direction of
  trend ``T`` is the unit vector ``(sin T, cos T, 0)`` - note ``atan2(x, y)``,
  the same compass convention ``plane_fit._fit_core`` uses for dip_direction.
- **Plunge**: the downward inclination of a line from horizontal, in [0, 90].
  A line of trend ``T`` and plunge ``p`` is ``(cos p sin T, cos p cos T, -sin p)``
  - i.e. plunge is positive *downward* (``z <= 0``).
- **Strike / dip (right-hand rule)**: ``dip_direction = (strike + 90) mod 360``
  (the plane dips down to the right of the strike azimuth). The plane's UPWARD
  normal is ``(sin d sin a, sin d cos a, cos d)`` with ``d=dip, a=dip_direction``
  - exactly ``plane_fit``'s normal. The **pole** we return is the DOWNWARD unit
  normal (lower hemisphere, ``z <= 0``): pole trend = ``(strike - 90) mod 360``,
  pole plunge = ``90 - dip``.

Round-trip identities (unit-tested): ``strike_dip_to_pole`` then
``pole_to_strike_dip`` recovers strike/dip; a vertical pole (horizontal plane)
projects to the disk centre and a horizontal line to the rim, both exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Angular epsilon below which a horizontal projection is treated as degenerate
# (trend undefined for a vertical line / a pole at the disk centre).
_EPS = 1e-12


# --- trend/plunge <-> xyz -----------------------------------------------------


def line_to_xyz(trend, plunge):
    """Unit vector(s) for a line of given trend and plunge (degrees).

    Returns ``(x=East, y=North, z=Up)`` with ``z <= 0`` for a downward
    (plunge >= 0) line. Accepts scalars or array-likes (broadcast together);
    the result has shape ``(..., 3)``.
    """
    t = np.radians(np.asarray(trend, dtype=float))
    p = np.radians(np.asarray(plunge, dtype=float))
    cp = np.cos(p)
    return np.stack([cp * np.sin(t), cp * np.cos(t), -np.sin(p)], axis=-1)


def xyz_to_line(vec, lower_hemisphere=True):
    """Trend and plunge (degrees) of a vector ``(x=East, y=North, z=Up)``.

    With ``lower_hemisphere=True`` (the default) the vector is treated as an
    undirected **axis**: any upward vector is flipped to its downward antipode
    first, so plunge is always in [0, 90]. With ``lower_hemisphere=False`` the
    vector keeps its direction and plunge may be negative (upward).

    A vertical line has an undefined trend; ``0.0`` is returned for it.
    """
    v = np.asarray(vec, dtype=float)
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n < _EPS:
        return float("nan"), float("nan")
    v = v / n
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    if lower_hemisphere and z > 0.0:
        x, y, z = -x, -y, -z
    plunge = np.degrees(np.arcsin(np.clip(-z, -1.0, 1.0)))
    horiz = np.hypot(x, y)
    trend = 0.0 if horiz < _EPS else np.degrees(np.arctan2(x, y)) % 360.0
    return float(trend), float(plunge)


# --- strike/dip <-> pole (downward unit normal) -------------------------------


def strike_dip_to_pole(strike, dip):
    """Downward unit normal (pole) of a plane given strike/dip (RHR, degrees).

    pole trend = ``strike - 90``; pole plunge = ``90 - dip``. The returned
    vector has ``z <= 0`` (lower hemisphere) and equals the negated
    ``plane_fit`` upward normal.
    """
    return line_to_xyz((np.asarray(strike, float) - 90.0) % 360.0,
                       90.0 - np.asarray(dip, float))


def pole_to_strike_dip(pole):
    """Strike/dip (RHR, degrees) of a plane from its pole (any unit normal).

    The pole is treated as an axis (flipped to the lower hemisphere if it points
    up), so this inverts :func:`strike_dip_to_pole`. For a horizontal plane the
    pole is vertical and strike is undefined; ``strike=0`` is returned.
    """
    trend, plunge = xyz_to_line(pole, lower_hemisphere=True)
    if not np.isfinite(trend):
        return float("nan"), float("nan")
    dip = 90.0 - plunge
    strike = (trend + 90.0) % 360.0
    return float(strike), float(dip)


# --- equal-area (Schmidt) lower-hemisphere projection -------------------------


def equal_area_xy(trend, plunge):
    """Lower-hemisphere equal-area (Schmidt) projection of a line -> (x, y).

    Returns plot coordinates in the unit disk with North up (``+y``) and East
    right (``+x``). The radius uses the identity ``r = sqrt(1 - sin(plunge))``
    (equivalently ``sqrt(2) sin((90-plunge)/2)``), which is *exact* at the two
    end-points: a vertical line (plunge 90) maps to the centre ``(0, 0)`` and a
    horizontal line (plunge 0) to the rim (``r = 1``). Accepts scalars or
    array-likes; result shape is ``(..., 2)``.
    """
    t = np.radians(np.asarray(trend, dtype=float))
    p = np.radians(np.asarray(plunge, dtype=float))
    r = np.sqrt(np.clip(1.0 - np.sin(p), 0.0, 2.0))
    return np.stack([r * np.sin(t), r * np.cos(t)], axis=-1)


def _equal_area_from_xyz(v):
    """Equal-area projection of lower-hemisphere unit vectors ``(...,3)``.

    Vectors with ``z > 0`` are flipped to their downward antipode first. Uses
    ``r = sqrt(1 + z)`` (= ``sqrt(1 - sin(plunge))``) and the horizontal
    direction of the vector, avoiding a trend round-trip.
    """
    v = np.asarray(v, dtype=float)
    up = v[..., 2] > 0.0
    v = np.where(up[..., None], -v, v)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    r = np.sqrt(np.clip(1.0 + z, 0.0, 2.0))
    horiz = np.hypot(x, y)
    safe = horiz > _EPS
    scale = np.where(safe, r / np.where(safe, horiz, 1.0), 0.0)
    return np.stack([x * scale, y * scale], axis=-1)


def great_circle(strike, dip, n=181):
    """Equal-area great-circle trace of a plane (for plotting) -> ``(n, 2)``.

    Sweeps the in-plane lines from one strike end, through the down-dip line, to
    the other strike end, projecting each (all lie in the lower hemisphere).
    ``n`` points span the half great circle.
    """
    if n < 2:
        raise ValueError("need at least 2 points for a great circle")
    s = float(strike)
    d = float(dip)
    strike_vec = line_to_xyz(s, 0.0)                 # horizontal, in-plane
    dip_vec = line_to_xyz((s + 90.0) % 360.0, d)     # down-dip, in-plane
    t = np.linspace(0.0, np.pi, n)
    # cos(t)*strike + sin(t)*downdip; z = -sin(t)*sin(dip) <= 0 for t in [0, pi].
    pts = (np.cos(t)[:, None] * strike_vec[None, :]
           + np.sin(t)[:, None] * dip_vec[None, :])
    return _equal_area_from_xyz(pts)


# --- unit-vector hygiene ------------------------------------------------------


def _clean_unit_vectors(vectors):
    """Return finite, unit-normalised rows of an ``(N, 3)`` array.

    Drops any row with a non-finite component or (near-)zero length, so callers
    degrade gracefully on NaN / zero input rather than crashing.
    """
    v = np.asarray(vectors, dtype=float)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError("vectors must have shape (N, 3)")
    finite = np.isfinite(v).all(axis=1)
    v = v[finite]
    if v.shape[0] == 0:
        return v
    norms = np.linalg.norm(v, axis=1)
    keep = norms > _EPS
    return v[keep] / norms[keep, None]


# --- Fisher (1953) directional statistics -------------------------------------


@dataclass(frozen=True)
class FisherStats:
    """Fisher (1953) mean direction and dispersion of DIRECTED unit vectors.

    Fields: ``n`` (vectors used), ``r`` (resultant length |sum v_i|),
    ``mean_vec`` (unit mean direction, x/y/z), ``trend``/``plunge`` of the mean,
    ``kappa`` (precision estimate ``(n-1)/(n-r)``), ``alpha95`` (95% confidence
    cone half-angle, degrees) and ``csd`` (angular/circular standard deviation
    ``81/sqrt(kappa)``). For ``n < 2`` or fully dispersed data (r ~ 0) the
    dispersion fields are NaN.

    Fisher statistics assume DIRECTED data all on one hemisphere; for axial data
    (poles, fold elements) use :func:`axial_mean` instead.
    """

    n: int
    r: float
    mean_vec: np.ndarray
    trend: float
    plunge: float
    kappa: float
    alpha95: float
    csd: float


def fisher_mean(vectors, confidence=0.95):
    """Fisher (1953) mean of directed unit vectors ``(N, 3)`` -> :class:`FisherStats`.

    ``confidence`` sets the cone level (0.95 -> alpha95). The cone half-angle is
    ``cos(alpha) = 1 - (n - r)/r * ((1/(1-confidence))^(1/(n-1)) - 1)`` (Fisher
    1953; Butler 1992, eq. A.57). Reproduces the PmagPy ``fisher_mean`` worked
    example (dec/inc) to ~1e-2 deg - see ``tests/test_stereonet.py``.
    """
    v = _clean_unit_vectors(vectors)
    n = int(v.shape[0])
    if n == 0:
        nan = float("nan")
        return FisherStats(0, nan, np.array([nan, nan, nan]), nan, nan, nan, nan, nan)
    resultant = v.sum(axis=0)
    r = float(np.linalg.norm(resultant))
    if r < _EPS:  # fully dispersed - mean direction undefined
        nan = float("nan")
        return FisherStats(n, 0.0, np.array([nan, nan, nan]), nan, nan, nan, nan, nan)
    mean_vec = resultant / r
    trend, plunge = xyz_to_line(mean_vec, lower_hemisphere=False)
    if n < 2:  # a single direction: mean is itself, dispersion undefined
        nan = float("nan")
        return FisherStats(n, r, mean_vec, trend, plunge, nan, nan, nan)
    kappa = (n - 1) / (n - r) if (n - r) > _EPS else float("inf")
    p = 1.0 - confidence
    cos_alpha = 1.0 - ((n - r) / r) * (p ** (-1.0 / (n - 1)) - 1.0)
    alpha = float(np.degrees(np.arccos(np.clip(cos_alpha, -1.0, 1.0))))
    csd = 81.0 / np.sqrt(kappa) if np.isfinite(kappa) and kappa > 0 else float("nan")
    return FisherStats(n, r, mean_vec, trend, plunge, float(kappa), alpha, float(csd))


# --- orientation tensor + eigen-analysis --------------------------------------


def orientation_tensor(vectors, normalize=False):
    """Orientation tensor ``T = sum_i v_i v_i^T`` of unit vectors ``(N, 3)``.

    Symmetric 3x3. With ``normalize=True`` it is divided by N (the normalised
    orientation tensor whose eigenvalues sum to 1). Sign-invariant: ``v`` and
    ``-v`` contribute identically, which is why it is the right tool for AXIAL
    data (poles, fold elements).
    """
    v = _clean_unit_vectors(vectors)
    t = v.T @ v
    if normalize and v.shape[0] > 0:
        t = t / v.shape[0]
    return t


@dataclass(frozen=True)
class PrincipalOrientations:
    """Sorted eigen-decomposition of the orientation tensor.

    ``eigenvalues`` are normalised (sum to 1) and DESCENDING ``S1 >= S2 >= S3``.
    ``eigenvectors`` is 3x3 with **columns** ``v1, v2, v3`` (the principal axes,
    each a downward-oriented unit vector). ``woodcock_k`` =
    ``ln(S1/S2) / ln(S2/S3)`` and ``strength_c`` = ``ln(S1/S3)`` (Woodcock 1977):
    K < 1 is girdle-like (planar / fold), K > 1 cluster-like (point maximum),
    larger C = stronger preferred orientation.
    """

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    woodcock_k: float
    strength_c: float
    n: int


def _woodcock(s):
    """Woodcock (1977) shape K and strength C from descending normalised eigvals.

    Guards the log ratios so an exact girdle (S2==S3) -> K=inf-safe and an exact
    cluster (S1==S2) -> K=0 without raising; returns (K, C) as plain floats.
    """
    s1, s2, s3 = (max(float(x), _EPS) for x in s)
    ln12 = np.log(s1 / s2)
    ln23 = np.log(s2 / s3)
    if ln23 < _EPS:
        k = float("inf") if ln12 > _EPS else float("nan")
    else:
        k = ln12 / ln23
    c = float(np.log(s1 / s3))
    return float(k), c


def principal_orientations(vectors):
    """Eigen-analysis of the orientation tensor -> :class:`PrincipalOrientations`.

    Uses ``numpy.linalg.eigh`` (symmetric solver), sorted descending. Eigenvalues
    are normalised by N. Each eigenvector is oriented to the lower hemisphere
    (``z <= 0``) for a consistent axial reading.
    """
    v = _clean_unit_vectors(vectors)
    n = int(v.shape[0])
    if n == 0:
        nan3 = np.full(3, float("nan"))
        return PrincipalOrientations(nan3, np.full((3, 3), float("nan")),
                                     float("nan"), float("nan"), 0)
    t = (v.T @ v) / n
    w, vecs = np.linalg.eigh(t)            # ascending eigenvalues
    order = np.argsort(w)[::-1]            # -> descending S1 >= S2 >= S3
    w = w[order]
    vecs = vecs[:, order]
    # orient each principal axis downward (lower hemisphere) for axial reading
    flip = vecs[2, :] > 0.0
    vecs[:, flip] *= -1.0
    k, c = _woodcock(w)
    return PrincipalOrientations(w, vecs, k, c, n)


# --- axial mean (poles are AXES, +/- equivalent) ------------------------------


@dataclass(frozen=True)
class AxialMean:
    """Mean of AXIAL data (sign-insensitive), from the orientation tensor.

    ``vector`` is the principal (largest-eigenvalue) eigenvector, oriented
    downward; ``trend``/``plunge`` are its orientation; ``concentration`` is the
    largest normalised eigenvalue S1 (1 = perfectly clustered, 1/3 = isotropic).
    A vector Fisher mean is WRONG for axial data without hemisphere
    normalisation - this is the correct estimator.
    """

    trend: float
    plunge: float
    vector: np.ndarray
    concentration: float
    n: int


def axial_mean(vectors):
    """Axial mean of unit vectors ``(N, 3)`` -> :class:`AxialMean`.

    The mean axis is the principal eigenvector of the orientation tensor, so a
    vector and its antipode are treated identically. Robust to mixed-hemisphere
    input (e.g. poles that were not pre-flipped).
    """
    po = principal_orientations(vectors)
    if po.n == 0:
        nan = float("nan")
        return AxialMean(nan, nan, np.array([nan, nan, nan]), nan, 0)
    v1 = po.eigenvectors[:, 0]
    trend, plunge = xyz_to_line(v1, lower_hemisphere=True)
    return AxialMean(trend, plunge, v1, float(po.eigenvalues[0]), po.n)


# --- fold axis (girdle pole) --------------------------------------------------


@dataclass(frozen=True)
class FoldAxis:
    """Best-fit fold axis (beta) from poles to folded bedding.

    Poles to cylindrically folded bedding lie on a GIRDLE; its pole - the
    MINIMUM-eigenvalue eigenvector of the orientation tensor - is the fold axis.
    ``trend``/``plunge`` give beta. ``eigenvalues`` are the normalised
    S1>=S2>=S3. ``woodcock_k`` / ``strength_c`` are the Woodcock (1977) shape and
    strength; ``classification`` is ``"girdle"`` (K < 1, a real fold), ``"cluster"``
    (K > 1, a point maximum - do NOT trust beta), or ``"weak"`` (C small, no
    preferred orientation). ``is_girdle`` is True only for a trustworthy fold.
    """

    trend: float
    plunge: float
    eigenvalues: np.ndarray
    woodcock_k: float
    strength_c: float
    classification: str
    is_girdle: bool
    n: int


def fold_axis(strikes, dips, weak_strength=0.3):
    """Fold axis (beta) from arrays of bedding strike/dip (RHR, degrees).

    Builds the bedding poles, eigen-decomposes their orientation tensor, and
    returns the girdle pole (S3 eigenvector) as the fold axis, together with a
    girdle-vs-cluster shape statistic so a point cluster is never mis-reported as
    a fold. ``weak_strength`` is the Woodcock C below which the orientation is
    called ``"weak"`` (no reliable axis). Needs >= 3 measurements.
    """
    strikes = np.asarray(strikes, dtype=float)
    dips = np.asarray(dips, dtype=float)
    if strikes.shape != dips.shape:
        raise ValueError("strikes and dips must have the same shape")
    poles = strike_dip_to_pole(strikes, dips)
    po = principal_orientations(poles)
    if po.n < 3:
        nan = float("nan")
        return FoldAxis(nan, nan, po.eigenvalues, po.woodcock_k, po.strength_c,
                        "undefined", False, po.n)
    beta = po.eigenvectors[:, 2]  # minimum-eigenvalue eigenvector = girdle pole
    trend, plunge = xyz_to_line(beta, lower_hemisphere=True)
    k, c = po.woodcock_k, po.strength_c
    if not np.isfinite(c) or c < weak_strength:
        classification = "weak"
    elif np.isfinite(k) and k < 1.0:
        classification = "girdle"
    else:
        classification = "cluster"
    is_girdle = classification == "girdle"
    return FoldAxis(trend, plunge, po.eigenvalues, k, c, classification,
                    is_girdle, po.n)


# --- rose diagram -------------------------------------------------------------


def rose_bins(strikes, bin_deg=10.0):
    """Bidirectional strike-azimuth histogram -> ``(counts, bin_edges)``.

    Strikes are AXES (mod 180), so each measurement is counted in both its
    azimuth bin and the opposite bin: ``counts`` is therefore symmetric across
    the diameter (``counts[i] == counts[i + nbins/2]``). ``bin_deg`` must divide
    360 evenly. ``bin_edges`` has ``nbins + 1`` entries spanning ``[0, 360]``.
    Non-finite strikes are ignored.
    """
    if bin_deg <= 0 or not np.isclose(360.0 / bin_deg, round(360.0 / bin_deg)):
        raise ValueError("bin_deg must evenly divide 360")
    nbins = int(round(360.0 / bin_deg))
    half = nbins // 2
    s = np.asarray(strikes, dtype=float)
    s = s[np.isfinite(s)]
    bin_edges = np.arange(nbins + 1) * bin_deg
    if s.size == 0:
        return np.zeros(nbins, dtype=int), bin_edges
    folded = np.mod(s, 180.0)                       # axis -> [0, 180)
    half_counts, _ = np.histogram(folded, bins=np.arange(half + 1) * bin_deg)
    counts = np.concatenate([half_counts, half_counts])  # mirror to [180, 360)
    return counts.astype(int), bin_edges
