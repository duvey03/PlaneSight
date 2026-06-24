"""Strike/dip via best-fit plane (PCA / SVD).

Recovers a plane's orientation from 3D points sampled along a trace where it
crosses topography. The algorithm and its degeneracy handling are specified in
ARCHITECTURE.md Section 6:

  - The plane normal is the smallest-singular-value direction of the demeaned
    points (the direction of least variance).
  - Two independent quality metrics, NOT one (decision D12):
      * conditioning = lambda2/lambda1 - is the fit constrained at all? Near 0
        means the points are collinear (a straight trace), so the normal is
        undefined *even at high relief*. This is the real degeneracy guard.
      * planarity = lambda3/lambda2 - given 2D spread, how planar is it? Near 0
        is a clean planar fit; larger means scatter or folding.
  - Angles use a compass convention: azimuth clockwise from North (y = North,
    x = East). Strike follows the right-hand rule (dip is 90 deg clockwise from
    strike). qgSurf/GeoTrace alignment is documented here and can be matched
    exactly later.

Validated against synthetic planes of known attitude (the D14 gate) in
tests/test_plane_fit.py before any use on real data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Attitude:
    """A single strike/dip measurement with quality metrics.

    Angles are in degrees. ``conditioning`` and ``planarity`` are the two
    eigenvalue-ratio metrics from ARCHITECTURE.md S6.3:

    - ``conditioning`` (lambda2/lambda1): ~0 = collinear/degenerate (reject);
      larger = better-constrained 2D spread.
    - ``planarity`` (lambda3/lambda2): ~0 = cleanly planar; larger = scatter/fold.

    ``dip_uncertainty`` and ``dip_direction_uncertainty`` (1-sigma, degrees) are
    populated when ``fit_plane`` is given a DEM vertical error ``sigma_z``
    (ARCHITECTURE.md S6.4); otherwise NaN. They are estimated by Monte-Carlo
    perturbation of the sampled elevations, so low-relief / poorly-conditioned
    traces automatically get large error bars.

    ``map_conditioning`` (lambda2/lambda1 of the (x, y) projection only) ~0 means
    the trace is straight *in map view* and so cannot constrain dip - even with
    relief, where the 3D ``conditioning`` is fooled into passing. It is the guard
    against the straight-segment / near-vertical pseudo-plane artifact seen in
    automatic detection (planesight-2je).
    """

    strike: float
    dip: float
    dip_direction: float
    conditioning: float
    planarity: float
    residual_rms: float
    relief: float
    n_samples: int
    dip_uncertainty: float = float("nan")
    dip_direction_uncertainty: float = float("nan")
    map_conditioning: float = float("nan")


def _fit_core(pts):
    """Core PCA fit: return (dip, dip_direction, strike, normal, eigvals, demeaned)."""
    q = pts - pts.mean(axis=0)
    _, s, vt = np.linalg.svd(q, full_matrices=False)
    normal = vt[2].copy()  # smallest singular value -> direction of least variance
    if normal[2] < 0.0:  # orient upward so dip in [0, 90] and azimuth downslope
        normal = -normal
    nz = max(-1.0, min(1.0, float(normal[2])))
    dip = math.degrees(math.acos(nz))
    dip_direction = math.degrees(math.atan2(normal[0], normal[1])) % 360.0
    strike = (dip_direction - 90.0) % 360.0
    return dip, dip_direction, strike, normal, s**2, q


def _estimate_uncertainty(pts, sigma_z, n_mc, seed):
    """1-sigma dip and dip-direction uncertainty via Monte-Carlo z-perturbation.

    NOTE: assumes *independent* per-point vertical noise, so for well-sampled,
    high-relief traces it is an optimistic LOWER BOUND - real DEM error is
    spatially correlated (it does not average down by sqrt(N)). It does, however,
    correctly blow up for low-relief / poorly-conditioned traces. A correlated-
    error (random-tilt) term is a planned refinement.
    """
    rng = np.random.default_rng(seed)
    n = pts.shape[0]
    samples = np.broadcast_to(pts, (n_mc, n, 3)).copy()
    samples[:, :, 2] += rng.normal(0.0, sigma_z, size=(n_mc, n))
    centred = samples - samples.mean(axis=1, keepdims=True)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)  # batched SVD
    normals = vt[:, 2, :]
    normals[normals[:, 2] < 0.0] *= -1.0
    nz = np.clip(normals[:, 2], -1.0, 1.0)
    dips = np.degrees(np.arccos(nz))
    dds = np.radians(np.degrees(np.arctan2(normals[:, 0], normals[:, 1])))
    # circular standard deviation for the (azimuthal) dip direction
    r = float(np.hypot(np.cos(dds).mean(), np.sin(dds).mean()))
    r = min(1.0, max(1e-12, r))
    dd_unc = math.degrees(math.sqrt(-2.0 * math.log(r)))
    return float(np.std(dips)), float(dd_unc)


def fit_plane(points, sigma_z=None, n_mc=200, seed=0) -> Attitude:
    """Fit a best-fit plane to 3D points and return the recovered Attitude.

    Args:
        points: array-like of shape (n, 3) - (x, y, z) in a metric CRS, with at
            least 3 points.
        sigma_z: optional DEM vertical 1-sigma error (metres). When given,
            dip/dip-direction uncertainty is estimated by Monte-Carlo
            perturbation (ARCHITECTURE.md S6.4). This assumes independent
            per-point noise and is an optimistic lower bound (see
            ``_estimate_uncertainty``).
        n_mc: Monte-Carlo iterations for the uncertainty estimate.
        seed: RNG seed, for reproducible uncertainty.

    Raises:
        ValueError: if fewer than 3 points or the wrong shape is given.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    n = pts.shape[0]
    if n < 3:
        raise ValueError("need at least 3 points to fit a plane")

    dip, dip_direction, strike, normal, lam, q = _fit_core(pts)
    l1, l2, l3 = float(lam[0]), float(lam[1]), float(lam[2])
    conditioning = l2 / l1 if l1 > 0.0 else 0.0
    planarity = l3 / l2 if l2 > 0.0 else 0.0

    # map-view conditioning: eigenvalue ratio of the (x, y) projection alone. ~0 =
    # straight map trace -> dip unconstrained, even when 3D conditioning passes due
    # to relief (planesight-2je).
    mev = np.linalg.eigvalsh(q[:, :2].T @ q[:, :2])  # ascending
    map_conditioning = float(mev[0] / mev[1]) if mev[1] > 0.0 else 0.0

    dists = q @ normal  # signed distance of each point to the fitted plane
    residual_rms = float(np.sqrt(np.mean(dists**2)))
    relief = float(pts[:, 2].max() - pts[:, 2].min())

    dip_unc = dd_unc = float("nan")
    if sigma_z is not None and sigma_z > 0.0:
        dip_unc, dd_unc = _estimate_uncertainty(pts, sigma_z, n_mc, seed)

    return Attitude(
        strike=strike,
        dip=dip,
        dip_direction=dip_direction,
        conditioning=conditioning,
        planarity=planarity,
        residual_rms=residual_rms,
        relief=relief,
        n_samples=n,
        dip_uncertainty=dip_unc,
        dip_direction_uncertainty=dd_unc,
        map_conditioning=map_conditioning,
    )
