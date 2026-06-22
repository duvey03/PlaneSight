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
    """

    strike: float
    dip: float
    dip_direction: float
    conditioning: float
    planarity: float
    residual_rms: float
    relief: float
    n_samples: int


def fit_plane(points) -> Attitude:
    """Fit a best-fit plane to 3D points and return the recovered Attitude.

    Args:
        points: array-like of shape (n, 3) - (x, y, z) in a metric CRS, with at
            least 3 points.

    Raises:
        ValueError: if fewer than 3 points or the wrong shape is given.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    n = pts.shape[0]
    if n < 3:
        raise ValueError("need at least 3 points to fit a plane")

    centroid = pts.mean(axis=0)
    q = pts - centroid
    # SVD of the demeaned points; rows of vt are principal axes, s descending.
    _, s, vt = np.linalg.svd(q, full_matrices=False)
    normal = vt[2]  # smallest singular value -> direction of least variance
    if normal[2] < 0.0:  # orient upward so dip in [0, 90] and azimuth downslope
        normal = -normal
    nx, ny, nz = normal

    nz = max(-1.0, min(1.0, nz))  # guard arccos against float overshoot
    dip = math.degrees(math.acos(nz))
    dip_direction = math.degrees(math.atan2(nx, ny)) % 360.0
    strike = (dip_direction - 90.0) % 360.0

    lam = s**2  # eigenvalues of the covariance = squared singular values
    l1, l2, l3 = float(lam[0]), float(lam[1]), float(lam[2])
    conditioning = l2 / l1 if l1 > 0.0 else 0.0
    planarity = l3 / l2 if l2 > 0.0 else 0.0

    dists = q @ normal  # signed distance of each point to the fitted plane
    residual_rms = float(np.sqrt(np.mean(dists**2)))
    relief = float(pts[:, 2].max() - pts[:, 2].min())

    return Attitude(
        strike=strike,
        dip=dip,
        dip_direction=dip_direction,
        conditioning=conditioning,
        planarity=planarity,
        residual_rms=residual_rms,
        relief=relief,
        n_samples=n,
    )
