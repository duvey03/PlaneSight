"""Strike/dip via best-fit plane (PCA/SVD) - SCAFFOLD STUB.

The implementation lands in Phase 3 (issue planesight-lph). The algorithm and its
critical degeneracy handling are specified in ARCHITECTURE.md Section 6:

  - Fit a plane to DEM-sampled 3D trace points; the plane normal is the
    smallest-eigenvalue eigenvector of the demeaned point covariance.
  - Report TWO metrics, not one: conditioning ``lambda2/lambda1`` (the real
    degeneracy guard - straight/collinear traces are undefined even at high
    relief) and planarity ``lambda3/lambda2``.
  - Propagate dip/strike uncertainty from the DEM vertical-error budget.
  - Validate against a synthetic plane DEM before any real data (decision D14).

This stub fixes the public interface so the rest of the package can be built and
tested around it; it is intentionally not yet implemented.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Attitude:
    """A single strike/dip measurement with quality metrics.

    Angles are in degrees. ``conditioning`` and ``planarity`` are the two
    eigenvalue-ratio metrics from ARCHITECTURE.md S6.3.
    """

    strike: float
    dip: float
    dip_direction: float
    conditioning: float  # lambda2 / lambda1  (is the fit constrained?)
    planarity: float  # lambda3 / lambda2   (is the feature planar?)
    residual_rms: float
    relief: float
    n_samples: int


def fit_plane(points) -> Attitude:
    """Fit a plane to 3D points and return an Attitude. Not yet implemented.

    Args:
        points: array-like of shape (n, 3) - (x, y, z) points in a metric CRS.
    """
    raise NotImplementedError(
        "Strike/dip plane fitting is implemented in Phase 3 (planesight-lph). "
        "See ARCHITECTURE.md Section 6."
    )
