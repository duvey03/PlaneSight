"""Structural-analysis math (stereonet, Fisher stats, fold axis).

Pure-numpy core for GUI milestone M3 (the "Analyze" tool): lower-hemisphere
equal-area (Schmidt) projection, Fisher (1953) directional statistics, the
orientation tensor + eigen-analysis, axial means, fold-axis (girdle) recovery,
and rose diagrams. Dependency-free (numpy only); no plotting - rendering is a
later, separate GUI decision.

Conventions match ``planesight.core.attitude.plane_fit`` exactly (right-hand-rule
strike, dip-direction = strike + 90 measured clockwise from North); see
``stereonet`` module docstring and ``docs/STEREONET_MATH.md``.
"""

from .stereonet import (
    AxialMean,
    FisherStats,
    FoldAxis,
    PrincipalOrientations,
    axial_mean,
    equal_area_xy,
    fisher_mean,
    fold_axis,
    great_circle,
    line_to_xyz,
    orientation_tensor,
    pole_to_strike_dip,
    principal_orientations,
    rose_bins,
    strike_dip_to_pole,
    xyz_to_line,
)

__all__ = [
    "AxialMean",
    "FisherStats",
    "FoldAxis",
    "PrincipalOrientations",
    "axial_mean",
    "equal_area_xy",
    "fisher_mean",
    "fold_axis",
    "great_circle",
    "line_to_xyz",
    "orientation_tensor",
    "pole_to_strike_dip",
    "principal_orientations",
    "rose_bins",
    "strike_dip_to_pole",
    "xyz_to_line",
]
