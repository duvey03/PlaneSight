"""Strike/dip (attitude) computation subpackage."""

from .plane_fit import Attitude, fit_plane
from .sample import densify_line, sample_bilinear, sample_trace

__all__ = [
    "Attitude",
    "fit_plane",
    "densify_line",
    "sample_bilinear",
    "sample_trace",
]
