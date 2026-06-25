"""Strike/dip (attitude) computation subpackage."""

from .plane_fit import Attitude, fit_plane
from .sample import densify_line, sample_bilinear, sample_trace
from .variability import (
    CONTOUR_PARALLEL,
    STRAIGHT,
    V,
    circular_mean_strike,
    circular_resultant_length,
    classify_morphology,
    polyline_length,
    strike_difference,
    windowed_deviations,
)

__all__ = [
    "Attitude",
    "fit_plane",
    "densify_line",
    "sample_bilinear",
    "sample_trace",
    "strike_difference",
    "circular_mean_strike",
    "circular_resultant_length",
    "windowed_deviations",
    "classify_morphology",
    "polyline_length",
    "STRAIGHT",
    "CONTOUR_PARALLEL",
    "V",
]
