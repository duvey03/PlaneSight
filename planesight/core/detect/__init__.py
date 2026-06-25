"""Trace-detection subpackage: the detector interface and its registry."""

from .base import (
    TraceDetector,
    available_detectors,
    get_detector,
    register_detector,
)
from .canny import canny, gaussian_gradient, hysteresis, non_max_suppression
from .classical import ClassicalTraceDetector
from .linearity import (
    linearity_at_budget,
    linearity_metrics,
)
from .score import (
    detect_at_budget,
    disk,
    recall_at_budget,
    recall_curve,
)
from .structure import (
    linear_response,
    structure_tensor,
    tensor_eigenvalues,
    tensor_response,
)
from .vectorize import (
    close_gaps,
    extract_polylines,
    link_polylines,
    pixels_to_world,
    polylines_from_mask,
    simplify,
    thin,
    trace_skeleton,
)

__all__ = [
    "TraceDetector",
    "available_detectors",
    "get_detector",
    "register_detector",
    "disk",
    "detect_at_budget",
    "recall_at_budget",
    "recall_curve",
    "linearity_metrics",
    "linearity_at_budget",
    "structure_tensor",
    "tensor_eigenvalues",
    "tensor_response",
    "linear_response",
    "thin",
    "trace_skeleton",
    "simplify",
    "pixels_to_world",
    "polylines_from_mask",
    "extract_polylines",
    "link_polylines",
    "close_gaps",
    "gaussian_gradient",
    "non_max_suppression",
    "hysteresis",
    "canny",
    "ClassicalTraceDetector",
]
