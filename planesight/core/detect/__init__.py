"""Trace-detection subpackage: the detector interface and its registry."""

from .base import (
    TraceDetector,
    available_detectors,
    get_detector,
    register_detector,
)
from .score import (
    detect_at_budget,
    disk,
    recall_at_budget,
    recall_curve,
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
]
