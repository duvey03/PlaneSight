"""Trace-detection subpackage: the detector interface and its registry."""

from .base import (
    TraceDetector,
    available_detectors,
    get_detector,
    register_detector,
)

__all__ = [
    "TraceDetector",
    "available_detectors",
    "get_detector",
    "register_detector",
]
