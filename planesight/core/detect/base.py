"""Trace-detector interface and registry.

Detectors turn a multi-band raster stack into candidate trace polylines. v1 is a
classical (numpy/scipy) detector; an optional ONNX ML detector arrives later.
All detectors implement TraceDetector so the pipeline can swap backends without
change (ARCHITECTURE.md S5.3, S8).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Sequence, Tuple, Type

# A trace is an ordered sequence of (x, y) vertices.
Trace = Sequence[Tuple[float, float]]


class TraceDetector(ABC):
    """Abstract base class for trace detectors."""

    #: Short, unique registry key (e.g. "classical", "onnx"). Subclasses set it.
    name: str = ""

    @abstractmethod
    def detect(self, stack, transform=None) -> List[Trace]:
        """Detect candidate traces in a multi-band stack.

        Args:
            stack: array-like of shape (bands, rows, cols) - the input stack.
            transform: optional affine geotransform mapping pixel -> CRS coords.
                When given, returned vertices are in CRS coordinates; otherwise
                in pixel coordinates.

        Returns:
            A list of traces, each an ordered sequence of (x, y) vertices.
        """
        raise NotImplementedError


_REGISTRY: Dict[str, Type[TraceDetector]] = {}


def register_detector(cls: Type[TraceDetector]) -> Type[TraceDetector]:
    """Class decorator registering a detector under its ``name``."""
    key = getattr(cls, "name", "") or ""
    if not key:
        raise ValueError(f"{cls.__name__} must define a non-empty 'name'.")
    if key in _REGISTRY:
        raise ValueError(f"Detector name '{key}' is already registered.")
    _REGISTRY[key] = cls
    return cls


def get_detector(name: str) -> TraceDetector:
    """Instantiate a registered detector by name."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown detector '{name}'. Available: {available_detectors()}"
        )
    return _REGISTRY[name]()


def available_detectors() -> List[str]:
    """Return the sorted list of registered detector names."""
    return sorted(_REGISTRY)
