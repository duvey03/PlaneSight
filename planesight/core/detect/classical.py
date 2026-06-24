"""ClassicalTraceDetector - the v1 trace detector (numpy/scipy only, D11).

Wires the Phase 1 conclusions into one swappable detector (ARCHITECTURE.md S5.3,
S8.1): combine the input bands into a response, run Canny (the chosen front-end,
planesight-0pb), then vectorise to polylines (the back-end). Registered as
``"classical"`` so the pipeline can later swap in an ONNX ML detector unchanged.

Pass the top DEM bands (profile_curvature / curvature / slope) for the
experiment-winning behaviour; iron-oxide / seasonal SWIR can be added as secondary
channels. Bands are normalised and weighted-averaged into a single response before
Canny - simple and matching the finding that a single strong band already detects
well and multi-band fusion adds little recall.
"""

from __future__ import annotations

import numpy as np

from .base import TraceDetector, register_detector
from .canny import canny
from .vectorize import polylines_from_mask


@register_detector
class ClassicalTraceDetector(TraceDetector):
    """Canny-on-DEM-bands classical detector."""

    name = "classical"

    def __init__(self, weights=None, sigma: float = 1.0,
                 low_quantile: float = 0.85, high_quantile: float = 0.95,
                 min_length: int = 5, simplify_tol: float = 1.0):
        self.weights = weights
        self.sigma = sigma
        self.low_quantile = low_quantile
        self.high_quantile = high_quantile
        self.min_length = min_length
        self.simplify_tol = simplify_tol

    def _response(self, stack):
        """Normalise each band to [0, 1] and weighted-average into one response.

        A pixel that is NaN in any band stays NaN (consistent valid footprint).
        """
        from ..derivatives import normalize01  # lazy: avoids import-order coupling

        bands = [normalize01(stack[i]) for i in range(stack.shape[0])]
        w = np.ones(len(bands)) if self.weights is None else np.asarray(self.weights, float)
        if w.shape[0] != len(bands):
            raise ValueError("weights must match the number of bands")
        acc = np.zeros(bands[0].shape)
        for wt, b in zip(w, bands):
            acc += wt * b
        return acc / w.sum()

    def detect(self, stack, transform=None):
        """Detect candidate traces in a multi-band stack.

        Args:
            stack: array of shape (bands, rows, cols), or a single (rows, cols)
                response/band.
            transform: optional GDAL geotransform; when given, vertices are world
                (x, y), else pixel (col, row).

        Returns:
            List of traces, each an ``(n, 2)`` array of vertices.
        """
        stack = np.asarray(stack, dtype=float)
        if stack.ndim == 2:
            stack = stack[None]
        elif stack.ndim != 3:
            raise ValueError("stack must be (bands, rows, cols) or (rows, cols)")
        response = self._response(stack)
        edges = canny(response, sigma=self.sigma, low_quantile=self.low_quantile,
                      high_quantile=self.high_quantile,
                      valid_mask=np.isfinite(response))
        return polylines_from_mask(edges, min_length=self.min_length,
                                   simplify_tol=self.simplify_tol, transform=transform)
