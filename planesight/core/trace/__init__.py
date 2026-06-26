"""Assisted (live-wire) contact tracing core - dependency-free (planesight-fe7).

A cost surface where contact-like pixels are cheap + a least-cost path solver, so the
GUI map tool can snap a "wire" along a geological contact between a few user clicks. Pure
numpy/scipy; PyQGIS/Qt live in the gui layer.
"""

from planesight.core.trace.cost import build_cost_surface, curvature_magnitude
from planesight.core.trace.livewire import (
    LiveWireField,
    backtrace,
    cost_to_all,
    least_cost_path,
)

__all__ = [
    "LiveWireField",
    "backtrace",
    "build_cost_surface",
    "cost_to_all",
    "curvature_magnitude",
    "least_cost_path",
]
