"""Assisted (live-wire) contact tracing core - dependency-free (planesight-fe7).

A cost surface where contact-like pixels are cheap + a least-cost path solver, so the
GUI map tool can snap a "wire" along a geological contact between a few user clicks. Pure
numpy/scipy; PyQGIS/Qt live in the gui layer.
"""

from planesight.core.trace.cost import (
    build_cost_surface,
    contact_strength,
    curvature_magnitude,
    trace_cost_surface,
)
from planesight.core.trace.livewire import (
    LiveWireField,
    backtrace,
    cost_to_all,
    least_cost_path,
)
from planesight.core.trace.penalties import drainage_penalty, orientation_incoherence
from planesight.core.trace.snap import (
    SnapField,
    build_snap_field,
    edge_mask,
    snap_point,
)

__all__ = [
    "LiveWireField",
    "SnapField",
    "backtrace",
    "build_cost_surface",
    "build_snap_field",
    "contact_strength",
    "cost_to_all",
    "curvature_magnitude",
    "drainage_penalty",
    "edge_mask",
    "least_cost_path",
    "orientation_incoherence",
    "snap_point",
    "trace_cost_surface",
]
