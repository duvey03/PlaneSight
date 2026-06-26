"""Headless detection -> strike/dip pipeline orchestration (dependency-free core).

One place for the "DEM array in -> attitudes out" sequence (detect -> drainage-flag ->
link -> plane-fit) that the QGIS GUI (M1 QgsTask), the in-QGIS checkpoint, and the
headless demo scripts all share. GDAL/I/O stays at the edges (callers fetch + warp the
DEM and build the geotransform); this module is pure numpy on arrays, so it is unit-
testable without GDAL or QGIS. See ARCHITECTURE.md S5.3 (execution model).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from planesight.core.attitude import fit_plane, sample_trace
from planesight.core.attitude.plane_fit import Attitude

# NOTE: the detection stack (core.detect/core.derivatives) pulls in scipy, which is a
# slow cold-import (esp. on Windows). It is imported LAZILY inside detect_attitudes()
# so the strike/dip-only path (fit_traces -> M2) never loads it. fit_traces below uses
# only fit_plane/sample_trace (pure numpy).

# Phase 1 winners (docs/PHASE1_REPORT.md): DEM curvature/slope dominate detection.
DEFAULT_BANDS = ("profile_curvature", "curvature", "slope")


class AttitudePoint(NamedTuple):
    """A fitted attitude with the map location (world coords) of its trace centroid."""

    x: float
    y: float
    attitude: Attitude
    reliable: bool


class PipelineResult(NamedTuple):
    """Everything the detection->strike/dip pass produces, for the GUI/checkpoint."""

    traces: list          # all detected pixel-space (col,row) polylines
    flags: list           # one DrainageFlag per trace (parallel to `traces`)
    kept: list            # traces NOT drainage-flagged (pixel space)
    linked: list          # kept traces after continuity linking (pixel space)
    attitudes: list       # list[AttitudePoint] from the linked, kept traces


def fit_traces(
    world_traces,
    dem: np.ndarray,
    gt,
    *,
    res: float = 30.0,
    sigma_z: float = 2.0,
    min_trace_pts: int = 8,
    cond_reliable: float = 1e-2,
    map_cond_reliable: float = 1e-3,
) -> list:
    """Fit strike/dip along each world-coordinate polyline trace.

    The shared trace -> attitude step used by both detection (detect_attitudes,
    on detected traces) and the GUI's "strike/dip on supplied traces" (M2, on
    user/hand-drawn traces). Traces must already be in the DEM's metric CRS.

    Args:
        world_traces: iterable of ``(N, 2)`` arrays of (x, y) world coordinates.
        dem: 2D elevation array (metric CRS, NaN nodata).
        gt: GDAL geotransform mapping pixel <-> world for ``dem``.
        res: along-trace sampling spacing (m).
        sigma_z, min_trace_pts, cond_reliable, map_cond_reliable: as in
            :func:`detect_attitudes`.

    Returns:
        list[AttitudePoint] - one per trace that yields a finite fit with enough
        samples; the point location is the trace centroid in world coordinates.
    """
    out: list[AttitudePoint] = []
    for trace in world_traces:
        world = np.asarray(trace, dtype=float)
        if world.ndim != 2 or world.shape[0] < 2:
            continue
        pts = sample_trace(world, dem, gt, spacing=res, nodata=None)
        if len(pts) < min_trace_pts:
            continue
        att = fit_plane(pts, sigma_z=sigma_z)
        if not np.isfinite(att.dip):
            continue
        cx, cy = float(np.mean(world[:, 0])), float(np.mean(world[:, 1]))
        reliable = bool(
            att.conditioning >= cond_reliable
            and att.map_conditioning >= map_cond_reliable
        )
        out.append(AttitudePoint(cx, cy, att, reliable))
    return out


class CandidateTrace(NamedTuple):
    """A detected candidate trace for the review gate (detector-agnostic).

    ``geometry`` is (N, 2) world coords. ``is_drainage`` marks a drainage-flagged trace
    (a review/rescue candidate). ``score`` is a normalized confidence in [0, 1] that the
    trace is a genuine geological contact - the classical detector fills it from the
    drainage signal; a future ML detector would fill it from its probability, so the
    review UI never needs to change. ``rank`` is the review priority (higher = sooner).
    ``length`` is in metres.
    """

    geometry: np.ndarray
    is_drainage: bool
    score: float
    rank: float
    length: float


def _polyline_length(world):
    w = np.asarray(world, dtype=float)
    if w.shape[0] < 2:
        return 0.0
    return float(np.hypot(*np.diff(w, axis=0).T).sum())


def _detect_and_link(
    dem, *, res, bands, min_trace_pts, simplify_tol,
    drain_downsample, drain_accum, drain_overlap,
):
    """Shared detection core: terrain stack -> detect -> drainage-flag -> link kept.

    Returns ``(traces, flags, kept, linked)`` in PIXEL space. The scipy-heavy detection
    imports are lazy (kept off the strike/dip-only path). Used by both detect_traces
    (review gate, no fit) and detect_attitudes (full pipeline).
    """
    from planesight.core.derivatives import build_terrain_stack
    from planesight.core.detect import ClassicalTraceDetector
    from planesight.core.detect.drainage import flag_drainage
    from planesight.core.detect.vectorize import link_polylines

    valid = np.isfinite(dem)
    _, terr = build_terrain_stack(dem, res, names=bands)
    stack = np.stack([terr[b] for b in bands])
    traces = ClassicalTraceDetector(
        min_length=min_trace_pts, simplify_tol=simplify_tol
    ).detect(stack)
    flags = flag_drainage(
        traces, dem, downsample=drain_downsample, min_accum_cells=drain_accum,
        min_overlap_fraction=drain_overlap, valid_mask=valid,
    )
    kept = [t for t, f in zip(traces, flags) if not f.is_drainage]
    linked = link_polylines(kept)   # AFTER drainage removal (planesight-61f contract)
    return traces, flags, kept, linked


def detect_traces(
    dem: np.ndarray,
    gt,
    *,
    res: float = 30.0,
    bands=DEFAULT_BANDS,
    min_trace_pts: int = 8,
    simplify_tol: float = 1.0,
    drain_downsample: int = 3,
    drain_accum: int = 15,
    drain_overlap: float = 0.5,
) -> list:
    """Detect candidate contact traces (detect -> drainage-flag -> link), WITHOUT fitting.

    The detection step for the GUI review gate (M4): returns world-coordinate
    :class:`CandidateTrace` objects so a human can accept/reject/classify before
    strike/dip is fitted (via :func:`fit_traces`) on the accepted set. Kept (linked)
    traces score 1.0; drainage-flagged traces carry a creek-vs-contact confidence
    (1 - monotonicity) and the drainage rescue rank. Detector-agnostic output.
    """
    from planesight.core.detect.vectorize import pixels_to_world

    traces, flags, _kept, linked = _detect_and_link(
        dem, res=res, bands=bands, min_trace_pts=min_trace_pts, simplify_tol=simplify_tol,
        drain_downsample=drain_downsample, drain_accum=drain_accum,
        drain_overlap=drain_overlap,
    )
    out: list[CandidateTrace] = []
    for tr_px in linked:                         # kept + linked -> high-confidence contacts
        world = pixels_to_world(tr_px[:, ::-1], gt)
        out.append(CandidateTrace(world, False, 1.0, 0.0, _polyline_length(world)))
    for tr_px, f in zip(traces, flags):          # drainage-flagged -> review/rescue queue
        if not f.is_drainage:
            continue
        world = pixels_to_world(tr_px[:, ::-1], gt)
        mono = 1.0 if not np.isfinite(f.monotonicity) else f.monotonicity
        out.append(CandidateTrace(
            world, True, float(np.clip(1.0 - mono, 0.0, 1.0)),
            float(f.rank), _polyline_length(world),
        ))
    return out


def detect_attitudes(
    dem: np.ndarray,
    gt,
    *,
    res: float = 30.0,
    bands=DEFAULT_BANDS,
    sigma_z: float = 2.0,
    min_trace_pts: int = 8,
    simplify_tol: float = 1.0,
    drain_downsample: int = 3,
    drain_accum: int = 15,
    drain_overlap: float = 0.5,
    cond_reliable: float = 1e-2,
    map_cond_reliable: float = 1e-3,
) -> PipelineResult:
    """Run detect -> drainage-flag -> link -> strike/dip on a metric-CRS DEM array.

    Args:
        dem: 2D elevation array in a metric CRS (NaN = nodata). Caller is responsible
            for fetching/warping to UTM; this stays pure-array.
        gt: GDAL-style geotransform (6-tuple) mapping pixel -> world for `dem`.
        res: ground sample distance (m); used as the along-trace sampling spacing.
        bands: terrain derivative bands to detect on (Phase 1 winners by default).
        sigma_z: DEM vertical noise (m) for the plane-fit uncertainty budget.
        min_trace_pts: minimum sampled points for a usable fit (also the detector's
            minimum trace length).
        simplify_tol: Douglas-Peucker tolerance (px) for the detector.
        drain_downsample, drain_accum, drain_overlap: drainage flag parameters
            (planesight-61f: overlap-based, sinuosity-robust).
        cond_reliable, map_cond_reliable: conditioning gates marking an attitude
            `reliable` (3D conditioning and map-view conditioning, planesight-2je).

    Returns:
        A PipelineResult. Drainage-flagged traces are retained (review-flag, not a
        delete) in `flags`/excluded from `linked`; `attitudes` come only from the
        kept+linked traces.
    """
    from planesight.core.detect.vectorize import pixels_to_world

    traces, flags, kept, linked = _detect_and_link(
        dem, res=res, bands=bands, min_trace_pts=min_trace_pts, simplify_tol=simplify_tol,
        drain_downsample=drain_downsample, drain_accum=drain_accum,
        drain_overlap=drain_overlap,
    )
    world_traces = [pixels_to_world(tr[:, ::-1], gt) for tr in linked]
    attitudes = fit_traces(
        world_traces, dem, gt, res=res, sigma_z=sigma_z, min_trace_pts=min_trace_pts,
        cond_reliable=cond_reliable, map_cond_reliable=map_cond_reliable,
    )
    return PipelineResult(
        traces=traces, flags=flags, kept=kept, linked=linked, attitudes=attitudes
    )
