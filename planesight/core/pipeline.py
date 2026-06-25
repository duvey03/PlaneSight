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
from planesight.core.derivatives import build_terrain_stack
from planesight.core.detect import ClassicalTraceDetector
from planesight.core.detect.drainage import flag_drainage
from planesight.core.detect.vectorize import link_polylines, pixels_to_world

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
    # Continuity link runs AFTER drainage removal (the planesight-61f contract):
    # linking before would reconnect creek fragments.
    linked = link_polylines(kept)

    attitudes: list[AttitudePoint] = []
    for tr_px in linked:
        world = pixels_to_world(tr_px[:, ::-1], gt)
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
        attitudes.append(AttitudePoint(cx, cy, att, reliable))

    return PipelineResult(
        traces=traces, flags=flags, kept=kept, linked=linked, attitudes=attitudes
    )
