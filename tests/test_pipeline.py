"""Contract tests for the headless detect->strike/dip orchestration (core/pipeline.py).

The component algorithms (detect/drainage/link/fit) have their own unit tests; these
check the ORCHESTRATION: the stages wire together, the result structure is internally
consistent, and degenerate input doesn't crash. Pure numpy - no GDAL/QGIS.
"""

import numpy as np

from planesight.core.pipeline import (
    AttitudePoint,
    CandidateTrace,
    PipelineResult,
    detect_attitudes,
    detect_traces,
    fit_traces,
)

# A standard north-up geotransform (origin, +x res, 0, origin, 0, -y res).
GT = (500000.0, 30.0, 0.0, 3000000.0, 0.0, -30.0)


def _escarpment_dem(h=120, w=120, res=30.0):
    """A gentle tilt with a sharp diagonal escarpment -> a detectable contact trace."""
    rr, cc = np.indices((h, w))
    tilt = (h - 1 - rr).astype(float) * 0.5 * res            # relief for plane-fitting
    step = 200.0 / (1.0 + np.exp(-(cc - rr) / 2.0))          # sigmoid step across cc==rr
    return tilt + step


def test_pipeline_returns_consistent_structure():
    dem = _escarpment_dem()
    res = detect_attitudes(dem, GT, res=30.0)

    assert isinstance(res, PipelineResult)
    # one drainage flag per detected trace
    assert len(res.flags) == len(res.traces)
    # kept = the non-flagged traces; never more than detected
    assert len(res.kept) == sum(not f.is_drainage for f in res.flags)
    assert len(res.kept) <= len(res.traces)
    assert isinstance(res.linked, list)
    # every attitude is well-formed and tied to a finite map location
    for ap in res.attitudes:
        assert isinstance(ap, AttitudePoint)
        assert np.isfinite(ap.x) and np.isfinite(ap.y)
        assert np.isfinite(ap.attitude.dip)
        assert isinstance(ap.reliable, bool)


def test_pipeline_detects_a_clear_escarpment():
    # the engineered linear contact must surface at least one trace through the
    # whole orchestration (detection sanity, end to end).
    res = detect_attitudes(_escarpment_dem(), GT, res=30.0)
    assert len(res.traces) > 0


def test_pipeline_flat_dem_yields_no_traces_without_crashing():
    flat = np.full((80, 80), 100.0)
    res = detect_attitudes(flat, GT, res=30.0)
    assert res.traces == [] or len(res.traces) == 0
    assert res.attitudes == []
    assert len(res.flags) == len(res.traces)


def test_pipeline_handles_nodata_nan():
    dem = _escarpment_dem()
    dem[:10, :10] = np.nan                 # a nodata corner
    res = detect_attitudes(dem, GT, res=30.0)
    assert isinstance(res, PipelineResult)
    for ap in res.attitudes:
        assert np.isfinite(ap.x) and np.isfinite(ap.y)


# --- fit_traces (M2: strike/dip on supplied traces) ---

GT10 = (0.0, 10.0, 0.0, 0.0, 0.0, -10.0)   # origin 0,0; 10 m pixels; north-up


def _east_dipping_dem(dip_deg=15.0, res=10.0, h=60, w=60):
    """A planar surface dipping due East at a known angle (elevation falls with +x)."""
    g = np.tan(np.radians(dip_deg))
    _, cc = np.indices((h, w))
    return -g * (cc * res)


def test_fit_traces_recovers_known_dip_and_direction():
    # an L-shaped trace (spans x AND y) on a 15 deg East-dipping plane: the fit must
    # recover dip 15, dip_direction 090 exactly (DEM sampling is exact on a linear field).
    dem = _east_dipping_dem(15.0)
    trace = np.array([[50.0, -50.0], [400.0, -50.0], [400.0, -400.0]])
    atts = fit_traces([trace], dem, GT10, res=10.0)
    assert len(atts) == 1
    a = atts[0].attitude
    assert abs(a.dip - 15.0) < 0.5
    assert abs(((a.dip_direction - 90.0 + 180) % 360) - 180) < 1.5
    assert atts[0].reliable is True


def test_fit_traces_skips_empty_and_too_short():
    dem = _east_dipping_dem()
    assert fit_traces([], dem, GT10, res=10.0) == []
    assert fit_traces([np.array([[10.0, -10.0]])], dem, GT10, res=10.0) == []


# --- detect_traces (M4: review-gate detection without fitting) ---


def test_detect_traces_returns_candidates():
    dem = _escarpment_dem()
    cands = detect_traces(dem, GT, res=30.0)
    assert isinstance(cands, list) and len(cands) > 0
    for c in cands:
        assert isinstance(c, CandidateTrace)
        assert c.geometry.ndim == 2 and c.geometry.shape[1] == 2
        assert isinstance(c.is_drainage, bool)
        assert 0.0 <= c.score <= 1.0          # detector-agnostic normalized confidence
        assert c.rank >= 0.0 and c.length >= 0.0
    # kept (non-drainage) candidates are high-confidence by construction
    assert all(c.score == 1.0 for c in cands if not c.is_drainage)
