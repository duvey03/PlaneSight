"""Contract tests for the headless detect->strike/dip orchestration (core/pipeline.py).

The component algorithms (detect/drainage/link/fit) have their own unit tests; these
check the ORCHESTRATION: the stages wire together, the result structure is internally
consistent, and degenerate input doesn't crash. Pure numpy - no GDAL/QGIS.
"""

import numpy as np

from planesight.core.pipeline import AttitudePoint, PipelineResult, detect_attitudes

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
