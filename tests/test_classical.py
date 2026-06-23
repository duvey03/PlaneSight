"""Tests for the ClassicalTraceDetector end-to-end (pure numpy/scipy)."""

import numpy as np

from planesight.core.attitude import fit_plane, sample_trace
from planesight.core.detect import (
    ClassicalTraceDetector,
    available_detectors,
    get_detector,
)


def _largest(traces):
    """The spatially largest trace (by bounding-box diagonal) - robust to the
    vertex count collapsing under simplification."""
    return max(traces, key=lambda t: np.hypot(np.ptp(t[:, 0]), np.ptp(t[:, 1])))


def test_registered_in_the_registry():
    assert "classical" in available_detectors()
    assert isinstance(get_detector("classical"), ClassicalTraceDetector)


def test_detects_a_synthetic_contact_as_a_polyline():
    # A band with a sharp linear contact (a step across the middle).
    band = np.zeros((40, 60))
    band[:, 30:] = 1.0
    traces = ClassicalTraceDetector(min_length=5).detect(band)
    assert len(traces) >= 1
    main = _largest(traces)
    # the contact runs vertically near col 30 -> x roughly constant ~30
    xs = main[:, 0]
    assert abs(np.median(xs) - 30) <= 2
    assert main[:, 1].max() - main[:, 1].min() > 20  # spans most of the height


def test_accepts_multiband_stack_and_weights():
    b1 = np.zeros((30, 30))
    b1[15, :] = 1.0
    b2 = np.zeros((30, 30))
    b2[15, :] = 1.0
    det = ClassicalTraceDetector(weights=[2.0, 1.0], min_length=5)
    traces = det.detect(np.stack([b1, b2]))
    assert len(traces) >= 1


def test_transform_yields_world_coordinates():
    band = np.zeros((30, 30))
    band[:, 15:] = 1.0
    gt = (1000.0, 10.0, 0.0, 5000.0, 0.0, -10.0)
    traces = ClassicalTraceDetector(min_length=5).detect(band, transform=gt)
    main = _largest(traces)
    # vertical contact near col 15 -> world x ~ 1000 + 15.5*10 = 1155
    assert abs(np.median(main[:, 0]) - 1155) <= 20


def test_detected_trace_feeds_the_attitude_engine():
    # Wiring check: a detected polyline must be consumable by the strike/dip path
    # (sample_trace -> fit_plane) without error. Geological validation on real
    # sinuous traces is the job of the wiu eval, not a synthetic - a straight
    # trace is degenerate for the plane fit by design (low conditioning).
    band = np.zeros((50, 50))
    band[:, 25:] = 1.0  # a vertical contact
    gt = (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)  # identity-ish pixel grid
    traces = ClassicalTraceDetector(min_length=10).detect(band, transform=gt)
    assert traces
    line = _largest(traces)
    dem = np.indices((50, 50))[1] * 0.2  # z = 0.2 * col (dips east)
    pts = sample_trace(line, dem, gt, spacing=1.0)
    assert len(pts) >= 3  # the polyline densified into samplable 3D points
    att = fit_plane(pts)
    assert np.isfinite(att.conditioning)  # returns a valid Attitude, not a crash
