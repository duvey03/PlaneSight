"""Tests for the trace-detector interface and registry (pure, no QGIS)."""

import pytest

from planesight.core.detect import (
    TraceDetector,
    available_detectors,
    get_detector,
    register_detector,
)


def test_cannot_instantiate_abstract():
    with pytest.raises(TypeError):
        TraceDetector()


def test_register_and_get():
    @register_detector
    class _Dummy(TraceDetector):
        name = "dummy-test"

        def detect(self, stack, transform=None):
            return []

    assert "dummy-test" in available_detectors()
    det = get_detector("dummy-test")
    assert isinstance(det, TraceDetector)
    assert det.detect(None) == []


def test_unknown_detector_raises():
    with pytest.raises(KeyError):
        get_detector("nope-not-registered")


def test_unnamed_detector_rejected():
    with pytest.raises(ValueError):

        @register_detector
        class _Unnamed(TraceDetector):
            def detect(self, stack, transform=None):
                return []
