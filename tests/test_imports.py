"""Smoke tests: the pure-Python core imports without a QGIS runtime."""

import planesight
from planesight.core import attitude, detect


def test_version():
    assert isinstance(planesight.__version__, str)
    assert planesight.__version__.count(".") >= 2


def test_core_subpackages_import():
    # core must import with no QGIS runtime present
    assert hasattr(attitude, "fit_plane")
    assert hasattr(detect, "TraceDetector")
