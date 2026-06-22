"""PlaneSight QGIS plugin package entry point.

Exposes classFactory(), which QGIS calls to instantiate the plugin. QGIS/Qt
imports are deferred into classFactory so the pure-Python core
(``planesight.core``) stays importable outside QGIS - e.g. in CI and unit tests.
"""

__version__ = "0.0.1"


def classFactory(iface):
    """Instantiate and return the PlaneSight plugin for the given QGIS iface."""
    from .plugin import PlaneSightPlugin

    return PlaneSightPlugin(iface)
