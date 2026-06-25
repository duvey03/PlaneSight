"""AggregateTask: run the data aggregator off the UI thread (GUI milestone M1).

Wraps planesight.core.aggregate.aggregate_terrain in a QgsTask so fetching the DEM /
Sentinel-2 and computing derivatives never freezes QGIS. Reports fractional progress
(via QgsTask.setProgress) and a human status line (via the ``message`` signal), and
honours cancellation at stage boundaries.
"""

from __future__ import annotations

from qgis.core import Qgis, QgsMessageLog, QgsTask
from qgis.PyQt.QtCore import pyqtSignal


class _CanceledError(Exception):
    """Internal: raised from the progress callback to unwind on cancel."""


class AggregateTask(QgsTask):
    """Background AOI -> raster-layer-specs task. Read ``specs``/``error`` when done."""

    message = pyqtSignal(str)

    def __init__(self, bbox, out_dir, *, include_s2: bool = True):
        super().__init__("PlaneSight: aggregate AOI layers", QgsTask.CanCancel)
        self._bbox = bbox
        self._out_dir = out_dir
        self._include_s2 = include_s2
        self.specs: list = []
        self.error: Exception | None = None

    def run(self) -> bool:
        """Executes on a worker thread; returns True on success."""
        def _progress(frac, msg):
            if self.isCanceled():
                raise _CanceledError()
            self.setProgress(max(1.0, frac * 100.0))
            self.message.emit(msg)

        try:
            # heavy import (GDAL/numpy/derivatives) - kept off the UI thread
            from planesight.core.aggregate import aggregate_terrain

            self.specs = aggregate_terrain(
                self._bbox, self._out_dir, include_s2=self._include_s2,
                progress=_progress,
            )
            return True
        except _CanceledError:
            return False
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI via `error`
            self.error = exc
            QgsMessageLog.logMessage(
                f"Aggregate failed: {exc}", "PlaneSight", Qgis.Critical
            )
            return False
