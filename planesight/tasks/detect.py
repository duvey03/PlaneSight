"""DetectTask: run candidate-trace detection off the UI thread (GUI M4).

Wraps planesight.core.pipeline.detect_traces (detect -> drainage-flag -> link, no fit)
in a QgsTask. The heavy, scipy-using detection import happens inside run() on the worker
thread, so enabling/using the plugin never triggers the slow scipy cold-load on the UI
thread. Read ``candidates``/``gt``/``error`` when finished.
"""

from __future__ import annotations

import numpy as np
from qgis.core import Qgis, QgsMessageLog, QgsTask


class DetectTask(QgsTask):
    """Background candidate-trace detection on a DEM raster."""

    def __init__(self, dem_path):
        super().__init__("PlaneSight: detect candidate traces", QgsTask.CanCancel)
        self._dem_path = dem_path
        self.candidates: list = []
        self.gt = None
        self.error: Exception | None = None

    def run(self) -> bool:
        try:
            from osgeo import gdal

            from planesight.core.pipeline import detect_traces  # heavy (scipy); off-thread

            gdal.UseExceptions()
            ds = gdal.Open(self._dem_path)
            if ds is None:
                raise ValueError(f"could not open DEM raster {self._dem_path!r}")
            dem = ds.ReadAsArray().astype(float)
            gt = ds.GetGeoTransform()
            nodata = ds.GetRasterBand(1).GetNoDataValue()
            ds = None
            if nodata is not None:
                dem = np.where(dem == nodata, np.nan, dem)
            self.gt = gt
            self.candidates = detect_traces(dem, gt, res=abs(gt[1]) or 30.0)
            return True
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI via `error`
            self.error = exc
            QgsMessageLog.logMessage(
                f"Detection failed: {exc}", "PlaneSight", Qgis.Critical
            )
            return False
