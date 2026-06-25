"""AttitudeTask: fit strike/dip along supplied world-coordinate traces (GUI M2).

Wraps planesight.core.pipeline.fit_traces in a QgsTask. The caller (dockwidget)
extracts + reprojects the trace vertices to the DEM's CRS on the main thread (QGIS
geometry access is not thread-safe) and passes plain numpy arrays here; this task
reads the DEM array via GDAL and runs the (pure-numpy) fit off the UI thread.
"""

from __future__ import annotations

import numpy as np
from qgis.core import Qgis, QgsMessageLog, QgsTask


class AttitudeTask(QgsTask):
    """Background strike/dip fit. Read ``attitudes``/``error`` when finished."""

    def __init__(self, world_traces, dem_path, *, sigma_z: float = 2.0):
        super().__init__("PlaneSight: strike/dip on supplied traces", QgsTask.CanCancel)
        self._traces = world_traces
        self._dem_path = dem_path
        self._sigma_z = sigma_z
        self.attitudes: list = []
        self.error: Exception | None = None

    def run(self) -> bool:
        try:
            from osgeo import gdal

            from planesight.core.pipeline import fit_traces  # heavy (scipy); off-UI-thread

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
            res = abs(gt[1]) or 30.0
            self.attitudes = fit_traces(
                self._traces, dem, gt, res=res, sigma_z=self._sigma_z
            )
            return True
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI via `error`
            self.error = exc
            QgsMessageLog.logMessage(
                f"Strike/dip failed: {exc}", "PlaneSight", Qgis.Critical
            )
            return False
