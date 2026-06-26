"""BuildCostTask: precompute the live-wire cost surface + snap field off the UI thread.

The assisted-tracing map tool needs a cost surface (cheap on contacts) and a snap field
before it can trace. Both are scipy-heavy (curvature, flow-accumulation drainage, the
distance-transform snap), so they are built once per DEM in a QgsTask - the UI thread
never does the cold scipy load or the per-AOI grid work. Read ``cost``/``snap``/``gt``/
``error`` when finished and hand them to LiveWireMapTool. Mirrors tasks/detect.py.
"""

from __future__ import annotations

import numpy as np
from qgis.core import Qgis, QgsMessageLog, QgsTask


class BuildCostTask(QgsTask):
    """Background build of the live-wire cost surface + snap field from a DEM raster."""

    def __init__(self, dem_path, *, drainage_weight: float = 3.0, snap_budget: float = 0.07):
        super().__init__("PlaneSight: build trace cost surface", QgsTask.CanCancel)
        self._dem_path = dem_path
        self._drainage_weight = float(drainage_weight)
        self._snap_budget = float(snap_budget)
        self.cost = None        # (H, W) float cost grid
        self.snap = None        # SnapField
        self.gt = None          # GDAL geotransform
        self.error: Exception | None = None

    def run(self) -> bool:
        try:
            from osgeo import gdal

            # heavy (scipy) - imported off the UI thread
            from planesight.core.trace import (
                build_cost_surface,
                build_snap_field,
                curvature_magnitude,
                drainage_penalty,
            )

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
            valid = np.isfinite(dem)
            px = abs(gt[1]) or 30.0

            # curvature + drainage. The cross-AOI sensitivity study (Nepal/Pakistan/Canada,
            # debug/trace_sensitivity_study.py) found that adding slope/TPI/Canny to the
            # cost surface consistently REDUCES trace adherence (more cheap pixels -> the
            # wire drifts onto parallel edges); curv+drainage is best or tied everywhere.
            # The richer blend (core.trace.cost.trace_cost_surface) stays available for the
            # curvature-invisible case the ML probability map (T3) will target.
            curv_mag = curvature_magnitude(dem, px)
            kwargs = {"w_edge": 1.0}
            if self._drainage_weight > 0:
                kwargs["drainage_penalty"] = drainage_penalty(dem)
                kwargs["w_drain"] = self._drainage_weight
            self.cost = build_cost_surface(curv_mag, valid=valid, **kwargs)
            self.snap = build_snap_field(curv_mag, valid=valid, budget=self._snap_budget)
            self.gt = gt
            return True
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI via `error`
            self.error = exc
            QgsMessageLog.logMessage(
                f"Cost-surface build failed: {exc}", "PlaneSight", Qgis.Critical
            )
            return False
