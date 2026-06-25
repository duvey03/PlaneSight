"""PlaneSight dockwidget (GUI milestone M1): the data-aggregator panel.

The first real workflow UI: choose an AOI (current map extent or a drawn rectangle),
fetch the DEM + Sentinel-2 + terrain derivatives via a background AggregateTask, and
load the styled rasters. Later milestones add the strike/dip, structural-analysis, and
detection/review panels to this same dock. All heavy work runs off the UI thread.
"""

from __future__ import annotations

import tempfile

from qgis.core import (
    QgsApplication,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsProject,
    QgsRasterLayer,
)
from qgis.gui import QgsDockWidget, QgsMapToolExtent
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QLabel,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from ..tasks.aggregate import AggregateTask

WGS84 = QgsCoordinateReferenceSystem("EPSG:4326")
MAX_AOI_DEG = 0.5      # v1 guard (~55 km/side) against accidental continent-sized fetches


class PlaneSightDockWidget(QgsDockWidget):
    """Dockable AOI -> layers panel."""

    def __init__(self, iface, parent=None):
        super().__init__("PlaneSight", parent)
        self.iface = iface
        self.canvas = iface.mapCanvas()
        self._task = None
        self._pending_bbox = None               # AOI of the in-flight fetch (for naming)
        self._drawn = None                      # QgsRectangle in canvas CRS, or None
        self._prev_tool = None
        self._extent_tool = QgsMapToolExtent(self.canvas)
        self._extent_tool.extentChanged.connect(self._on_extent_drawn)
        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        root = QWidget()
        layout = QVBoxLayout(root)

        aoi = QGroupBox("Area of interest")
        av = QVBoxLayout(aoi)
        self.rb_extent = QRadioButton("Use current map extent")
        self.rb_draw = QRadioButton("Draw rectangle on map")
        self.rb_extent.setChecked(True)
        self.rb_extent.toggled.connect(self._on_aoi_mode)
        av.addWidget(self.rb_extent)
        av.addWidget(self.rb_draw)
        self.lbl_bbox = QLabel("-")
        self.lbl_bbox.setWordWrap(True)
        av.addWidget(self.lbl_bbox)
        layout.addWidget(aoi)

        opts = QGroupBox("Layers")
        ov = QVBoxLayout(opts)
        self.cb_s2 = QCheckBox("Include Sentinel-2 true color")
        self.cb_s2.setChecked(True)
        ov.addWidget(self.cb_s2)
        layout.addWidget(opts)

        self.btn_fetch = QPushButton("Fetch layers")
        self.btn_fetch.clicked.connect(self._on_fetch)
        layout.addWidget(self.btn_fetch)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)
        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)

        layout.addStretch(1)
        self.setWidget(root)
        self.canvas.extentsChanged.connect(self._refresh_bbox)
        self._refresh_bbox()

    # ----------------------------------------------------------------- AOI
    def _on_aoi_mode(self):
        if self.rb_draw.isChecked():
            self._prev_tool = self.canvas.mapTool()
            self.canvas.setMapTool(self._extent_tool)
            self.lbl_status.setText("Drag a rectangle on the map to set the AOI.")
        else:
            if self.canvas.mapTool() is self._extent_tool and self._prev_tool:
                self.canvas.setMapTool(self._prev_tool)
            self._refresh_bbox()

    def _on_extent_drawn(self, rect):
        self._drawn = rect
        self._refresh_bbox()

    def _aoi_bbox(self):
        """AOI as [west, south, east, north] in EPSG:4326 (lon/lat)."""
        if self.rb_draw.isChecked() and self._drawn is not None:
            rect = self._drawn
        else:
            rect = self.canvas.extent()
        src = self.canvas.mapSettings().destinationCrs()
        xform = QgsCoordinateTransform(src, WGS84, QgsProject.instance())
        geo = xform.transformBoundingBox(rect)
        return [geo.xMinimum(), geo.yMinimum(), geo.xMaximum(), geo.yMaximum()]

    def _refresh_bbox(self):
        try:
            b = self._aoi_bbox()
        except Exception:                       # noqa: BLE001 - transform can fail pre-CRS
            self.lbl_bbox.setText("-")
            return
        self.lbl_bbox.setText(
            f"W {b[0]:.4f}   S {b[1]:.4f}\nE {b[2]:.4f}   N {b[3]:.4f}"
        )

    # ----------------------------------------------------------------- run
    def _on_fetch(self):
        if self._task is not None:
            return
        try:
            bbox = self._aoi_bbox()
        except Exception as exc:                # noqa: BLE001
            self.lbl_status.setText(f"Could not read AOI: {exc}")
            return
        span = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        if span > MAX_AOI_DEG:
            self.lbl_status.setText(
                f"AOI too large ({span:.2f} deg/side). Zoom in to under "
                f"{MAX_AOI_DEG} deg for v1."
            )
            return
        out_dir = tempfile.mkdtemp(prefix="planesight_")
        self._pending_bbox = bbox
        self._task = AggregateTask(bbox, out_dir, include_s2=self.cb_s2.isChecked())
        self._task.progressChanged.connect(self._on_progress)
        self._task.message.connect(self.lbl_status.setText)
        self._task.taskCompleted.connect(self._on_done)
        self._task.taskTerminated.connect(self._on_failed)
        self.btn_fetch.setEnabled(False)
        self.progress.setValue(0)
        QgsApplication.taskManager().addTask(self._task)

    def _on_progress(self):
        if self._task is not None:
            self.progress.setValue(int(self._task.progress()))

    @staticmethod
    def _group_name(bbox):
        """A short AOI-centre name for the layer group, e.g. 'PlaneSight 27.750N 82.450E'."""
        lat = 0.5 * (bbox[1] + bbox[3])
        lon = 0.5 * (bbox[0] + bbox[2])
        ns = "N" if lat >= 0 else "S"
        ew = "E" if lon >= 0 else "W"
        return f"PlaneSight {abs(lat):.3f}{ns} {abs(lon):.3f}{ew}"

    def _on_done(self):
        specs = list(self._task.specs)
        project = QgsProject.instance()
        group = project.layerTreeRoot().insertGroup(0, self._group_name(self._pending_bbox))
        added = 0
        # reversed so the imagery/derivatives sit above the DEM in the group's draw order
        for spec in reversed(specs):
            layer = QgsRasterLayer(spec.path, spec.name)
            if layer.isValid():
                project.addMapLayer(layer, False)   # False: place it under the group, not root
                group.addLayer(layer)
                added += 1
        self.progress.setValue(100)
        self.lbl_status.setText(f"Loaded {added} layers into '{group.name()}'.")
        self._reset_task()

    def _on_failed(self):
        err = getattr(self._task, "error", None)
        self.lbl_status.setText("Canceled." if err is None else f"Failed: {err}")
        self._reset_task()

    def _reset_task(self):
        self.btn_fetch.setEnabled(True)
        self._task = None
