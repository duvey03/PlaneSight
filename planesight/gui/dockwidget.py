"""PlaneSight dockwidget: the workflow UI, one tab per capability.

Tabs grow with the milestones:
- **Data** (M1): AOI -> fetch DEM + Sentinel-2 + derivatives -> styled, grouped layers.
- **Strike/Dip** (M2): pick a trace layer + a DEM layer -> fit strike/dip along each
  trace -> styled attitude markers. Works on ANY traces (incl. hand-drawn), decoupled
  from detection.
Later milestones add Analyze (stereonet) and Detect/Review tabs. All heavy work runs
off the UI thread in QgsTasks.
"""

from __future__ import annotations

import tempfile

import numpy as np
from qgis.core import (
    QgsApplication,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsFeature,
    QgsField,
    QgsGeometry,
    QgsLineSymbol,
    QgsMapLayerProxyModel,
    QgsPointXY,
    QgsProject,
    QgsRasterLayer,
    QgsSingleSymbolRenderer,
    QgsVectorLayer,
    QgsWkbTypes,
)
from qgis.gui import QgsDockWidget, QgsMapLayerComboBox, QgsMapToolExtent
from qgis.PyQt.QtCore import QVariant
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..tasks.aggregate import AggregateTask
from ..tasks.attitudes import AttitudeTask
from ..tasks.detect import DetectTask
from ..tasks.trace_cost import BuildCostTask
from .stereonet_widget import StereonetWidget
from .styling import style_attitudes, style_candidates

# NOTE: .trace_tool imports core.trace (scipy) at module load, so it is imported LAZILY
# inside _on_cost_done() - by then BuildCostTask has already loaded scipy off the UI
# thread. Importing it here would cold-load scipy on plugin enable and freeze the UI.

WGS84 = QgsCoordinateReferenceSystem("EPSG:4326")
MAX_AOI_DEG = 0.5      # v1 guard (~55 km/side) against accidental continent-sized fetches


class PlaneSightDockWidget(QgsDockWidget):
    """Dockable, tabbed PlaneSight workflow panel."""

    def __init__(self, iface, parent=None):
        super().__init__("PlaneSight", parent)
        self.iface = iface
        self.canvas = iface.mapCanvas()
        self._task = None                       # aggregate task
        self._att_task = None                   # attitude task
        self._detect_task = None                # detection task
        self._det_fit_task = None               # fit-accepted task (Detect tab)
        self._candidate_layer = None            # last candidate-trace layer
        self._cost_task = None                  # live-wire cost-surface build task
        self._trace_tool = None                 # active LiveWireMapTool
        self._traces_layer = None               # 'PlaneSight traces' (assisted tracing)
        self._n_traced = 0                      # committed assisted traces
        self._pending_bbox = None
        self._drawn = None                      # QgsRectangle in canvas CRS, or None
        self._prev_tool = None
        self._extent_tool = QgsMapToolExtent(self.canvas)
        self._extent_tool.extentChanged.connect(self._on_extent_drawn)
        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_data_tab(), "Data")
        self._tabs.addTab(self._build_detect_tab(), "Detect")
        self._tabs.addTab(self._build_trace_tab(), "Trace")
        self._tabs.addTab(self._build_attitude_tab(), "Strike/Dip")
        self._tabs.addTab(self._build_analyze_tab(), "Analyze")
        self._tabs.currentChanged.connect(self._on_tab_changed)
        self.setWidget(self._tabs)
        self.canvas.extentsChanged.connect(self._refresh_bbox)
        self._refresh_bbox()

    def _on_tab_changed(self, index):
        name = self._tabs.tabText(index)
        if name == "Detect":
            self._default_detect_dem()
        elif name == "Trace":
            self._default_trace_dem()
        elif name == "Strike/Dip":
            self._apply_default_selections()
        elif name == "Analyze":
            self._default_attitude_layer()
            # ensure selectionChanged is wired even if the combo didn't change
            self._on_analyze_layer_changed(self.cmb_att.currentLayer())

    def _apply_default_selections(self):
        """Point the Strike/Dip combos at sensible layers (PlaneSight DEM + user traces)."""
        project = QgsProject.instance()
        cur_dem = self.cmb_dem.currentLayer()
        if cur_dem is None or not cur_dem.name().startswith("PlaneSight DEM"):
            for lyr in project.mapLayers().values():
                if lyr.type() == lyr.RasterLayer and lyr.name().startswith("PlaneSight DEM"):
                    self.cmb_dem.setLayer(lyr)
                    break
        cur_tr = self.cmb_traces.currentLayer()
        if cur_tr is None or (cur_tr.name().startswith("PlaneSight")
                              and not cur_tr.name().startswith("PlaneSight traces")):
            # prefer the assisted-tracing output, else the first user (non-PlaneSight) line
            pick = fallback = None
            for lyr in project.mapLayers().values():
                if (lyr.type() != lyr.VectorLayer
                        or QgsWkbTypes.geometryType(lyr.wkbType()) != QgsWkbTypes.LineGeometry):
                    continue
                if lyr.name().startswith("PlaneSight traces"):
                    pick = lyr
                    break
                if not lyr.name().startswith("PlaneSight") and fallback is None:
                    fallback = lyr
            chosen = pick or fallback
            if chosen is not None:
                self.cmb_traces.setLayer(chosen)

    def _default_attitude_layer(self):
        cur = self.cmb_att.currentLayer()
        if cur is None or not cur.name().startswith("PlaneSight attitudes"):
            for lyr in QgsProject.instance().mapLayers().values():
                if lyr.type() == lyr.VectorLayer and lyr.name().startswith("PlaneSight attitudes"):
                    self.cmb_att.setLayer(lyr)
                    break

    def _default_detect_dem(self):
        cur = self.cmb_detect_dem.currentLayer()
        if cur is None or not cur.name().startswith("PlaneSight DEM"):
            for lyr in QgsProject.instance().mapLayers().values():
                if lyr.type() == lyr.RasterLayer and lyr.name().startswith("PlaneSight DEM"):
                    self.cmb_detect_dem.setLayer(lyr)
                    break

    def _default_trace_dem(self):
        cur = self.cmb_trace_dem.currentLayer()
        if cur is None or not cur.name().startswith("PlaneSight DEM"):
            for lyr in QgsProject.instance().mapLayers().values():
                if lyr.type() == lyr.RasterLayer and lyr.name().startswith("PlaneSight DEM"):
                    self.cmb_trace_dem.setLayer(lyr)
                    break

    def _build_data_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

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
        layout.addWidget(self.progress)
        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)
        layout.addStretch(1)
        return tab

    def _build_detect_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("DEM layer (raster):"))
        self.cmb_detect_dem = QgsMapLayerComboBox()
        self.cmb_detect_dem.setFilters(QgsMapLayerProxyModel.RasterLayer)
        layout.addWidget(self.cmb_detect_dem)
        self.btn_detect = QPushButton("Detect candidate traces")
        self.btn_detect.clicked.connect(self._on_detect)
        layout.addWidget(self.btn_detect)
        self.detect_progress = QProgressBar()
        self.detect_progress.setRange(0, 0)     # busy indicator
        self.detect_progress.hide()
        layout.addWidget(self.detect_progress)
        self.btn_fit_accepted = QPushButton("Compute strike/dip on accepted")
        self.btn_fit_accepted.setToolTip(
            "Fit accepted candidates (class not drainage/reject; or just the selected "
            "features) -> a PlaneSight attitudes layer."
        )
        self.btn_fit_accepted.setEnabled(False)
        self.btn_fit_accepted.clicked.connect(self._on_fit_accepted)
        layout.addWidget(self.btn_fit_accepted)
        self.detect_status = QLabel("Detect candidate traces, reclassify the 'class' field, "
                                    "then fit the accepted ones.")
        self.detect_status.setWordWrap(True)
        layout.addWidget(self.detect_status)
        layout.addStretch(1)
        return tab

    def _build_trace_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("DEM layer (raster):"))
        self.cmb_trace_dem = QgsMapLayerComboBox()
        self.cmb_trace_dem.setFilters(QgsMapLayerProxyModel.RasterLayer)
        layout.addWidget(self.cmb_trace_dem)

        tune = QGroupBox("Tracing")
        tv = QVBoxLayout(tune)
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Snap radius (px):"))
        self.spin_snap = QSpinBox()
        self.spin_snap.setRange(0, 25)
        self.spin_snap.setValue(4)
        row1.addWidget(self.spin_snap)
        tv.addLayout(row1)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Drainage avoidance:"))
        self.spin_drain = QDoubleSpinBox()
        self.spin_drain.setRange(0.0, 10.0)
        self.spin_drain.setSingleStep(0.5)
        self.spin_drain.setValue(3.0)
        self.spin_drain.setToolTip("Weight of the creek-avoidance penalty in the cost "
                                   "surface (0 = curvature only).")
        row2.addWidget(self.spin_drain)
        tv.addLayout(row2)
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Max segment (px):"))
        self.spin_margin = QSpinBox()
        self.spin_margin.setRange(50, 1000)
        self.spin_margin.setSingleStep(50)
        self.spin_margin.setValue(200)
        self.spin_margin.setToolTip("Reach of the live wire between clicks: the Dijkstra "
                                    "search window around each anchor. Larger = longer "
                                    "segments preview, but a heavier per-click solve.")
        row3.addWidget(self.spin_margin)
        tv.addLayout(row3)
        layout.addWidget(tune)

        self.btn_trace = QPushButton("Activate trace tool")
        self.btn_trace.clicked.connect(self._on_activate_trace)
        layout.addWidget(self.btn_trace)
        self.trace_progress = QProgressBar()
        self.trace_progress.setRange(0, 0)      # busy indicator while building the surface
        self.trace_progress.hide()
        layout.addWidget(self.trace_progress)
        self.trace_status = QLabel(
            "Pick a DEM, then Activate. Click to start a trace; move to preview the wire; "
            "click to add a point; right-click or Enter to finish; Ctrl-click to free-draw."
        )
        self.trace_status.setWordWrap(True)
        layout.addWidget(self.trace_status)
        layout.addStretch(1)
        return tab

    def _build_attitude_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        box = QGroupBox("Strike / dip from traces")
        bv = QVBoxLayout(box)
        bv.addWidget(QLabel("Trace layer (lines):"))
        self.cmb_traces = QgsMapLayerComboBox()
        self.cmb_traces.setFilters(QgsMapLayerProxyModel.LineLayer)
        bv.addWidget(self.cmb_traces)
        bv.addWidget(QLabel("DEM layer (raster):"))
        self.cmb_dem = QgsMapLayerComboBox()
        self.cmb_dem.setFilters(QgsMapLayerProxyModel.RasterLayer)
        bv.addWidget(self.cmb_dem)
        layout.addWidget(box)

        self.btn_fit = QPushButton("Compute strike/dip")
        self.btn_fit.clicked.connect(self._on_compute_attitudes)
        layout.addWidget(self.btn_fit)
        self.att_progress = QProgressBar()
        self.att_progress.setRange(0, 0)        # busy indicator (fit is one quick step)
        self.att_progress.hide()
        layout.addWidget(self.att_progress)
        self.att_status = QLabel("")
        self.att_status.setWordWrap(True)
        layout.addWidget(self.att_status)
        layout.addStretch(1)
        return tab

    # ----------------------------------------------------------------- AOI (M1)
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

    # ----------------------------------------------------------------- fetch (M1)
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
        # append (not insert at 0) so PlaneSight rasters sit BELOW pre-existing
        # layers/groups - keeps the user's traces etc. visible on top.
        group = project.layerTreeRoot().addGroup(self._group_name(self._pending_bbox))
        added = 0
        dem_layer = None
        for spec in reversed(specs):   # imagery/derivatives above the DEM
            layer = QgsRasterLayer(spec.path, spec.name)
            if layer.isValid():
                project.addMapLayer(layer, False)
                group.addLayer(layer)
                if spec.kind == "dem":
                    dem_layer = layer
                added += 1
        if dem_layer is not None:
            self.cmb_dem.setLayer(dem_layer)        # point Strike/Dip at the new DEM
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

    # ------------------------------------------------------- strike/dip (M2)
    def _extract_world_traces(self, layer, dst_crs):
        """(Multi)line features -> list of (N, 2) numpy arrays in dst_crs world coords."""
        xform = QgsCoordinateTransform(layer.crs(), dst_crs, QgsProject.instance())
        traces = []
        for feat in layer.getFeatures():
            geom = feat.geometry()
            if geom is None or geom.isEmpty():
                continue
            parts = geom.asMultiPolyline() if geom.isMultipart() else [geom.asPolyline()]
            for part in parts:
                if len(part) < 2:
                    continue
                pts = [(q.x(), q.y()) for q in (xform.transform(p) for p in part)]
                traces.append(np.array(pts, dtype=float))
        return traces

    def _on_compute_attitudes(self):
        if self._att_task is not None:
            return
        trace_layer = self.cmb_traces.currentLayer()
        dem_layer = self.cmb_dem.currentLayer()
        if trace_layer is None or dem_layer is None:
            self.att_status.setText("Pick both a trace layer and a DEM layer.")
            return
        try:
            traces = self._extract_world_traces(trace_layer, dem_layer.crs())
        except Exception as exc:                # noqa: BLE001
            self.att_status.setText(f"Could not read traces: {exc}")
            return
        if not traces:
            self.att_status.setText("No line features found in the trace layer.")
            return
        self._att_crs = dem_layer.crs()
        self._att_task = AttitudeTask(traces, dem_layer.source())
        self._att_task.taskCompleted.connect(self._on_att_done)
        self._att_task.taskTerminated.connect(self._on_att_failed)
        self.btn_fit.setEnabled(False)
        self.att_progress.show()
        self.att_status.setText(f"Fitting strike/dip along {len(traces)} traces ...")
        QgsApplication.taskManager().addTask(self._att_task)

    def _build_attitude_layer(self, attitudes, crs):
        """Build + style a 'PlaneSight attitudes' point layer; returns the reliable count."""
        layer = QgsVectorLayer(f"Point?crs={crs.authid()}", "PlaneSight attitudes", "memory")
        prov = layer.dataProvider()
        prov.addAttributes([
            QgsField("strike", QVariant.Double),
            QgsField("dip", QVariant.Double),
            QgsField("dip_dir", QVariant.Double),
            QgsField("dip_unc", QVariant.Double),
            QgsField("conditioning", QVariant.Double),
            QgsField("reliable", QVariant.Int),
        ])
        layer.updateFields()
        feats = []
        for ap in attitudes:
            a = ap.attitude
            f = QgsFeature(layer.fields())
            f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(ap.x, ap.y)))
            f.setAttributes([
                float(a.strike), float(a.dip), float(a.dip_direction),
                float(a.dip_uncertainty), float(a.conditioning), int(ap.reliable),
            ])
            feats.append(f)
        prov.addFeatures(feats)
        layer.updateExtents()
        try:
            style_attitudes(layer)
        except Exception:                       # noqa: BLE001 - styling is non-fatal
            pass
        QgsProject.instance().addMapLayer(layer)
        return sum(ap.reliable for ap in attitudes)

    def _on_att_done(self):
        atts = list(self._att_task.attitudes)
        if not atts:
            self.att_status.setText(
                "0 attitudes fitted - do the trace layer and DEM cover the same area? "
                "Traces sampled outside the DEM footprint are skipped."
            )
            self._reset_att_task()
            return
        n_rel = self._build_attitude_layer(atts, self._att_crs)
        self.att_status.setText(
            f"Fitted {len(atts)} attitudes ({n_rel} reliable) -> 'PlaneSight attitudes'."
        )
        self._reset_att_task()

    def _on_att_failed(self):
        err = getattr(self._att_task, "error", None)
        self.att_status.setText("Canceled." if err is None else f"Failed: {err}")
        self._reset_att_task()

    def _reset_att_task(self):
        self.att_progress.hide()
        self.btn_fit.setEnabled(True)
        self._att_task = None

    # ---------------------------------------------------------- detect (M4)
    def _on_detect(self):
        if self._detect_task is not None:
            return
        dem_layer = self.cmb_detect_dem.currentLayer()
        if dem_layer is None:
            self.detect_status.setText("Pick a DEM layer.")
            return
        self._detect_task = DetectTask(dem_layer.source())
        self._detect_task.taskCompleted.connect(self._on_detect_done)
        self._detect_task.taskTerminated.connect(self._on_detect_failed)
        self.btn_detect.setEnabled(False)
        self.detect_progress.show()
        self.detect_status.setText("Detecting candidate traces (first run loads scipy) ...")
        QgsApplication.taskManager().addTask(self._detect_task)

    def _on_detect_done(self):
        cands = list(self._detect_task.candidates)
        dem_layer = self.cmb_detect_dem.currentLayer()
        crs = dem_layer.crs() if dem_layer is not None else WGS84
        layer = QgsVectorLayer(
            f"LineString?crs={crs.authid()}", "PlaneSight candidates", "memory"
        )
        prov = layer.dataProvider()
        prov.addAttributes([
            QgsField("class", QVariant.String),
            QgsField("is_drainage", QVariant.Int),
            QgsField("score", QVariant.Double),
            QgsField("rank", QVariant.Double),
            QgsField("length_m", QVariant.Double),
        ])
        layer.updateFields()
        feats = []
        n_drain = 0
        for c in cands:
            f = QgsFeature(layer.fields())
            f.setGeometry(QgsGeometry.fromPolylineXY(
                [QgsPointXY(float(x), float(y)) for x, y in c.geometry]
            ))
            n_drain += int(c.is_drainage)
            f.setAttributes([
                "drainage" if c.is_drainage else "contact", int(c.is_drainage),
                float(c.score), float(c.rank), float(c.length),
            ])
            feats.append(f)
        prov.addFeatures(feats)
        layer.updateExtents()
        try:
            style_candidates(layer)
        except Exception:                       # noqa: BLE001 - styling is non-fatal
            pass
        QgsProject.instance().addMapLayer(layer)
        self._candidate_layer = layer
        self.detect_status.setText(
            f"{len(cands)} candidates ({n_drain} drainage-flagged) -> 'PlaneSight "
            f"candidates'. Reclassify the 'class' field, then fit the accepted ones."
        )
        self.btn_fit_accepted.setEnabled(len(cands) > 0)
        self._reset_detect_task()

    def _on_detect_failed(self):
        err = getattr(self._detect_task, "error", None)
        self.detect_status.setText("Canceled." if err is None else f"Failed: {err}")
        self._reset_detect_task()

    def _reset_detect_task(self):
        self.detect_progress.hide()
        self.btn_detect.setEnabled(True)
        self._detect_task = None

    @staticmethod
    def _polylines_world(layer, predicate=None):
        """Collect (multi)line features as numpy (N,2) arrays; optional feature predicate."""
        out = []
        feats = (layer.selectedFeatures() if layer.selectedFeatureCount()
                 else layer.getFeatures())
        for feat in feats:
            if predicate is not None and not predicate(feat):
                continue
            geom = feat.geometry()
            if geom is None or geom.isEmpty():
                continue
            parts = geom.asMultiPolyline() if geom.isMultipart() else [geom.asPolyline()]
            for part in parts:
                if len(part) >= 2:
                    out.append(np.array([(p.x(), p.y()) for p in part], dtype=float))
        return out

    def _on_fit_accepted(self):
        if self._det_fit_task is not None:
            return
        layer = self._candidate_layer
        dem_layer = self.cmb_detect_dem.currentLayer()
        if layer is None or dem_layer is None:
            self.detect_status.setText("Detect candidate traces and pick a DEM first.")
            return
        traces = self._polylines_world(
            layer, predicate=lambda f: f["class"] not in ("drainage", "reject")
        )
        if not traces:
            self.detect_status.setText("No accepted traces (all drainage/reject?).")
            return
        crs = dem_layer.crs()
        task = AttitudeTask(traces, dem_layer.source())
        self._det_fit_task = task

        def _done():
            atts = list(task.attitudes)
            if atts:
                n_rel = self._build_attitude_layer(atts, crs)
                self.detect_status.setText(
                    f"Fitted {len(atts)} attitudes ({n_rel} reliable) from accepted traces."
                )
            else:
                self.detect_status.setText("0 attitudes (do the traces overlap the DEM?).")
            self.btn_fit_accepted.setEnabled(True)
            self._det_fit_task = None

        def _failed():
            self.detect_status.setText(f"Fit failed: {getattr(task, 'error', None)}")
            self.btn_fit_accepted.setEnabled(True)
            self._det_fit_task = None

        task.taskCompleted.connect(_done)
        task.taskTerminated.connect(_failed)
        self.btn_fit_accepted.setEnabled(False)
        self.detect_status.setText(f"Fitting {len(traces)} accepted traces ...")
        QgsApplication.taskManager().addTask(task)

    # ----------------------------------------------------- assisted trace (T2)
    def _on_activate_trace(self):
        if self._cost_task is not None:
            return
        dem_layer = self.cmb_trace_dem.currentLayer()
        if dem_layer is None:
            self.trace_status.setText("Pick a DEM layer.")
            return
        self._cost_task = BuildCostTask(
            dem_layer.source(), drainage_weight=self.spin_drain.value()
        )
        self._cost_task.taskCompleted.connect(self._on_cost_done)
        self._cost_task.taskTerminated.connect(self._on_cost_failed)
        self.btn_trace.setEnabled(False)
        self.trace_progress.show()
        self.trace_status.setText("Building cost surface (first run loads scipy) ...")
        QgsApplication.taskManager().addTask(self._cost_task)

    def _ensure_traces_layer(self, crs):
        """Create (or reuse) the editable 'PlaneSight traces' line layer in ``crs``."""
        existing = self._traces_layer
        if existing is not None and existing.id() in QgsProject.instance().mapLayers():
            if existing.crs().authid() == crs.authid():
                return existing
        layer = QgsVectorLayer(f"LineString?crs={crs.authid()}", "PlaneSight traces", "memory")
        try:
            sym = QgsLineSymbol.createSimple({"color": "#e01e1e", "width": "0.6"})
            layer.setRenderer(QgsSingleSymbolRenderer(sym))
        except Exception:                       # noqa: BLE001 - styling is non-fatal
            pass
        QgsProject.instance().addMapLayer(layer)
        self._traces_layer = layer
        self._n_traced = 0
        if hasattr(self, "cmb_traces"):
            self.cmb_traces.setLayer(layer)     # default Strike/Dip at the traced lines
        return layer

    def _commit_trace(self, world_pts):
        """Append a finished assisted trace (DEM-CRS (x, y) points) to the traces layer."""
        layer = self._traces_layer
        if layer is None or len(world_pts) < 2:
            return
        feat = QgsFeature(layer.fields())
        feat.setGeometry(QgsGeometry.fromPolylineXY(
            [QgsPointXY(float(x), float(y)) for x, y in world_pts]
        ))
        layer.dataProvider().addFeatures([feat])
        layer.updateExtents()
        layer.triggerRepaint()
        self._n_traced += 1
        self.trace_status.setText(
            f"{self._n_traced} trace(s) committed to 'PlaneSight traces'. Keep tracing, or "
            "switch to Strike/Dip to measure them."
        )

    def _on_cost_done(self):
        task = self._cost_task
        dem_layer = self.cmb_trace_dem.currentLayer()
        crs = dem_layer.crs() if dem_layer is not None else WGS84
        self._ensure_traces_layer(crs)
        from .trace_tool import LiveWireMapTool  # lazy: scipy already loaded off-thread
        self._trace_tool = LiveWireMapTool(
            self.canvas, task.cost, task.snap, task.gt, crs, self._commit_trace,
            snap_radius=self.spin_snap.value(), margin=self.spin_margin.value(),
        )
        self.canvas.setMapTool(self._trace_tool)
        self.trace_status.setText(
            "Trace tool active. Click to start; move to preview the wire; click to add a "
            "point; right-click or Enter to finish; Ctrl-click to free-draw."
        )
        self._reset_cost_task()

    def _on_cost_failed(self):
        err = getattr(self._cost_task, "error", None)
        self.trace_status.setText("Canceled." if err is None else f"Failed: {err}")
        self._reset_cost_task()

    def _reset_cost_task(self):
        self.trace_progress.hide()
        self.btn_trace.setEnabled(True)
        self._cost_task = None

    # ------------------------------------------------------- analyze (M3)
    def _build_analyze_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("Attitude layer (points):"))
        self.cmb_att = QgsMapLayerComboBox()
        self.cmb_att.setFilters(QgsMapLayerProxyModel.PointLayer)
        self.cmb_att.layerChanged.connect(self._on_analyze_layer_changed)
        layout.addWidget(self.cmb_att)
        self.btn_map_select = QPushButton("Select attitudes on map")
        self.btn_map_select.setToolTip(
            "Activate the attitude layer + QGIS freehand select; drag on the map to "
            "select attitudes (they highlight on the stereonet)."
        )
        self.btn_map_select.clicked.connect(self._on_select_on_map)
        layout.addWidget(self.btn_map_select)
        self.btn_clear = QPushButton("Clear selection")
        self.btn_clear.setToolTip("Clear the selection on both the map and the stereonet.")
        self.btn_clear.clicked.connect(self._on_clear_selection)
        layout.addWidget(self.btn_clear)
        self.btn_plot = QPushButton("Refresh")
        self.btn_plot.clicked.connect(self._on_analyze_update)
        layout.addWidget(self.btn_plot)
        self.stereonet = StereonetWidget()
        self.stereonet.lassoed.connect(self._on_net_lassoed)
        layout.addWidget(self.stereonet, 1)
        self.lbl_stats = QLabel("Lasso poles to select traces; map selection highlights here.")
        self.lbl_stats.setWordWrap(True)
        layout.addWidget(self.lbl_stats)
        self._att_layer = None      # the layer whose selectionChanged we're listening to
        return tab

    def _on_analyze_layer_changed(self, layer):
        if self._att_layer is not None:
            try:
                self._att_layer.selectionChanged.disconnect(self._on_map_selection_changed)
            except (TypeError, RuntimeError):
                pass
        self._att_layer = layer
        if layer is not None:
            layer.selectionChanged.connect(self._on_map_selection_changed)
        self._on_analyze_update()

    def _on_analyze_update(self):
        layer = self.cmb_att.currentLayer()
        if layer is None:
            self.stereonet.set_attitudes([], [], [])
            self.lbl_stats.setText("Pick an attitude point layer.")
            return
        names = {f.name() for f in layer.fields()}
        if "strike" not in names or "dip" not in names:
            self.stereonet.set_attitudes([], [], [])
            self.lbl_stats.setText(
                "Layer has no 'strike'/'dip' fields - pick a PlaneSight attitudes layer."
            )
            return
        strikes, dips, ids = [], [], []
        for feat in layer.getFeatures():
            s, d = feat["strike"], feat["dip"]
            if s is not None and d is not None:
                strikes.append(float(s))
                dips.append(float(d))
                ids.append(feat.id())
        self.stereonet.set_attitudes(strikes, dips, ids)
        self.stereonet.set_selected_ids(layer.selectedFeatureIds())
        self.lbl_stats.setText(self.stereonet.stats_text())

    def _on_map_selection_changed(self, *args):
        layer = self.cmb_att.currentLayer()
        if layer is not None:
            self.stereonet.set_selected_ids(layer.selectedFeatureIds())
            self.lbl_stats.setText(self.stereonet.stats_text())

    def _on_net_lassoed(self, ids):
        layer = self.cmb_att.currentLayer()
        if layer is not None:
            layer.selectByIds(list(ids))   # drives the map highlight + re-syncs the net

    def _on_select_on_map(self):
        """Arm QGIS's native freehand-select on the attitude layer (mirrors the net lasso)."""
        layer = self.cmb_att.currentLayer()
        if layer is None:
            return
        self.iface.setActiveLayer(layer)
        self.iface.actionSelectFreehand().trigger()

    def _on_clear_selection(self):
        """Clear the selection on both the map and the stereonet (one shared selection)."""
        layer = self.cmb_att.currentLayer()
        if layer is not None:
            layer.removeSelection()   # fires selectionChanged -> stereonet drops highlight
