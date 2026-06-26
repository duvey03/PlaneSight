"""LiveWireMapTool: the assisted (live-wire) contact-tracing map tool (GUI T2).

The interaction front-end of planesight.core.trace. The user clicks an anchor, moves the
cursor, and a "wire" snaps along the cheapest contact path (live preview via a rubber
band); each click commits a segment, right-click / Enter finishes the polyline into an
editable trace layer. Holding Ctrl free-draws a straight segment for concealed ground.

Heavy work is precomputed (BuildCostTask gives us cost + snap field); this tool only does
per-click Dijkstra (cost_to_all once per committed anchor) and O(path) backtraces per
cursor move - the seismic/Photoshop trick. The geometry helpers are kept free of Qt
events so they can be unit-tested headless; the canvas*Event handlers are thin adapters.
"""

from __future__ import annotations

import numpy as np
from qgis.core import (
    QgsCoordinateTransform,
    QgsPointXY,
    QgsProject,
    QgsWkbTypes,
)
from qgis.gui import QgsMapTool, QgsRubberBand
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor

from ..core.trace import backtrace, cost_to_all, snap_point


def _straight_pixels(a, b):
    """Rasterised straight pixel path a->b (free-draw, no wire)."""
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) + 1
    rr = np.round(np.linspace(a[0], b[0], n)).astype(int)
    cc = np.round(np.linspace(a[1], b[1], n)).astype(int)
    return np.stack([rr, cc], axis=1)


class LiveWireMapTool(QgsMapTool):
    """Click-to-snap contact tracing over a precomputed cost surface."""

    def __init__(self, canvas, cost, snap, gt, dem_crs, commit_cb, *,
                 snap_radius: int = 4, margin: int = 200):
        super().__init__(canvas)
        self._canvas = canvas
        self._cost = cost
        self._snap = snap
        self._gt = gt
        self._dem_crs = dem_crs
        self._commit_cb = commit_cb
        self._snap_radius = int(snap_radius)
        self._margin = int(margin)
        self._h, self._w = cost.shape

        # affine: world = origin + M @ [col, row]  (pixel corners); centers add 0.5
        self._origin = np.array([gt[0], gt[3]], dtype=float)
        self._M = np.array([[gt[1], gt[2]], [gt[4], gt[5]]], dtype=float)
        self._Minv = np.linalg.inv(self._M)

        self._anchor = None          # committed-anchor pixel (row, col)
        self._field = None           # LiveWireField from the anchor
        self._committed = []         # accumulated pixel path (list of (row, col))

        self._band = QgsRubberBand(canvas, QgsWkbTypes.LineGeometry)   # committed segments
        self._band.setColor(QColor(220, 30, 30))
        self._band.setWidth(2)
        self._live = QgsRubberBand(canvas, QgsWkbTypes.LineGeometry)   # cursor preview
        self._live.setColor(QColor(255, 150, 0, 180))
        self._live.setWidth(2)

    # ----------------------------------------------------- geometry (Qt-free)
    def _world_to_pixel(self, x, y):
        col, row = self._Minv @ (np.array([x, y]) - self._origin)
        return int(np.floor(row)), int(np.floor(col))

    def _pixel_to_world(self, rc):
        x, y = self._origin + self._M @ np.array([rc[1] + 0.5, rc[0] + 0.5])
        return float(x), float(y)

    def _in_bounds(self, rc):
        return 0 <= rc[0] < self._h and 0 <= rc[1] < self._w

    def _snap_rc(self, rc):
        snapped = snap_point(self._snap, rc, self._snap_radius)
        return snapped if snapped is not None else rc

    def _segment(self, target_rc, *, free=False):
        """Pixel path from the current anchor to ``target_rc`` (wire, or straight if free)."""
        if self._anchor is None:
            return np.empty((0, 2), dtype=int)
        if free:
            return _straight_pixels(self._anchor, target_rc)
        return backtrace(self._field, target_rc)

    def _start_anchor(self, rc):
        rc = self._snap_rc(rc)
        self._anchor = rc
        self._committed = [rc]
        self._field = cost_to_all(self._cost, rc, margin=self._margin)

    def _commit_anchor(self, rc, *, free=False):
        seg = self._segment(rc, free=free)
        if seg.shape[0] >= 2:
            self._committed.extend(tuple(p) for p in seg[1:])
        new_anchor = rc if free else self._snap_rc(rc)
        self._anchor = new_anchor
        self._field = cost_to_all(self._cost, new_anchor, margin=self._margin)

    def world_polyline(self):
        """The committed pixel path as DEM-CRS (x, y) world points (>= 2 or empty list)."""
        if len(self._committed) < 2:
            return []
        return [self._pixel_to_world(rc) for rc in self._committed]

    # ----------------------------------------------------- canvas <-> drawing
    def _to_dem(self, map_pt):
        xform = QgsCoordinateTransform(
            self._canvas.mapSettings().destinationCrs(), self._dem_crs,
            QgsProject.instance(),
        )
        return xform.transform(map_pt)

    def _map_to_pixel(self, map_pt):
        p = self._to_dem(map_pt)
        rc = self._world_to_pixel(p.x(), p.y())
        return rc if self._in_bounds(rc) else None

    def _rc_to_canvas(self, rc):
        x, y = self._pixel_to_world(rc)
        xform = QgsCoordinateTransform(
            self._dem_crs, self._canvas.mapSettings().destinationCrs(),
            QgsProject.instance(),
        )
        p = xform.transform(QgsPointXY(x, y))
        return p

    def _draw(self, band, pixels):
        band.reset(QgsWkbTypes.LineGeometry)
        for rc in pixels:
            band.addPoint(self._rc_to_canvas(rc))

    # ------------------------------------------------------------ Qt events
    def canvasPressEvent(self, event):
        if event.button() == Qt.RightButton:
            self._finish()
            return
        if event.button() != Qt.LeftButton:
            return
        rc = self._map_to_pixel(event.mapPoint())
        if rc is None:
            return
        free = bool(event.modifiers() & Qt.ControlModifier)
        if self._anchor is None:
            self._start_anchor(rc)
        else:
            self._commit_anchor(rc, free=free)
        self._draw(self._band, self._committed)
        self._live.reset(QgsWkbTypes.LineGeometry)

    def canvasMoveEvent(self, event):
        if self._anchor is None:
            return
        rc = self._map_to_pixel(event.mapPoint())
        if rc is None:
            return
        free = bool(event.modifiers() & Qt.ControlModifier)
        seg = self._segment(rc, free=free)
        if seg.shape[0] >= 2:
            self._draw(self._live, seg)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self._cancel()
        elif event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self._finish()

    def _finish(self):
        pts = self.world_polyline()
        if pts:
            self._commit_cb(pts)
        self._cancel()

    def _cancel(self):
        self._anchor = None
        self._field = None
        self._committed = []
        self._band.reset(QgsWkbTypes.LineGeometry)
        self._live.reset(QgsWkbTypes.LineGeometry)

    def deactivate(self):
        self._cancel()
        super().deactivate()
