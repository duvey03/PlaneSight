"""Stereonet widget (GUI M3): equal-area lower-hemisphere plot with lasso selection.

Pure QPainter - no matplotlib, so no heavy cold-import (the projection math is the merged,
tested core/structural/stereonet.py). Plots poles to bedding; supports highlight-in-context
selection (selected poles emphasized, the rest dimmed) and a freehand LASSO that emits the
enclosed feature ids for bidirectional map<->stereonet linkage. The mean plane + best-fit
girdle + fold axis (beta) are computed on the ACTIVE set (the selection if any, else all).
North is up, east is right.
"""

from __future__ import annotations

import numpy as np
from qgis.PyQt.QtCore import QPointF, QRectF, Qt, pyqtSignal
from qgis.PyQt.QtGui import QBrush, QColor, QPainter, QPen, QPolygonF
from qgis.PyQt.QtWidgets import QWidget

from planesight.core.structural import (
    axial_mean,
    equal_area_xy,
    fold_axis,
    great_circle,
    line_to_xyz,
    pole_to_strike_dip,
    strike_dip_to_pole,
)


class StereonetWidget(QWidget):
    """Equal-area stereonet of bedding poles with highlight + lasso selection."""

    lassoed = pyqtSignal(object)   # list of feature ids enclosed by the lasso

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(240, 240)
        self._strikes = np.zeros(0)
        self._dips = np.zeros(0)
        self._ids = np.zeros(0, dtype=object)
        self._xy = np.zeros((0, 2))          # pole positions in unit-disk coords
        self._selected = set()               # selected feature ids (highlight)
        self._lasso = None                   # list[QPointF] while dragging
        self._fold = None
        self._mean = None

    # ----------------------------------------------------------------- data
    def set_attitudes(self, strikes, dips, ids=None):
        s = np.asarray(strikes, dtype=float)
        d = np.asarray(dips, dtype=float)
        ids = np.arange(s.size) if ids is None else np.asarray(list(ids), dtype=object)
        ok = np.isfinite(s) & np.isfinite(d)
        self._strikes, self._dips, self._ids = s[ok], d[ok], ids[ok]
        if self._strikes.size:
            self._xy = np.atleast_2d(
                equal_area_xy((self._strikes - 90.0) % 360.0, 90.0 - self._dips)
            )
        else:
            self._xy = np.zeros((0, 2))
        self._selected = set()
        self._recompute()
        self.update()

    def set_selected_ids(self, ids):
        self._selected = set(ids)
        self._recompute()
        self.update()

    def _active_mask(self):
        if self._selected:
            return np.array([i in self._selected for i in self._ids], dtype=bool)
        return np.ones(self._ids.size, dtype=bool)

    def _recompute(self):
        m = self._active_mask()
        s, d = self._strikes[m], self._dips[m]
        if s.size >= 3:
            self._fold = fold_axis(s, d)
            self._mean = axial_mean(strike_dip_to_pole(s, d))
        else:
            self._fold = self._mean = None

    def stats_text(self):
        n_active = int(self._active_mask().sum())
        scope = f"selection ({n_active})" if self._selected else f"all ({self._strikes.size})"
        if self._mean is None:
            return f"{scope}: need >= 3 attitudes for statistics"
        ms, md = pole_to_strike_dip(self._mean.vector)
        lines = [
            scope,
            f"mean plane: {ms:.0f}/{md:.0f} strike/dip  (S1={self._mean.concentration:.2f})",
        ]
        f = self._fold
        k = "n/a" if not np.isfinite(f.woodcock_k) else f"{f.woodcock_k:.2f}"
        lines.append(f"distribution: {f.classification} (Woodcock K={k})")
        if f.is_girdle:
            lines.append(f"fold axis (beta): {f.trend:.0f} -> {f.plunge:.0f} plunge")
        return "\n".join(lines)

    # ------------------------------------------------------------- geometry
    def _geom(self):
        w, h = self.width(), self.height()
        return w / 2.0, h / 2.0, 0.5 * min(w, h) - 14.0

    def _to_px(self, xy):
        cx, cy, r = self._geom()
        return QPointF(cx + float(xy[0]) * r, cy - float(xy[1]) * r)

    def _ids_in_path(self, path):
        """Feature ids whose pole falls inside the closed lasso path (widget px)."""
        if len(path) < 3 or self._xy.shape[0] == 0:
            return []
        poly = QPolygonF(path)
        return [self._ids[i] for i in range(self._xy.shape[0])
                if poly.containsPoint(self._to_px(self._xy[i]), Qt.OddEvenFill)]

    # --------------------------------------------------------------- mouse
    @staticmethod
    def _pos(ev):
        return ev.position() if hasattr(ev, "position") else QPointF(ev.pos())

    def mousePressEvent(self, ev):
        self._lasso = [self._pos(ev)]
        self.update()

    def mouseMoveEvent(self, ev):
        if self._lasso is not None:
            self._lasso.append(self._pos(ev))
            self.update()

    def mouseReleaseEvent(self, ev):
        path = self._lasso or []
        self._lasso = None
        self.lassoed.emit(self._ids_in_path(path))   # empty -> clears selection
        self.update()

    # --------------------------------------------------------------- paint
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        cx, cy, r = self._geom()

        def px(xy):
            return QPointF(cx + float(xy[0]) * r, cy - float(xy[1]) * r)

        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor("#333333"), 1.5))
        p.drawEllipse(QRectF(cx - r, cy - r, 2 * r, 2 * r))
        p.setPen(QPen(QColor("#333333"), 1.0))
        for az in (0.0, 90.0, 180.0, 270.0):
            a = np.radians(az)
            p.drawLine(px((np.sin(a) * 0.96, np.cos(a) * 0.96)), px((np.sin(a), np.cos(a))))
        p.drawText(QPointF(cx - 4, cy - r - 3), "N")

        if self._fold is not None and self._fold.is_girdle:
            gs, gd = pole_to_strike_dip(line_to_xyz(self._fold.trend, self._fold.plunge))
            gc = great_circle(gs, gd, 181)
            p.setPen(QPen(QColor("#e67e22"), 1.6))
            pts = [px(row) for row in gc]
            for i in range(len(pts) - 1):
                p.drawLine(pts[i], pts[i + 1])

        mask = self._active_mask()
        for i in range(self._xy.shape[0]):
            pt = px(self._xy[i])
            if self._selected and mask[i]:
                p.setPen(QPen(QColor("#ffffff"), 0.8))
                p.setBrush(QBrush(QColor("#d7301f")))
                p.drawEllipse(pt, 3.4, 3.4)
            elif self._selected:
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(QColor(120, 120, 120, 110)))
                p.drawEllipse(pt, 2.2, 2.2)
            else:
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(QColor(26, 26, 26, 170)))
                p.drawEllipse(pt, 2.6, 2.6)

        if self._mean is not None and np.isfinite(self._mean.trend):
            pt = px(equal_area_xy(self._mean.trend, self._mean.plunge))
            p.setPen(QPen(QColor("#ffffff"), 1.2))
            p.setBrush(QBrush(QColor("#2c7fb8")))
            p.drawEllipse(pt, 4.5, 4.5)
        if self._fold is not None and self._fold.is_girdle and np.isfinite(self._fold.trend):
            pt = px(equal_area_xy(self._fold.trend, self._fold.plunge))
            p.setPen(QPen(QColor("#ffffff"), 1.2))
            p.setBrush(QBrush(QColor("#e67e22")))
            p.drawRect(QRectF(pt.x() - 3.5, pt.y() - 3.5, 7.0, 7.0))

        if self._lasso and len(self._lasso) >= 2:
            p.setPen(QPen(QColor("#d7301f"), 1.2, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawPolyline(QPolygonF(self._lasso))
        p.end()
