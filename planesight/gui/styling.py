"""Attitude-layer styling: the qgSurf-style strike/dip symbology (GUI M2).

A two-layer SVG strike/dip marker (white casing + reliability-coloured symbol, rotated
by dip direction) with a dip-value label offset in the dip direction. Built
programmatically (not a .qml) so it is always valid for the running QGIS. Verified to
render on QGIS 3.44 (headless offscreen test); the earlier crash was a QGIS 3.28
renderer bug. Mirrors scripts/qgis_style_attitudes.py.
"""

from __future__ import annotations

import os

from qgis.core import (
    Qgis,
    QgsMarkerSymbol,
    QgsPalLayerSettings,
    QgsProperty,
    QgsSingleSymbolRenderer,
    QgsSvgMarkerSymbolLayer,
    QgsSymbolLayer,
    QgsUnitTypes,
    QgsVectorLayerSimpleLabeling,
)
from qgis.PyQt.QtGui import QColor

_SVG = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "resources", "symbols", "strike_dip_bedding.svg",
)
_RELIABLE_COLOR = "if(\"reliable\" = 1, '#1a1a1a', '#b0b0b0')"


def _svg_layer(stroke_width):
    """An SVG strike/dip marker layer (~12x7.8 mm), rotated by the dip_dir field."""
    sl = QgsSvgMarkerSymbolLayer(_SVG)
    sl.setSize(7.8)                       # height; SVG's 12:7.8 aspect -> ~12 mm wide
    sl.setSizeUnit(QgsUnitTypes.RenderMillimeters)
    sl.setStrokeWidth(stroke_width)
    sl.setStrokeWidthUnit(QgsUnitTypes.RenderMillimeters)
    sl.setDataDefinedProperty(
        QgsSymbolLayer.PropertyAngle, QgsProperty.fromField("dip_dir")
    )
    return sl


def style_attitudes(layer):
    """Apply the white-cased, reliability-coloured strike/dip markers + dip labels.

    Requires ``dip_dir``, ``dip`` and ``reliable`` fields. Bottom layer is a white
    casing; top layer is the symbol, darkened when reliable. The dip value labels each
    point, offset in the dip direction.
    """
    halo = _svg_layer(4.2)
    halo.setStrokeColor(QColor("#ffffff"))
    symbol = _svg_layer(2.4)
    symbol.setStrokeColor(QColor("#1a1a1a"))
    symbol.setDataDefinedProperty(
        QgsSymbolLayer.PropertyStrokeColor, QgsProperty.fromExpression(_RELIABLE_COLOR)
    )
    layer.setRenderer(QgsSingleSymbolRenderer(QgsMarkerSymbol([halo, symbol])))

    labels = QgsPalLayerSettings()
    labels.fieldName = "format_number(\"dip\", 0)"
    labels.isExpression = True
    labels.placement = Qgis.LabelPlacement.OverPoint
    labels.offsetUnits = QgsUnitTypes.RenderPixels
    ddp = labels.dataDefinedProperties()
    ddp.setProperty(
        QgsPalLayerSettings.OffsetXY,
        # +x = east, -y = north (screen); flip -15 -> +15 if labels land up-dip
        QgsProperty.fromExpression(
            "array(15 * sin(radians(\"dip_dir\")), -15 * cos(radians(\"dip_dir\")))"
        ),
    )
    labels.setDataDefinedProperties(ddp)
    layer.setLabeling(QgsVectorLayerSimpleLabeling(labels))
    layer.setLabelsEnabled(True)
    layer.triggerRepaint()
