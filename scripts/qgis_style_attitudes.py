"""Style the Nepal attitudes layer via PyQGIS - paste into the QGIS Python Console.

A robust alternative to the emitted nepal_attitudes.qml: it builds the strike/dip
style with the PyQGIS API (always valid for your QGIS version), applies it, and
re-saves a fresh .qml. Loads the hillshade + attitudes too.
"""

import os

from qgis.core import (
    QgsMarkerSymbol,
    QgsPalLayerSettings,
    QgsProject,
    QgsProperty,
    QgsRasterLayer,
    QgsSingleSymbolRenderer,
    QgsSvgMarkerSymbolLayer,
    QgsSymbolLayer,
    QgsUnitTypes,
    QgsVectorLayer,
    QgsVectorLayerSimpleLabeling,
)
from qgis.PyQt.QtGui import QColor

DEBUG = r"C:\PlaneSight\debug"
SVG = r"C:\PlaneSight\planesight\resources\symbols\strike_dip_bedding.svg"

# Load hillshade (drawn first / underneath) then the attitude points.
hs = QgsRasterLayer(os.path.join(DEBUG, "nepal_hillshade.tif"), "Nepal hillshade")
QgsProject.instance().addMapLayer(hs)
gpkg = os.path.join(DEBUG, "nepal_attitudes.gpkg") + "|layername=attitudes"
lyr = QgsVectorLayer(gpkg, "Nepal attitudes", "ogr")
QgsProject.instance().addMapLayer(lyr)


def svg_layer(stroke_width):
    """An SVG strike/dip marker layer (~12x7.8 mm), rotated by dip_dir."""
    sl = QgsSvgMarkerSymbolLayer(SVG)
    sl.setSize(7.8)  # height; the SVG's native 12:7.8 aspect makes width ~12 mm
    sl.setSizeUnit(QgsUnitTypes.RenderMillimeters)
    sl.setStrokeWidth(stroke_width)
    sl.setStrokeWidthUnit(QgsUnitTypes.RenderMillimeters)
    sl.setDataDefinedProperty(QgsSymbolLayer.PropertyAngle, QgsProperty.fromField("dip_dir"))
    return sl

# Bottom layer: white casing (wider stroke). Top layer: symbol, coloured by reliability.
halo = svg_layer(4.2)
halo.setStrokeColor(QColor("#ffffff"))
symbol = svg_layer(2.4)
symbol.setStrokeColor(QColor("#1a1a1a"))
symbol.setDataDefinedProperty(
    QgsSymbolLayer.PropertyStrokeColor,
    QgsProperty.fromExpression("if(\"reliable\" = 1, '#1a1a1a', '#b0b0b0')"),
)
lyr.setRenderer(QgsSingleSymbolRenderer(QgsMarkerSymbol([halo, symbol])))

# Label each point with its dip value.
labels = QgsPalLayerSettings()
labels.fieldName = "format_number(\"dip\", 0)"
labels.isExpression = True
labels.dist = 2
lyr.setLabeling(QgsVectorLayerSimpleLabeling(labels))
lyr.setLabelsEnabled(True)
lyr.triggerRepaint()

qml = os.path.join(DEBUG, "nepal_attitudes.qml")
lyr.saveNamedStyle(qml)
print(f"[OK] styled Nepal attitudes and saved {qml}")
