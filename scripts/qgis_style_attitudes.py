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
    QgsVectorLayer,
    QgsVectorLayerSimpleLabeling,
)

DEBUG = r"C:\PlaneSight\debug"
SVG = r"C:\PlaneSight\planesight\resources\symbols\strike_dip_bedding.svg"

# Load hillshade (drawn first / underneath) then the attitude points.
hs = QgsRasterLayer(os.path.join(DEBUG, "nepal_hillshade.tif"), "Nepal hillshade")
QgsProject.instance().addMapLayer(hs)
gpkg = os.path.join(DEBUG, "nepal_attitudes.gpkg") + "|layername=attitudes"
lyr = QgsVectorLayer(gpkg, "Nepal attitudes", "ogr")
QgsProject.instance().addMapLayer(lyr)

# SVG strike/dip marker: rotate by dip_dir, stroke colour by reliability.
sl = QgsSvgMarkerSymbolLayer(SVG)
sl.setSize(7)
sl.setDataDefinedProperty(QgsSymbolLayer.PropertyAngle, QgsProperty.fromField("dip_dir"))
sl.setDataDefinedProperty(
    QgsSymbolLayer.PropertyStrokeColor,
    QgsProperty.fromExpression("if(\"reliable\" = 1, '#1a1a1a', '#b0b0b0')"),
)
lyr.setRenderer(QgsSingleSymbolRenderer(QgsMarkerSymbol([sl])))

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
