"""PlaneSight QGIS integration checkpoint (M0) - paste into the QGIS Python Console.

Runs the FULL headless pipeline (fetch -> derivatives -> detect -> drainage-flag ->
link -> strike/dip) inside QGIS's own Python/GDAL, then builds in-memory QGIS vector
layers (traces: kept vs drainage-flagged; attitude points) and adds them to the canvas.

This is the M0 de-risking gate for the GUI (beads epic planesight-pzz / planesight-4vw):
it proves the core runs in QGIS AND that results convert to QgsVectorLayers the way M1/M2
will need - BEFORE any widget code. If correct traces + attitudes render on the canvas,
the integration is sound and the dockwidget work can proceed on solid ground.

How to run:
  1. Edit REPO and (optionally) BBOX below.
  2. QGIS: Plugins > Python Console.
  3. Paste this whole file and run.

(Headless smoke-test of the same pipeline, outside QGIS: scripts/detect_attitudes_nepal.py.)
"""

import os
import sys
import tempfile

import numpy as np

REPO = r"C:\PlaneSight"                 # <- path to your local clone
BBOX = [82.40, 27.70, 82.50, 27.80]    # <- small AOI (default: Nepal foothills, fast)

# Pipeline parameters (mirror scripts/detect_attitudes_nepal.py - the tested headless run)
RES = 30.0
BANDS = ("profile_curvature", "curvature", "slope")   # Phase 1 winners
SIGMA_Z = 2.0
COND_RELIABLE, MAP_COND_RELIABLE = 1e-2, 1e-3
MIN_TRACE_PTS = 8
DRAIN_DS, DRAIN_ACCUM, DRAIN_OVERLAP = 3, 15, 0.5

if REPO not in sys.path:
    sys.path.insert(0, REPO)

from osgeo import gdal  # noqa: E402
from qgis.core import (  # noqa: E402
    QgsCategorizedSymbolRenderer,
    QgsFeature,
    QgsField,
    QgsGeometry,
    QgsPointXY,
    QgsProject,
    QgsRasterLayer,
    QgsRendererCategory,
    QgsSymbol,
    QgsVectorLayer,
)
from qgis.PyQt.QtCore import QVariant  # noqa: E402
from qgis.PyQt.QtGui import QColor  # noqa: E402

from planesight.core.data import fetch_dem  # noqa: E402
from planesight.core.detect.vectorize import pixels_to_world  # noqa: E402
from planesight.core.geo import utm_epsg  # noqa: E402
from planesight.core.pipeline import detect_attitudes  # noqa: E402


def _line_feature(fields, poly_px, gt, attrs):
    """Build a LineString QgsFeature from a pixel-space polyline (col,row -> world)."""
    world = pixels_to_world(np.asarray(poly_px, float)[:, ::-1], gt)
    geom = QgsGeometry.fromPolylineXY(
        [QgsPointXY(float(x), float(y)) for x, y in world]
    )
    feat = QgsFeature(fields)
    feat.setGeometry(geom)
    feat.setAttributes(attrs)
    return feat, world


def _categorized_lines(layer):
    """Style the trace layer: kept = green, drainage = orange."""
    cats = []
    for value, color, label in (
        ("kept", "#2ca02c", "kept (-> strike/dip)"),
        ("drainage", "#ff7f0e", "drainage-flagged"),
    ):
        sym = QgsSymbol.defaultSymbol(layer.geometryType())
        sym.setColor(QColor(color))
        sym.setWidth(0.5)
        cats.append(QgsRendererCategory(value, sym, label))
    layer.setRenderer(QgsCategorizedSymbolRenderer("class", cats))


def main():
    gdal.UseExceptions()
    epsg = utm_epsg(BBOX)
    crs = f"EPSG:{epsg}"
    tmp = tempfile.mkdtemp()
    dem4326 = os.path.join(tmp, "dem.tif")
    dem_utm = os.path.join(tmp, "dem_utm.tif")

    print(f"[1/3] Fetching GLO-30 DEM over {BBOX} ...")
    fetch_dem(BBOX, dem4326)
    gdal.Warp(dem_utm, dem4326, dstSRS=crs, xRes=RES, yRes=RES, resampleAlg="bilinear")
    ds = gdal.Open(dem_utm)
    dem = ds.ReadAsArray().astype(float)
    gt = ds.GetGeoTransform()
    nodata = ds.GetRasterBand(1).GetNoDataValue()
    ds = None
    if nodata is not None:
        dem = np.where(dem == nodata, np.nan, dem)
    print(f"      grid {dem.shape[1]} x {dem.shape[0]} @ {RES:.0f} m, {crs}")

    print(f"[2/3] Pipeline (detect {'+'.join(BANDS)} -> drainage-flag -> link "
          f"-> strike/dip) ...")
    # the SAME core orchestration M1's QgsTask will call (planesight.core.pipeline)
    result = detect_attitudes(
        dem, gt, res=RES, bands=BANDS, sigma_z=SIGMA_Z, min_trace_pts=MIN_TRACE_PTS,
        drain_downsample=DRAIN_DS, drain_accum=DRAIN_ACCUM, drain_overlap=DRAIN_OVERLAP,
        cond_reliable=COND_RELIABLE, map_cond_reliable=MAP_COND_RELIABLE,
    )
    flagged = [t for t, f in zip(result.traces, result.flags) if f.is_drainage]
    n_reliable = sum(a.reliable for a in result.attitudes)
    print(f"      {len(result.traces)} traces -> {len(flagged)} flagged / "
          f"{len(result.kept)} kept -> {len(result.linked)} linked; "
          f"{len(result.attitudes)} fits, {n_reliable} reliable")

    print("[3/3] Building QGIS layers ...")
    # --- trace lines (kept linked + drainage-flagged), attribute 'class' ---
    tlayer = QgsVectorLayer(f"LineString?crs={crs}", "PlaneSight traces", "memory")
    tprov = tlayer.dataProvider()
    tprov.addAttributes([QgsField("class", QVariant.String)])
    tlayer.updateFields()
    feats = []
    for poly in result.linked:
        feat, _ = _line_feature(tlayer.fields(), poly, gt, ["kept"])
        feats.append(feat)
    for poly in flagged:
        feat, _ = _line_feature(tlayer.fields(), poly, gt, ["drainage"])
        feats.append(feat)
    tprov.addFeatures(feats)
    tlayer.updateExtents()
    _categorized_lines(tlayer)

    # --- attitude points ---
    alayer = QgsVectorLayer(f"Point?crs={crs}", "PlaneSight attitudes", "memory")
    aprov = alayer.dataProvider()
    aprov.addAttributes([
        QgsField("strike", QVariant.Double),
        QgsField("dip", QVariant.Double),
        QgsField("dip_dir", QVariant.Double),
        QgsField("conditioning", QVariant.Double),
        QgsField("reliable", QVariant.Int),
    ])
    alayer.updateFields()
    apts = []
    for ap in result.attitudes:
        att = ap.attitude
        feat = QgsFeature(alayer.fields())
        feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(ap.x, ap.y)))
        feat.setAttributes([
            float(att.strike), float(att.dip), float(att.dip_direction),
            float(att.conditioning), int(ap.reliable),
        ])
        apts.append(feat)
    aprov.addFeatures(apts)
    alayer.updateExtents()

    # DEM at the bottom for visual context, then traces, then attitudes on top.
    dem_layer = QgsRasterLayer(dem_utm, "PlaneSight DEM")
    if dem_layer.isValid():
        QgsProject.instance().addMapLayer(dem_layer)
    QgsProject.instance().addMapLayer(tlayer)
    QgsProject.instance().addMapLayer(alayer)
    print(f"[OK] added {2 + int(dem_layer.isValid())} layers "
          f"(DEM + traces + attitudes): {tlayer.featureCount()} traces "
          f"({len(flagged)} flagged), {alayer.featureCount()} attitudes "
          f"- check the QGIS canvas")


main()
