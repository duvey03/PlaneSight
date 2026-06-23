"""Nepal vertical slice: first REAL strike/dip measurements end-to-end.

Fetches the Copernicus GLO-30 DEM over the Nepal seed AOI, reprojects it to the
traces' metric CRS (UTM 44N / EPSG:32644), samples the DEM along each hand-drawn
trace, fits a plane (PCA), and reports the resulting attitudes + quality metrics.
Writes the measurements to a GeoPackage.

Run under the headless GDAL env:
  MAMBA_ROOT_PREFIX=$HOME/micromamba PYTHONPATH=$PWD \
    $HOME/bin/micromamba run -n gdal python scripts/nepal_slice.py
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
from osgeo import gdal, ogr, osr

from planesight.core.attitude import fit_plane, sample_trace
from planesight.core.data import fetch_dem

gdal.UseExceptions()

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACES = os.path.join(REPO, "data", "raw", "nepal", "nepal_traces.shp")
AOI_WGS84 = [82.0, 27.6, 83.0, 28.0]   # covers the Nepal traces
TARGET_EPSG = 32644                     # UTM 44N (the traces' CRS)
SPACING = 30.0                          # DEM-resolution sampling along traces
MIN_SAMPLES = 5                         # need a few points for a meaningful fit
COND_RELIABLE = 1e-3                    # conditioning above this = usable fit
OUT_DIR = os.path.join(REPO, "debug")


def load_dem_utm():
    """Fetch GLO-30 over the AOI and reproject to the metric target CRS."""
    tmp = tempfile.mkdtemp()
    dem4326 = os.path.join(tmp, "dem4326.tif")
    print(f"Fetching GLO-30 DEM over {AOI_WGS84} ...")
    fetch_dem(AOI_WGS84, dem4326)
    dem_utm = os.path.join(tmp, "dem_utm.tif")
    print(f"Reprojecting DEM to EPSG:{TARGET_EPSG} @ {SPACING} m ...")
    gdal.Warp(
        dem_utm, dem4326, dstSRS=f"EPSG:{TARGET_EPSG}",
        xRes=SPACING, yRes=SPACING, resampleAlg="bilinear",
    )
    ds = gdal.Open(dem_utm)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray().astype(float)
    return arr, ds.GetGeoTransform(), band.GetNoDataValue()


def load_traces():
    """Read trace polylines (one list of (x, y) per line part) from the shapefile."""
    ds = ogr.Open(TRACES)
    layer = ds.GetLayer()
    traces = []
    for feat in layer:
        geom = feat.GetGeometryRef()
        if geom is None:
            continue
        parts = [geom] if geom.GetGeometryType() == ogr.wkbLineString else \
            [geom.GetGeometryRef(i) for i in range(geom.GetGeometryCount())]
        for part in parts:
            pts = [(part.GetX(i), part.GetY(i)) for i in range(part.GetPointCount())]
            if len(pts) >= 2:
                traces.append(np.asarray(pts, dtype=float))
    return traces


def write_geopackage(records):
    """Write attitude points (at trace centroids) to a GeoPackage."""
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "nepal_attitudes.gpkg")
    drv = ogr.GetDriverByName("GPKG")
    if os.path.exists(out):
        drv.DeleteDataSource(out)
    ds = drv.CreateDataSource(out)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(TARGET_EPSG)
    layer = ds.CreateLayer("attitudes", srs, ogr.wkbPoint)
    fields = [("strike", ogr.OFTReal), ("dip", ogr.OFTReal), ("dip_dir", ogr.OFTReal),
              ("cond", ogr.OFTReal), ("planarity", ogr.OFTReal), ("relief", ogr.OFTReal),
              ("n_samp", ogr.OFTInteger), ("reliable", ogr.OFTInteger)]
    for name, ftype in fields:
        layer.CreateField(ogr.FieldDefn(name, ftype))
    for cx, cy, att in records:
        feat = ogr.Feature(layer.GetLayerDefn())
        pt = ogr.Geometry(ogr.wkbPoint)
        pt.AddPoint_2D(float(cx), float(cy))
        feat.SetGeometry(pt)
        for name, val in [("strike", att.strike), ("dip", att.dip),
                          ("dip_dir", att.dip_direction), ("cond", att.conditioning),
                          ("planarity", att.planarity), ("relief", att.relief),
                          ("n_samp", att.n_samples),
                          ("reliable", int(att.conditioning >= COND_RELIABLE))]:
            feat.SetField(name, val)
        layer.CreateFeature(feat)
    ds = None
    return out


def main():
    arr, gt, nodata = load_dem_utm()
    print(f"DEM: {arr.shape[1]}x{arr.shape[0]} px, elevation "
          f"{np.nanmin(arr):.0f}-{np.nanmax(arr):.0f} m")
    traces = load_traces()
    print(f"Traces: {len(traces)} line parts loaded\n")

    records, skipped = [], 0
    for tr in traces:
        pts3d = sample_trace(tr, arr, gt, spacing=SPACING, nodata=nodata)
        if len(pts3d) < MIN_SAMPLES:
            skipped += 1
            continue
        att = fit_plane(pts3d)
        cx, cy = pts3d[:, 0].mean(), pts3d[:, 1].mean()
        records.append((cx, cy, att))

    fitted = [r[2] for r in records]
    reliable = [a for a in fitted if a.conditioning >= COND_RELIABLE]
    print(f"Fitted {len(fitted)} traces ({skipped} skipped for too few samples)")
    print(f"Reliable (conditioning >= {COND_RELIABLE}): {len(reliable)} "
          f"({100*len(reliable)//max(1,len(fitted))}%)\n")

    if reliable:
        dips = np.array([a.dip for a in reliable])
        reliefs = np.array([a.relief for a in reliable])
        print("Reliable-trace attitudes:")
        print(f"  dip      : median {np.median(dips):.0f} deg, "
              f"IQR {np.percentile(dips,25):.0f}-{np.percentile(dips,75):.0f}")
        print(f"  relief   : median {np.median(reliefs):.0f} m, max {reliefs.max():.0f} m")
        # circular mean of dip-direction
        dd = np.radians([a.dip_direction for a in reliable])
        mean_dd = np.degrees(np.arctan2(np.sin(dd).mean(), np.cos(dd).mean())) % 360
        print(f"  mean dip-direction: {mean_dd:.0f} deg "
              f"(=> mean strike ~{(mean_dd-90)%360:.0f} deg)")

    out = write_geopackage(records)
    print(f"\nWrote {len(records)} measurements -> {out}")


if __name__ == "__main__":
    main()
