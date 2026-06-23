"""Phase 1 detector/derivative experiment harness (planesight-bc1 / planesight-c3r).

Answers the first Phase 1 question empirically: which input bands best EXPOSE the
mapped contacts? It assembles the full candidate stack - DEM-derived terrain bands
AND Sentinel-2-derived spectral bands, co-registered on one grid - rasterises the
hand-drawn Nepal traces as (incomplete) ground truth, and scores each band by how
well a simple edge response recovers those traces.

Scoring is positive-unlabeled by design (the drawn traces are a small subset of
the truly mappable contacts): we rank on RECALL at an equal detection budget, not
precision/F1 (see planesight/core/detect/score.py). The winning bands and the
edge-response baseline feed the c3r detector choice; this is a derivative-ranking
baseline, not the final detector.

Run under the headless GDAL env:
  MAMBA_ROOT_PREFIX=$HOME/micromamba PYTHONPATH=$PWD \
    $HOME/bin/micromamba run -n gdal python scripts/detector_experiment.py

Outputs (debug/, gitignored): per-band response GeoTIFFs, the truth mask, and
detector_leaderboard.csv.
"""

from __future__ import annotations

import csv
import logging
import os
import tempfile

import numpy as np
from osgeo import gdal, ogr

from planesight.core.data import (
    align_to_grid,
    asset_href,
    clearest_months,
    fetch_dem,
    pick_scene,
    search_clear_sentinel2,
)
from planesight.core.derivatives import build_spectral_stack, build_terrain_stack, normalize01
from planesight.core.derivatives.terrain import gradients
from planesight.core.detect import recall_curve

gdal.UseExceptions()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("detector_experiment")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACES = os.path.join(REPO, "data", "raw", "nepal", "nepal_traces.shp")
OUT_DIR = os.path.join(REPO, "debug")
AOI_WGS84 = [82.0, 27.6, 83.0, 28.0]
TARGET_EPSG = 32644          # UTM 44N (the traces' CRS)
RES = 30.0                   # analysis grid resolution (m) = GLO-30 native
BUDGETS = (0.01, 0.02, 0.05, 0.10, 0.20)
TOLERANCE_PX = 1             # ~1 pixel registration buffer for matching
RANK_BUDGET = 0.05           # leaderboard is sorted by recall at this budget

# S2 bands the spectral indices need (fetched from one clear dry-season scene).
S2_NEEDED = ("blue", "red", "nir", "swir16", "swir22")
TERRAIN = ("elevation", "slope", "multi_hillshade", "tpi",
           "curvature", "plan_curvature", "profile_curvature")
SPECTRAL = ("ndvi", "clay", "iron_oxide", "ferrous", "swir16_s", "swir22_s")


def dem_on_grid(tmp):
    """Fetch GLO-30 and reproject to the UTM analysis grid; return (array, ds_path)."""
    dem4326 = os.path.join(tmp, "dem4326.tif")
    log.info("Fetching GLO-30 DEM over %s ...", AOI_WGS84)
    fetch_dem(AOI_WGS84, dem4326)
    dem_utm = os.path.join(tmp, "dem_utm.tif")
    gdal.Warp(dem_utm, dem4326, dstSRS=f"EPSG:{TARGET_EPSG}",
              xRes=RES, yRes=RES, resampleAlg="bilinear")
    return dem_utm


def fetch_s2_bands(dem_path, tmp):
    """Fetch a clear dry-season S2 scene and warp each needed band onto the DEM grid.

    Returns a dict {band_key: array}, or {} if no clear scene / network failure -
    the experiment then runs terrain-only.
    """
    gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    try:
        items, _ = search_clear_sentinel2(AOI_WGS84, cloud_max=5.0)
        if not items:
            log.warning("No clear Sentinel-2 scene; running terrain-only.")
            return {}
        scene = pick_scene(items, prefer_months=clearest_months(items))
        log.info("S2 scene %s (month-pref from %d clear scenes)",
                 scene.get("id"), len(items))
        bands = {}
        for key in S2_NEEDED:
            href = asset_href(scene, key)
            out = os.path.join(tmp, f"s2_{key}.tif")
            # warp the remote COG straight onto the DEM grid (streams via overviews)
            align_to_grid("/vsicurl/" + href, out, dem_path, resampling="bilinear")
            bands[key] = gdal.Open(out).ReadAsArray().astype(float)
        return bands
    except Exception as exc:  # network / STAC / asset issues - degrade gracefully
        log.warning("Sentinel-2 fetch failed (%s); running terrain-only.", exc)
        return {}


def rasterize_traces(dem_path):
    """Burn the trace polylines onto the DEM grid -> boolean ground-truth mask."""
    ref = gdal.Open(dem_path)
    gt, proj = ref.GetGeoTransform(), ref.GetProjection()
    w, h = ref.RasterXSize, ref.RasterYSize
    mem = gdal.GetDriverByName("MEM").Create("", w, h, 1, gdal.GDT_Byte)
    mem.SetGeoTransform(gt)
    mem.SetProjection(proj)
    src = ogr.Open(TRACES)
    layer = src.GetLayer()  # traces are already in the grid CRS (UTM 44N)
    gdal.RasterizeLayer(mem, [1], layer, burn_values=[1],
                        options=["ALL_TOUCHED=TRUE"])
    return mem.ReadAsArray().astype(bool)


def edge_response(band):
    """Baseline contact response: gradient magnitude of the normalized band.

    Contacts read as sharp changes in a band; this is the simplest detector-
    agnostic edge strength, the baseline the c3r detectors must beat. NaNs
    propagate so they are excluded by the scorer's validity mask.
    """
    norm = normalize01(band)
    fx, fy = gradients(norm, px=1.0)  # pixel units; magnitude only
    return np.hypot(fx, fy)


def save_raster(arr, like_path, out_name):
    """Write a float array on the reference grid to debug/ for visual QA."""
    ref = gdal.Open(like_path)
    out = os.path.join(OUT_DIR, out_name)
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(out, ref.RasterXSize, ref.RasterYSize, 1, gdal.GDT_Float32,
                    options=["COMPRESS=DEFLATE", "TILED=YES"])
    ds.SetGeoTransform(ref.GetGeoTransform())
    ds.SetProjection(ref.GetProjection())
    ds.GetRasterBand(1).WriteArray(np.where(np.isfinite(arr), arr, -9999).astype("float32"))
    ds.GetRasterBand(1).SetNoDataValue(-9999)
    ds = None
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = tempfile.mkdtemp()
    dem_path = dem_on_grid(tmp)
    ds = gdal.Open(dem_path)  # keep the dataset alive while we read the band
    dem = ds.ReadAsArray().astype(float)
    nodata = ds.GetRasterBand(1).GetNoDataValue()
    ds = None
    if nodata is not None:
        dem = np.where(dem == nodata, np.nan, dem)
    valid = np.isfinite(dem)
    log.info("Analysis grid: %d x %d px @ %.0f m", dem.shape[1], dem.shape[0], RES)

    truth = rasterize_traces(dem_path) & valid
    log.info("Ground-truth trace pixels: %d (incomplete - recall-only scoring)",
             int(truth.sum()))
    save_raster(truth.astype(float), dem_path, "truth_mask.tif")

    # --- assemble candidate bands (terrain + spectral) ---
    _, terr = build_terrain_stack(dem, RES, names=TERRAIN)
    layers = dict(terr)
    s2 = fetch_s2_bands(dem_path, tmp)
    if s2:
        _, spec = build_spectral_stack(s2, names=SPECTRAL)
        layers.update(spec)
    log.info("Scoring %d bands: %s", len(layers), ", ".join(layers))

    # --- score each band's edge response by recall at equal budget ---
    rows = []
    for name, band in layers.items():
        resp = edge_response(band)
        curve = dict(recall_curve(resp, truth, budgets=BUDGETS,
                                  tolerance_px=TOLERANCE_PX, valid_mask=valid))
        rows.append((name, curve))
        save_raster(resp, dem_path, f"response_{name}.tif")

    rows.sort(key=lambda r: (r[1].get(RANK_BUDGET) or 0.0), reverse=True)

    # --- report + CSV ---
    hdr = "band".ljust(20) + "".join(f"r@{int(b*100)}%".rjust(9) for b in BUDGETS)
    print("\n=== Derivative ranking: recall vs drawn traces at equal budget ===")
    print(f"(higher = exposes contacts better; sorted by recall@{int(RANK_BUDGET*100)}%, "
          f"tolerance {TOLERANCE_PX}px)\n")
    print(hdr)
    print("-" * len(hdr))
    for name, curve in rows:
        line = name.ljust(20) + "".join(
            (f"{curve[b]:.3f}" if np.isfinite(curve[b]) else "  nan").rjust(9)
            for b in BUDGETS)
        print(line)

    csv_path = os.path.join(OUT_DIR, "detector_leaderboard.csv")
    with open(csv_path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["band"] + [f"recall_at_{int(b*100)}pct" for b in BUDGETS])
        for name, curve in rows:
            wr.writerow([name] + [f"{curve[b]:.4f}" for b in BUDGETS])

    print(f"\nWrote {csv_path}")
    print("Per-band response GeoTIFFs + truth_mask.tif in debug/ (load over the "
          "hillshade in QGIS to eyeball the off-label detections).")
    print("NOTE: detections far from drawn traces are UNCONFIRMED, not false - "
          "the labels are a small subset of all mappable contacts.")


if __name__ == "__main__":
    main()
