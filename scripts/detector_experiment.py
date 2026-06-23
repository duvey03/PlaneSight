"""Phase 1 detector/derivative experiment harness (planesight-bc1 / planesight-c3r).

Answers the first Phase 1 question empirically: which input bands best EXPOSE the
mapped contacts? It assembles the full candidate stack - DEM-derived terrain bands
AND Sentinel-2-derived spectral bands, co-registered on one grid - rasterises the
hand-drawn traces as (incomplete) ground truth, and scores each band by how well a
simple edge response recovers those traces.

Scoring is positive-unlabeled by design (the drawn traces are a small subset of
the truly mappable contacts): we rank on RECALL at an equal detection budget, not
precision/F1 (see planesight/core/detect/score.py). The winning bands and the
edge-response baseline feed the c3r detector choice; this is a derivative-ranking
baseline, not the final detector.

Two regions of differing trace PROVENANCE let us de-bias the S2 verdict:
  - nepal    - monsoonal Himalaya, traces digitized from topography -> DEM-biased.
  - pakistan - arid Balochistan, traces digitized from SATELLITE IMAGERY
               (source: ".../Satellite Imagery/Pakistan/...") -> the fair test of
               whether spectral bands expose contacts.

Run under the headless GDAL env (default region nepal; pass a region name):
  MAMBA_ROOT_PREFIX=$HOME/micromamba PYTHONPATH=$PWD \
    $HOME/bin/micromamba run -n gdal python scripts/detector_experiment.py pakistan

Outputs (debug/, gitignored, region-prefixed): per-band response GeoTIFFs, the
truth mask, and <region>_detector_leaderboard.csv.
"""

from __future__ import annotations

import csv
import logging
import os
import sys
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
from planesight.core.detect import linear_response, recall_curve

gdal.UseExceptions()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("detector_experiment")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "debug")

# Per-region config. epsg = a local metric (UTM) CRS for the analysis grid.
REGIONS = {
    "nepal": {
        "aoi": [82.0, 27.6, 83.0, 28.0],
        "traces": os.path.join(REPO, "data", "raw", "nepal", "nepal_traces.shp"),
        "epsg": 32644,  # UTM 44N
        "provenance": "topography-digitized (DEM-biased)",
    },
    "pakistan": {
        "aoi": [62.0, 25.0, 63.0, 26.0],
        "traces": os.path.join(REPO, "data", "raw", "pakistan", "pakistan_traces.shp"),
        "epsg": 32641,  # UTM 41N
        "provenance": "satellite-imagery-digitized (fair S2 test)",
    },
    "canada": {
        "aoi": [-117.0, 52.0, -116.0, 53.0],
        "traces": os.path.join(REPO, "data", "raw", "canada", "canada_traces.shp"),
        "epsg": 32611,  # UTM 11N (Alberta/BC Cordillera - high relief, NOT the Shield)
        "provenance": "Cordilleran mountains, EPSG:4326 traces",
    },
}

RES = 30.0                   # analysis grid resolution (m) = GLO-30 native
BUDGETS = (0.01, 0.02, 0.05, 0.10, 0.20)
TOLERANCE_PX = 1             # ~1 pixel registration buffer for matching
RANK_BUDGET = 0.05           # leaderboard is sorted by recall at this budget
SIGMA_D = 1.0                # structure-tensor derivative scale (px)
SIGMA_I = 4.0                # structure-tensor integration scale (px ~ 120 m)

S2_NEEDED = ("blue", "red", "nir", "swir16", "swir22")
TERRAIN = ("elevation", "slope", "multi_hillshade", "tpi",
           "curvature", "plan_curvature", "profile_curvature")
SPECTRAL = ("ndvi", "clay", "iron_oxide", "ferrous", "swir16_s", "swir22_s")


def dem_on_grid(aoi, epsg, tmp):
    """Fetch GLO-30 over the AOI and reproject to the UTM analysis grid."""
    dem4326 = os.path.join(tmp, "dem4326.tif")
    log.info("Fetching GLO-30 DEM over %s ...", aoi)
    fetch_dem(aoi, dem4326)
    dem_utm = os.path.join(tmp, "dem_utm.tif")
    gdal.Warp(dem_utm, dem4326, dstSRS=f"EPSG:{epsg}",
              xRes=RES, yRes=RES, resampleAlg="bilinear")
    return dem_utm


def fetch_s2_bands(aoi, dem_path, tmp):
    """Fetch a clear dry-season S2 scene and warp each needed band onto the DEM grid.

    Returns {band_key: array}, or {} if no clear scene / network failure - the
    experiment then runs terrain-only.
    """
    gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    try:
        items, _ = search_clear_sentinel2(aoi, cloud_max=5.0)
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
            align_to_grid("/vsicurl/" + href, out, dem_path, resampling="bilinear")
            ds = gdal.Open(out)
            bands[key] = ds.ReadAsArray().astype(float)
            ds = None
        return bands
    except Exception as exc:  # network / STAC / asset issues - degrade gracefully
        log.warning("Sentinel-2 fetch failed (%s); running terrain-only.", exc)
        return {}


def rasterize_traces(traces_path, dem_path, epsg):
    """Reproject traces to the grid CRS, then burn them onto the DEM grid -> mask."""
    ref = gdal.Open(dem_path)
    gt, proj = ref.GetGeoTransform(), ref.GetProjection()
    w, h = ref.RasterXSize, ref.RasterYSize
    ref = None
    # reproject the vector to the analysis CRS (no-op if already there)
    tmpdir = tempfile.mkdtemp()
    reproj = os.path.join(tmpdir, "traces.gpkg")
    gdal.VectorTranslate(
        reproj, traces_path,
        options=gdal.VectorTranslateOptions(dstSRS=f"EPSG:{epsg}", reproject=True),
    )
    mem = gdal.GetDriverByName("MEM").Create("", w, h, 1, gdal.GDT_Byte)
    mem.SetGeoTransform(gt)
    mem.SetProjection(proj)
    src = ogr.Open(reproj)
    layer = src.GetLayer()
    gdal.RasterizeLayer(mem, [1], layer, burn_values=[1], options=["ALL_TOUCHED=TRUE"])
    mask = mem.ReadAsArray().astype(bool)
    mem = None
    src = None
    return mask


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
    ds = gdal.GetDriverByName("GTiff").Create(
        out, ref.RasterXSize, ref.RasterYSize, 1, gdal.GDT_Float32,
        options=["COMPRESS=DEFLATE", "TILED=YES"])
    ds.SetGeoTransform(ref.GetGeoTransform())
    ds.SetProjection(ref.GetProjection())
    ds.GetRasterBand(1).WriteArray(np.where(np.isfinite(arr), arr, -9999).astype("float32"))
    ds.GetRasterBand(1).SetNoDataValue(-9999)
    ds = None
    ref = None
    return out


def run_region(key):
    cfg = REGIONS[key]
    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = tempfile.mkdtemp()
    log.info("=== Region: %s (%s) ===", key, cfg["provenance"])

    dem_path = dem_on_grid(cfg["aoi"], cfg["epsg"], tmp)
    ds = gdal.Open(dem_path)
    dem = ds.ReadAsArray().astype(float)
    nodata = ds.GetRasterBand(1).GetNoDataValue()
    ds = None
    if nodata is not None:
        dem = np.where(dem == nodata, np.nan, dem)
    valid = np.isfinite(dem)
    log.info("Analysis grid: %d x %d px @ %.0f m", dem.shape[1], dem.shape[0], RES)

    truth = rasterize_traces(cfg["traces"], dem_path, cfg["epsg"]) & valid
    log.info("Ground-truth trace pixels: %d (incomplete - recall-only scoring)",
             int(truth.sum()))
    save_raster(truth.astype(float), dem_path, f"{key}_truth_mask.tif")

    _, terr = build_terrain_stack(dem, RES, names=TERRAIN)
    layers = dict(terr)
    s2 = fetch_s2_bands(cfg["aoi"], dem_path, tmp)
    if s2:
        _, spec = build_spectral_stack(s2, names=SPECTRAL)
        layers.update(spec)
    log.info("Scoring %d bands: %s", len(layers), ", ".join(layers))

    rows = []
    for name, band in layers.items():
        resp = edge_response(band)
        curve = dict(recall_curve(resp, truth, budgets=BUDGETS,
                                  tolerance_px=TOLERANCE_PX, valid_mask=valid))
        rows.append((name, curve))
        save_raster(resp, dem_path, f"{key}_response_{name}.tif")
    rows.sort(key=lambda r: (r[1].get(RANK_BUDGET) or 0.0), reverse=True)

    hdr = "band".ljust(20) + "".join(f"r@{int(b*100)}%".rjust(9) for b in BUDGETS)
    print(f"\n=== {key}: recall vs drawn traces at equal budget "
          f"({cfg['provenance']}) ===")
    print(f"(sorted by recall@{int(RANK_BUDGET*100)}%, tolerance {TOLERANCE_PX}px)\n")
    print(hdr)
    print("-" * len(hdr))
    for name, curve in rows:
        print(name.ljust(20) + "".join(
            (f"{curve[b]:.3f}" if np.isfinite(curve[b]) else "  nan").rjust(9)
            for b in BUDGETS))

    csv_path = os.path.join(OUT_DIR, f"{key}_detector_leaderboard.csv")
    with open(csv_path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["band"] + [f"recall_at_{int(b*100)}pct" for b in BUDGETS])
        for name, curve in rows:
            wr.writerow([name] + [f"{curve[b]:.4f}" for b in BUDGETS])
    print(f"\nWrote {csv_path}")

    score_linear_response(key, layers, truth, valid, dem_path, rows)
    return rows


def score_linear_response(key, layers, truth, valid, dem_path, band_rows):
    """Compare the multi-channel structure-tensor linear response (DEM-only,
    S2-only, fused) against the best single-band gradient baseline.

    Tests the contact-vs-mound hypothesis: a linearity-aware, cross-modal operator
    should match/beat per-band gradient magnitude, and fusing DEM + S2 should beat
    either alone (Sentinel supporting the DEM edge).
    """
    dem_ch = [normalize01(layers[n]) for n in TERRAIN if n in layers]
    s2_ch = [normalize01(layers[n]) for n in SPECTRAL if n in layers]
    variants = [("st_dem", "DEM-only (struct)", dem_ch)]
    if s2_ch:
        variants.append(("st_s2", "S2-only (struct)", s2_ch))
        variants.append(("st_demS2", "DEM+S2 (struct)", dem_ch + s2_ch))

    results = []
    # reference: the best single band under the plain gradient-magnitude baseline
    best_band, best_curve = band_rows[0]
    results.append((f"gradient baseline ({best_band})", best_curve))
    for tag, label, channels in variants:
        resp = linear_response(channels, sigma_d=SIGMA_D, sigma_i=SIGMA_I,
                               kind="anisotropy")
        curve = dict(recall_curve(resp, truth, budgets=BUDGETS,
                                  tolerance_px=TOLERANCE_PX, valid_mask=valid))
        results.append((label, curve))
        save_raster(resp, dem_path, f"{key}_{tag}.tif")
        if tag == "st_demS2":  # coherence raster for the visual mound-suppression QA
            coh = linear_response(channels, sigma_d=SIGMA_D, sigma_i=SIGMA_I,
                                  kind="coherence")
            save_raster(coh, dem_path, f"{key}_st_demS2_coherence.tif")

    hdr = "method".ljust(28) + "".join(f"r@{int(b*100)}%".rjust(9) for b in BUDGETS)
    print(f"\n=== {key}: structure-tensor linear response vs gradient baseline ===")
    print(f"(recall@equal budget, tolerance {TOLERANCE_PX}px; "
          f"sigma_d={SIGMA_D}, sigma_i={SIGMA_I})\n")
    print(hdr)
    print("-" * len(hdr))
    for label, curve in results:
        print(label.ljust(28) + "".join(
            (f"{curve[b]:.3f}" if np.isfinite(curve[b]) else "  nan").rjust(9)
            for b in BUDGETS))


def main():
    keys = sys.argv[1:] or ["nepal"]
    for key in keys:
        if key not in REGIONS:
            raise SystemExit(f"unknown region {key!r}; choose from {sorted(REGIONS)}")
    for key in keys:
        run_region(key)
    print("\nNOTE: detections far from drawn traces are UNCONFIRMED, not false - "
          "the labels are a small subset of all mappable contacts.")


if __name__ == "__main__":
    main()
