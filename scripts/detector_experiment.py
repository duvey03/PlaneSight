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
    item_month,
    search_clear_sentinel2,
)
from planesight.core.derivatives import build_spectral_stack, build_terrain_stack, normalize01
from planesight.core.derivatives.terrain import gradients
from planesight.core.detect import linear_response, linearity_at_budget, recall_curve

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


def _aoi_coverage(items, aoi, n=120):
    """Fraction of the AOI covered by the union of the items' footprints (bbox)."""
    minx, miny, maxx, maxy = aoi
    xs = np.linspace(minx, maxx, n)
    ys = np.linspace(miny, maxy, n)
    gx, gy = np.meshgrid(xs, ys)
    covered = np.zeros((n, n), dtype=bool)
    for it in items:
        bx0, by0, bx1, by1 = it["bbox"][:4]
        covered |= (gx >= bx0) & (gx <= bx1) & (gy >= by0) & (gy <= by1)
    return float(covered.mean())


def fetch_s2_bands(aoi, dem_path, tmp, force_months=None):
    """Fetch a clear dry-season S2 mosaic and warp each band onto the DEM grid.

    A single S2 scene often covers only part of an AOI (the tile grid is fixed), so
    we pick the acquisition DATE whose tiles best cover the AOI and mosaic them
    (same date -> no temporal seam). This is the experiment-grade stand-in for the
    deferred multi-scene compositing (planesight-bcn). ``force_months`` (e.g. [11])
    overrides the empirical dry-season pick, for seasonal-sensitivity tests.
    Returns {band_key: array} or {} if no usable coverage / network failure (then
    terrain-only).
    """
    gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    try:
        items, _ = search_clear_sentinel2(aoi, cloud_max=5.0)
        if not items:
            log.warning("No clear Sentinel-2 scene; running terrain-only.")
            return {}
        months = set(force_months) if force_months else set(clearest_months(items))
        if force_months:
            log.info("Forcing S2 months %s (seasonal test)", sorted(months))
        candidates = [it for it in items if item_month(it) in months] or items
        by_date = {}
        for it in candidates:
            by_date.setdefault(it["properties"]["datetime"][:10], []).append(it)
        best_date = max(by_date, key=lambda d: _aoi_coverage(by_date[d], aoi))
        scenes = by_date[best_date]
        cover = _aoi_coverage(scenes, aoi)
        log.info("S2 date %s: %d tile(s), AOI coverage %.0f%%",
                 best_date, len(scenes), 100 * cover)
        if cover < 0.5:
            log.warning("Best S2 date covers only %.0f%% of AOI; spectral bands "
                        "will be largely nodata.", 100 * cover)
        bands = {}
        for key in S2_NEEDED:
            hrefs = ["/vsicurl/" + asset_href(s, key) for s in scenes]
            out = os.path.join(tmp, f"s2_{key}.tif")
            align_to_grid(hrefs, out, dem_path, resampling="bilinear", src_nodata=0)
            ds = gdal.Open(out)
            arr = ds.ReadAsArray().astype(float)
            nd = ds.GetRasterBand(1).GetNoDataValue()
            ds = None
            bands[key] = np.where(arr == nd, np.nan, arr) if nd is not None else arr
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
    fm = os.environ.get("PS_FORCE_MONTHS")
    force_months = [int(m) for m in fm.split(",")] if fm else None
    s2 = fetch_s2_bands(cfg["aoi"], dem_path, tmp, force_months=force_months)
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
    """Compare the structure-tensor linear response against the gradient baseline,
    on BOTH recall (vs incomplete labels) and label-free LINEARITY.

    Two clean comparisons: (a) same-band - gradient vs structure tensor on the
    single best band, isolating the operator from the band set; (b) fusion -
    structure tensor on the DEM stack, S2 stack, and both. The linearity column
    measures what recall cannot: whether detections form long clean lines (the
    contact-vs-mound goal) rather than scattered blobs.
    """
    dem_ch = [normalize01(layers[n]) for n in TERRAIN if n in layers]
    s2_ch = [normalize01(layers[n]) for n in SPECTRAL if n in layers]
    best_band, _ = band_rows[0]
    best_norm = normalize01(layers[best_band])

    def st(channels):
        return linear_response(channels, sigma_d=SIGMA_D, sigma_i=SIGMA_I,
                               kind="anisotropy")

    methods = [
        (f"gradient [{best_band}]", "grad_best", edge_response(layers[best_band])),
        (f"struct-tensor [{best_band}]", "st_best", st(best_norm)),
        ("struct DEM-stack", "st_dem", st(dem_ch)),
    ]
    if s2_ch:
        methods.append(("struct S2-stack", "st_s2", st(s2_ch)))
        methods.append(("struct DEM+S2", "st_demS2", st(dem_ch + s2_ch)))

    rows = []
    for label, tag, resp in methods:
        curve = dict(recall_curve(resp, truth, budgets=BUDGETS,
                                  tolerance_px=TOLERANCE_PX, valid_mask=valid))
        lin = linearity_at_budget(resp, budget=RANK_BUDGET, valid_mask=valid)
        rows.append((label, curve, lin))
        save_raster(resp, dem_path, f"{key}_{tag}.tif")

    hdr = ("method".ljust(26) + "".join(f"r@{int(b*100)}%".rjust(8) for b in BUDGETS)
           + f"  lin@{int(RANK_BUDGET*100)}%".rjust(9))
    print(f"\n=== {key}: structure tensor vs gradient (recall AND linearity) ===")
    print(f"(tolerance {TOLERANCE_PX}px; sigma_d={SIGMA_D}, sigma_i={SIGMA_I}; "
          f"linearity is label-free: 1=clean lines, 0=blobs/speckle)\n")
    print(hdr)
    print("-" * len(hdr))
    for label, curve, lin in rows:
        print(label.ljust(26)
              + "".join((f"{curve[b]:.3f}" if np.isfinite(curve[b]) else "nan").rjust(8)
                        for b in BUDGETS)
              + f"{lin:.3f}".rjust(9))


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
