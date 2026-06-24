"""Multi-region detection evaluation (planesight-wiu).

Scores the ClassicalTraceDetector itself (not just per-band responses) on each
region: how well its auto-drawn traces recover the hand-drawn contacts (recall +
label-free linearity), and whether they yield sane, well-conditioned attitudes.
DEM-only (the Phase 1 winning bands), so it runs fast and regions parallelise.

  MAMBA_ROOT_PREFIX=$HOME/micromamba PYTHONPATH=$PWD \
    $HOME/bin/micromamba run -n gdal python scripts/detector_eval.py <region>
"""

from __future__ import annotations

import logging
import os
import tempfile

import numpy as np
from osgeo import gdal, ogr
from scipy.ndimage import binary_dilation

from planesight.core.attitude import fit_plane, sample_trace
from planesight.core.data import fetch_dem
from planesight.core.derivatives import build_terrain_stack, normalize01
from planesight.core.detect import canny, disk, linearity_metrics, polylines_from_mask

gdal.UseExceptions()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("detector_eval")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGIONS = {
    "nepal": ([82.0, 27.6, 83.0, 28.0], 32644, "nepal/nepal_traces.shp"),
    "pakistan": ([62.0, 25.0, 63.0, 26.0], 32641, "pakistan/pakistan_traces.shp"),
    "canada": ([-117.0, 52.0, -116.0, 53.0], 32611, "canada/canada_traces.shp"),
}
RES = 30.0
BANDS = ("profile_curvature", "curvature", "slope")
TOL_PX = 1
SIGMA_Z = 2.0
COND = 1e-2
MIN_PTS = 8


def _dem(aoi, epsg, tmp):
    d4326 = os.path.join(tmp, "d.tif")
    fetch_dem(aoi, d4326)
    dutm = os.path.join(tmp, "dutm.tif")
    gdal.Warp(dutm, d4326, dstSRS=f"EPSG:{epsg}", xRes=RES, yRes=RES, resampleAlg="bilinear")
    ds = gdal.Open(dutm)
    dem = ds.ReadAsArray().astype(float)
    gt = ds.GetGeoTransform()
    nd = ds.GetRasterBand(1).GetNoDataValue()
    ds = None
    if nd is not None:
        dem = np.where(dem == nd, np.nan, dem)
    return dem, gt, dutm


def _truth_mask(traces_rel, dem_path, epsg):
    ref = gdal.Open(dem_path)
    gt, proj, w, h = ref.GetGeoTransform(), ref.GetProjection(), ref.RasterXSize, ref.RasterYSize
    ref = None
    tmp = tempfile.mkdtemp()
    reproj = os.path.join(tmp, "t.gpkg")
    gdal.VectorTranslate(reproj, os.path.join(REPO, "data", "raw", traces_rel),
                         options=gdal.VectorTranslateOptions(dstSRS=f"EPSG:{epsg}", reproject=True))
    mem = gdal.GetDriverByName("MEM").Create("", w, h, 1, gdal.GDT_Byte)
    mem.SetGeoTransform(gt)
    mem.SetProjection(proj)
    src = ogr.Open(reproj)
    gdal.RasterizeLayer(mem, [1], src.GetLayer(), burn_values=[1], options=["ALL_TOUCHED=TRUE"])
    mask = mem.ReadAsArray().astype(bool)
    mem = None
    src = None
    return mask


def run(region):
    aoi, epsg, traces_rel = REGIONS[region]
    tmp = tempfile.mkdtemp()
    log.info("[%s] fetching DEM ...", region)
    dem, gt, dem_path = _dem(aoi, epsg, tmp)
    valid = np.isfinite(dem)
    _, terr = build_terrain_stack(dem, RES, names=BANDS)
    response = np.zeros(dem.shape)
    for b in BANDS:
        response += normalize01(terr[b])
    response /= len(BANDS)
    response = np.where(valid, response, np.nan)

    edges = canny(response, sigma=1.0, valid_mask=valid)
    truth = _truth_mask(traces_rel, dem_path, epsg) & valid

    reached = binary_dilation(edges, structure=disk(TOL_PX))
    recall = float((truth & reached).sum()) / max(int(truth.sum()), 1)
    budget = float(edges.sum()) / max(int(valid.sum()), 1)
    lin = linearity_metrics(edges)["linearity"]

    traces = polylines_from_mask(edges, min_length=MIN_PTS, simplify_tol=1.0, transform=gt)
    reliable = []
    for tr in traces:
        pts = sample_trace(tr, dem, gt, spacing=RES)
        if len(pts) < MIN_PTS:
            continue
        att = fit_plane(pts, sigma_z=SIGMA_Z)
        if np.isfinite(att.dip) and att.conditioning >= COND:
            reliable.append(att)

    dips = np.array([a.dip for a in reliable]) if reliable else np.array([np.nan])
    strikes = np.array([a.strike % 180 for a in reliable]) if reliable else np.array([np.nan])
    if reliable:
        ang = np.radians(2 * strikes)
        dom = (np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) / 2) % 180
    else:
        dom = float("nan")

    print(f"\n=== {region}: detector eval (Canny on {'+'.join(BANDS)}) ===")
    print(f"detected edge budget : {budget*100:.1f}% of valid pixels")
    print(f"recall of drawn traces: {recall:.3f}  (@tol {TOL_PX}px)")
    print(f"linearity (label-free): {lin:.3f}")
    print(f"candidate traces      : {len(traces)}")
    print(f"reliable attitudes    : {len(reliable)}  (conditioning >= {COND:.0e})")
    print(f"dip median            : {np.median(dips):.1f}  IQR "
          f"[{np.percentile(dips,25):.1f}, {np.percentile(dips,75):.1f}]")
    print(f"near-vertical (>=85)  : {100*np.mean(dips>=85):.0f}%")
    print(f"dominant strike       : {dom:.0f} deg")
    # map_conditioning calibration: how an added map-view gate trims the
    # straight-map-trace (near-vertical) artifacts (planesight-2je).
    if reliable:
        mc = np.array([a.map_conditioning for a in reliable])
        print("map-view gate effect (on top of conditioning>=1e-2):")
        for thr in (0.0, 1e-4, 1e-3, 1e-2):
            keep = mc >= thr
            nv = 100 * np.mean(dips[keep] >= 85) if keep.any() else float("nan")
            print(f"  map_cond>={thr:.0e}: kept {int(keep.sum()):5d}/{len(reliable)}  "
                  f"near-vertical {nv:.0f}%  dip median {np.median(dips[keep]):.1f}")


def main():
    import sys
    for r in (sys.argv[1:] or list(REGIONS)):
        if r not in REGIONS:
            raise SystemExit(f"unknown region {r}; choose {list(REGIONS)}")
        run(r)


if __name__ == "__main__":
    main()
