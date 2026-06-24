"""Training-bootstrap probe (planesight-9vt): can draped map linework be labels?

The v2 ML story rests on draping published vector geology maps over the DEM to
auto-generate (input stack, trace mask) training pairs at scale (ARCHITECTURE.md
S8.3). The central risks are REGISTRATION (a label offset from the true feature
teaches blurry/wrong boundaries) and CONCEALED contacts (labels with no surface
signal teach noise). This probe quantifies both using our hand-drawn traces as a
BEST-CASE stand-in for map linework: for every drawn-trace pixel, measure the
distance to the nearest detected DEM edge.

  - small offsets + few concealed  -> draping is viable (with filtering)
  - large offsets / many concealed  -> draping poisons labels; fall back to the
    hand-labeled seed + active learning

Published 1:100k maps are positioned to ~50-100 m ~ 2-3 GLO-30 px WORSE than our
hand-drawn traces, so these numbers are an optimistic lower bound on map-draping.

  MAMBA_ROOT_PREFIX=$HOME/micromamba PYTHONPATH=$PWD \
    $HOME/bin/micromamba run -n gdal python scripts/bootstrap_probe.py <region>
"""

from __future__ import annotations

import logging
import os
import tempfile

import numpy as np
from osgeo import gdal, ogr
from scipy.ndimage import distance_transform_edt

from planesight.core.data import fetch_dem
from planesight.core.derivatives import build_terrain_stack, normalize01
from planesight.core.detect import canny

gdal.UseExceptions()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("bootstrap_probe")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGIONS = {
    "nepal": ([82.0, 27.6, 83.0, 28.0], 32644, "nepal/nepal_traces.shp"),
    "pakistan": ([62.0, 25.0, 63.0, 26.0], 32641, "pakistan/pakistan_traces.shp"),
    "canada": ([-117.0, 52.0, -116.0, 53.0], 32611, "canada/canada_traces.shp"),
}
RES = 30.0
BANDS = ("profile_curvature", "curvature", "slope")
CONCEALED_PX = 5  # a label > this far from any detected edge has no usable signal


def _dem(aoi, epsg, tmp):
    d4326 = os.path.join(tmp, "d.tif")
    fetch_dem(aoi, d4326)
    dutm = os.path.join(tmp, "dutm.tif")
    gdal.Warp(dutm, d4326, dstSRS=f"EPSG:{epsg}", xRes=RES, yRes=RES, resampleAlg="bilinear")
    ds = gdal.Open(dutm)
    dem = ds.ReadAsArray().astype(float)
    nd = ds.GetRasterBand(1).GetNoDataValue()
    ds = None
    if nd is not None:
        dem = np.where(dem == nd, np.nan, dem)
    return dem, dutm


def _truth(traces_rel, dem_path, epsg):
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
    dem, dem_path = _dem(aoi, epsg, tmp)
    valid = np.isfinite(dem)
    _, terr = build_terrain_stack(dem, RES, names=BANDS)
    resp = np.zeros(dem.shape)
    for b in BANDS:
        resp += normalize01(terr[b])
    resp = np.where(valid, resp / len(BANDS), np.nan)
    edges = canny(resp, sigma=1.0, valid_mask=valid)
    truth = _truth(traces_rel, dem_path, epsg) & valid

    # distance (px) from every pixel to the nearest detected edge; sample at labels
    dist = distance_transform_edt(~edges)
    off = dist[truth]
    if off.size == 0:
        log.warning("[%s] no truth pixels", region)
        return
    within = {k: float(np.mean(off <= k)) for k in (0, 1, 2, 3, 5)}
    concealed = float(np.mean(off > CONCEALED_PX))

    print(f"\n=== {region}: label-signal registration probe (n={off.size} label px) ===")
    print(f"offset to nearest DEM edge (px @ {RES:.0f} m):")
    print(f"  median {np.median(off):.1f}  p75 {np.percentile(off,75):.1f}  "
          f"p90 {np.percentile(off,90):.1f}")
    print(f"label pixels within: 0px {within[0]*100:.0f}%  1px {within[1]*100:.0f}%  "
          f"2px {within[2]*100:.0f}%  3px {within[3]*100:.0f}%  5px {within[5]*100:.0f}%")
    print(f"concealed (> {CONCEALED_PX}px from any edge): {concealed*100:.0f}%  "
          f"<- labels with no usable DEM signal")


def main():
    import sys
    for r in (sys.argv[1:] or list(REGIONS)):
        if r not in REGIONS:
            raise SystemExit(f"unknown region {r}; choose {list(REGIONS)}")
        run(r)


if __name__ == "__main__":
    main()
