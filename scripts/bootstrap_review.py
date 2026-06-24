"""Fairer bootstrap probe + visual review panels (planesight-9vt revisit).

Re-tests map-draping label quality without the original probe's confounds (it used
thresholded DEM-only edges and treated noisy hand-traces as ground truth), and -
the point of this script - exports side-by-side PNG panels so a human can judge
alignment by eye. Edges are now MULTI-MODAL (DEM curvature/slope + Sentinel-2
iron-oxide/SWIR), and scoring is positive-unlabeled: the hand-traces are a small
subset, so far more detections than annotations is expected; we only check the
detections are accurate and align with the markups where they exist.

Panels (4 windows centred on the densest hand-traced areas), each row:
  [ hillshade | S2 false-colour | multi-modal edge strength | overlay ]
overlay colours: green = system agrees with your trace, red = your trace the system
missed, cyan = system-only (expected extra coverage).

  MAMBA_ROOT_PREFIX=$HOME/micromamba PYTHONPATH=$PWD \
    $HOME/bin/micromamba run -n gdal python scripts/bootstrap_review.py nepal
"""

from __future__ import annotations

import logging
import os
import tempfile

import numpy as np
from osgeo import gdal, ogr
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, distance_transform_edt, uniform_filter

from planesight.core.attitude.sample import densify_line
from planesight.core.data import (
    align_to_grid,
    asset_href,
    clearest_months,
    fetch_dem,
    item_month,
    search_clear_sentinel2,
)
from planesight.core.derivatives import build_terrain_stack, normalize01
from planesight.core.derivatives import spectral as sp
from planesight.core.derivatives import terrain as tr
from planesight.core.detect import ClassicalTraceDetector, canny, disk
from planesight.core.detect.drainage import (
    channel_proximity,
    flow_accumulation,
    flow_azimuth,
    is_drainage,
)

gdal.UseExceptions()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("bootstrap_review")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGIONS = {
    "nepal": ([82.0, 27.6, 83.0, 28.0], 32644, "nepal/nepal_traces.shp"),
    "pakistan": ([62.0, 25.0, 63.0, 26.0], 32641, "pakistan/pakistan_traces.shp"),
    "canada": ([-117.0, 52.0, -116.0, 53.0], 32611, "canada/canada_traces.shp"),
}
RES = 30.0
WIN_M = 2000           # review window size (m)
N_WINDOWS = 4
# drainage pre-filter (the chosen knee from scripts/drainage_test.py)
DRAIN_BANDS = ("profile_curvature", "curvature", "slope")
DRAIN_DS = 3           # flow downsample
DRAIN_ACCUM = 8        # channel accumulation threshold (downsampled cells)
DRAIN_ALIGNED = 0.5    # >=50% along-flow length -> drainage
UPSCALE = 5            # render scale for legibility
S2_BANDS = ("blue", "red", "nir", "swir16", "swir22")
TOL_BANDS = (1, 2, 3)


def fetch(region, tmp):
    aoi, epsg, traces_rel = REGIONS[region]
    d4326 = os.path.join(tmp, "d.tif")
    log.info("[%s] fetching GLO-30 ...", region)
    fetch_dem(aoi, d4326)
    dem_path = os.path.join(tmp, "dem.tif")
    gdal.Warp(dem_path, d4326, dstSRS=f"EPSG:{epsg}", xRes=RES, yRes=RES, resampleAlg="bilinear")
    ds = gdal.Open(dem_path)
    dem = ds.ReadAsArray().astype(float)
    nd = ds.GetRasterBand(1).GetNoDataValue()
    ds = None
    if nd is not None:
        dem = np.where(dem == nd, np.nan, dem)

    bands = fetch_s2(aoi, dem_path, tmp)
    hand = rasterize(traces_rel, dem_path, epsg)
    return dem, dem_path, bands, hand


def fetch_s2(aoi, dem_path, tmp):
    gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    try:
        items, _ = search_clear_sentinel2(aoi, cloud_max=5.0)
        if not items:
            return {}
        months = set(clearest_months(items))
        cand = [it for it in items if item_month(it) in months] or items
        by_date = {}
        for it in cand:
            by_date.setdefault(it["properties"]["datetime"][:10], []).append(it)

        def cover(scs):
            n = 80
            xs = np.linspace(aoi[0], aoi[2], n)
            ys = np.linspace(aoi[1], aoi[3], n)
            gx, gy = np.meshgrid(xs, ys)
            cov = np.zeros((n, n), bool)
            for it in scs:
                b = it["bbox"]
                cov |= (gx >= b[0]) & (gx <= b[2]) & (gy >= b[1]) & (gy <= b[3])
            return cov.mean()

        best = max(by_date, key=lambda d: cover(by_date[d]))
        scenes = by_date[best]
        log.info("S2 date %s (%d tiles, %.0f%% cover)", best, len(scenes), 100 * cover(scenes))
        out = {}
        for key in S2_BANDS:
            hrefs = ["/vsicurl/" + asset_href(s, key) for s in scenes]
            o = os.path.join(tmp, f"s2_{key}.tif")
            align_to_grid(hrefs, o, dem_path, resampling="bilinear", src_nodata=0)
            ds = gdal.Open(o)
            arr = ds.ReadAsArray().astype(float)
            ndv = ds.GetRasterBand(1).GetNoDataValue()
            ds = None
            out[key] = np.where(arr == ndv, np.nan, arr) if ndv is not None else arr
        return out
    except Exception as exc:
        log.warning("S2 fetch failed (%s); DEM-only panels.", exc)
        return {}


def rasterize(traces_rel, dem_path, epsg):
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


def build(dem, bands):
    """Hillshade, S2 false-colour RGB, multi-modal edge strength, detected edges."""
    valid = np.isfinite(dem)
    hill = tr.multi_hillshade(dem, RES)
    dem_resp = 0.5 * (normalize01(tr.curvature(dem, RES, kind="total"))
                      + normalize01(tr.slope(dem, RES)))
    parts = [dem_resp]
    if bands:
        s2_resp = 0.5 * (normalize01(sp.iron_oxide_index(bands["red"], bands["blue"]))
                         + normalize01(sp.percentile_stretch(bands["swir16"])))
        parts.append(s2_resp)
        rgb = np.dstack([sp.percentile_stretch(bands["swir22"]),
                         sp.percentile_stretch(bands["swir16"]),
                         sp.percentile_stretch(bands["blue"])])
    else:
        rgb = np.dstack([hill, hill, hill])
    resp = np.where(valid, np.nanmean(parts, axis=0), np.nan)
    fx, fy = tr.gradients(np.nan_to_num(resp, nan=float(np.nanmedian(resp))), RES)
    edge_strength = normalize01(np.hypot(fx, fy))
    detected = canny(resp, sigma=1.0, valid_mask=valid)
    return hill, rgb, edge_strength, detected, valid


def classify_detections(dem, valid):
    """Run the detector and split its traces into kept (geology) vs drainage-flagged
    pixel masks, using the flow-accumulation channel network."""
    _, terr = build_terrain_stack(dem, RES, names=DRAIN_BANDS)
    polys = ClassicalTraceDetector(min_length=8).detect(
        np.stack([terr[b] for b in DRAIN_BANDS]))  # (col, row)
    sub = dem[::DRAIN_DS, ::DRAIN_DS]
    acc, az = flow_accumulation(sub), flow_azimuth(sub)
    h, w = dem.shape
    ri = np.minimum(np.arange(h) // DRAIN_DS, sub.shape[0] - 1)
    ci = np.minimum(np.arange(w) // DRAIN_DS, sub.shape[1] - 1)
    channel = (acc >= DRAIN_ACCUM)[np.ix_(ri, ci)] & valid
    buf, near = channel_proximity(channel, az[np.ix_(ri, ci)], buffer_px=2)
    keep = np.zeros(dem.shape, dtype=bool)
    drain = np.zeros(dem.shape, dtype=bool)
    n_drain = 0
    for p in polys:
        d = densify_line(p, spacing=1.0)
        cols = np.clip(np.round(d[:, 0]).astype(int), 0, w - 1)
        rows = np.clip(np.round(d[:, 1]).astype(int), 0, h - 1)
        if is_drainage(p, buf, near, min_aligned_fraction=DRAIN_ALIGNED):
            drain[rows, cols] = True
            n_drain += 1
        else:
            keep[rows, cols] = True
    log.info("[detections] %d traces, %d drainage-flagged", len(polys), n_drain)
    return keep, drain


def stats(region, hand, detected, valid):
    truth = hand & valid
    n_hand, n_det = int(truth.sum()), int(detected.sum())
    dist = distance_transform_edt(~detected)
    off = dist[truth]
    within = {k: float(np.mean(off <= k)) for k in TOL_BANDS}
    concealed = float(np.mean(off > 3))
    print(f"\n=== {region}: fairer multi-modal probe (positive-unlabeled) ===")
    print(f"hand-trace label px : {n_hand}  | system edge px: {n_det} "
          f"({n_det/max(n_hand,1):.1f}x - more is expected, you labeled a subset)")
    print("your labels with a detected edge within: "
          + "  ".join(f"{k}px {within[k]*100:.0f}%" for k in TOL_BANDS))
    print(f"  (2px is the fair operating point: you trace ridge/contact CRESTS, while")
    print(f"   gradient edges fire on the FLANKS ~1-2px off - a geometry offset, not")
    print(f"   a registration error. {within[2]*100:.0f}% of your labels sit within 2px of signal.)")
    print(f"concealed (> 3px from any DEM+S2 edge): {concealed*100:.0f}% "
          f"(candidate no-signal labels - judge visually whether real or tracing slips)")


def windows(hand, shape):
    win = int(round(WIN_M / RES))
    dens = uniform_filter(hand.astype(float), win)
    d = dens.copy()
    picks = []
    for _ in range(N_WINDOWS):
        r, c = np.unravel_index(int(np.argmax(d)), d.shape)
        if d[r, c] <= 0:
            break
        picks.append((r, c))
        d[max(0, r - win):r + win, max(0, c - win):c + win] = 0.0
    return win, picks


def _gray_rgb(a01):
    g = (np.clip(np.nan_to_num(a01), 0, 1) * 255).astype(np.uint8)
    return np.dstack([g, g, g])


def _rgb8(rgb01):
    return (np.clip(np.nan_to_num(rgb01), 0, 1) * 255).astype(np.uint8)


def panel(region, out_dir, win, idx, center, hill, rgb, edge, hand, detected, det_drain):
    r, c = center
    half = win // 2
    r0, c0 = max(0, r - half), max(0, c - half)
    r1, c1 = r0 + win, c0 + win
    sl = (slice(r0, r1), slice(c0, c1))

    base = _gray_rgb(hill[sl])
    # keep the original dense detection display (the multi-modal edge map); just split
    # it into drainage vs kept by proximity to the drainage-flagged traces.
    hand_w, det_w = hand[sl], detected[sl]
    drain_near = binary_dilation(det_drain[sl], structure=disk(2))
    sys_drain = det_w & drain_near
    sys_keep = det_w & ~drain_near
    # match within a 2px tolerance (absorbs the crest-vs-flank geometry: a geologist
    # traces the ridge/contact crest, while gradient edges fire on the flanks ~1-2px off).
    keep_near = binary_dilation(sys_keep, structure=disk(2))
    hand_near = binary_dilation(hand_w, structure=disk(2))
    overlay = base.copy()
    overlay[sys_drain] = [255, 150, 30]              # ORANGE = excluded drainage
    overlay[sys_keep & ~hand_near] = [45, 110, 150]  # dim cyan = system-only (kept)
    overlay[hand_w & ~keep_near] = [255, 60, 60]      # red  = your trace, missed
    overlay[hand_w & keep_near] = [60, 255, 80]       # green = agreement (within 2px)

    tiles = [("hillshade (raw)", base),
             ("S2 false-colour", _rgb8(rgb[sl])),
             ("edge strength (DEM+S2)", _gray_rgb(edge[sl])),
             ("overlay: you=red/green  kept=cyan  drainage=orange", overlay)]

    imgs = []
    for label, arr in tiles:
        im = Image.fromarray(arr).resize((win * UPSCALE, win * UPSCALE), Image.NEAREST)
        canvas = Image.new("RGB", (im.width, im.height + 18), (20, 20, 20))
        canvas.paste(im, (0, 18))
        ImageDraw.Draw(canvas).text((3, 4), label, fill=(255, 255, 255))
        imgs.append(canvas)

    gap = 6
    total_w = sum(i.width for i in imgs) + gap * (len(imgs) - 1)
    strip = Image.new("RGB", (total_w, imgs[0].height), (20, 20, 20))
    x = 0
    for im in imgs:
        strip.paste(im, (x, 0))
        x += im.width + gap
    path = os.path.join(out_dir, f"{region}_window{idx}.png")
    strip.save(path)
    return path


def main():
    import sys
    region = (sys.argv[1:] or ["nepal"])[0]
    if region not in REGIONS:
        raise SystemExit(f"unknown region {region}; choose {list(REGIONS)}")
    out_dir = os.path.join(REPO, "debug", "bootstrap_review", region)
    os.makedirs(out_dir, exist_ok=True)
    tmp = tempfile.mkdtemp()

    dem, dem_path, bands, hand = fetch(region, tmp)
    log.info("[%s] grid %dx%d, hand-trace px %d",
             region, dem.shape[1], dem.shape[0], int(hand.sum()))
    hill, rgb, edge, detected, valid = build(dem, bands)
    stats(region, hand, detected, valid)
    _, det_drain = classify_detections(dem, valid)

    win, picks = windows(hand & valid, dem.shape)
    print(f"\n{len(picks)} review windows ({WIN_M:.0f} m) over your densest markups:")
    for i, ctr in enumerate(picks, 1):
        p = panel(region, out_dir, win, i, ctr, hill, rgb, edge, hand, detected, det_drain)
        print(f"  window {i} @ row,col {ctr}: {p}")
    print(f"\nPanels in {out_dir}")


if __name__ == "__main__":
    main()
