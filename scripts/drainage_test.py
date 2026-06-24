"""Drainage pre-filter test (planesight-2je follow-up / S9.1).

Decisive experiment for the flow-accumulation drainage filter: run the detector on
Nepal, flag traces that run ALONG the drainage network, and measure the trade-off -
(a) how much detected lineament length the filter removes vs (b) how many of the
geologist's HAND-traced contacts it wrongly flags (the false-negative cost, using
the hand-traces as the yardstick). Plus visual panels showing kept (green) vs
drainage-flagged (red) traces over the hillshade + channel network.

  MAMBA_ROOT_PREFIX=$HOME/micromamba PYTHONPATH=$PWD \
    $HOME/bin/micromamba run -n gdal python scripts/drainage_test.py nepal
"""

from __future__ import annotations

import logging
import os
import tempfile

import numpy as np
from osgeo import gdal, ogr
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, uniform_filter

from planesight.core.attitude.sample import densify_line
from planesight.core.data import fetch_dem
from planesight.core.derivatives import build_terrain_stack
from planesight.core.derivatives import terrain as tr
from planesight.core.detect import ClassicalTraceDetector
from planesight.core.detect.drainage import (
    channel_proximity,
    flow_accumulation,
    flow_azimuth,
    is_drainage,
)

gdal.UseExceptions()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("drainage_test")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGIONS = {
    "nepal": ([82.0, 27.6, 83.0, 28.0], 32644, "nepal/nepal_traces.shp"),
    "pakistan": ([62.0, 25.0, 63.0, 26.0], 32641, "pakistan/pakistan_traces.shp"),
    "canada": ([-117.0, 52.0, -116.0, 53.0], 32611, "canada/canada_traces.shp"),
}
RES = 30.0
BANDS = ("profile_curvature", "curvature", "slope")
DOWNSAMPLE = 3                 # flow at ~90 m
MIN_ACCUM = 60                 # channel threshold (downsampled cells)
BUFFER_PX = 2
MIN_ALIGNED = 0.5              # >=50% along-flow length -> drainage
THRESHOLDS = (120, 60, 30, 15, 8, 4)   # accumulation sweep (downsampled cells)
MAX_FALSE_NEG = 0.05           # pick the densest channel net under this hand-FN cost
WIN_M, N_WIN, UPSCALE = 2000, 4, 5


def fetch(region, tmp):
    aoi, epsg, traces_rel = REGIONS[region]
    d4326 = os.path.join(tmp, "d.tif")
    log.info("[%s] fetching GLO-30 ...", region)
    fetch_dem(aoi, d4326)
    dem_path = os.path.join(tmp, "dem.tif")
    gdal.Warp(dem_path, d4326, dstSRS=f"EPSG:{epsg}", xRes=RES, yRes=RES, resampleAlg="bilinear")
    ds = gdal.Open(dem_path)
    dem = ds.ReadAsArray().astype(float)
    gt = ds.GetGeoTransform()
    nd = ds.GetRasterBand(1).GetNoDataValue()
    ds = None
    if nd is not None:
        dem = np.where(dem == nd, np.nan, dem)
    return dem, gt, dem_path, epsg, traces_rel


def hand_polylines(traces_rel, dem_path, gt, epsg):
    """Load hand-traced linework as lists of (col,row) pixel vertices."""
    tmp = tempfile.mkdtemp()
    reproj = os.path.join(tmp, "t.gpkg")
    gdal.VectorTranslate(reproj, os.path.join(REPO, "data", "raw", traces_rel),
                         options=gdal.VectorTranslateOptions(dstSRS=f"EPSG:{epsg}", reproject=True))
    src = ogr.Open(reproj)
    layer = src.GetLayer()
    out = []
    for feat in layer:
        geom = feat.GetGeometryRef()
        if geom is None:
            continue
        parts = [geom.GetGeometryRef(i) for i in range(geom.GetGeometryCount())] \
            if geom.GetGeometryCount() > 0 else [geom]
        for part in parts:
            pts = np.array([(part.GetX(k), part.GetY(k)) for k in range(part.GetPointCount())])
            if len(pts) < 2:
                continue
            col = (pts[:, 0] - gt[0]) / gt[1]
            row = (pts[:, 1] - gt[3]) / gt[5]
            out.append(np.column_stack([col, row]))
    src = None
    return out


def length(poly):
    d = densify_line(poly, spacing=1.0)
    return float(np.sum(np.hypot(*np.diff(d, axis=0).T))) if len(d) > 1 else 0.0


def classify(polys, buf, near_az):
    flagged = [is_drainage(p, buf, near_az, min_aligned_fraction=MIN_ALIGNED) for p in polys]
    total = sum(length(p) for p in polys)
    drain = sum(length(p) for p, f in zip(polys, flagged) if f)
    return flagged, total, drain


def mark(polys, flags, shape):
    keep = np.zeros(shape, bool)
    drain = np.zeros(shape, bool)
    for p, f in zip(polys, flags):
        d = densify_line(p, spacing=1.0)
        cols = np.clip(np.round(d[:, 0]).astype(int), 0, shape[1] - 1)
        rows = np.clip(np.round(d[:, 1]).astype(int), 0, shape[0] - 1)
        (drain if f else keep)[rows, cols] = True
    return keep, drain


def windows(hand_mask, shape):
    win = int(round(WIN_M / RES))
    dens = uniform_filter(hand_mask.astype(float), win)
    d = dens.copy()
    picks = []
    for _ in range(N_WIN):
        r, c = np.unravel_index(int(np.argmax(d)), d.shape)
        if d[r, c] <= 0:
            break
        picks.append((r, c))
        d[max(0, r - win):r + win, max(0, c - win):c + win] = 0.0
    return win, picks


def _tile(base_gray, overlays, label):
    """base_gray: (h,w) uint8; overlays: list of (mask, rgb). Returns a labeled PIL tile."""
    img = np.dstack([base_gray, base_gray, base_gray])
    for mask, rgb in overlays:
        img[binary_dilation(mask, iterations=1)] = rgb
    im = Image.fromarray(img).resize((img.shape[1] * UPSCALE, img.shape[0] * UPSCALE),
                                     Image.NEAREST)
    canvas = Image.new("RGB", (im.width, im.height + 18), (20, 20, 20))
    canvas.paste(im, (0, 18))
    ImageDraw.Draw(canvas).text((3, 4), label, fill=(255, 255, 255))
    return canvas


def panel(out_dir, region, win, idx, ctr, hill, channel, keep, drain):
    r, c = ctr
    half = win // 2
    sl = (slice(max(0, r - half), max(0, r - half) + win),
          slice(max(0, c - half), max(0, c - half) + win))
    g = (np.clip(np.nan_to_num(hill[sl]), 0, 1) * 255).astype(np.uint8)
    cyan, blue, green, red = [60, 200, 255], [70, 120, 255], [60, 255, 80], [255, 60, 60]
    tiles = [
        _tile(g, [], "1. hillshade (raw)"),
        _tile(g, [(keep[sl] | drain[sl], cyan)], "2. all detections"),
        _tile(g, [(channel[sl], blue)], "3. drainage network (flow)"),
        _tile(g, [(keep[sl], green), (drain[sl], red)],
              "4. result: green=kept  red=drainage"),
    ]
    gap = 6
    total_w = sum(t.width for t in tiles) + gap * (len(tiles) - 1)
    strip = Image.new("RGB", (total_w, tiles[0].height), (20, 20, 20))
    x = 0
    for t in tiles:
        strip.paste(t, (x, 0))
        x += t.width + gap
    path = os.path.join(out_dir, f"{region}_drainage_window{idx}.png")
    strip.save(path)
    return path


def main():
    import sys
    region = (sys.argv[1:] or ["nepal"])[0]
    out_dir = os.path.join(REPO, "debug", "drainage_test", region)
    os.makedirs(out_dir, exist_ok=True)
    tmp = tempfile.mkdtemp()

    dem, gt, dem_path, epsg, traces_rel = fetch(region, tmp)
    valid = np.isfinite(dem)
    _, terr = build_terrain_stack(dem, RES, names=BANDS)
    stack = np.stack([terr[b] for b in BANDS])
    log.info("[%s] detecting ...", region)
    detected = ClassicalTraceDetector(min_length=8).detect(stack)  # pixel (col,row)
    hand = hand_polylines(traces_rel, dem_path, gt, epsg)

    # flow accumulation computed ONCE (downsampled); only the channel threshold sweeps
    log.info("[%s] flow accumulation (downsample %dx) ...", region, DOWNSAMPLE)
    sub = dem[::DOWNSAMPLE, ::DOWNSAMPLE]
    acc = flow_accumulation(sub)
    az = flow_azimuth(sub)
    h, w = dem.shape
    ri = np.minimum(np.arange(h) // DOWNSAMPLE, sub.shape[0] - 1)
    ci = np.minimum(np.arange(w) // DOWNSAMPLE, sub.shape[1] - 1)

    def up(a2):
        return a2[np.ix_(ri, ci)]

    az_full = up(az)
    print(f"\n=== {region}: drainage pre-filter sweep "
          f"(buffer {BUFFER_PX}px, >= {int(MIN_ALIGNED*100)}% along-flow) ===")
    print("accum  channel%   detected drainage-removed   hand-trace FALSE-NEG")
    print("-" * 64)
    chosen = None
    for thr in THRESHOLDS:
        mask = up(acc >= thr) & valid
        buf, near = channel_proximity(mask, az_full, buffer_px=BUFFER_PX)
        df, dtot, ddr = classify(detected, buf, near)
        hf, htot, hdr = classify(hand, buf, near)
        cov = 100 * mask.sum() / valid.sum()
        det_pct, hand_pct = 100 * ddr / max(dtot, 1), 100 * hdr / max(htot, 1)
        print(f"{thr:5d}  {cov:6.1f}%   {det_pct:5.0f}% ({sum(df)} traces)        "
              f"   {hand_pct:5.1f}% ({sum(hf)}/{len(hand)} traces)")
        if hdr / max(htot, 1) <= MAX_FALSE_NEG:
            chosen = (thr, mask, df)
    if chosen is None:
        chosen = (THRESHOLDS[0], up(acc >= THRESHOLDS[0]) & valid, None)
    thr, channel, det_flags = chosen
    if det_flags is None:
        buf, near = channel_proximity(channel, az_full, buffer_px=BUFFER_PX)
        det_flags, _, _ = classify(detected, buf, near)
    print(f"\nchosen threshold {thr} (densest channel net with hand-FN <= "
          f"{int(MAX_FALSE_NEG*100)}%); rendering panels.")

    hill = tr.multi_hillshade(dem, RES)
    keep_m, drain_m = mark(detected, det_flags, dem.shape)
    hand_mask = np.zeros(dem.shape, bool)
    for p in hand:
        d = densify_line(p, spacing=1.0)
        hand_mask[np.clip(np.round(d[:, 1]).astype(int), 0, dem.shape[0] - 1),
                  np.clip(np.round(d[:, 0]).astype(int), 0, dem.shape[1] - 1)] = True
    win, picks = windows(hand_mask & valid, dem.shape)
    for i, ctr in enumerate(picks, 1):
        print("  panel:", panel(out_dir, region, win, i, ctr, hill, channel, keep_m, drain_m))
    print(f"\nPanels in {out_dir}")


if __name__ == "__main__":
    main()
