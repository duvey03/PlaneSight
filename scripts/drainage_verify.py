"""Drainage filter VERIFICATION (planesight-5p3) - the decisive check.

The drainage_test.py headline was oversold (critical review): the "1.3% false-neg"
diluted the few at-risk contacts among 408 mostly-cross-cutting hand-traces, and the
~31% removed set was never confirmed to actually be drainage. This script settles
those three questions honestly:

  1. AUDIT SHEET - render ~30 RANDOM drainage-flagged traces, each as a tile over the
     hillshade and over Sentinel-2 false colour (channel network in blue, flagged
     trace in orange), so the geologist can judge each one: creek or contact? This is
     the test the removed set never got.
  2. CONDITIONED FALSE-NEGATIVE - among hand-traces that actually OVERLAP the channel
     network (the only ones the filter can possibly mis-flag), what fraction get
     flagged? Reported with the small-N caveat, not diluted across all 408.
  3. PARAMETER SENSITIVITY - sweep angle_tol x min_aligned and report removed-% and
     conditioned-FN as a RANGE, not a single fragile 31%.

  MAMBA_ROOT_PREFIX=$HOME/micromamba PYTHONPATH=$PWD \
    $HOME/bin/micromamba run -n gdal python scripts/drainage_verify.py nepal
"""

from __future__ import annotations

import logging
import os
import tempfile

import numpy as np
from osgeo import gdal, ogr
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation

from planesight.core.attitude.sample import densify_line
from planesight.core.data import (
    align_to_grid,
    asset_href,
    clearest_months,
    fetch_dem,
    item_month,
    search_clear_sentinel2,
)
from planesight.core.derivatives import build_terrain_stack
from planesight.core.derivatives import spectral as sp
from planesight.core.derivatives import terrain as tr
from planesight.core.detect import ClassicalTraceDetector
from planesight.core.detect.drainage import (
    channel_proximity,
    flow_network,
    trace_drainage_fraction,
)

gdal.UseExceptions()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("drainage_verify")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGIONS = {
    "nepal": ([82.0, 27.6, 83.0, 28.0], 32644, "nepal/nepal_traces.shp"),
    "pakistan": ([62.0, 25.0, 63.0, 26.0], 32641, "pakistan/pakistan_traces.shp"),
    "canada": ([-117.0, 52.0, -116.0, 53.0], 32611, "canada/canada_traces.shp"),
}
RES = 30.0
BANDS = ("profile_curvature", "curvature", "slope")
DRAIN_DS = 3            # flow downsample (~90 m)
# channel accumulation threshold (downsampled cells). Recalibrated 8->15 for the
# hardened depression-filled network (j8t): filling raises accumulation, so the old
# 8 flagged ~22% of the map as channel; 15 restores a ~15% network at the same ~31%
# removal the geologist audited, on the flat part of the false-negative curve.
DRAIN_ACCUM = 15
S2_BANDS = ("blue", "red", "nir", "swir16", "swir22")

# the operating point being verified
ANGLE_TOL = 30.0
MIN_ALIGNED = 0.5
# at-risk hand-trace = at least this fraction sits on the channel buffer
OVERLAP_AT_RISK = 0.5

# audit sheet layout
N_AUDIT = 30
GRID_COLS = 6
TILE = 150              # px per tile after resize
MIN_HALF_PX = 18        # min half-window around a trace (~540 m)
MARGIN_PX = 6
SEED = 20260624        # deterministic random sample (reproducible audit)

# sensitivity sweep
SWEEP_ANGLE = (20.0, 25.0, 30.0, 35.0, 45.0)
SWEEP_ALIGNED = (0.4, 0.5, 0.6)


def fetch(region, tmp):
    aoi, epsg, traces_rel = REGIONS[region]
    d4326 = os.path.join(tmp, "d.tif")
    log.info("[%s] fetching GLO-30 ...", region)
    fetch_dem(aoi, d4326)
    dem_path = os.path.join(tmp, "dem.tif")
    gdal.Warp(dem_path, d4326, dstSRS=f"EPSG:{epsg}", xRes=RES, yRes=RES,
              resampleAlg="bilinear")
    ds = gdal.Open(dem_path)
    dem = ds.ReadAsArray().astype(float)
    gt = ds.GetGeoTransform()
    nd = ds.GetRasterBand(1).GetNoDataValue()
    ds = None
    if nd is not None:
        dem = np.where(dem == nd, np.nan, dem)
    bands = fetch_s2(aoi, dem_path, tmp)
    return dem, gt, dem_path, epsg, traces_rel, bands


def fetch_s2(aoi, dem_path, tmp):
    """Coverage-aware same-date S2 mosaic aligned to the DEM grid (or {} on failure)."""
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
        log.info("S2 date %s (%d tiles, %.0f%% cover)", best, len(scenes),
                 100 * cover(scenes))
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
        log.warning("S2 fetch failed (%s); hillshade-only audit sheet.", exc)
        return {}


def hand_polylines(traces_rel, gt, epsg):
    """Hand-traced linework as lists of (col,row) pixel vertices."""
    tmp = tempfile.mkdtemp()
    reproj = os.path.join(tmp, "t.gpkg")
    gdal.VectorTranslate(reproj, os.path.join(REPO, "data", "raw", traces_rel),
                         options=gdal.VectorTranslateOptions(dstSRS=f"EPSG:{epsg}",
                                                             reproject=True))
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
            pts = np.array([(part.GetX(k), part.GetY(k))
                            for k in range(part.GetPointCount())])
            if len(pts) < 2:
                continue
            col = (pts[:, 0] - gt[0]) / gt[1]
            row = (pts[:, 1] - gt[3]) / gt[5]
            out.append(np.column_stack([col, row]))
    src = None
    return out


def build_channel(dem, valid):
    """Channel buffer + nearest-channel flow azimuth, and the raw channel mask.

    Uses the hardened flow network (block-mean downsample + depression fill)."""
    acc, az = flow_network(dem, downsample=DRAIN_DS, fill=True)
    channel = (acc >= DRAIN_ACCUM) & valid
    buf, near = channel_proximity(channel, az, buffer_px=2)
    return channel, buf, near


def _gray_rgb(a01):
    g = (np.clip(np.nan_to_num(a01), 0, 1) * 255).astype(np.uint8)
    return np.dstack([g, g, g])


def _s2_rgb(bands):
    if not bands:
        return None
    return (np.clip(np.nan_to_num(np.dstack([
        sp.percentile_stretch(bands["swir22"]),
        sp.percentile_stretch(bands["swir16"]),
        sp.percentile_stretch(bands["blue"])])), 0, 1) * 255).astype(np.uint8)


def _trace_tile(base_rgb, channel, poly, shape, label):
    """One audit tile: crop around the trace, draw channel (blue) + trace (orange)."""
    d = densify_line(poly, spacing=1.0)
    cols = np.clip(np.round(d[:, 0]).astype(int), 0, shape[1] - 1)
    rows = np.clip(np.round(d[:, 1]).astype(int), 0, shape[0] - 1)
    cr, cc = int(rows.mean()), int(cols.mean())
    half = max(MIN_HALF_PX,
               int(max(np.ptp(rows), np.ptp(cols)) / 2) + MARGIN_PX)
    r0, c0 = max(0, cr - half), max(0, cc - half)
    r1, c1 = min(shape[0], cr + half), min(shape[1], cc + half)
    crop = base_rgb[r0:r1, c0:c1].copy()
    chan_local = channel[r0:r1, c0:c1]
    crop[binary_dilation(chan_local, iterations=1)] = [70, 120, 255]   # channel blue
    tmask = np.zeros(crop.shape[:2], bool)
    rr = np.clip(rows - r0, 0, crop.shape[0] - 1)
    ccl = np.clip(cols - c0, 0, crop.shape[1] - 1)
    tmask[rr, ccl] = True
    crop[binary_dilation(tmask, iterations=1)] = [255, 150, 30]        # flagged orange
    im = Image.fromarray(crop).resize((TILE, TILE), Image.NEAREST)
    canvas = Image.new("RGB", (TILE, TILE + 14), (20, 20, 20))
    canvas.paste(im, (0, 14))
    ImageDraw.Draw(canvas).text((3, 2), label, fill=(255, 255, 255))
    return canvas


def audit_sheet(out_dir, region, name, base_rgb, channel, polys, shape):
    """Contact sheet of the sampled flagged traces over one base layer."""
    rows = (len(polys) + GRID_COLS - 1) // GRID_COLS
    gap = 6
    cell = TILE + 14
    sheet = Image.new("RGB", (GRID_COLS * TILE + (GRID_COLS - 1) * gap,
                              rows * cell + (rows - 1) * gap), (20, 20, 20))
    for i, poly in enumerate(polys):
        tile = _trace_tile(base_rgb, channel, poly, shape, f"#{i + 1}")
        r, c = divmod(i, GRID_COLS)
        sheet.paste(tile, (c * (TILE + gap), r * (cell + gap)))
    path = os.path.join(out_dir, f"{region}_audit_{name}.png")
    sheet.save(path)
    return path


def conditioned_fn(hand, buf, near, angle_tol, min_aligned):
    """(n_at_risk, n_flagged, flagged_fraction) over hand-traces that overlap the
    channel buffer for >= OVERLAP_AT_RISK of their length."""
    n_at_risk = n_flagged = 0
    for p in hand:
        overlap, aligned = trace_drainage_fraction(p, buf, near, angle_tol_deg=angle_tol)
        if overlap >= OVERLAP_AT_RISK:
            n_at_risk += 1
            if aligned >= min_aligned:
                n_flagged += 1
    frac = n_flagged / n_at_risk if n_at_risk else 0.0
    return n_at_risk, n_flagged, frac


def length(poly):
    d = densify_line(poly, spacing=1.0)
    return float(np.sum(np.hypot(*np.diff(d, axis=0).T))) if len(d) > 1 else 0.0


def main():
    import sys
    region = (sys.argv[1:] or ["nepal"])[0]
    if region not in REGIONS:
        raise SystemExit(f"unknown region {region}; choose {list(REGIONS)}")
    out_dir = os.path.join(REPO, "debug", "drainage_verify", region)
    os.makedirs(out_dir, exist_ok=True)
    tmp = tempfile.mkdtemp()

    dem, gt, dem_path, epsg, traces_rel, bands = fetch(region, tmp)
    valid = np.isfinite(dem)
    _, terr = build_terrain_stack(dem, RES, names=BANDS)
    stack = np.stack([terr[b] for b in BANDS])
    log.info("[%s] detecting ...", region)
    detected = ClassicalTraceDetector(min_length=8).detect(stack)   # (col,row)
    hand = hand_polylines(traces_rel, gt, epsg)
    channel, buf, near = build_channel(dem, valid)

    # ---- classify detections at the operating point ----
    det_align = [trace_drainage_fraction(p, buf, near, angle_tol_deg=ANGLE_TOL)[1]
                 for p in detected]
    flagged = [a >= MIN_ALIGNED for a in det_align]
    flagged_polys = [p for p, f in zip(detected, flagged) if f]
    tot_len = sum(length(p) for p in detected)
    drain_len = sum(length(p) for p, f in zip(detected, flagged) if f)

    print(f"\n=== {region}: drainage filter VERIFICATION "
          f"(angle_tol {ANGLE_TOL:.0f}, min_aligned {MIN_ALIGNED:.0%}) ===")
    print(f"detected traces      : {len(detected)}")
    print(f"flagged as drainage  : {len(flagged_polys)} traces "
          f"({len(flagged_polys)/max(len(detected),1)*100:.0f}% of count), "
          f"{drain_len/max(tot_len,1)*100:.0f}% of total length")

    # ---- (2) conditioned false-negative ----
    n_risk, n_flag, frac = conditioned_fn(hand, buf, near, ANGLE_TOL, MIN_ALIGNED)
    n_overlap_any = sum(
        1 for p in hand
        if trace_drainage_fraction(p, buf, near, angle_tol_deg=ANGLE_TOL)[0] > 0.0)
    print("\n-- false-negative, CONDITIONED on valley-overlapping hand-traces --")
    print(f"hand-traces total            : {len(hand)}")
    print(f"  touching a channel at all  : {n_overlap_any}")
    print(f"  AT RISK (>= {OVERLAP_AT_RISK:.0%} on buffer) : {n_risk}")
    print(f"  of those, FLAGGED (lost)   : {n_flag}  -> {frac*100:.0f}% conditioned FN")
    print(f"  (the diluted number was {n_flag}/{len(hand)} = "
          f"{n_flag/max(len(hand),1)*100:.1f}% across ALL hand-traces - misleading)")
    if n_risk < 30:
        print(f"  CAVEAT: small N (only {n_risk} at-risk traces) - treat as indicative.")

    # ---- (3) parameter sensitivity ----
    print("\n-- parameter sensitivity (removed % of length / conditioned FN %) --")
    header = "min_aligned\\angle " + "".join(f"{a:>10.0f}" for a in SWEEP_ANGLE)
    print(header)
    for ma in SWEEP_ALIGNED:
        cells = []
        for at in SWEEP_ANGLE:
            da = [trace_drainage_fraction(p, buf, near, angle_tol_deg=at)[1]
                  for p in detected]
            dl = sum(length(p) for p, a in zip(detected, da) if a >= ma)
            _, _, ffn = conditioned_fn(hand, buf, near, at, ma)
            cells.append(f"{dl/max(tot_len,1)*100:3.0f}/{ffn*100:>3.0f}")
        print(f"        {ma:>5.0%}      " + "".join(f"{c:>10}" for c in cells))
    print("        (cell = removed%/condFN%;  removed = of detected length)")

    # ---- (1) audit sheets: RANDOM sample of flagged traces ----
    rng = np.random.default_rng(SEED)
    n = min(N_AUDIT, len(flagged_polys))
    pick = sorted(rng.choice(len(flagged_polys), size=n, replace=False).tolist()) \
        if flagged_polys else []
    sample = [flagged_polys[i] for i in pick]
    print(f"\n-- audit sheet: {n} RANDOM flagged traces (seed {SEED}) --")
    hill = tr.multi_hillshade(dem, RES)
    p1 = audit_sheet(out_dir, region, "hillshade", _gray_rgb(hill), channel, sample,
                     dem.shape)
    print(f"  hillshade: {p1}")
    rgb = _s2_rgb(bands)
    if rgb is not None:
        p2 = audit_sheet(out_dir, region, "s2", rgb, channel, sample, dem.shape)
        print(f"  S2 false-colour: {p2}")
    else:
        print("  S2 unavailable (fetch failed) - hillshade sheet only.")
    print("\nJudge each tile: orange trace ON a blue channel = creek (correct flag); "
          "orange crossing/independent of blue = contact wrongly removed.")
    print(f"Sheets in {out_dir}")


if __name__ == "__main__":
    main()
