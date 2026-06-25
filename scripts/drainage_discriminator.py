"""Along-vs-across drainage discriminators (planesight-4l8, step 2).

The geologist confirmed the on-channel KEPT band is mostly creeks we missed: meander
sinuosity makes flow-azimuth alignment brittle, so genuine creeks land just under the
0.5 cut. This tests sinuosity-robust discriminators that separate creek-follows-thalweg
from contact-crosses-valley (rule of V's):

  - ELEV monotonicity: a creek descends the thalweg monotonically; a contact crossing
    a valley is a V in elevation (interior minimum). Score = |net drop| / total
    variation (1 = monotonic creek, ~0 = V-shaped contact). Immune to azimuth wiggle.
  - PROFILE-CURVATURE aggregate (the geologist's convexity instinct, as a per-trace
    MEAN not a binary pixel sign): creeks sit in concave thalwegs.

Scorecard per candidate rule: flagged count / % length, the on-channel kept band it
recovers (recall), and the hand-trace conditioned false-negative (must not blow up).

  MAMBA_ROOT_PREFIX=$HOME/micromamba PYTHONPATH=$PWD \
    $HOME/bin/micromamba run -n gdal python scripts/drainage_discriminator.py nepal
"""

from __future__ import annotations

import logging
import os
import tempfile

import numpy as np
from osgeo import gdal, ogr

from planesight.core.attitude.sample import densify_line
from planesight.core.data import fetch_dem
from planesight.core.derivatives import build_terrain_stack
from planesight.core.detect import ClassicalTraceDetector
from planesight.core.detect.drainage import (
    channel_proximity,
    flow_network,
    trace_drainage_fraction,
)

gdal.UseExceptions()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("drainage_discriminator")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGIONS = {
    "nepal": ([82.0, 27.6, 83.0, 28.0], 32644, "nepal/nepal_traces.shp"),
    "pakistan": ([62.0, 25.0, 63.0, 26.0], 32641, "pakistan/pakistan_traces.shp"),
    "canada": ([-117.0, 52.0, -116.0, 53.0], 32611, "canada/canada_traces.shp"),
}
RES = 30.0
BANDS = ("profile_curvature", "curvature", "slope")
DRAIN_DS = 3
DRAIN_ACCUM = 15
ANGLE_TOL = 30.0
MIN_ALIGNED = 0.5
OVERLAP_ON = 0.5        # "on the channel" band
MONO_T = 0.6           # monotonic-descent threshold for "follows thalweg"


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
    return dem, gt, epsg, traces_rel


def hand_polylines(traces_rel, gt, epsg):
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


def length(poly):
    d = densify_line(poly, spacing=1.0)
    return float(np.sum(np.hypot(*np.diff(d, axis=0).T))) if len(d) > 1 else 0.0


def _sample(poly, grid):
    """Nearest-pixel values of `grid` along a densified trace (finite only)."""
    d = densify_line(poly, spacing=1.0)
    rows = np.clip(np.round(d[:, 1]).astype(int), 0, grid.shape[0] - 1)
    cols = np.clip(np.round(d[:, 0]).astype(int), 0, grid.shape[1] - 1)
    v = grid[rows, cols]
    return v[np.isfinite(v)]


def monotonicity(poly, dem):
    """|net elevation change| / total variation along the trace.

    1.0 = strictly monotonic (creek descending the thalweg); ~0 = V-shaped in
    elevation (a contact crossing a valley, down-then-up). NaN if too short."""
    z = _sample(poly, dem)
    if z.size < 3:
        return np.nan
    tv = float(np.sum(np.abs(np.diff(z))))
    if tv <= 0:
        return 1.0
    return float(abs(z[-1] - z[0]) / tv)


def mean_profcurv(poly, pc):
    v = _sample(poly, pc)
    return float(np.mean(v)) if v.size else np.nan


def cond_fn(hand, flag_fn):
    """(at_risk, flagged, fraction) over hand-traces overlapping the channel >=0.5."""
    nr = nf = 0
    for p, ov in hand:
        if ov >= OVERLAP_ON:
            nr += 1
            if flag_fn(p):
                nf += 1
    return nr, nf, (nf / nr if nr else 0.0)


def main():
    import sys
    region = (sys.argv[1:] or ["nepal"])[0]
    tmp = tempfile.mkdtemp()
    dem, gt, epsg, traces_rel = fetch(region, tmp)
    valid = np.isfinite(dem)
    _, terr = build_terrain_stack(dem, RES, names=BANDS)
    pc = terr["profile_curvature"]
    stack = np.stack([terr[b] for b in BANDS])
    log.info("[%s] detecting ...", region)
    det = ClassicalTraceDetector(min_length=8).detect(stack)        # (col,row)

    acc, az = flow_network(dem, downsample=DRAIN_DS, fill=True)
    channel = (acc >= DRAIN_ACCUM) & valid
    buf, near = channel_proximity(channel, az, buffer_px=2)

    lens = np.array([length(p) for p in det])
    ov = np.array([trace_drainage_fraction(p, buf, near, angle_tol_deg=ANGLE_TOL)[0]
                   for p in det])
    al = np.array([trace_drainage_fraction(p, buf, near, angle_tol_deg=ANGLE_TOL)[1]
                   for p in det])
    mono = np.array([monotonicity(p, dem) for p in det])
    mpc = np.array([mean_profcurv(p, pc) for p in det])
    tot = lens.sum()
    on = ov >= OVERLAP_ON
    band = on & (al < MIN_ALIGNED)          # the on-channel KEPT band (the gap)

    # candidate flag rules (boolean masks over detections)
    rules = {
        "OLD align>=0.5": al >= MIN_ALIGNED,
        "MONO on&mono>=.6": on & (mono >= MONO_T),
        "OLD or MONO": (al >= MIN_ALIGNED) | (on & (mono >= MONO_T)),
        "CURV on&pc<0": on & (mpc < 0),
        "OLD or CURV": (al >= MIN_ALIGNED) | (on & (mpc < 0)),
    }

    # hand-trace overlap for conditioned-FN
    hand = hand_polylines(traces_rel, gt, epsg)
    hand_ov = [(p, trace_drainage_fraction(p, buf, near, angle_tol_deg=ANGLE_TOL)[0])
               for p in hand]

    def hand_flag(rule_name):
        # recompute the rule on a single hand trace
        def f(p):
            o, a = trace_drainage_fraction(p, buf, near, angle_tol_deg=ANGLE_TOL)
            m = monotonicity(p, dem)
            c = mean_profcurv(p, pc)
            onp = o >= OVERLAP_ON
            return {
                "OLD align>=0.5": a >= MIN_ALIGNED,
                "MONO on&mono>=.6": onp and (m >= MONO_T),
                "OLD or MONO": (a >= MIN_ALIGNED) or (onp and (m >= MONO_T)),
                "CURV on&pc<0": onp and (c < 0),
                "OLD or CURV": (a >= MIN_ALIGNED) or (onp and (c < 0)),
            }[rule_name]
        return f

    print(f"\n=== {region}: along-vs-across drainage discriminators ===")
    print(f"detected {len(det)} | on-channel band (ov>=.5 & al<.5) = "
          f"{int(band.sum())} traces, {lens[band].sum()/tot*100:.1f}% len (the gap)")
    print(f"mono available for {int(np.isfinite(mono).sum())}/{len(det)} traces\n")
    print(f"{'rule':18s} {'flagged':>8s} {'%len':>5s} {'band caught':>12s} "
          f"{'hand condFN':>12s}")
    print("-" * 60)
    for name, mask in rules.items():
        caught = band & mask
        nr, nf, ff = cond_fn(hand_ov, hand_flag(name))
        print(f"{name:18s} {int(mask.sum()):8d} {lens[mask].sum()/tot*100:4.0f}% "
              f"{int(caught.sum()):6d}/{int(band.sum())} "
              f"{nf:5d}/{nr} ({ff*100:3.0f}%)")
    print("\nband caught = on-channel KEPT creeks now flagged (recall gain);")
    print("hand condFN = real contacts wrongly flagged among valley-overlapping hand traces.")
    print("Goal: high 'band caught', low 'hand condFN' vs the OLD baseline.")


if __name__ == "__main__":
    main()
