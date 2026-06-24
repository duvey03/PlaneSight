"""Residual-drainage analysis (planesight-4l8): does the filter leave creeks in the KEPT set?

The geologist still sees creek/drainage flagging in the results, so before trusting
the flow-only filter we MEASURE the recall gap directly: of the traces we KEEP (do
not flag), how many still run along the channel network just under the threshold, or
along sub-threshold tributaries the channel mask never captured? Reports:
  (1) overlap/alignment distribution, kept vs flagged;
  (2) recall-gap-vs-min_aligned (how many MORE flag as we loosen the along-flow cut);
  (3) channel-density sensitivity (how many more flag as we lower the accum threshold
      to capture smaller tributaries);
  (4) an audit sheet of the most drainage-like KEPT traces, for an eyeball check.

  MAMBA_ROOT_PREFIX=$HOME/micromamba PYTHONPATH=$PWD \
    $HOME/bin/micromamba run -n gdal python scripts/drainage_residual.py nepal
"""

from __future__ import annotations

import logging
import os
import tempfile

import numpy as np
from osgeo import gdal
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation

from planesight.core.attitude.sample import densify_line
from planesight.core.data import fetch_dem
from planesight.core.derivatives import build_terrain_stack
from planesight.core.derivatives import terrain as tr
from planesight.core.detect import ClassicalTraceDetector
from planesight.core.detect.drainage import (
    channel_proximity,
    flow_network,
    trace_drainage_fraction,
)

gdal.UseExceptions()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("drainage_residual")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGIONS = {
    "nepal": ([82.0, 27.6, 83.0, 28.0], 32644),
    "pakistan": ([62.0, 25.0, 63.0, 26.0], 32641),
    "canada": ([-117.0, 52.0, -116.0, 53.0], 32611),
}
RES = 30.0
BANDS = ("profile_curvature", "curvature", "slope")
DRAIN_DS = 3
DRAIN_ACCUM = 15        # current operating channel threshold
ANGLE_TOL = 30.0
MIN_ALIGNED = 0.5       # current operating along-flow cut (flagged if >= this)
OVERLAP_ON_CHANNEL = 0.5  # "sits on the channel buffer" if overlap >= this

N_AUDIT = 30
GRID_COLS = 6
TILE = 150
MIN_HALF_PX = 18
MARGIN_PX = 6


def fetch(region, tmp):
    aoi, epsg = REGIONS[region]
    d4326 = os.path.join(tmp, "d.tif")
    log.info("[%s] fetching GLO-30 ...", region)
    fetch_dem(aoi, d4326)
    dem_path = os.path.join(tmp, "dem.tif")
    gdal.Warp(dem_path, d4326, dstSRS=f"EPSG:{epsg}", xRes=RES, yRes=RES,
              resampleAlg="bilinear")
    ds = gdal.Open(dem_path)
    dem = ds.ReadAsArray().astype(float)
    nd = ds.GetRasterBand(1).GetNoDataValue()
    ds = None
    if nd is not None:
        dem = np.where(dem == nd, np.nan, dem)
    return dem


def length(poly):
    d = densify_line(poly, spacing=1.0)
    return float(np.sum(np.hypot(*np.diff(d, axis=0).T))) if len(d) > 1 else 0.0


def _gray_rgb(a01):
    g = (np.clip(np.nan_to_num(a01), 0, 1) * 255).astype(np.uint8)
    return np.dstack([g, g, g])


def _trace_tile(base_rgb, channel, poly, shape, label):
    d = densify_line(poly, spacing=1.0)
    cols = np.clip(np.round(d[:, 0]).astype(int), 0, shape[1] - 1)
    rows = np.clip(np.round(d[:, 1]).astype(int), 0, shape[0] - 1)
    cr, cc = int(rows.mean()), int(cols.mean())
    half = max(MIN_HALF_PX, int(max(np.ptp(rows), np.ptp(cols)) / 2) + MARGIN_PX)
    r0, c0 = max(0, cr - half), max(0, cc - half)
    r1, c1 = min(shape[0], cr + half), min(shape[1], cc + half)
    crop = base_rgb[r0:r1, c0:c1].copy()
    crop[binary_dilation(channel[r0:r1, c0:c1], iterations=1)] = [70, 120, 255]
    tmask = np.zeros(crop.shape[:2], bool)
    tmask[np.clip(rows - r0, 0, crop.shape[0] - 1),
          np.clip(cols - c0, 0, crop.shape[1] - 1)] = True
    crop[binary_dilation(tmask, iterations=1)] = [255, 230, 40]   # KEPT-but-suspect: yellow
    im = Image.fromarray(crop).resize((TILE, TILE), Image.NEAREST)
    canvas = Image.new("RGB", (TILE, TILE + 14), (20, 20, 20))
    canvas.paste(im, (0, 14))
    ImageDraw.Draw(canvas).text((3, 2), label, fill=(255, 255, 255))
    return canvas


def audit_sheet(out_dir, region, base_rgb, channel, items, shape):
    rows = (len(items) + GRID_COLS - 1) // GRID_COLS
    gap, cell = 6, TILE + 14
    sheet = Image.new("RGB", (GRID_COLS * TILE + (GRID_COLS - 1) * gap,
                              rows * cell + (rows - 1) * gap), (20, 20, 20))
    for i, (poly, ov, al) in enumerate(items):
        tile = _trace_tile(base_rgb, channel, poly, shape,
                           f"#{i + 1} ov{ov:.2f} al{al:.2f}")
        r, c = divmod(i, GRID_COLS)
        sheet.paste(tile, (c * (TILE + gap), r * (cell + gap)))
    path = os.path.join(out_dir, f"{region}_kept_drainage_like.png")
    sheet.save(path)
    return path


def main():
    import sys
    region = (sys.argv[1:] or ["nepal"])[0]
    out_dir = os.path.join(REPO, "debug", "drainage_residual", region)
    os.makedirs(out_dir, exist_ok=True)
    tmp = tempfile.mkdtemp()

    dem = fetch(region, tmp)
    valid = np.isfinite(dem)
    _, terr = build_terrain_stack(dem, RES, names=BANDS)
    stack = np.stack([terr[b] for b in BANDS])
    log.info("[%s] detecting ...", region)
    detected = ClassicalTraceDetector(min_length=8).detect(stack)   # (col,row)

    # operating-point channel network + per-trace overlap/alignment
    acc, az = flow_network(dem, downsample=DRAIN_DS, fill=True)
    channel = (acc >= DRAIN_ACCUM) & valid
    buf, near = channel_proximity(channel, az, buffer_px=2)
    frac = [trace_drainage_fraction(p, buf, near, angle_tol_deg=ANGLE_TOL)
            for p in detected]              # (overlap, aligned) per trace
    lens = np.array([length(p) for p in detected])
    overlap = np.array([f[0] for f in frac])
    aligned = np.array([f[1] for f in frac])
    flagged = aligned >= MIN_ALIGNED
    kept = ~flagged
    tot = lens.sum()

    print(f"\n=== {region}: residual-drainage in the KEPT set "
          f"(accum {DRAIN_ACCUM}, min_aligned {MIN_ALIGNED:.0%}, angle {ANGLE_TOL:.0f}) ===")
    print(f"detected {len(detected)} | flagged {int(flagged.sum())} "
          f"({lens[flagged].sum()/tot*100:.0f}% len) | KEPT {int(kept.sum())} "
          f"({lens[kept].sum()/tot*100:.0f}% len)")

    # (1) of the KEPT traces, how many still sit ON the channel network?
    kept_on = kept & (overlap >= OVERLAP_ON_CHANNEL)
    kept_border = kept & (aligned >= 0.3)
    print("\n-- residual drainage among KEPT traces --")
    print(f"  KEPT but sitting on channel buffer (overlap >= {OVERLAP_ON_CHANNEL:.0%}): "
          f"{int(kept_on.sum())} traces, {lens[kept_on].sum()/tot*100:.1f}% of all length")
    print(f"  KEPT but along-flow 0.3-0.5 (just under the cut): {int(kept_border.sum())} "
          f"traces, {lens[kept_border].sum()/tot*100:.1f}% of length")

    # (2) recall gap vs the along-flow cut: how much MORE flags as we loosen min_aligned
    print("\n-- recall gap vs min_aligned (cumulative flagged) --")
    for ma in (0.5, 0.4, 0.3, 0.2):
        m = aligned >= ma
        print(f"  min_aligned {ma:.1f}: {int(m.sum())} flagged "
              f"({lens[m].sum()/tot*100:.0f}% len)  (+{int(m.sum()-flagged.sum())} vs 0.5)")

    # (3) channel-density sensitivity: lower accum to capture smaller tributaries
    print("\n-- channel-density sensitivity (min_aligned 0.5, vary accum) --")
    for ac in (30, 15, 8, 4):
        ch = (acc >= ac) & valid
        b2, n2 = channel_proximity(ch, az, buffer_px=2)
        al2 = np.array([trace_drainage_fraction(p, b2, n2, angle_tol_deg=ANGLE_TOL)[1]
                        for p in detected])
        m = al2 >= MIN_ALIGNED
        covpct = 100 * ch.sum() / valid.sum()
        print(f"  accum {ac:3d} (channel {covpct:4.1f}% of map): {int(m.sum())} flagged "
              f"({lens[m].sum()/tot*100:.0f}% len)  (+{int(m.sum()-flagged.sum())} vs accum15)")

    # (4) audit sheet: the most drainage-like KEPT traces (rank by overlap, then aligned)
    kept_idx = np.where(kept)[0]
    order = sorted(kept_idx, key=lambda i: (overlap[i], aligned[i]), reverse=True)
    pick = order[:N_AUDIT]
    items = [(detected[i], float(overlap[i]), float(aligned[i])) for i in pick]
    hill = tr.multi_hillshade(dem, RES)
    p = audit_sheet(out_dir, region, _gray_rgb(hill), channel, items, dem.shape)
    print(f"\n-- audit: {len(items)} most drainage-like KEPT traces (yellow=kept, blue=channel) --")
    print(f"  {p}")
    print("  If these are creeks, the flow-only filter has a real recall gap (expected:")
    print("  sub-threshold tributaries + along-flow-but-just-under-cut). If they cross the")
    print("  blue, they are legit kept contacts.")


if __name__ == "__main__":
    main()
