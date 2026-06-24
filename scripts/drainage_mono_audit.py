"""Visual validation of the elevation-monotonicity discriminator (planesight-4l8).

Renders the on-channel KEPT band (ov>=.5 & al<.5 - the creeks the alignment cut
missed) split by monotonicity: ORANGE = MONO would now flag (should be creeks that
descend a thalweg), GREEN = MONO keeps (should be contact-Vs that cross the valley,
down-then-up). Each tile shows its monotonicity score. The geologist confirms MONO
catches creeks AND spares the V's before we adopt OLD-or-MONO.

  MAMBA_ROOT_PREFIX=$HOME/micromamba PYTHONPATH=$PWD \
    $HOME/bin/micromamba run -n gdal python scripts/drainage_mono_audit.py nepal
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
log = logging.getLogger("drainage_mono_audit")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGIONS = {"nepal": ([82.0, 27.6, 83.0, 28.0], 32644),
           "pakistan": ([62.0, 25.0, 63.0, 26.0], 32641),
           "canada": ([-117.0, 52.0, -116.0, 53.0], 32611)}
RES = 30.0
BANDS = ("profile_curvature", "curvature", "slope")
DRAIN_DS, DRAIN_ACCUM, ANGLE_TOL, MIN_ALIGNED = 3, 15, 30.0, 0.5
OVERLAP_ON, MONO_T = 0.5, 0.6
N_EACH, GRID_COLS, TILE, MIN_HALF_PX, MARGIN_PX, SEED = 15, 6, 150, 18, 6, 20260624


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


def _sample(poly, grid):
    d = densify_line(poly, spacing=1.0)
    rows = np.clip(np.round(d[:, 1]).astype(int), 0, grid.shape[0] - 1)
    cols = np.clip(np.round(d[:, 0]).astype(int), 0, grid.shape[1] - 1)
    v = grid[rows, cols]
    return v[np.isfinite(v)]


def monotonicity(poly, dem):
    z = _sample(poly, dem)
    if z.size < 3:
        return np.nan
    tv = float(np.sum(np.abs(np.diff(z))))
    return 1.0 if tv <= 0 else float(abs(z[-1] - z[0]) / tv)


def _gray_rgb(a01):
    g = (np.clip(np.nan_to_num(a01), 0, 1) * 255).astype(np.uint8)
    return np.dstack([g, g, g])


def _tile(base_rgb, channel, poly, shape, label, color):
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
    crop[binary_dilation(tmask, iterations=1)] = color
    im = Image.fromarray(crop).resize((TILE, TILE), Image.NEAREST)
    canvas = Image.new("RGB", (TILE, TILE + 14), (20, 20, 20))
    canvas.paste(im, (0, 14))
    ImageDraw.Draw(canvas).text((3, 2), label, fill=(255, 255, 255))
    return canvas


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
    det = ClassicalTraceDetector(min_length=8).detect(stack)
    acc, az = flow_network(dem, downsample=DRAIN_DS, fill=True)
    channel = (acc >= DRAIN_ACCUM) & valid
    buf, near = channel_proximity(channel, az, buffer_px=2)

    band = []
    for p in det:
        ov, al = trace_drainage_fraction(p, buf, near, angle_tol_deg=ANGLE_TOL)
        if ov >= OVERLAP_ON and al < MIN_ALIGNED:
            m = monotonicity(p, dem)
            if np.isfinite(m):
                band.append((p, m))
    flagged = [(p, m) for p, m in band if m >= MONO_T]   # MONO would flag (creeks?)
    kept = [(p, m) for p, m in band if m < MONO_T]        # MONO keeps (contact-Vs?)
    log.info("band %d: MONO-flag %d, MONO-keep %d", len(band), len(flagged), len(kept))

    rng = np.random.default_rng(SEED)

    def pick(items, n):
        if len(items) <= n:
            return items
        idx = sorted(rng.choice(len(items), size=n, replace=False).tolist())
        return [items[i] for i in idx]

    fl = [(p, m, "F", [255, 150, 30]) for p, m in pick(flagged, N_EACH)]
    kp = [(p, m, "K", [60, 255, 80]) for p, m in pick(kept, N_EACH)]
    items = fl + kp                      # flagged (orange) block, then kept (green)

    hill = _gray_rgb(tr.multi_hillshade(dem, RES))
    rows = (len(items) + GRID_COLS - 1) // GRID_COLS
    gap, cell = 6, TILE + 14
    sheet = Image.new("RGB", (GRID_COLS * TILE + (GRID_COLS - 1) * gap,
                              rows * cell + (rows - 1) * gap), (20, 20, 20))
    for i, (poly, m, tag, color) in enumerate(items):
        t = _tile(hill, channel, poly, dem.shape, f"#{i + 1} {tag} mono{m:.2f}", color)
        r, c = divmod(i, GRID_COLS)
        sheet.paste(t, (c * (TILE + gap), r * (cell + gap)))
    path = os.path.join(out_dir, f"{region}_mono_split.png")
    sheet.save(path)
    print(f"\nband {len(band)} | MONO-flag {len(flagged)} (orange) | "
          f"MONO-keep {len(kept)} (green)")
    print(f"sheet: {path}")
    print("orange (F) should be creeks descending a thalweg; green (K) should be")
    print("contact-Vs that cross the valley (down-then-up).")


if __name__ == "__main__":
    main()
