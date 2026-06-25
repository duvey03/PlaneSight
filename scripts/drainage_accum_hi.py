"""High-end accumulation sweep (planesight-amn follow-up).

drainage_test.py only sweeps accum down to 4 and up to 120. For low-relief arid
terrain (Pakistan) the ~15% channel-density operating point sits AT or ABOVE 120,
with no clean knee in the standard range. This extends the sweep upward
(240..60) to locate (or rule out) a knee for the recommended Pakistan threshold,
reusing the exact same flow_network + channel_proximity + alignment classifier so
the numbers are comparable to drainage_test.py. Channel% + drainage-removed% only
(no hand-FN; it is a weak positive-unlabeled signal - see docs/DRAINAGE_SWEEP.md).

  MAMBA_ROOT_PREFIX=$HOME/micromamba PYTHONPATH=$PWD \
    $HOME/bin/micromamba run -n gdal python scripts/drainage_accum_hi.py pakistan
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile

import numpy as np
from osgeo import gdal

from planesight.core.attitude.sample import densify_line
from planesight.core.data import fetch_dem
from planesight.core.derivatives import build_terrain_stack
from planesight.core.detect import ClassicalTraceDetector
from planesight.core.detect.drainage import (
    channel_proximity,
    flow_network,
    is_drainage,
)

gdal.UseExceptions()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("drainage_accum_hi")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGIONS = {
    "nepal": ([82.0, 27.6, 83.0, 28.0], 32644),
    "pakistan": ([62.0, 25.0, 63.0, 26.0], 32641),
    "canada": ([-117.0, 52.0, -116.0, 53.0], 32611),
}
RES = 30.0
BANDS = ("profile_curvature", "curvature", "slope")
DOWNSAMPLE = 3
BUFFER_PX = 2
MIN_ALIGNED = 0.5
THRESHOLDS = (240, 180, 120, 90, 60)   # high-end (drainage_test stops at 120)


def length(poly):
    d = densify_line(poly, spacing=1.0)
    return float(np.sum(np.hypot(*np.diff(d, axis=0).T))) if len(d) > 1 else 0.0


def main():
    region = (sys.argv[1:] or ["pakistan"])[0]
    if region not in REGIONS:
        raise SystemExit(f"unknown region {region}; choose {list(REGIONS)}")
    aoi, epsg = REGIONS[region]
    tmp = tempfile.mkdtemp()
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
    valid = np.isfinite(dem)

    _, terr = build_terrain_stack(dem, RES, names=BANDS)
    stack = np.stack([terr[b] for b in BANDS])
    log.info("[%s] detecting ...", region)
    detected = ClassicalTraceDetector(min_length=8).detect(stack)
    log.info("[%s] flow network (downsample %dx, filled) ...", region, DOWNSAMPLE)
    acc, az = flow_network(dem, downsample=DOWNSAMPLE, fill=True)
    tot = sum(length(p) for p in detected)

    print(f"\n=== {region}: HIGH-END accum sweep (buffer {BUFFER_PX}px, "
          f">= {int(MIN_ALIGNED*100)}% along-flow) ===")
    print("accum  channel%   drainage-removed%   d(channel%)")
    print("-" * 50)
    prev = None
    for thr in THRESHOLDS:
        mask = (acc >= thr) & valid
        buf, near = channel_proximity(mask, az, buffer_px=BUFFER_PX)
        flags = [is_drainage(p, buf, near, min_aligned_fraction=MIN_ALIGNED)
                 for p in detected]
        cov = 100 * mask.sum() / valid.sum()
        drain = sum(length(p) for p, f in zip(detected, flags) if f)
        rem = 100 * drain / max(tot, 1)
        delta = "" if prev is None else f"{cov - prev:+5.1f}"
        print(f"{thr:5d}  {cov:6.1f}%      {rem:5.0f}% ({sum(flags)} traces)   {delta}")
        prev = cov


if __name__ == "__main__":
    main()
