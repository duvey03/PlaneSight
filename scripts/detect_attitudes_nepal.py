"""End-to-end demo: automatic detection -> strike/dip on real Nepal data.

Closes the whole loop for the first time without hand-drawn traces: fetch GLO-30
over the Nepal AOI, build the top DEM bands (the Phase 1 winners), run the
ClassicalTraceDetector to auto-draw candidate contact polylines, then sample the
DEM along each and fit a plane -> strike/dip. Reports how many automatic traces
yield well-conditioned attitudes and their dip/strike distribution, for comparison
with the hand-drawn baseline (scripts/nepal_slice.py).

Run under the headless GDAL env:
  MAMBA_ROOT_PREFIX=$HOME/micromamba PYTHONPATH=$PWD \
    $HOME/bin/micromamba run -n gdal python scripts/detect_attitudes_nepal.py
"""

from __future__ import annotations

import csv
import logging
import os
import tempfile

import numpy as np
from osgeo import gdal

from planesight.core.attitude import fit_plane, sample_trace
from planesight.core.data import fetch_dem
from planesight.core.derivatives import build_terrain_stack
from planesight.core.detect import ClassicalTraceDetector
from planesight.core.detect.drainage import flag_drainage
from planesight.core.detect.vectorize import pixels_to_world

gdal.UseExceptions()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("detect_attitudes_nepal")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "debug")
AOI = [82.0, 27.6, 83.0, 28.0]
EPSG = 32644            # UTM 44N
RES = 30.0
BANDS = ("profile_curvature", "curvature", "slope")  # Phase 1 winners
SIGMA_Z = 2.0          # GLO-30 vertical noise (m)
# Conditioning gate. 1e-3 (used for curated hand-drawn traces) is too lenient for
# AUTO-detected traces: straight map-view segments with relief make 2D vertical
# "planes" (dip ~90) that pass it. They cluster just above 1e-3 while real fits sit
# ~4.6e-2, so 1e-2 cleanly rejects them (near-vertical share 29% -> 5%). See
# docs/PHASE1_REPORT.md S5 and the gate-calibration follow-up issue.
COND_RELIABLE = 1e-2
# Map-view conditioning gate (planesight-2je): straight map traces draped on relief
# pass the 3D conditioning but make degenerate near-vertical fits. A 1e-3 gate
# removes them (near-vertical -> 0%) while keeping 95-98% of fits, calibrated
# consistently across Nepal/Pakistan/Canada (the artifacts cluster below 1e-4).
MAP_COND_RELIABLE = 1e-3
MIN_TRACE_PTS = 8      # min sampled points for a meaningful fit
# Drainage review-flag (planesight-xx2): tag creek-following traces and keep them OUT
# of the attitude stats (recoverable, not deleted). Calibrated operating point from
# the verification + hardening (planesight-5p3/j8t): filled flow net, accum 15.
DRAIN_DS, DRAIN_ACCUM, DRAIN_ALIGNED = 3, 15, 0.5


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = tempfile.mkdtemp()
    dem4326 = os.path.join(tmp, "dem.tif")
    log.info("Fetching GLO-30 over %s ...", AOI)
    fetch_dem(AOI, dem4326)
    dem_utm = os.path.join(tmp, "dem_utm.tif")
    gdal.Warp(dem_utm, dem4326, dstSRS=f"EPSG:{EPSG}", xRes=RES, yRes=RES,
              resampleAlg="bilinear")
    ds = gdal.Open(dem_utm)
    dem = ds.ReadAsArray().astype(float)
    gt = ds.GetGeoTransform()
    nodata = ds.GetRasterBand(1).GetNoDataValue()
    ds = None
    if nodata is not None:
        dem = np.where(dem == nodata, np.nan, dem)
    log.info("Grid %d x %d @ %.0f m", dem.shape[1], dem.shape[0], RES)

    _, terr = build_terrain_stack(dem, RES, names=BANDS)
    stack = np.stack([terr[b] for b in BANDS])

    log.info("Detecting traces (Canny on %s)...", "+".join(BANDS))
    # detect in PIXEL (col,row) space so the drainage classifier can index the grid;
    # kept traces are converted to world coords for sampling below.
    traces = ClassicalTraceDetector(min_length=MIN_TRACE_PTS, simplify_tol=1.0).detect(
        stack)
    log.info("Auto-detected %d candidate traces", len(traces))

    # drainage review-flag: tag creek-following traces and keep them out of the
    # attitude stats (retained, not deleted - recoverable in review).
    labels = flag_drainage(traces, dem, downsample=DRAIN_DS, min_accum_cells=DRAIN_ACCUM,
                           min_aligned_fraction=DRAIN_ALIGNED, valid_mask=np.isfinite(dem))
    kept = [(t, s) for t, (d, s) in zip(traces, labels) if not d]
    flagged = [(t, s) for t, (d, s) in zip(traces, labels) if d]
    log.info("Drainage review-flag: %d kept, %d flagged (retained, excluded from "
             "attitudes)", len(kept), len(flagged))

    # fit strike/dip along each KEPT trace (pixel (col,row) -> world for sampling)
    atts, reliable = [], []
    for tr_px, _ in kept:
        world = pixels_to_world(tr_px[:, ::-1], gt)
        pts = sample_trace(world, dem, gt, spacing=RES, nodata=None)
        if len(pts) < MIN_TRACE_PTS:
            continue
        att = fit_plane(pts, sigma_z=SIGMA_Z)
        if not np.isfinite(att.dip):
            continue
        atts.append(att)
        if att.conditioning >= COND_RELIABLE and att.map_conditioning >= MAP_COND_RELIABLE:
            reliable.append(att)

    log.info("Fitted %d attitudes; %d reliable (conditioning >= %.0e)",
             len(atts), len(reliable), COND_RELIABLE)
    if reliable:
        dips = np.array([a.dip for a in reliable])
        strikes = np.array([a.strike % 180 for a in reliable])
        print("\n=== Automatic strike/dip on Nepal (reliable fits) ===")
        print(f"traces detected : {len(traces)}")
        print(f"drainage-flagged: {len(flagged)} (retained for review, not fitted)")
        print(f"kept for fitting: {len(kept)}")
        print(f"reliable fits   : {len(reliable)}")
        print(f"dip    median {np.median(dips):.1f}  IQR [{np.percentile(dips,25):.1f}, "
              f"{np.percentile(dips,75):.1f}]  range [{dips.min():.1f}, {dips.max():.1f}]")
        print(f"strike median {np.median(strikes):.1f} (mod 180)  "
              f"IQR [{np.percentile(strikes,25):.1f}, {np.percentile(strikes,75):.1f}]")
        # dominant strike direction (circular, on doubled angles)
        ang = np.radians(2 * strikes)
        mean_strike = (np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) / 2) % 180
        print(f"dominant strike : {mean_strike:.0f} deg (vector mean) - compare to the "
              f"Himalayan ~NW-SE structural grain")

        csv_path = os.path.join(OUT_DIR, "nepal_auto_attitudes.csv")
        with open(csv_path, "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["strike", "dip", "dip_direction", "conditioning",
                         "planarity", "n_samples"])
            for a in reliable:
                wr.writerow([f"{a.strike:.1f}", f"{a.dip:.1f}", f"{a.dip_direction:.1f}",
                             f"{a.conditioning:.3e}", f"{a.planarity:.3e}", a.n_samples])
        print(f"\nWrote {csv_path}")
    else:
        log.warning("No reliable attitudes - check detection/conditioning gate.")

    # drainage review queue: flagged traces ranked most-creek-like first (score = the
    # fraction of the trace running along the flow). The future GUI review gate
    # consumes this; for now it is the auditable record of what was set aside.
    if flagged:
        rq_path = os.path.join(OUT_DIR, "nepal_drainage_review_queue.csv")
        with open(rq_path, "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["rank", "drainage_score", "n_vertices"])
            for i, (tr_px, s) in enumerate(sorted(flagged, key=lambda x: -x[1]), 1):
                wr.writerow([i, f"{s:.3f}", len(tr_px)])
        print(f"Wrote {rq_path} ({len(flagged)} flagged, ranked by score)")


if __name__ == "__main__":
    main()
