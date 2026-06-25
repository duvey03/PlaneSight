"""Spectral creek-vs-contact discriminator probe (planesight-l2c / option A).

The drainage FILTER (flow accumulation) fails on low-relief arid Pakistan. This tests
the only correct secondary use of Sentinel-2 here: not as a detector (DEM-curvature is
the strongest detector everywhere, including Pakistan - Phase 1), but as a creek-vs-
contact DISCRIMINATOR. Hypothesis: creeks carry riparian vegetation / moisture, dry
rock contacts do not, so NDVI / NDMI separate them where flow accumulation can't.

Clean labels (NOT the broken Pakistan flag): CONTACT = the hand-traced contacts (ground
truth); CREEK = confident high-accumulation channel pixels. Compares the NDVI and NDMI
distributions and reports an AUC (0.5 = no separation, >~0.65 = useful). Runs Pakistan
(arid target) and Nepal (vegetated control) so we can tell "spectral fails in arid"
from "spectral fails everywhere".

  MAMBA_ROOT_PREFIX=$HOME/micromamba PYTHONPATH=$PWD \
    $HOME/bin/micromamba run -n gdal python scripts/spectral_drainage_probe.py pakistan
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drainage_verify import REGIONS, fetch, hand_polylines  # noqa: E402

from planesight.core.attitude.sample import densify_line  # noqa: E402
from planesight.core.derivatives import spectral as sp  # noqa: E402
from planesight.core.detect.drainage import flow_network  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("spectral_drainage_probe")

DRAIN_DS = 3
CREEK_ACCUM = 120        # confident high-accumulation channels = "creek" label
RNG = np.random.default_rng(20260625)
MAX_PIX = 20000          # subsample cap per population for the AUC


def sample_along(poly, grid):
    d = densify_line(np.asarray(poly, float), spacing=1.0)
    rows = np.clip(np.round(d[:, 1]).astype(int), 0, grid.shape[0] - 1)
    cols = np.clip(np.round(d[:, 0]).astype(int), 0, grid.shape[1] - 1)
    v = grid[rows, cols]
    return v[np.isfinite(v)]


def subsample(a):
    a = a[np.isfinite(a)]
    if a.size > MAX_PIX:
        a = a[RNG.choice(a.size, size=MAX_PIX, replace=False)]
    return a


def auc_creek_gt_contact(creek, contact):
    """P(creek index value > a random contact value); 0.5 = no separation."""
    if creek.size == 0 or contact.size == 0:
        return float("nan")
    c = np.sort(contact)
    ranks = np.searchsorted(c, creek, side="right")
    return float(np.mean(ranks / c.size))


def stat(a):
    return (float(np.median(a)), float(np.percentile(a, 25)), float(np.percentile(a, 75)))


def main():
    region = (sys.argv[1:] or ["pakistan"])[0]
    if region not in REGIONS:
        raise SystemExit(f"unknown region {region}; choose {list(REGIONS)}")
    tmp = tempfile.mkdtemp()
    dem, gt, dem_path, epsg, traces_rel, bands = fetch(region, tmp)
    if not bands:
        raise SystemExit("S2 fetch failed (no bands) - re-run; the probe needs imagery.")
    valid = np.isfinite(dem)

    ndvi = sp.ndvi(bands["nir"], bands["red"])
    ndmi = sp.normalized_ratio(bands["nir"], bands["swir16"])   # moisture
    indices = {"NDVI": ndvi, "NDMI": ndmi}

    # CREEK label = confident channel pixels (high accumulation)
    acc, _ = flow_network(dem, downsample=DRAIN_DS, fill=True)
    creek_mask = (acc >= CREEK_ACCUM) & valid

    # CONTACT label = hand-traced contact pixels
    hand = hand_polylines(traces_rel, gt, epsg)

    print(f"\n=== {region}: spectral creek-vs-contact discriminator "
          f"(creek=accum>={CREEK_ACCUM}, contact=hand traces) ===")
    print(f"creek pixels {int(creek_mask.sum())} | hand contacts {len(hand)}")
    print(f"{'index':6s} {'contact med [IQR]':22s} {'creek med [IQR]':22s} {'AUC':>6s}  verdict")
    print("-" * 74)
    for name, grid in indices.items():
        contact_vals = subsample(np.concatenate(
            [sample_along(p, grid) for p in hand]) if hand else np.array([]))
        creek_vals = subsample(grid[creek_mask])
        if contact_vals.size == 0 or creek_vals.size == 0:
            print(f"{name:6s} (insufficient samples)")
            continue
        cm, clo, chi = stat(contact_vals)
        rm, rlo, rhi = stat(creek_vals)
        a = auc_creek_gt_contact(creek_vals, contact_vals)
        sep = abs(a - 0.5)
        verdict = ("USEFUL" if sep >= 0.15 else
                   "weak" if sep >= 0.08 else "NONE")
        print(f"{name:6s} {cm:+.3f} [{clo:+.2f},{chi:+.2f}]   "
              f"{rm:+.3f} [{rlo:+.2f},{rhi:+.2f}]   {a:.3f}  {verdict}")
    print("\nAUC = P(creek index > contact index). 0.5 = no separation; a clear creek>"
          "contact (veg/moisture) or contact>creek gap means S2 can discriminate.")


if __name__ == "__main__":
    main()
