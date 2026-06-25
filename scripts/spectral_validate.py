"""Validate the spectral NDMI drainage discriminator on the REAL use case (planesight-0bf).

The probe (`spectral_drainage_probe.py`) found NDMI separates creek-from-contact on arid
Pakistan at AUC 0.94 - but using a PROXY label (channel *pixels* vs hand *contacts*),
which partly reflects "wet valley vs dry slope", not the actual operation: classify
DETECTED traces. This is the build gate. It answers, on the units we actually act on:

  Does per-trace mean-NDMI flag creek-following DETECTED traces without eating
  valley-crossing CONTACTS - and is it cleaner than the flow-accumulation filter?

Method (Pakistan, arid target). Detect traces (ClassicalTraceDetector), fetch S2, compute
NDMI = normalized_ratio(nir, swir16). For every detected trace, take the mean/median NDMI
along its length. Label detected traces by the flow network (the only creek reference we
have): CREEK-proxy = overlaps the channel buffer >= OVERLAP_AT_RISK; OFF-CHANNEL = no
overlap. Pick a moisture threshold (Youden's J separating creek-proxy detected traces from
the hand contacts), report the flagged fraction of the detected set, and the CONDITIONED
FALSE-NEGATIVE: the fraction of hand-traced CONTACTS the threshold would eat. Render audit
panels (high-NDMI flagged-creek vs low-NDMI kept-contact, over hillshade + S2) and compare
head-to-head with the flow filter on the same detected set + hand contacts.

  MAMBA_ROOT_PREFIX=$HOME/micromamba PYTHONPATH=$PWD \
    $HOME/bin/micromamba run -n gdal python scripts/spectral_validate.py pakistan
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Reuse the drainage-verify S2 fetch + audit-panel machinery and the probe's NDMI helpers.
from drainage_verify import (  # noqa: E402
    ANGLE_TOL,
    BANDS,
    GRID_COLS,
    MIN_ALIGNED,
    OVERLAP_AT_RISK,
    REGIONS,
    RES,
    TILE,
    _gray_rgb,
    _s2_rgb,
    _trace_tile,
    build_channel,
    conditioned_fn,
    fetch,
    hand_polylines,
    length,
)
from spectral_drainage_probe import auc_creek_gt_contact, sample_along  # noqa: E402

from planesight.core.derivatives import build_terrain_stack  # noqa: E402
from planesight.core.derivatives import spectral as sp  # noqa: E402
from planesight.core.derivatives import terrain as tr  # noqa: E402
from planesight.core.detect import ClassicalTraceDetector  # noqa: E402
from planesight.core.detect.drainage import trace_drainage_fraction  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("spectral_validate")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_PANEL = 18           # traces per audit sheet (high-NDMI and low-NDMI)
MIN_SAMPLES = 5        # a trace needs this many finite NDMI pixels to get a per-trace value


def per_trace_ndmi(polys, ndmi):
    """Mean and median NDMI along each trace; NaN for traces with too few finite pixels.

    Returns (mean, median) float arrays aligned to ``polys`` (NaN where unsampled)."""
    means = np.full(len(polys), np.nan)
    medians = np.full(len(polys), np.nan)
    for i, p in enumerate(polys):
        v = sample_along(p, ndmi)
        if v.size >= MIN_SAMPLES:
            means[i] = float(np.mean(v))
            medians[i] = float(np.median(v))
    return means, medians


def youden_threshold(pos, neg):
    """Threshold on a 'high = positive' score maximising Youden's J = TPR - FPR.

    pos = creek-proxy per-trace NDMI, neg = contact per-trace NDMI. Returns
    (threshold, tpr, fpr, j). Candidates are the midpoints between sorted unique values,
    so the operating point sits between samples rather than on one."""
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    vals = np.unique(np.concatenate([pos, neg]))
    mids = (vals[:-1] + vals[1:]) / 2.0 if vals.size > 1 else vals
    best = (float("nan"), 0.0, 0.0, -np.inf)
    for t in mids:
        tpr = float(np.mean(pos >= t))
        fpr = float(np.mean(neg >= t))
        j = tpr - fpr
        if j > best[3]:
            best = (float(t), tpr, fpr, j)
    return best


def panel_sheet(out_dir, region, name, base_rgb, channel, polys, labels, shape):
    """Audit contact-sheet of `polys` over one base layer, each tile labelled (NDMI).

    Reuses drainage_verify._trace_tile (channel = blue, trace = orange) but with our own
    NDMI labels, so the geologist can read the per-trace value off each tile."""
    if not polys:
        return None
    rows = (len(polys) + GRID_COLS - 1) // GRID_COLS
    gap, cell = 6, TILE + 14
    sheet = Image.new("RGB", (GRID_COLS * TILE + (GRID_COLS - 1) * gap,
                              rows * cell + (rows - 1) * gap), (20, 20, 20))
    for i, (poly, lab) in enumerate(zip(polys, labels)):
        tile = _trace_tile(base_rgb, channel, poly, shape, lab)
        r, c = divmod(i, GRID_COLS)
        sheet.paste(tile, (c * (TILE + gap), r * (cell + gap)))
    path = os.path.join(out_dir, f"{region}_ndmi_{name}.png")
    sheet.save(path)
    return path


def main():
    region = (sys.argv[1:] or ["pakistan"])[0]
    if region not in REGIONS:
        raise SystemExit(f"unknown region {region}; choose {list(REGIONS)}")
    out_dir = os.path.join(REPO, "debug", "spectral_validate", region)
    os.makedirs(out_dir, exist_ok=True)
    tmp = tempfile.mkdtemp()

    dem, gt, dem_path, epsg, traces_rel, bands = fetch(region, tmp)
    if not bands:
        raise SystemExit("S2 fetch failed (no bands) - re-run in the FOREGROUND; the "
                         "validation needs imagery.")
    valid = np.isfinite(dem)
    ndmi = sp.normalized_ratio(bands["nir"], bands["swir16"])   # high = moist

    # ---- detect traces (the real units) + the flow-network channel reference ----
    _, terr = build_terrain_stack(dem, RES, names=BANDS)
    stack = np.stack([terr[b] for b in BANDS])
    log.info("[%s] detecting ...", region)
    detected = ClassicalTraceDetector(min_length=8).detect(stack)   # (col,row) pixels
    hand = hand_polylines(traces_rel, gt, epsg)
    channel, buf, near = build_channel(dem, valid)

    # ---- per-trace NDMI on DETECTED traces + their channel overlap (creek proxy) ----
    det_mean, det_med = per_trace_ndmi(detected, ndmi)
    det_overlap = np.array([trace_drainage_fraction(p, buf, near,
                            angle_tol_deg=ANGLE_TOL)[0] for p in detected])
    sampled = np.isfinite(det_mean)
    creek_proxy = sampled & (det_overlap >= OVERLAP_AT_RISK)   # on a channel = creek-like
    offchannel = sampled & (det_overlap == 0.0)                # clearly off-channel

    # ---- per-trace NDMI on hand CONTACTS (ground truth) + their channel overlap ----
    hand_mean, _ = per_trace_ndmi(hand, ndmi)
    hand_overlap = np.array([trace_drainage_fraction(p, buf, near,
                             angle_tol_deg=ANGLE_TOL)[0] for p in hand])
    hand_ok = np.isfinite(hand_mean)
    hand_valley = hand_ok & (hand_overlap > 0.0)               # valley-crossing contacts
    hand_atrisk = hand_ok & (hand_overlap >= OVERLAP_AT_RISK)  # flow-filter's at-risk set

    print(f"\n=== {region}: per-trace NDMI validation (REAL use case, planesight-0bf) ===")
    print(f"detected traces        : {len(detected)} ({int(sampled.sum())} with >= "
          f"{MIN_SAMPLES} finite NDMI px)")
    print(f"  creek-proxy (overlap >= {OVERLAP_AT_RISK:.0%}) : {int(creek_proxy.sum())}")
    print(f"  off-channel (overlap == 0)      : {int(offchannel.sum())}")
    print(f"hand contacts          : {len(hand)} ({int(hand_ok.sum())} sampled; "
          f"{int(hand_valley.sum())} valley-crossing)")

    # ---- separation on the REAL units (trace-level), vs the probe's pixel 0.94 ----
    auc_proxy_vs_contact = auc_creek_gt_contact(det_mean[creek_proxy], hand_mean[hand_ok])
    auc_proxy_vs_off = auc_creek_gt_contact(det_mean[creek_proxy], det_mean[offchannel])
    print("\n-- per-trace NDMI separation (AUC = P(creek NDMI > contact NDMI)) --")

    def _ms(a):
        a = a[np.isfinite(a)]
        return (f"{np.median(a):+.3f} [{np.percentile(a, 25):+.2f},"
                f"{np.percentile(a, 75):+.2f}]") if a.size else "n/a"
    print(f"  creek-proxy detected NDMI : {_ms(det_mean[creek_proxy])}")
    print(f"  off-channel detected NDMI : {_ms(det_mean[offchannel])}")
    print(f"  hand-contact        NDMI : {_ms(hand_mean[hand_ok])}")
    print(f"  AUC creek-proxy vs hand-contact : {auc_proxy_vs_contact:.3f}  "
          f"(probe pixel-proxy was 0.94)")
    print(f"  AUC creek-proxy vs off-channel  : {auc_proxy_vs_off:.3f}")

    # ---- pick a moisture threshold (Youden's J: creek-proxy positive, contact negative) ----
    t, tpr, fpr, j = youden_threshold(det_mean[creek_proxy], hand_mean[hand_ok])
    flagged_all = sampled & (det_mean >= t)
    cond_fn = float(np.mean(hand_mean[hand_ok] >= t)) if hand_ok.any() else float("nan")
    cond_fn_valley = (float(np.mean(hand_mean[hand_valley] >= t))
                      if hand_valley.any() else float("nan"))
    print("\n-- recommended NDMI threshold (Youden's J on per-trace mean) --")
    print(f"  threshold  : NDMI >= {t:+.3f}  (flag as creek)   [J = {j:.2f}]")
    print(f"  flagged    : {int(flagged_all.sum())}/{int(sampled.sum())} detected traces "
          f"({flagged_all.sum()/max(sampled.sum(),1)*100:.0f}%); "
          f"TPR on creek-proxy = {tpr*100:.0f}%")
    print(f"  CONDITIONED FN (hand contacts eaten)        : {cond_fn*100:.0f}%  "
          f"({int(np.sum(hand_mean[hand_ok] >= t))}/{int(hand_ok.sum())})")
    print(f"  CONDITIONED FN on valley-crossing contacts  : {cond_fn_valley*100:.0f}%  "
          f"({int(np.sum(hand_mean[hand_valley] >= t))}/{int(hand_valley.sum())})")
    if hand_ok.sum() < 30:
        print(f"  CAVEAT: small N ({int(hand_ok.sum())} sampled hand contacts) - "
              f"treat FN as indicative.")

    # ---- head-to-head with the flow-accumulation filter (same detected set + contacts) ----
    flow_aligned = np.array([trace_drainage_fraction(p, buf, near,
                             angle_tol_deg=ANGLE_TOL)[1] for p in detected])
    flow_flagged = flow_aligned >= MIN_ALIGNED
    flow_flag_len = sum(length(p) for p, f in zip(detected, flow_flagged) if f)
    tot_len = sum(length(p) for p in detected)
    _, flow_nflag, flow_fn = conditioned_fn(hand, buf, near, ANGLE_TOL, MIN_ALIGNED)
    # NDMI cond-FN on the SAME at-risk subset the flow filter conditions on (>=50% overlap),
    # so the two cond-FN columns share a denominator.
    cond_fn_atrisk = (float(np.mean(hand_mean[hand_atrisk] >= t))
                      if hand_atrisk.any() else float("nan"))
    print("\n-- head-to-head: per-trace NDMI vs flow-accumulation filter (Pakistan) --")
    print(f"  {'method':22s} {'flagged (count)':>16s} {'flagged (length)':>17s} "
          f"{'cond-FN(at-risk)':>16s}")
    print(f"  {'NDMI per-trace':22s} "
          f"{flagged_all.sum()/max(sampled.sum(),1)*100:14.0f}% "
          f"{'(n/a)':>17s} {cond_fn_atrisk*100:15.0f}%")
    print(f"  {'flow-accum filter':22s} "
          f"{flow_flagged.sum()/max(len(detected),1)*100:14.0f}% "
          f"{flow_flag_len/max(tot_len,1)*100:16.0f}% {flow_fn*100:15.0f}%")
    print(f"  (at-risk = hand contacts with >= {OVERLAP_AT_RISK:.0%} channel overlap, "
          f"N = {int(hand_atrisk.sum())}; cleaner = flags creeks at LOWER cond-FN)")

    # ---- audit panels: high-NDMI (flagged creek) vs low-NDMI (kept contact) ----
    order = np.argsort(det_mean[sampled])              # ascending NDMI
    sidx = np.flatnonzero(sampled)[order]
    low_idx = sidx[:N_PANEL]                            # driest = kept contacts
    high_idx = sidx[::-1][:N_PANEL]                     # moistest = flagged creeks
    hill = tr.multi_hillshade(dem, RES)
    rgb = _s2_rgb(bands)
    print(f"\n-- audit panels ({N_PANEL} traces each) --")
    for tag, idx in (("highNDMI_flagged", high_idx), ("lowNDMI_kept", low_idx)):
        polys = [detected[i] for i in idx]
        labels = [f"{det_mean[i]:+.2f}" for i in idx]
        p1 = panel_sheet(out_dir, region, f"{tag}_hillshade", _gray_rgb(hill), channel,
                         polys, labels, dem.shape)
        print(f"  {tag} hillshade : {p1}")
        if rgb is not None:
            p2 = panel_sheet(out_dir, region, f"{tag}_s2", rgb, channel, polys, labels,
                             dem.shape)
            print(f"  {tag} S2        : {p2}")
    print("\nJudge: high-NDMI tiles should sit ON blue channels (genuine creeks/washes); "
          "low-NDMI tiles should cross/ignore channels (real contacts kept).")
    print(f"Panels in {out_dir}")


if __name__ == "__main__":
    main()
