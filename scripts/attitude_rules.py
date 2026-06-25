"""Data-driven attitude rules from the hand-traced datasets (planesight-5ug).

Mines the geologist's own Nepal/Pakistan/Canada hand traces to REPLACE guessed
thresholds with measured ones, feeding two blocked downstream pieces:

  - the skeptic's "implausible local strike/dip change" bar (planesight-gas), from
    the high percentiles of LOCAL attitude variability over sliding windows; and
  - the refined drainage rule's "minimum confident trace length" (planesight-61f),
    from where fit reliability stops improving with length.

For each region it: fetches GLO-30, warps to the region's metric CRS, samples the
DEM along each reprojected hand trace, fits a plane (reusing fit_plane /
sample_trace), then computes (1) windowed local strike/dip deviation at several
radii, (2) the rule-of-V's morphology mix, and (3) the length distribution +
length-vs-reliability knee. All circular strike maths is doubled-angle (mod 180).

Headless GDAL only (run manually; NOT part of the pure pytest suite):
  MAMBA_ROOT_PREFIX=$HOME/micromamba PYTHONPATH=$PWD \
    $HOME/bin/micromamba run -n gdal python scripts/attitude_rules.py

Per-region attitude CSVs land in debug/ (gitignored) for audit.
"""

from __future__ import annotations

import csv
import logging
import os
import tempfile

import numpy as np
from osgeo import gdal, ogr

from planesight.core.attitude import (
    classify_morphology,
    fit_plane,
    polyline_length,
    sample_trace,
    windowed_deviations,
)
from planesight.core.attitude.variability import CONTOUR_PARALLEL, STRAIGHT, V
from planesight.core.data import fetch_dem

gdal.UseExceptions()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("attitude_rules")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "debug")

# AOI (WGS84 bbox) / metric EPSG / traces shapefile, per region (matches the
# REGIONS table in scripts/drainage_verify.py).
REGIONS = {
    "nepal": ([82.0, 27.6, 83.0, 28.0], 32644, "nepal/nepal_traces.shp"),
    "pakistan": ([62.0, 25.0, 63.0, 26.0], 32641, "pakistan/pakistan_traces.shp"),
    "canada": ([-117.0, 52.0, -116.0, 53.0], 32611, "canada/canada_traces.shp"),
}

RES = 30.0              # DEM resolution / sampling spacing (m)
SIGMA_Z = 2.0          # GLO-30 ~1-sigma vertical error (m)
MIN_SAMPLES = 5        # min sampled points for a meaningful fit (matches nepal_slice)
# Hand-trace reliability gate. Hand traces are curated, so the lenient 1e-3
# conditioning gate (nepal_slice.py) applies, NOT the stricter 1e-2 used for noisy
# auto-detected traces. map_conditioning still guards the straight-map-trace /
# near-vertical pseudo-plane whose DIP is unconstrained (planesight-2je).
COND_RELIABLE = 1e-3
MAP_COND_RELIABLE = 1e-3
# Sliding-window radii (m) for local variability.
WINDOWS = (250.0, 500.0, 1000.0, 2000.0)
# Relief bins (m) for the relief-vs-reliability knee - the REAL confidence gate for
# 61f (relief / DEM-noise ratio), not length (length inverts in low-relief terrain).
RELIEF_BINS = (0, 10, 20, 30, 50, 80, 120, 200, 400, np.inf)
# Length bins (m) for the length-vs-reliability check (descriptive prior only - kept
# to demonstrate that length is NOT a clean confidence signal; see RELIEF_BINS).
LENGTH_BINS = (0, 250, 500, 1000, 2000, 4000, 8000, np.inf)


def load_dem(aoi, epsg, tmp):
    """Fetch GLO-30 over the AOI and warp to the region's metric CRS @ RES."""
    d4326 = os.path.join(tmp, "dem4326.tif")
    log.info("  fetching GLO-30 over %s ...", aoi)
    fetch_dem(aoi, d4326)
    dem_utm = os.path.join(tmp, "dem_utm.tif")
    gdal.Warp(dem_utm, d4326, dstSRS=f"EPSG:{epsg}", xRes=RES, yRes=RES,
              resampleAlg="bilinear")
    ds = gdal.Open(dem_utm)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray().astype(float)
    gt = ds.GetGeoTransform()
    nd = band.GetNoDataValue()
    ds = None
    if nd is not None:
        arr = np.where(arr == nd, np.nan, arr)
    return arr, gt, nd


def load_traces(traces_rel, epsg, tmp):
    """Reproject hand traces to the metric CRS; return lists of (x, y) world verts."""
    reproj = os.path.join(tmp, "traces.gpkg")
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
                            for k in range(part.GetPointCount())], dtype=float)
            if len(pts) >= 2:
                out.append(pts)
    src = None
    return out


def fit_region(region):
    """Fit attitudes along every hand trace of a region.

    Returns a list of dict records (one per trace part that yielded >= MIN_SAMPLES
    DEM samples) with geometry, fitted attitude, quality metrics and length.
    """
    aoi, epsg, traces_rel = REGIONS[region]
    tmp = tempfile.mkdtemp()
    arr, gt, _ = load_dem(aoi, epsg, tmp)
    traces = load_traces(traces_rel, epsg, tmp)
    log.info("  %s: %d trace parts; DEM %dx%d", region, len(traces),
             arr.shape[1], arr.shape[0])

    records = []
    for tr in traces:
        pts = sample_trace(tr, arr, gt, spacing=RES, nodata=None)
        if len(pts) < MIN_SAMPLES:
            continue
        att = fit_plane(pts, sigma_z=SIGMA_Z)
        if not np.isfinite(att.dip):
            continue
        reliable = (att.conditioning >= COND_RELIABLE
                    and att.map_conditioning >= MAP_COND_RELIABLE)
        records.append({
            "cx": float(pts[:, 0].mean()), "cy": float(pts[:, 1].mean()),
            "strike": att.strike, "dip": att.dip,
            "conditioning": att.conditioning, "map_conditioning": att.map_conditioning,
            "dip_unc": att.dip_uncertainty, "relief": att.relief,
            "length": polyline_length(tr), "n_samples": att.n_samples,
            "reliable": reliable,
        })
    return records


def _pct(a):
    """median / IQR / 90th formatted string for a 1D array (NaNs dropped)."""
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return "n/a (no data)"
    return (f"median {np.median(a):5.1f}  IQR [{np.percentile(a, 25):5.1f}, "
            f"{np.percentile(a, 75):5.1f}]  p90 {np.percentile(a, 90):5.1f}  (n={a.size})")


def variability_table(records):
    """Windowed local strike/dip deviation per radius, over RELIABLE attitudes.

    Returns {radius: (strike_dev_array, dip_dev_array)} so the caller can pool
    across regions for the recommended threshold.
    """
    rel = [r for r in records if r["reliable"]]
    if len(rel) < 3:
        log.info("    < 3 reliable attitudes; variability not computed")
        return {}
    xy = np.array([[r["cx"], r["cy"]] for r in rel])
    strikes = np.array([r["strike"] for r in rel])
    dips = np.array([r["dip"] for r in rel])
    out = {}
    for radius in WINDOWS:
        sdev, ddev, nbr = windowed_deviations(xy, strikes, dips, radius=radius)
        out[radius] = (sdev, ddev)
        covered = int(np.isfinite(sdev).sum())
        print(f"    r={radius:6.0f} m  ({covered}/{len(rel)} have >=2 neighbours)")
        print(f"      strike dev: {_pct(sdev)}")
        print(f"      dip    dev: {_pct(ddev)}")
    return out


def morphology_and_length(records):
    """Print the morphology mix (reliable fits) and length distribution."""
    rel = [r for r in records if r["reliable"]]
    counts = {STRAIGHT: 0, V: 0, CONTOUR_PARALLEL: 0, None: 0}
    for r in rel:
        counts[classify_morphology(r["dip"])] += 1
    total = max(1, len(rel))
    print(f"    morphology mix (n={len(rel)} reliable):")
    for label in (V, STRAIGHT, CONTOUR_PARALLEL):
        print(f"      {label:16s}: {counts[label]:4d}  ({100 * counts[label] / total:4.1f}%)")

    lengths_all = np.array([r["length"] for r in records])
    lengths_rel = np.array([r["length"] for r in rel])
    print(f"    length (all fitted, n={len(records)}): {_pct(lengths_all)} m")
    print(f"    length (reliable,  n={len(rel)}): {_pct(lengths_rel)} m")
    return lengths_all, lengths_rel


def _knee_table(records, key, bins, header):
    """Reliability% + median/p90 dip-uncertainty per bin of ``key`` (relief or length)."""
    vals = np.array([r[key] for r in records])
    reliable = np.array([r["reliable"] for r in records])
    dip_unc = np.array([r["dip_unc"] for r in records])
    print(f"    {header:>14s}     n   reliable%   median dip-unc   p90 dip-unc")
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (vals >= lo) & (vals < hi)
        n = int(m.sum())
        if n == 0:
            continue
        frac = 100.0 * reliable[m].mean()
        mu = dip_unc[m]
        mu = mu[np.isfinite(mu)]
        med = f"{np.median(mu):5.2f}" if mu.size else "  n/a"
        p90 = f"{np.percentile(mu, 90):5.2f}" if mu.size else "  n/a"
        hi_s = "inf" if np.isinf(hi) else f"{int(hi)}"
        print(f"    [{int(lo):5d}, {hi_s:>5s})  {n:5d}    {frac:5.1f}      "
              f"{med} deg      {p90} deg")


def relief_reliability_knee(records):
    """Relief-vs-reliability knee - the data-driven CONFIDENCE GATE for planesight-61f.

    Relief (vertical range a trace samples) drives dip-uncertainty monotonically in
    every region (no length-style inversion): as relief approaches the DEM vertical
    noise floor the dip is unconstrained (ARCHITECTURE.md S6.4). The knee where
    dip-uncertainty stabilises is the recommended relief gate (expressed as a
    relief / sigma_z ratio so it transfers across DEMs).
    """
    _knee_table(records, "relief", RELIEF_BINS, "relief bin (m)")


def length_reliability_knee(records):
    """Length-vs-reliability - DESCRIPTIVE ONLY (length is not a clean gate).

    Kept to show *why* length is rejected as the confidence signal: in low-relief
    terrain (Pakistan) the conditioning-pass fraction FALLS as length grows, because
    long traces there run contour-parallel / along drainage. Gate on relief, not this.
    """
    _knee_table(records, "length", LENGTH_BINS, "length bin (m)")


def write_csv(region, records):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{region}_attitude_rules.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cx", "cy", "strike", "dip", "conditioning", "map_conditioning",
                    "dip_unc", "relief", "length", "n_samples", "reliable"])
        for r in records:
            w.writerow([f"{r['cx']:.1f}", f"{r['cy']:.1f}", f"{r['strike']:.2f}",
                        f"{r['dip']:.2f}", f"{r['conditioning']:.3e}",
                        f"{r['map_conditioning']:.3e}", f"{r['dip_unc']:.2f}",
                        f"{r['relief']:.1f}", f"{r['length']:.1f}", r["n_samples"],
                        int(r["reliable"])])
    log.info("  wrote %s", path)


def main():
    pooled = {radius: ([], []) for radius in WINDOWS}  # strike_dev, dip_dev across regions
    region_n = {}
    for region in REGIONS:
        print(f"\n========== {region.upper()} ==========")
        records = fit_region(region)
        n_rel = sum(r["reliable"] for r in records)
        region_n[region] = (len(records), n_rel)
        print(f"  fitted {len(records)} attitudes; {n_rel} reliable "
              f"(conditioning >= {COND_RELIABLE:.0e} and map_cond >= {MAP_COND_RELIABLE:.0e})")
        write_csv(region, records)

        print("\n  -- local attitude variability (reliable fits) --")
        var = variability_table(records)
        for radius, (sdev, ddev) in var.items():
            pooled[radius][0].append(sdev[np.isfinite(sdev)])
            pooled[radius][1].append(ddev[np.isfinite(ddev)])

        print("\n  -- morphology + length priors --")
        morphology_and_length(records)
        print("\n  -- relief vs reliability (THE confidence gate: relief/sigma_z) --")
        relief_reliability_knee(records)
        print("\n  -- length vs reliability (descriptive: length is NOT a clean gate) --")
        length_reliability_knee(records)

    # Pooled recommended thresholds across all three regions.
    print("\n\n========== POOLED RECOMMENDATIONS (all regions) ==========")
    print("Ground-truth N per region (fitted / reliable):")
    for region, (n, nr) in region_n.items():
        flag = "  <-- small N, treat as indicative" if nr < 30 else ""
        print(f"  {region:9s}: {n:5d} / {nr:5d}{flag}")

    print("\nLocal strike-deviation distribution, pooled (the smoothness/outlier bar):")
    for radius in WINDOWS:
        s = np.concatenate(pooled[radius][0]) if pooled[radius][0] else np.array([])
        d = np.concatenate(pooled[radius][1]) if pooled[radius][1] else np.array([])
        print(f"  r={radius:6.0f} m  strike dev {_pct(s)}")
        print(f"             dip    dev {_pct(d)}")
    # Recommended outlier bar = pooled 90th percentile of strike deviation at the
    # 500 m-1 km window (local enough to be 'the same structure', wide enough for
    # >=2 neighbours). Printed explicitly for the doc + downstream skeptic.
    for radius in (500.0, 1000.0):
        s = np.concatenate(pooled[radius][0]) if pooled[radius][0] else np.array([])
        d = np.concatenate(pooled[radius][1]) if pooled[radius][1] else np.array([])
        if s.size:
            print(f"\n  -> @ {radius:.0f} m: strike p90={np.percentile(s, 90):.1f} deg, "
                  f"p95={np.percentile(s, 95):.1f} deg; "
                  f"dip p90={np.percentile(d, 90):.1f} deg, p95={np.percentile(d, 95):.1f} deg")


if __name__ == "__main__":
    main()
