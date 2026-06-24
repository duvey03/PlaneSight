# Data-driven attitude rules (planesight-5ug)

Empirically-measured thresholds mined from the geologist's hand-traced
Nepal/Pakistan/Canada datasets, to replace guessed numbers in two blocked pieces:

- **planesight-61f** (refined drainage rule) needs a *minimum confident trace length*.
- **planesight-gas** (skeptic verifier) needs an *implausible local strike/dip change* bar.

Driver: `scripts/attitude_rules.py` (headless GDAL; not in the pure suite). Pure
helpers it builds on: `planesight/core/attitude/variability.py` (circular strike
stats, windowed deviation, morphology, length) with `tests/test_variability.py`.

Method: fetch GLO-30, warp to each region's metric CRS @ 30 m, sample the DEM along
every reprojected hand trace, `fit_plane`, keep fits with `conditioning >= 1e-3`
**and** `map_conditioning >= 1e-3` ("reliable"; the lenient hand-trace gate from
`nepal_slice.py` plus the straight-map-trace guard from planesight-2je). Strike is
undirected (mod 180); all strike means/deviations use doubled-angle circular stats.

---

## Recommended thresholds

| Rule | Recommended | Basis |
|---|---|---|
| **Implausible local outlier** (skeptic, gas) | **strike Δ > 60°**, companion **dip Δ > 35°** | pooled **p95** of local deviation at the **1 km** window (n=691) |
| **Minimum confident length** (drainage, 61f) | **500 m** (≈17 DEM samples @ 30 m) | length-vs-reliability knee; below it dip-uncertainty climbs to 2.7–4.6° |

A more **aggressive** outlier bar (pooled p90 @ 1 km) would be strike Δ > 40°, dip Δ > 27°.
Use p95 (60°/35°) to trigger skeptic **review**, not auto-deletion — see caveats.

---

## 1. Local attitude variability → the smoothness / outlier bar

Per measurement, "local deviation" = how far its strike (undirected, mod-180
circular mean of neighbours) and dip (median of neighbours) depart from the OTHER
reliable attitudes within a radius. High percentiles = the "implausible" tail.

**Pooled across all three regions** (strike-deviation degrees):

| window | median | IQR | p90 | p95 | n (have ≥2 neighbours) |
|---|---|---|---|---|---|
| 250 m | 4.6 | 1.6–12.7 | 26.6 | — | **24** (too thin) |
| 500 m | 4.7 | 2.3–15.3 | 30.6 | 51.3 | 211 |
| **1 km** | **7.8** | 3.2–17.5 | **41.0** | **63.1** | **691** |
| 2 km | 8.0 | 3.3–19.2 | 42.1 | — | 1210 |

Dip deviation is flatter: pooled p90 ≈ 27°, p95 ≈ 34° at both 500 m and 1 km.

**Why 1 km / p95 → ~60° strike, ~35° dip.** The 1 km window is the smallest radius
at which all three regions are well-populated with neighbour pairs (250 m has only
6/17/1 per region — unusable; Canada has just 12 at 500 m). The median local strike
deviation is only ~8–11°, so the regional grain *is* locally smooth; the implausible
tail begins around the 90th–95th percentile. p95 (~60° strike) is the conservative
"this is almost surely wrong or a genuine cross-cutting structure" bar; p90 (~40°)
is the aggressive variant.

Per-region p90 strike deviation @ 1 km: Nepal 57.5° (n=207), Pakistan 28.3° (n=362),
Canada 44.8° (n=122) — Pakistan's grain is tighter, Nepal's noisier; the pooled
number sits between them.

## 2. Morphology + length priors

**Rule-of-V's morphology mix** (reliable fits; class from fitted dip: ≥75° =
`straight`/near-vertical strata, ≤20° = `contour-parallel`/near-horizontal bedding,
else `V`):

| region | V | straight | contour-parallel |
|---|---|---|---|
| Nepal | 62.1% | 1.1% | 36.7% |
| Pakistan | 55.9% | 0.4% | 43.7% |
| Canada | 71.7% | 2.6% | 25.7% |

The diagnostic **V** dominates everywhere (56–72%); near-vertical `straight` strata
are rare (<3%); a substantial `contour-parallel` minority (26–44%) reflects
shallow-dipping bedding hugging contours. **Prior for the skeptic:** an isolated
near-vertical (`straight`) attitude is a-priori unlikely and deserves scrutiny.

**Length distribution** (reliable traces, metres):

| region | median | IQR | p90 |
|---|---|---|---|
| Nepal | 793 | 558–1019 | 1485 |
| Pakistan | 618 | 437–960 | 1435 |
| Canada | 1333 | 926–1788 | 2354 |

**Length vs reliability (the 61f knee).** Reliability % = fraction passing the
conditioning gate; dip-unc = median MC dip 1-σ (lower = more confident):

```
            Nepal              Pakistan            Canada
len bin    rel%  dip-unc      rel%  dip-unc       rel%  dip-unc
0–250m     100%   1.4°        90%    4.6°          —      —
250–500    84%    0.8°        84%    2.7°         69%    1.0°
500–1k     85%    0.3°        73%    1.5°         80%    0.5°
1k–2k      94%    0.2°        50%    1.0°         86%    0.2°
2k–4k      89%    0.1°        28%    0.6°         84%    0.1°
```

Dip uncertainty falls **monotonically** with length in all three regions and crosses
~1.5° by the 500 m bin everywhere. Below ~250–500 m, the worst region (low-relief
Pakistan) carries 2.7–4.6° dip uncertainty — too coarse to trust. Hence **500 m**
as the minimum confident length: ~17 samples at 30 m, dip-unc ≤1.5° in all regions,
just below the reliable-trace medians (618–1333 m).

---

## Ground-truth N per region

| region | fitted | reliable | neighbour pairs @250m / @500m / @1km |
|---|---|---|---|
| Nepal | 408 | 354 | 6 / 57 / 207 |
| Pakistan | 853 | 556 | 17 / 142 / 362 |
| Canada | 652 | 541 | 1 / 12 / 122 |

The *attitude* N is healthy (the historical "~10 traces" worry does not apply here).
The binding small-N is **neighbour coverage at small radii**: trustworthy variability
needs the **≥1 km** window; the 250 m window (and Canada at 500 m, n=12) is too thin.

## Caveats (read before trusting the numbers)

- **BIGGEST: the outlier bar is window- and region-sensitive and its tail conflates
  real geology with bad fits.** p90 strike-deviation swings 31°→41° between the
  500 m and 1 km windows, and per-region from 28° (Pakistan) to 58° (Nepal); the
  small radii rest on as few as 1–17 neighbour pairs. Even at 1 km the tail mixes
  *genuine* structure (fold hinges, cross-cutting trends) with mis-fits. → The bar
  should gate skeptic **review**, never silent rejection, and should be applied at
  the ≥1 km scale only.
- **Positive-unlabeled.** These are curated hand traces, not a complete census of
  contacts. The distributions describe *plausible variation among what the geologist
  chose to draw*; they bound neither false-negatives nor the behaviour of
  auto-detected traces (which are noisier — `detect_attitudes_nepal.py` uses the
  stricter `conditioning >= 1e-2`).
- **Length is necessary, not sufficient (Pakistan inversion).** In low-relief
  Pakistan, conditioning-pass **falls** with length (90%→28%) because long traces
  there are contour-parallel / drainage-following. 61f must not read "longer = better":
  500 m is a floor; relief/conditioning must still gate independently above it.
- **Morphology depends on the fitted dip**, so it is only defined for the reliable
  subset and inherits the conditioning gate's choices.
- DEM dip-uncertainty assumes independent per-point noise (optimistic lower bound;
  see `plane_fit._estimate_uncertainty` and planesight-85g).
