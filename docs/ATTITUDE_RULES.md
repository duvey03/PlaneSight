# Data-driven attitude rules (planesight-5ug)

Empirically-measured thresholds mined from the geologist's hand-traced
Nepal/Pakistan/Canada datasets, to replace guessed numbers in two blocked pieces:

- **planesight-61f** (refined drainage rule) needs a *confidence gate* on attitudes.
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

## HEADLINE: gate confidence on conditioning + relief, NOT length

**A trace's length does not predict whether its attitude is trustworthy.** In
low-relief Pakistan the conditioning-pass fraction *falls* as length grows
(90%→28% from <250 m to 2–4 km) because long traces there run contour-parallel /
along drainage. **Relief** (the vertical range a trace samples) is the clean,
terrain-independent signal: it drives dip-uncertainty down monotonically in *every*
region, with no inversion. So the confidence rule for 61f is a **relief / DEM-noise
gate**, not a minimum length.

## Recommended thresholds

| Rule | Recommended | Basis |
|---|---|---|
| **Implausible local outlier** (skeptic, gas) | **strike Δ > 60°**, companion **dip Δ > 35°** | pooled **p95** of local deviation at the **1 km** window (n=691) |
| **Attitude confidence gate** (drainage, 61f) | `conditioning ≥ 1e-3` **and** `map_conditioning ≥ 1e-3` **and** **relief ≥ ~80 m (≈40·σ_z)** | relief-vs-reliability knee; below it median dip-unc climbs past ~0.8°, p90 past ~2° |

- Aggressive outlier variant (pooled p90 @ 1 km): strike Δ > 40°, dip Δ > 27°. Use
  p95 (60°/35°) to trigger skeptic **review**, not auto-deletion (see caveats).
- σ_z = GLO-30 vertical 1-σ ≈ 2 m, so ~80 m relief = ~40× the noise floor. Expressing
  the gate as a **ratio** (relief ≥ ~40·σ_z) lets it transfer to other DEMs.
- Length is retained only as a **descriptive prior** (median reliable length
  618–1333 m); it is NOT the gate.

---

## 1. Local attitude variability → the smoothness / outlier bar

Per measurement, "local deviation" = how far its strike (undirected, mod-180
circular mean of neighbours) and dip (median of neighbours) depart from the OTHER
reliable attitudes within a radius. High percentiles = the "implausible" tail.

**Pooled across all three regions** (strike-deviation degrees):

| window | median | IQR | p90 | p95 | n (≥2 neighbours) |
|---|---|---|---|---|---|
| 250 m | 4.6 | 1.6–12.7 | 26.6 | — | **24** (too thin) |
| 500 m | 4.7 | 2.3–15.3 | 30.6 | 51.3 | 211 |
| **1 km** | **7.8** | 3.2–17.5 | **41.0** | **63.1** | **691** |
| 2 km | 8.0 | 3.3–19.2 | 42.1 | — | 1210 |

Dip deviation is flatter: pooled p90 ≈ 27°, p95 ≈ 34° at both 500 m and 1 km.

The 1 km window is the smallest radius at which all three regions are well-populated
(250 m has only 6/17/1 neighbour pairs per region; Canada has 12 at 500 m). The
median local strike deviation is only ~8°, so the regional grain *is* locally smooth;
the implausible tail begins around p90–p95. **p95 (~60° strike / ~35° dip)** is the
conservative "almost surely wrong or a genuine cross-cutting structure" bar.

## 2. The confidence gate — relief vs reliability (61f)

Reliability % = fraction passing the conditioning gate; dip-unc = median MC dip 1-σ.
Relief drives both monotonically in all three regions:

```
            Nepal               Pakistan             Canada
relief(m)  rel%  med  p90      rel%  med  p90        rel%  med  p90   (dip-unc°)
 0–30      ~90   1.5  4.7      ~55  2.7  7.4         ~60  1.1  1.8
 30–50      62   0.7  2.1       67  1.4  4.6          66  0.8  1.6
 50–80      78   0.5  1.2       80  0.8  2.8          75  0.6  1.6
 80–120     90   0.4  1.0       74  0.5  1.2          75  0.4  1.6
 120–200    91   0.2  1.3       80  0.2  0.6          83  0.3  0.9
 200+       97   0.2  0.5       82  0.2  0.5          88+ 0.1  0.6
```

Below ~30 m relief (relief ≈ the 2 m DEM noise floor × ~15) dip-uncertainty is
1–2.9° and the fit often fails conditioning. By **~80 m relief (≈40·σ_z)** median
dip-uncertainty is ≤0.5° and p90 ≤1.5° in every region — the recommended floor. A
stricter 120 m floor buys p90 ≤1.3°; a lenient 50 m floor still risks p90 ~2.8°
(Pakistan).

**Length, by contrast, is not a clean gate** (kept in the driver only to show this):

```
length bin   Nepal rel%   Pakistan rel%   Canada rel%
250–500m       84            84              69
500–1k         85            73              80
1k–2k          94            50              86
2k–4k          89            28              84
```

Pakistan inverts — longer traces are *less* reliable. Gate on relief, not length.

## 3. Morphology priors

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

- **BIGGEST: length is a misleading confidence proxy — gate on relief + conditioning.**
  In low-relief terrain the longest traces are the *least* reliable (Pakistan
  conditioning-pass 90%→28% with length) because they run contour-parallel / along
  drainage. Any rule that reads "longer = more trustworthy" is wrong here; 61f must
  gate on relief ≥ ~40·σ_z **and** conditioning, with length used at most as a weak
  secondary prior.
- **The outlier bar is window- and region-sensitive and its tail mixes real geology
  with bad fits.** p90 strike-deviation swings 31°→41° between the 500 m and 1 km
  windows, and per-region from 28° (Pakistan) to 58° (Nepal); the small radii rest on
  as few as 1–17 neighbour pairs. Even at 1 km the tail includes *genuine* structure
  (fold hinges, cross-cutting trends). → Apply the bar at the ≥1 km scale only, and as
  skeptic **review**, never silent rejection.
- **Positive-unlabeled.** These are curated hand traces, not a complete census. The
  distributions describe *plausible variation among what the geologist drew*; they
  bound neither false-negatives nor auto-detected traces (which are noisier —
  `detect_attitudes_nepal.py` uses the stricter `conditioning >= 1e-2`).
- **Morphology depends on the fitted dip**, so it is only defined for the reliable
  subset and inherits the conditioning gate's choices.
- DEM dip-uncertainty assumes independent per-point noise (optimistic lower bound;
  see `plane_fit._estimate_uncertainty` and planesight-85g). The relief gate is the
  practical guard against this: it keeps the signal well above the noise floor.
