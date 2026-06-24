# PlaneSight Phase 1 Report - Research Spike Findings

**Status:** Phase 1 (research spike) complete. Phase 2 (classical detection MVP)
underway. **Date:** 2026-06-23.

This report records what the Phase 1 experiments established about (a) which input
bands best expose geological contacts, (b) which detection operator to ship in v1,
and (c) the methodology that got us there - including a wrong conclusion we caught
and corrected. It resolves issues `bc1`, `c3r`, and `0pb`, and informs the
Phase 2 build (`8gw`).

---

## 1. Method

All findings come from a reproducible experiment harness
(`scripts/detector_experiment.py`) run on three structurally distinct regions with
hand-drawn ground-truth traces:

| Region | Terrain | Trace provenance |
|---|---|---|
| Nepal | Monsoonal Himalaya, high relief | Topography-digitized |
| Pakistan | Arid Balochistan, moderate relief | **Satellite-imagery-digitized** |
| Canada | Alberta/BC Cordillera, high relief | (mixed) |

Two methodology choices were essential to getting honest answers:

- **Positive-unlabeled scoring (`detect/score.py`).** The drawn traces are an
  *incomplete* subset of all mappable contacts, so we rank on **recall at an equal
  detection budget** (top-k, tie-proof) and never on precision/F1 - a detection far
  from a drawn line may be a real, un-digitized contact, not a false positive.
- **Label-free linearity (`detect/linearity.py`).** Recall cannot see whether
  detections form long clean lines vs scattered blobs (the contact-vs-mound goal).
  We added a metric that scores connected-component elongation - the same
  eigenvalue/fabric idea used by the plane fit and structure tensor, applied to
  each blob's pixel-coordinate cloud.

---

## 2. Finding: which bands expose contacts (`bc1`)

Recall@5% per band, corrected (verified 100% Sentinel-2 AOI coverage):

| Band | Nepal | Pakistan | Canada |
|---|---|---|---|
| profile_curvature | 0.33 | **0.69** | 0.32 |
| curvature (total) | 0.43 | 0.64 | 0.36 |
| plan_curvature | **0.43** | 0.40 | 0.30 |
| slope | 0.39 | 0.52 | **0.53** |
| multi_hillshade | 0.29 | 0.47 | 0.43 |
| iron_oxide (S2) | 0.16 | 0.30 | 0.26 |
| swir16_s (S2) | 0.05 | 0.20 (Nov 0.45) | 0.27 |
| clay / ndvi (S2) | <0.18 | <0.06 | <0.07 |

**Conclusions:**
- **DEM curvature and slope are the robust, dominant contact exposers in every
  region** - the best band per region is always a DEM derivative.
- **Sentinel-2 is a useful secondary signal, not a primary detector.** `iron_oxide`
  is the most consistent spectral band; SWIR is **seasonally sensitive** (Pakistan
  `swir16_s` recall@5% rises from 0.20 in May to 0.45 in November), which validates
  the empirical dry-season scene selection (D15).
- **Curated defaults** (set in `derivatives/stack.py`):
  `DEFAULT_TERRAIN = (profile_curvature, curvature, slope, multi_hillshade)`,
  `DEFAULT_SPECTRAL = (iron_oxide, ferrous, swir16_s)`.

### 2.1 A wrong conclusion, caught and corrected

An earlier run reported a dramatic "Sentinel-2 SWIR dominates in arid Pakistan"
(recall@5% **0.76**, beating every DEM band). This was **substantially a coverage
artifact**: the single chosen S2 scene covered ~0.09% of the AOI, and uncovered
nodata read as 0, manufacturing false edges that inflated SWIR recall. Fixing scene
selection (coverage-aware same-date mosaic + nodata handling, `data/align.py`) and
re-running dropped it to 0.20 (May) / 0.45 (November). The lesson - **verify AOI
coverage and nodata before trusting any spectral number** - is now baked into the
harness, which logs per-date AOI coverage and warns below 50%.

---

## 3. Finding: which operator to ship (`c3r` / `0pb`)

We compared a plain gradient response against a multi-channel **structure tensor**
(`detect/structure.py`) on both recall and linearity (Pakistan, same best band):

| Method | recall@5% | linearity@5% |
|---|---|---|
| gradient [profile_curvature] | **0.69** | 0.65 |
| structure tensor [same band] | 0.58 | 0.64 |
| structure tensor, DEM+S2 stack | 0.47 | **0.71** |

**Decision: the v1 detector front-end is a Canny-style gradient operator on the top
DEM bands.** The structure tensor loses recall without buying linearity on a single
band; multi-band fusion buys ~10% linearity for ~32% recall, and DEM+S2 ≈ DEM-only.
The structure tensor is retained as an optional "clean-lines" mode, not the primary.
Canny is preferred over plain gradient magnitude because its non-maximum suppression
thins edges natively and its hysteresis links weak edges to strong ones (better than
a blunt top-k threshold).

---

## 4. Deliverables built in Phase 1

All pure numpy/scipy/GDAL (decision D11), CI-tested:

- **Derivative engine** (`core/derivatives/`): terrain (slope/aspect/hillshade/
  tpi/curvature) and Sentinel-2 spectral indices, named-band registries.
- **Co-registration** (`core/data/align.py`): warp + coverage-aware S2 mosaic onto
  the DEM analysis grid.
- **Scoring** (`core/detect/score.py`, `linearity.py`): positive-unlabeled
  recall-at-budget + label-free linearity.
- **Detection operators** (`core/detect/structure.py`, `canny.py`): structure
  tensor (evaluated, secondary) and Canny (v1 front-end).
- **Vectorization back-end** (`core/detect/vectorize.py`): Zhang-Suen thinning,
  deterministic skeleton tracing, gentle Douglas-Peucker.
- **`ClassicalTraceDetector`** (`core/detect/classical.py`): the v1 detector behind
  the `TraceDetector` ABC, registered "classical".

---

## 5. End-to-end validation on real Nepal data

The full chain - AOI -> GLO-30 -> top DEM bands -> `ClassicalTraceDetector` ->
auto-drawn polylines -> DEM sampling -> `fit_plane` - was run on the Nepal AOI
(`scripts/detect_attitudes_nepal.py`):

From a raw 1°×0.4° AOI the detector auto-drew **7,063 candidate traces**; sampling
and fitting each gave attitudes, of which **3,324 are well-conditioned** (gate
`conditioning ≥ 1e-2`):

| Metric | Value |
|---|---|
| candidate traces detected | 7,063 |
| well-conditioned attitudes | 3,324 |
| dip | median **19.8°**, IQR 11.8–28.5° |
| dominant strike (vector mean) | **~86°** |
| near-vertical (dip ≥ 85°) share | 5% |

The median dip ~20° and the dominant ~E–W strike are consistent with the Himalayan
structural grain at this longitude (western Nepal) - the fully automatic result
recovers the same regional structure the hand-drawn baseline did
(`scripts/nepal_slice.py`), without any manual digitizing.

**Finding - the conditioning gate must be tighter for automatic traces.** At the
old `1e-3` gate, **29%** of "reliable" fits were near-vertical (dip ≥ 85°): these
are *straight* detected segments, which - draped over relief - form a 2D vertical
strip that passes a 3D conditioning test (the metric catches 1D-collinear traces,
not straight-but-topographically-varying ones). They are cleanly separable - they
cluster at conditioning ~2.5e-3, just above the gate, while genuine fits sit at
~4.6e-2 - so raising the gate to `1e-2` removes them (29% → 5%). A principled gate
calibration and/or a map-view sinuosity filter is the recommended Phase 3 follow-up
(strike/dip signal lives in trace sinuosity; a straight segment cannot constrain
dip). This is exactly the kind of degenerate case the human-in-the-loop triage
(S9) and the conditioning-ranked review queue are designed to surface.

### 5.1 Multi-region detector evaluation (`wiu`)

The detector was run on all three regions (`scripts/detector_eval.py`, DEM-only,
gate `conditioning ≥ 1e-2`):

| Region | recall | linearity | reliable attitudes | dip median | dominant strike | known grain |
|---|---|---|---|---|---|---|
| Nepal | 0.39 | 0.68 | 3,324 | 19.8° | ~86° (E–W) | Himalaya ≈ E–W ✓ |
| Pakistan | 0.54 | 0.65 | 7,015 | 9.3° | ~90° (E–W) | Makran ranges ≈ E–W ✓ |
| Canada | 0.41 | 0.76 | 4,390 | 24.6° | ~148° (NW–SE) | Cordillera ≈ NW–SE ✓ |

The detector recovers 39–54% of the (incomplete) hand-drawn traces at a ~6–7%
edge budget with clean, linear detections, and yields thousands of
well-conditioned attitudes per region. **The decisive result: each region's
dominant automatic strike matches its known regional structural grain** -
E–W for the Himalaya and the Makran ranges, NW–SE for the Cordillera. The
near-vertical artifact share is 2–5% at the `1e-2` conditioning gate, further
reducible by the new `map_conditioning` guard (Section 2 of issue `2je`).

---

## 6. Outstanding / deferred

- **Detection eval (`wiu`)** - systematic recall + linearity + attitude-sanity of
  the detector across regions (this report's Section 5 is the first cut).
- **Training-bootstrap go/no-go (`9vt`)** - the v2 ML gate (map-draping for labels)
  is **not yet evaluated**; v1 classical ships independently of it.
- **Adaptive per-AOI weighting (`472`)** - deferred until more diverse labeled
  regions exist; blocked on acquiring a low-relief/Shield AOI (`luj`).

**Bottom line:** the v1 classical detection path is validated end to end and does
not depend on the unresolved ML bootstrap. DEM curvature/slope are the workhorses;
Sentinel-2 is a seasonal secondary; Canny is the operator; the structure tensor is
an optional mode.
