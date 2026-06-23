# PlaneSight - Architecture & Guidance

> A QGIS plugin that aggregates free global remote-sensing data for any area of
> interest, automatically detects geological bedding/contact traces, and
> mass-computes strike and dip measurements from trace-topography geometry.

**Status:** Foundational design document (living)
**Last updated:** 2026-06-22
**Owner:** Mike Duvall (duvey03)

This document is the single source of truth for *why* PlaneSight exists, *what*
it must do, and *how* it is structured. It is intentionally implementation-light
in places where decisions are still open; those are flagged in the Decisions Log
(Section 14). Code-level tracking lives in the Beads issue tracker, not here.

---

## 1. Purpose

Global-scale Digital Elevation Models (DEMs) and Sentinel-2 imagery are freely
available for the entire planet. Where geological strata are exposed and have
topographic expression, the *trace* of a bedding plane or lithological contact
across the land surface encodes its 3D orientation. A geologist can, in
principle, recover strike and dip from that trace and the underlying terrain -
the classic three-point problem, generalised.

Doing this by hand is slow and local. PlaneSight aims to do it **automatically,
at regional scale, anywhere on Earth**, from data anyone can obtain for free,
inside a tool a first-time QGIS user can operate.

### Mission statement

> Turn freely available global terrain and imagery into quantitative structural
> geology measurements - quickly, flexibly, and intuitively - for everyone from
> students to professional mappers.

---

## 2. Positioning & Prior Art

A prior-art scan (June 2026) established that PlaneSight's three pillars each
exist independently, and some pairs are integrated, but **the specific
end-to-end pipeline - auto-detect traces from raw raster, then mass-compute
attitude from trace-topography geometry - does not exist as a single tool.**

| Tool | What it does | Relationship to PlaneSight |
|---|---|---|
| **GeoTrace** (QGIS; Thiele/Grose) | Semi-automatic least-cost-path tracing + best-fit plane from trace eigenvectors draped on a DEM -> strike/dip with a planarity metric | Closest match to our **strike/dip engine** and uses the *same eigenvector method*. Stale (last meaningful release 2017). Tracing is user-seeded, not automatic. **Study and modernise.** |
| **qgSurf** (QGIS; M. Alberti) | SVD best-fit-plane attitude from manually picked points; plane-DEM intersection; stereonets | Maintained. Manual-only. Reference implementation for the attitude math. |
| **map2loop / Loop3D** (Jessell et al. 2021) | Fully automated regional mass-orientation field | Closest to the *end goal*, but **map-fed** - consumes existing vector geology maps, does not detect from raw raster. |
| **Aghaee et al. 2021** (`LineamentLearning`) | CNN per-pixel probability -> clustered, vectorised polylines; targets contacts and folds | Closest **detection** match; open source. **Stops before orientation** (attitude deferred to "future work"). Validates the segmentation-then-vectorise approach. |
| **CloudCompare FACETS / Compass** | Automatic/semi-auto plane detection + dip on 3D point clouds | Same math, wrong data domain (outcrop point clouds, not map-scale DEM traces). |

### Our differentiators

1. **The missing front-end.** Chain an automated raster trace-detector to a
   proven mass-attitude back-end - no other tool does this.
2. **One-AOI global data aggregation.** Bundle DEM + Sentinel-2 auto-fetch into
   the structural workflow. The data access exists; nobody packages it for this
   purpose.
3. **A modern, maintained QGIS structural plugin.** The only comparable QGIS
   tool (GeoTrace) is abandoned; qgSurf is manual-only.

### The hard, unsolved problem (scope honestly)

Every classical and deep-learning detector in the literature produces
**unlabeled** lines. Reliably distinguishing *bedding/contacts* from faults,
drainage networks, and anthropogenic lineaments is the genuine open scientific
risk, and labeled bedding training sets essentially do not exist. **PlaneSight
therefore treats human-in-the-loop review and classification as a first-class,
required part of the workflow - not an afterthought.**

---

## 3. Scope

### In scope

- Area-of-interest driven aggregation of global DEM and Sentinel-2 data (no
  account/signup friction for the end user).
- Computation of a configurable stack of terrain derivatives and imagery
  composites.
- Automated detection of candidate bedding/contact traces (classical first;
  machine-learning later).
- Human review, editing, and classification of detected traces.
- Mass computation of strike/dip from traces + DEM, with per-measurement
  uncertainty.
- Standard structural-geology output (symbolised attitude points, trace layers,
  exportable tables).

### Out of scope (for now)

- 3D structural/geological modelling (this is map2loop / Loop3D territory; we may
  export *to* such tools later).
- Subsurface inference, stratigraphic thickness, cross-sections.
- Non-planar / heavily folded structure reconstruction beyond per-trace planar
  fits.
- Field-data capture (Strabo / FieldMove territory).
- Mineral-prospectivity / targeting.

---

## 4. Users & Use Cases

**Primary persona - "first-time QGIS geologist."** Has geological knowledge,
limited GIS/coding skill, no patience for environment setup. Must be able to
install the plugin and get a result with a few clicks.

**Secondary persona - "power mapper / researcher."** Wants tunable parameters,
batch processing over large regions, the ability to bring custom rasters, and
clean exports for downstream analysis.

Representative use cases:

- *Reconnaissance:* "I have a remote AOI with no field access - give me a
  first-pass structural picture."
- *Augmentation:* "I have sparse field measurements - fill the gaps and check for
  consistency."
- *Teaching:* "Show students how trace geometry encodes orientation."
- *Pre-fieldwork planning:* "Where should I focus ground-truthing?"

---

## 5. System Architecture

### 5.1 High-level pipeline

```
  +---------------------------------------------------------------+
  |                        QGIS (PyQGIS)                          |
  |                                                               |
  |  1. AOI selection (draw / import / map extent)                |
  |            |                                                  |
  |            v                                                  |
  |  2. Data Aggregation  ---- no-auth STAC ----> DEM + Sentinel-2|
  |            |                                                  |
  |            v                                                  |
  |  3. Derivative Engine  (TPI, slope, hillshade, curvature, ... |
  |            |            + S2 false-color composites)          |
  |            v                                                  |
  |  4. Trace Detection                                           |
  |       v1: classical CV   v2: ONNX semantic-seg model          |
  |            |                                                  |
  |            v          -> candidate trace polylines            |
  |  5. HUMAN REVIEW GATE  (edit / classify / accept)  <--- core  |
  |            |                                                  |
  |            v                                                  |
  |  6. Strike/Dip Engine  (sample DEM along trace -> PCA plane   |
  |            |            fit -> attitude + uncertainty)         |
  |            v                                                  |
  |  7. Output  (attitude point layer w/ geologic symbology,      |
  |              trace layer, confidence attrs, exports)          |
  +---------------------------------------------------------------+
```

A fully-automatic "one-click" mode runs steps 1-7 with default parameters and an
*optional* (skippable) review gate. The default/guided mode makes step 5
required.

### 5.2 Components

1. **Data Aggregation** - resolves the AOI, queries STAC catalogs, downloads /
   streams the needed tiles, mosaics and clips to the AOI, reprojects to an
   appropriate local UTM zone (metric CRS - required for correct plane
   geometry). Caches results.
2. **Derivative Engine** - computes a configurable multi-band input stack from
   the DEM and Sentinel-2 (Section 7).
3. **Trace Detection** - pluggable detector interface. v1 = classical computer
   vision; v2 = ONNX semantic-segmentation model. Both output raster
   probability/edge maps that are then vectorised.
4. **Vectorisation** - threshold -> skeletonise -> trace -> simplify
   (Douglas-Peucker) -> `QgsVectorLayer` of candidate polylines.
5. **Human Review Gate** - QGIS editing tools + a classification panel (bedding /
   contact / fault / drainage / reject). Persists human decisions.
6. **Strike/Dip Engine** - the scientific core (Section 6).
7. **Output & Export** - symbolised layers + tabular export.

### 5.3 Execution model

- All heavy work runs in background `QgsTask` workers so the QGIS UI never
  freezes; progress is reported incrementally (not per-item spam).
- The detector is behind an interface so classical and ML backends are
  swappable without touching the rest of the pipeline.
- Long/verbose diagnostics are logged to files; the UI shows summaries first.

---

## 6. The Science: Strike & Dip from a Trace

This is the heart of PlaneSight and the part we must get provably right.

### 6.1 Principle

A bedding plane or contact is (locally) a plane in 3D. Its line of intersection
with the topographic surface is the **trace** we detect. Sampling the DEM
elevation along that trace yields a set of 3D points that, in the ideal planar
case, all lie on the geological plane. Fitting a plane to those points recovers
the plane's orientation - i.e. strike and dip.

### 6.2 Method (PCA / SVD best-fit plane)

For a trace with sampled 3D points `P_i = (x_i, y_i, z_i)` in a metric CRS:

1. Compute the centroid `c` and the demeaned points `Q_i = P_i - c`.
2. Form the 3x3 covariance matrix `C = Q^T Q` (or take the SVD of `Q`).
3. Eigen-decompose: eigenvalues `lambda_1 >= lambda_2 >= lambda_3` with
   eigenvectors `v_1, v_2, v_3`.
4. The **plane normal** is `n = v_3` (the eigenvector of the *smallest*
   eigenvalue - the direction of least variance). This is valid **only when the
   points span two dimensions**; for straight/collinear traces `v_3` is
   ill-defined (see the conditioning guard in 6.3).
5. Derive attitude from `n = (n_x, n_y, n_z)`:
   - **Dip** = angle between the plane and horizontal = `arccos(|n_z|)` for a
     unit normal (horizontal plane -> dip 0; vertical plane -> dip 90).
   - **Dip direction (azimuth)** = horizontal azimuth of `n`, i.e.
     `atan2(n_x, n_y)` measured clockwise from North, sign-corrected so it points
     downslope.
   - **Strike** = dip direction +/- 90 degrees, reported per the chosen
     convention (right-hand rule). We will match qgSurf/GeoTrace conventions
     exactly in implementation and document the choice in code.

This generalises the three-point problem: 3 points give an exact plane; >3
points give the least-squares best-fit plane, which is what PCA computes.

### 6.3 Degeneracy & conditioning (CRITICAL - corrects an earlier model)

A single-trace plane fit is undefined whenever the sampled 3D points are
**collinear** - which is *not* the same as "low relief". The trace is a 1D space
curve; PCA recovers the normal `v_3` reliably only when the points span **two**
dimensions. Two distinct failure modes produce collinearity:

1. **Low relief** - the trace approximates a near-horizontal straight line.
2. **Straight map-view trace, even with high relief** - e.g. a contact running
   straight down a planar hillslope or along a ridge. Here
   `lambda_1 >> lambda_2 ~ lambda_3 ~ 0`; `v_3` is arbitrary within the plane
   perpendicular to the trace, and the fit returns a confident-looking but
   **geometrically meaningless** attitude.

We therefore compute **two independent metrics**, not one:

- **Conditioning / 2D-spread: `lambda_2 / lambda_1`** - does the trace span a
  plane at all? This is the **real degeneracy guard**. Low value -> normal
  undefined -> reject or flag, *regardless of relief*.
- **Planarity: `lambda_3 / lambda_2`** - *given* the trace spans 2D, how planar
  is it (vs folded/curved)? Near 0 = good planar fit.

The earlier single metric `(lambda_2 - lambda_3)/lambda_1` is **deprecated**: it
conflates "is the feature planar" with "is the fit constrained" and silently
passes straight traces. Conditioning must gate *before* planarity is meaningful.

Practical consequence for detection and trace selection: the traces worth keeping
are **sinuous traces that "V" across topography** (the rule of Vs), not merely
long or high-relief ones. Selection and review triage (Section 9.1) should rank
by conditioning, not length.

### 6.4 Uncertainty & error budget

Beyond the two metrics above, each measurement also carries residual RMS
(distance of points to the plane, metres), relief / vertical range, and sample
count / trace length. These feed a **propagated dip/strike uncertainty** that
must be reported alongside the point estimate.

The error budget is set by **DEM vertical noise**: Copernicus GLO-30 has ~2-4 m
vertical RMSE. As a trace's relief approaches the DEM noise floor, dip
uncertainty explodes.

Worked floor (order-of-magnitude, to be replaced by a measured curve in Phase 3):
a trace spanning ~15 m of relief over ~1 km horizontal, with ~3 m DEM vertical
error, has a relief signal only a few times the noise; the implied dip 1-sigma is
on the order of +/-10-20 degrees for shallow dips. **This computed number - not a
guess - sets the default confidence thresholds.** Phase 3 must produce the actual
relief/sinuosity-vs-uncertainty curve for GLO-30 and choose defaults from it.

Behaviour (per user direction): attempt a fit even in marginal geometry, but
**always emit conditioning, planarity, and propagated uncertainty**, and default
to filtering fits below the computed threshold. We never present a geometrically
unconstrained number as authoritative.

### 6.5 Along-trace variation (single point vs windowed fits)

A single planar fit to a long, sinuous trace **averages away real along-strike
dip variation** - and the geometry that best constrains the fit (large Vs across
valleys) is exactly where dip is most likely to vary. The "one attitude point per
trace at the centroid" model (Section 10) quietly bakes in a planar-and-constant
assumption.

Direction (open item, Q5): support **windowed / moving plane fits** along the
trace - N attitudes per trace from overlapping sample windows - with conditioning
and planarity flagging where the single-plane assumption breaks. A simplified
single-centroid mode remains available.

### 6.6 Synthetic validation gate (before any real DEM)

Before trusting the engine on real data, validate against a **synthetic DEM that
is a plane of known attitude** and confirm exact recovery of strike and dip -
including the dip-direction downslope sign correction in 6.2 step 5, a classic
bug nest. This is a **hard gate within Phase 3**, ahead of (not part of) the
parallel field-benchmark workstream.

**Status: PASSED (synthetic-geometry portion).** `fit_plane` is implemented
(`planesight/core/attitude/plane_fit.py`) and validated in
`tests/test_plane_fit.py`: exact recovery of dip/dip-direction/strike across a
range of attitudes, the downslope sign, and the two metrics behaving -
conditioning flags a straight down-dip trace with 866 m of relief as
unconstrained (conditioning ~1e-32), and planarity flags folded traces. Pulled
forward as an early spike (it does not depend on the detector).

**First real-data slice (Nepal, June 2026):** `scripts/nepal_slice.py` fetched the
GLO-30 DEM, reprojected to UTM 44N, sampled all 408 hand-drawn Nepal traces
(`planesight/core/attitude/sample.py`), and fit each. Result: median dip 28 deg
(IQR 18-45), mean strike ~288 deg dipping NNE - **consistent with the Himalayan
structural grain (WNW-ESE strike, north-dipping)** - with 96% passing the
conditioning guard. The premise holds on real, messy traces. Still to do:
propagate the DEM-vertical-error budget into reported uncertainty (S6.4), add
windowed fits (S6.5), and wire to detected (not hand-drawn) traces.

### 6.7 Known limitations

- Assumes the trace samples a single, locally-planar feature. Folded/polyphase
  structures violate this; conditioning + planarity are the guards, and windowed
  fits (6.5) or trace segmentation may be needed.
- Sensitive to DEM vertical error (6.4) and horizontal misregistration between
  the detected trace and the DEM.
- **Undefined for collinear / straight traces regardless of relief** (6.3).
- DEM sampling method along the trace (vertex-densification interval, bilinear vs
  nearest sampling, raw vs lightly-smoothed DEM) materially affects residual RMS
  and must be fixed and documented (open item Q9, Phase 3).

---

## 7. Input Data & The Derivative Stack

### 7.1 Data sources (no-auth, free, global)

| Layer | Source | Access |
|---|---|---|
| DEM | Copernicus GLO-30 (30 m global) | AWS Open Data (anonymous), via STAC |
| Imagery | Sentinel-2 L2A surface reflectance | **AWS Earth Search / Element84 (genuinely anonymous)** - default; via STAC |

Rationale: STAC + anonymous cloud-hosted COGs means **zero account/signup
friction** for the end user - critical for the primary persona. SRTM remains a
fallback DEM. Users may also supply their own rasters.

**"No-auth" must be verified per source (D13).** AWS Earth Search (Element84) is
genuinely anonymous and is the **default**. Microsoft Planetary Computer is
**not** plain-anonymous: its assets require SAS-token signing via a free,
account-less token endpoint - plain `/vsicurl` streaming fails against MPC unless
URLs are signed (`planetary-computer.sign()`). Treat MPC as a secondary source
behind a signing shim, not a drop-in no-auth COG store. (~80% confidence on
current MPC behaviour; verify before relying on it.)

### 7.2 The input stack (configurable, with curated defaults)

The detector consumes a multi-band stack, not a single greyscale image (a key
upgrade over the 2020 prototype). Default candidates:

- **DEM-derived:** elevation, **TPI** (Topographic Position Index - already
  proven useful in the prototype), slope, multi-azimuth hillshade, plan/profile
  curvature. Phase 1 will evaluate additional derivatives (openness, sky-view
  factor, etc.).
- **Sentinel-2 derived:** a geology-oriented false-color composite (e.g. a SWIR
  combination such as bands 12-11-2, to be selected in Phase 1), possibly band
  ratios known to aid lithological discrimination.

All channels are tunable; the defaults exist so the first-time user gets a good
result without configuring anything.

> **Phase 1 explicitly investigates which derivatives are most diagnostic** for
> bedding/contacts and how best to port modern edge-detection techniques to
> multi-band geospatial data. The defaults above are the starting hypothesis, not
> the final answer.

### 7.3 Aggregation constraints (must be designed, not assumed)

- **CRS policy (Q6).** "Regional scale anywhere on Earth" conflicts with
  "reproject to a single local UTM zone": large/regional AOIs straddle UTM zones,
  and polar AOIs (>84 N / >80 S) have no UTM at all. Policy: derive a **per-AOI
  custom projection** (transverse Mercator or Lambert centred on the AOI), or
  **tile** the AOI - never assume one fixed UTM zone. The metric-CRS requirement
  for plane fitting (Section 6) is satisfied by this per-AOI projection.
- **Cloud cover & dry-season selection (Q7, D15).** Geology reads best with
  minimal cloud AND in the dry season (less vegetation/snow). Scene selection now
  (a) hard-filters to **<5% cloud by default** (configurable; escalates the cap
  for persistently cloudy AOIs and logs it), and (b) prefers the locality's
  **dry season, derived empirically** from the AOI's own clear-scene histogram -
  the months with the most near-clear scenes - so no global climate model is
  needed. Verified live: correctly recovers Oct-Dec for arid Pakistan and Nov for
  monsoonal Nepal. **Still P0d:** multi-scene SCL/QA cloud *masking* + temporal
  *compositing* (seasonal median) on top of this selection.
- **Reproducibility.** STAC query results drift as catalogs update, so the same
  AOI can yield different answers over time. **Pin resolved scene IDs and
  acquisition dates into the output provenance** so results reproduce - essential
  for a science tool.

---

## 8. Detection Strategy

### 8.1 Two-stage rollout

- **v1 - Classical computer vision.** The best technique identified in Phase 1
  (candidates: Canny, Hough, ridge/valley extraction, phase congruency) applied
  to the derivative stack. No machine-learning dependency, works globally on day
  one, highly tunable. Ships first.
- **v2 - Supervised semantic segmentation.** Built from scratch (the 2020 Mask
  R-CNN approach is dated and ill-suited to thin curvilinear features). A
  segmentation model (e.g. U-Net / lightweight transformer) predicts a per-pixel
  trace probability map, which is then skeletonised and vectorised - the approach
  validated by Aghaee et al. 2021. Shipped as an optional "smart detect" upgrade.

### 8.2 ML runtime (decided)

- Models are exported to **ONNX** and run with **onnxruntime** - a single
  lightweight, CPU-friendly dependency. No PyTorch/TensorFlow on the end user's
  machine, fully offline, free to distribute. This is the only runtime option
  that is simultaneously zero-friction for the user and zero-cost for the
  maintainer.

### 8.3 Training data - the bootstrap strategy (HIGH RISK / go-no-go gate)

Labeled bedding/contact training sets essentially do not exist - the central ML
risk. PlaneSight's *proposed* unlock:

> **Drape existing published vector geology maps (contact/bedding linework) over
> the DEM/imagery stack to auto-generate `(input stack, trace mask)` training
> pairs at scale - no manual digitising.**

The entire v2 ML story rests on this, so it is a **go/no-go gate evaluated in
Phase 1 (Q10)**, not an assumed deliverable. Failure modes to test before
committing:

- **Concealed / inferred contacts** (dashed/dotted on maps) have no surface
  expression. Draping them creates labels where there is no detectable signal -
  *actively poisoning* a thin-feature segmenter. Must filter by line-type
  attributes, which differ across every source schema.
- **Thin-feature registration catastrophe.** A 1:100k contact is positioned to
  ~50-100 m ~ 2-3 GLO-30 pixels; target features are 1-2 pixels wide. A 2-pixel
  label offset on a 1-pixel feature yields *mostly-wrong* labels, not merely
  noisy ones.
- **Generalisation smooths away the Vs** - removing exactly the trace sinuosity
  the strike/dip engine depends on (Section 6.3).
- **Heterogeneous licensing / datums** across USGS / GSC / Macrostrat /
  OneGeology (see Section 16 licensing note).

Note: **map2loop already does map-draping for orientation extraction** - review
what they learned about map-linework reliability before building this.

Label sources and the active-learning loop:

- Hand-labeled seed: ~2,400 lines across Canada / Nepal / Pakistan - see
  `data/DATA_MANIFEST.md`.
- Candidate map sources: USGS, GSC, Macrostrat, OneGeology.
- In-plugin labeling + active learning (accepted/edited traces become training
  data over time).

**Fallback if the gate fails:** rely on the hand-labeled seed + in-plugin active
learning, and keep v1 classical as the shipping detector. v2 is never assumed.

---

## 9. Human-in-the-Loop Review (Required by Design)

Because automated detectors cannot reliably distinguish bedding from faults,
drainage, and roads, the review gate is a core feature:

- After detection, candidate traces load as an editable layer.
- The user can add, delete, snap, split, and merge traces using native QGIS
  editing.
- A classification panel lets the user label each trace (bedding / contact /
  fault / drainage / reject). Only accepted geological traces flow to the
  strike/dip engine.
- Decisions are persisted (and feed the active-learning loop in Section 8.3).

Two modes:
- **Guided (default):** review gate required.
- **One-click:** runs end-to-end with defaults; review optional/skippable.

### 9.1 Triage design (resolving the scale-vs-gate tension)

The headline promise (automatic, regional, mass-compute) and the required
per-trace human gate are in **direct tension**: per-trace bedding-vs-fault
classification is O(n) manual work and, at thousands of traces, becomes the
throughput bottleneck that undercuts the scale value-proposition. The gate must
therefore be a **triage system, not just an editing panel**:

- **Confidence-ranked review queue** - surface low-conditioning / low-confidence
  traces first; let high-confidence ones pass with a glance.
- **Batch operations** - accept/reject by region, cluster, or attribute; "review
  only traces above conditioning/planarity X".
- **Geomorphic pre-filters** - drainage networks are **computable from the DEM**
  (flow accumulation) and can be auto-flagged/rejected before human review;
  similar pre-filters for roads where ancillary data exists. This removes a large
  fraction of non-geological lines automatically.
- **Active-learning payoff** - human decisions feed v2, so manual effort
  *decreases over time* rather than being pure O(n) grind.

Without triage, the workflow degrades to "mass-detect, then manually grind" -
closer to GeoTrace's user-seeded model than intended.

---

## 10. Outputs

- **Attitude point layer** - attitude measurements with attributes: strike, dip,
  dip direction, **conditioning (lambda_2/lambda_1)**, **planarity
  (lambda_3/lambda_2)**, residual RMS, relief, **propagated dip/strike
  uncertainty**, sample count, classification, detector/version. Styled with
  standard geologic strike/dip symbology. Output is **one point per fit window**
  (Section 6.5): one-per-trace in simplified mode, N-per-trace when windowed
  fitting is enabled.
- **Trace polyline layer** - the (reviewed) traces with their classification.
- **Exports** - GeoPackage (primary), CSV/table for attitudes, with a clear,
  documented schema.

**Project-state model (Q8, open).** Detection, human edits/classifications, and
fits must persist in a **versioned project-state schema**. Key unspecified
decision: when the user **re-runs detection with new parameters**, are prior
human classifications/edits **preserved or discarded**? Needs an explicit
re-detect-vs-preserve-edits policy with state versioning.

---

## 11. Technology Stack

| Concern | Choice |
|---|---|
| Host | QGIS plugin, PyQGIS, Qt UI (Processing framework + custom dialogs) |
| Concurrency | `QgsTask` background workers |
| Raster I/O & warp | GDAL (bundled with QGIS), rasterio |
| Data discovery | STAC (`pystac-client`); COG streaming via `/vsicurl` |
| Numerics | numpy, scipy (both bundled with QGIS) |
| Classical CV (v1) | **numpy / scipy / GDAL only** - `scipy.ndimage` filters, FFT for phase congruency; no opencv/scikit-image |
| Vectorisation | `scipy.ndimage` + shapely + QGIS geometry (skeletonise via scipy where possible) |
| ML inference (v2, optional) | ONNX + onnxruntime |
| ML training (offline, dev only) | PyTorch (not shipped to users) |

**Dependency-free v1 (decided, D11).** Installing third-party packages into
QGIS's bundled Python (OSGeo4W on Windows) is historically a top cause of plugin
breakage (ABI mismatches, missing compilers, write-permission issues) - and for
the primary "first-time user" persona it is the **most likely reason the plugin
fails to install at all**. QGIS bundles numpy, scipy and GDAL but **not** opencv
or scikit-image. Therefore **v1 is implemented with numpy/scipy/GDAL only** and
ships truly friction-free. Heavier deps (onnxruntime; opencv/scikit-image if ever
needed) are deferred to the **optional ML upgrade**, gated behind explicit user
opt-in where a larger install is justifiable. This makes the classical-first
decision (D2) a *packaging* strategy as well as a science one.

`pystac-client` is needed for data discovery (Section 7); evaluate whether plain
`requests` against the STAC API suffices to keep even Phase 0 dependency-free.

Packaging note: onnxruntime wheels are tens of MB per platform x3, inflating the
plugin-repo footprint - the ML model and runtime should likely be **fetched on
first use** rather than bundled.

---

## 12. Phased Roadmap

Phases are sequenced by dependency. Each delivers something independently
demonstrable.

| Phase | Goal | Key deliverable |
|---|---|---|
| **0 - Foundations & data backbone** | Plugin skeleton, repo, CI, dev env; AOI -> STAC fetch -> reproject to **per-AOI metric CRS** -> derivative stack; cloud-mask/composite (7.3) | Draw an AOI, get a clean multi-band raster stack in QGIS |
| **1 - Research spike** | Lit + experimental review: most-diagnostic derivatives; modern edge/line detection (numpy/scipy-implementable); porting to multi-band geodata; **training-bootstrap go/no-go gate** | Short report -> recommended input stack, detector, and bootstrap **verdict** |
| **2 - Classical detection MVP** | Implement Phase 1's chosen classical detector + vectorisation (**dependency-free**) | AOI -> auto-drawn candidate trace polylines |
| **3 - Strike/dip engine** | DEM sampling + PCA fit; **conditioning + planarity metrics; error budget; synthetic-DEM recovery gate**; symbolised output | Full end-to-end pipeline (classical), **validated on synthetic data first** |
| **4 - Human-in-the-loop UX** | Editing, classification panel, tunable parameters, one-click vs guided | Reviewable, tunable workflow |
| **5 - Supervised ML detector** | Assemble bootstrapped training set; train; export ONNX; ship as optional smart-detect | ML detector upgrade |
| **6 - Packaging & release** | QGIS plugin-repo submission, docs, first-time-user tutorial | Public, installable plugin |

**Parallel - Validation workstream (from Phase 3):** source published structural
maps + field strike/dip; benchmark PlaneSight predictions; track accuracy by
relief/terrain class.

---

## 13. Risks & Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| **Collinear/straight-trace degeneracy** (undefined normal, even at high relief) | High | Conditioning metric `lambda_2/lambda_1` as the primary guard (6.3); reject/flag below threshold |
| **DEM vertical noise vs low relief** (dip uncertainty explodes) | High | Computed error budget sets defaults (6.4); propagate & report uncertainty |
| Semantic labeling (bedding vs fault/drainage/road) | High | Human gate + **triage** (9.1); geomorphic auto-pre-filters |
| **Human-gate throughput is O(n)** at regional scale | High | Triage: ranked queues, batch ops, drainage auto-reject, active learning (9.1) |
| **Training bootstrap unreliable** (concealed contacts, registration, generalisation) | High | **Go/no-go gate** in Phase 1 (8.3); fallback to seed + active learning |
| QGIS Python dependency packaging | High | **Dependency-free v1** (D11; numpy/scipy/GDAL only); heavy deps gated behind optional ML |
| Imagery cloud cover wrecks the stack | Medium | Cloud masking + temporal/seasonal compositing + scene selection (7.3) |
| CRS/zone limits at regional/polar scale | Medium | Per-AOI custom projection or tiling (7.3) |
| Reproducibility drift (STAC catalog changes) | Medium | Pin scene IDs/dates in provenance (7.3) |
| MPC not truly no-auth | Low | Default to AWS Earth Search (anonymous); MPC behind a signing shim (7.1) |
| Scope creep into 3D modelling | Low | Explicit out-of-scope list (Section 3); export to map2loop/Loop3D instead |

---

## 14. Decisions Log

Confirmed decisions:

- **D1** - Plane fit via PCA/SVD eigenvector method; attempt even in low relief
  but always emit uncertainty. (Confirmed)
- **D2** - Supervised ML is the long-term detector, but **v1 ships classical**;
  ML is a later upgrade. (Confirmed)
- **D3** - ML runtime is **ONNX + onnxruntime**, local, offline. (Confirmed)
- **D4** - Replace Mask R-CNN with a **from-scratch semantic-segmentation ->
  vectorise** approach. (Confirmed)
- **D5** - Data access via **no-auth STAC** (Copernicus GLO-30 + Sentinel-2 L2A).
  (Confirmed)
- **D6** - Distributable, public QGIS plugin is the goal (not just a personal
  script). (Confirmed)
- **D7** - Human-in-the-loop review gate is required by default; a one-click
  auto mode is also provided. (Confirmed)
- **D8** - Input is a configurable multi-band stack (DEM derivatives + S2
  composite), not a single greyscale image. (Confirmed)
- **D9** - Phase 1 is a research spike that precedes committing to specific
  derivatives and detection techniques. (Confirmed)
- **D10** - Project name **PlaneSight**; verified clear on the QGIS plugin repo,
  PyPI, and GitHub (June 2026). (Confirmed)
- **D11** - **v1 is dependency-free** (numpy/scipy/GDAL only); heavy deps
  (onnxruntime, opencv) deferred to the optional ML upgrade. (Confirmed - review)
- **D12** - Degeneracy is guarded by **two metrics**: conditioning
  `lambda_2/lambda_1` (primary) + planarity `lambda_3/lambda_2`; the old single
  metric `(lambda_2-lambda_3)/lambda_1` is deprecated. (Confirmed - review)
- **D13** - Default imagery source is **AWS Earth Search (anonymous)**; MPC only
  behind a signing shim. (Confirmed - review)
- **D14** - **Synthetic-DEM recovery** is a hard gate inside Phase 3, before
  field validation. (Confirmed - review)
- **D15** - Sentinel-2 selection defaults to **<5% cloud** and prefers the
  **empirically-detected dry season** (clearest-months histogram per AOI), not a
  hardcoded climate model. (Confirmed - implemented & live-verified)

Open questions:

- **Q1** - Exact default Sentinel-2 band composite(s) and the final derivative
  set - resolved by Phase 1.
- **Q2** - Specific classical detection algorithm for v1 (numpy/scipy-feasible)
  - resolved by Phase 1.
- **Q3** - Dependency-bootstrap mechanism for the *optional ML upgrade* (fetch on
  first use vs bundle) - prototype in Phase 5.
- **Q4** - Validation datasets / regions - user to source.
- **Q5** - Windowed/moving plane fits (N attitudes per trace) vs single-centroid
  output: default mode and window size (6.5).
- **Q6** - CRS policy: per-AOI custom TM/Lambert vs tiling; behaviour at zone
  boundaries and poles (7.3).
- **Q7** - Cloud handling: scene selection (<5% + dry-season) is done (D15);
  remaining is multi-scene SCL/QA masking + temporal compositing in P0d (7.3).
- **Q8** - Re-detect vs preserve-edits project-state policy + state-schema
  versioning (Section 10).
- **Q9** - DEM sampling method along traces (densify interval, bilinear vs
  nearest, raw vs smoothed) (6.7).
- **Q10** - Training-bootstrap **go/no-go** - pending Phase 1 (8.3).

---

## 15. Provenance

This project builds on a 2020 prototype ("Strike Dip Predict") that generated
Mask R-CNN training labels from QGIS-exported INTERLIS polylines over a single
1-degree Canada DEM tile (~1,160 hand-labeled traces). PlaneSight is a
ground-up redesign: it discards the brittle INTERLIS text-parsing and hand-rolled
georeferencing in favour of native PyQGIS/GDAL, replaces Mask R-CNN with modern
segmentation, adds the global data-aggregation front-end, and - crucially -
implements the strike/dip computation that the prototype never reached. The
prototype's hand-labeled traces are retained as seed training data.

---

## 16. References

**Licensing note.** The plugin itself is GPL (QGIS ecosystem). A *shipped trained
model* inherits obligations from its training data: USGS maps are public domain,
but GSC (OGL-Canada), Macrostrat (CC-BY), and OneGeology (often restrictive) can
encumber a redistributed model. The model's data provenance and resulting licence
must be stated explicitly before any model is shipped (ties to the Section 8.3
bootstrap gate).

- GeoTrace - https://plugins.qgis.org/plugins/GeoTrace/ ; https://github.com/lachlangrose/GeoTrace
- qgSurf - https://plugins.qgis.org/plugins/qgSurf/ ; https://gitlab.com/mauroalberti/qgSurf ; https://peerj.com/preprints/27694/
- map2loop / Loop3D - https://gmd.copernicus.org/articles/14/5063/2021/ ; https://loop3d.org/map2loop/
- Aghaee et al. 2021, "A CNN for semi-automated lineament detection and
  vectorisation," Computers & Geosciences 151:104724 -
  https://github.com/aminrd/LineamentLearning
- Allmendinger, GMDE / Stereonet - https://www.rickallmendinger.net/
- Martinez-Torres et al. 2012, Computers & Geosciences v42 p.200 -
  https://www.researchgate.net/publication/234191038
- Copernicus GLO-30 DEM (AWS Open Data); Sentinel-2 L2A (AWS / Microsoft
  Planetary Computer)
