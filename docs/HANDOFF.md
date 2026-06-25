# PlaneSight - Session Handoff

**Updated:** 2026-06-25. NEXT PHASE: the QGIS plugin GUI (see end).

PlaneSight = a QGIS plugin that, for any AOI, aggregates global DEM (Copernicus
GLO-30) + Sentinel-2, auto-detects geological bedding/contact traces, and computes
strike/dip en masse from the DEM geometry. Everything to date is a validated,
**headless** Python core (`planesight/core/`) - the QGIS GUI is not yet built. Read
`ARCHITECTURE.md` for design/decisions (D1-D15), `docs/PHASE1_REPORT.md` +
`docs/BOOTSTRAP_VERDICT.md` for the early science, and `docs/ATTITUDE_RULES.md` for the
data-driven thresholds.

---

## TL;DR state

- **All science work is now MERGED to `main`** (PRs #1-#5). The headless core is
  complete and validated end-to-end; the **QGIS GUI is the next phase and is NOT yet
  built**. Merged this cycle on top of Phases 0-2:
  1. the **complete + BOUNDED drainage arc** - flow-accumulation filter, verified +
     hardened + integrated as a review-flag, refined to overlap-flag + confidence-rank
     (`3em` CLOSED); then *generalized*: transfers to Canada, **fails on low-relief
     Pakistan** (sweep `amn`), and the spectral fallback was **probed + falsified**
     (`0bf`). Net: works in steep/dissected terrain; low-relief has no auto drainage
     filter but **degrades gracefully** (review-flag + gates). See "Strategic landing".
  2. **three improvements** - data-driven attitude rules (`5ug`), continuity linking
     (`zod`), a correlated-error uncertainty floor (`85g`);
  3. **process tooling** - verification hooks, a parallelization kit, an adversarial
     skeptic verifier. The parallel worktree workflow was proven on **5 lanes**
     (`5ug`/`zod`/`85g`/`amn`/`0bf`), each independently adversarially verified.
- **~195 pure tests green; ruff clean.** Headless GDAL via micromamba env `gdal`:
  `MAMBA_ROOT_PREFIX=$HOME/micromamba PYTHONPATH=$PWD $HOME/bin/micromamba run -n gdal python ...`
- **Working tree is on `main`; no open worktrees.** Start the GUI from a fresh branch.

---

## What's done (verified)

| Capability | Where | Status |
|---|---|---|
| AOI -> STAC -> GLO-30 DEM + Sentinel-2 (anonymous AWS) | `core/data/` | on main |
| Derivative engine (slope/aspect/hillshade/tpi/curvature; S2 indices; named stacks) | `core/derivatives/` | on main |
| Plane-fit strike/dip (SVD) + conditioning/planarity/**map_conditioning** + MC uncertainty | `core/attitude/plane_fit.py` | on main |
| Positive-unlabeled scoring (recall-at-budget) + label-free linearity | `core/detect/score.py`, `linearity.py` | on main |
| Canny + vectorize (thin/trace/simplify) + `ClassicalTraceDetector` | `core/detect/` | on main |
| **Drainage filter** (D8 flow accum, depression-filled, overlap-flag + rank) | `core/detect/drainage.py` | **done (`3em`)** |
| **Continuity linking** (`link_polylines`, 3-condition guard) | `core/detect/vectorize.py` | done (`zod`) |
| **Correlated-error uncertainty floor** (`correlation_length`) | `core/attitude/plane_fit.py` | done (`85g`) |
| **Data-driven attitude rules** (local variability, morphology/length priors) | `core/attitude/variability.py`, `docs/ATTITUDE_RULES.md` | done (`5ug`) |
| End-to-end auto strike/dip on Nepal (drainage-flag + link + ranked queue) | `scripts/detect_attitudes_nepal.py` | done (`61f`) |
| **Stereonet math** (equal-area projection, Fisher mean/alpha95, orientation tensor, axial mean, fold axis + Woodcock K, rose) | `core/structural/stereonet.py`, `docs/STEREONET_MATH.md` | **done (`8et`)** |

---

## The drainage arc (the main story this cycle)

The detector is a generic topographic break-line detector, so in dissected terrain it
traces **creeks**, not just contacts. The fix and its validation:

1. **Flow-accumulation filter** (`drainage.py`): D8 `flow_directions/accumulation/
   azimuth`, **`fill_depressions`** (priority-flood + epsilon) and **`block_mean`**
   downsample via **`flow_network`** (`j8t` hardening - keeps channels continuous on
   low relief), `channel_network`, `channel_proximity`.
2. **Verified (`5p3`):** geologist audited 30 random flagged traces -> all creeks
   (removed set clean). Conditioned false-negative 25% (6/24 valley-overlapping hand
   traces, small N) - the old "1.3%/408" was denominator dilution.
3. **Integrated as a review-FLAG, not a delete (`xx2`):** flagged traces are retained,
   excluded from the attitude stats, and routed to a review queue. Strike-valley
   contacts (rare, recoverable) are the accepted false-negative cost.
4. **Recall-gap investigation (`4l8`):** the geologist still saw creeks in the kept set.
   Measured: ~848 on-channel traces (10.6% length) that flow-**alignment** missed
   because meander sinuosity drops the alignment score. Discriminator test:
   **convexity FAILS** even as a per-trace aggregate (confirms the old curvature-sign
   dead-end); **elevation monotonicity** (creek descends a thalweg; a contact-V crosses
   it) is the right, sinuosity-robust signal. Geologist verdict: the on-channel band is
   a **confidence** problem - flag it aggressively, rescue only the big cross-cutters.
5. **Refined rule (`61f`):** `flag_drainage` now flags on **overlap** (sinuosity-robust)
   and returns `DrainageFlag(is_drainage, overlap, monotonicity, length, rank)` with
   **rank = length x (1 - monotonicity)** so a long cross-cutter sits atop the rescue
   queue and a creek (mono~1) sinks to rank 0. `link_polylines` runs on the KEPT set
   **after** drainage removal (the contract), then the fit. Nepal end-to-end:
   7063 -> 3323 flagged / 3740 kept -> 3680 linked -> 1599 reliable; dominant strike
   **83 deg** (Himalayan grain preserved).

## Other science / engineering landed

- **Continuity (`zod`):** `link_polylines(polylines, max_gap_px=5, max_angle_deg=20)`
  rejoins fragments only on a **three-condition** undirected (mod-180) collinearity
  test - the two end-tangents collinear AND the gap vector collinear with each tangent.
  The gap-vector guard is what stops parallel-offset bedding layers from merging.
- **Uncertainty floor (`85g`):** `fit_plane(..., correlation_length=500.0)` adds a
  correlated random-tilt term so the MC budget stops averaging down ~1/sqrt(N). Floors
  the dense-trace estimate (~0.66 deg vs the old ~0.034 deg at n=1000). De-confounds
  `5ug`'s length-reliability metric. Disable with `None`/`<=0` (bit-for-bit legacy).
- **Attitude rules (`5ug`, `docs/ATTITUDE_RULES.md`):** implausible-local-outlier bar
  **strike Δ > 60° / dip Δ > 35°** (pooled p95 @ 1 km; review-trigger only, never
  auto-reject; median local strike dev ~8 deg confirms the regional grain is locally
  smooth). Morphology is V-dominant (56-72%), near-vertical `straight` rare (<3%).

## Process tooling added (the parallel-workflow experiment)

- **Verification hooks** (`.claude/settings.json` + `.claude/hooks/`): PostToolUse ruff
  autofix, PreToolUse blocks `rm -rf`/force-push/push-to-main, **Stop gate** runs the
  pure pytest suite (project-scoped; activate by launching Claude Code from the repo).
- **Parallelization kit:** `/handoff <bead>` writes a task brief; `scripts/spawn_task.sh
  <branch> <bead>` creates an isolated worktree (`BEADS_DB` wired to the canonical DB) +
  kickoff prompt; `docs/PARALLEL_WORKFLOW.md` is the protocol + adversarial-verify
  checklist. **Validated:** `5ug`, `zod`, `85g` all ran as fresh parallel sessions and
  passed an independent adversarial pass (each had a real soft spot the generator
  presented as settled).
- **Skeptic verifier (`gas`):** `.claude/agents/skeptic.md` + `/verify` - a fresh,
  unanchored agent that re-runs the evidence to break an empirical/geological claim
  (methodology + geology checklists, the `5ug` thresholds + caveats).

---

## Strategic landing: low-relief drainage (BOUNDED, negative result)

The drainage filter was chased to its terrain boundary and the boundary is now *measured*:

- **Detection** works everywhere (DEM-curvature is the strongest signal in all regions,
  including Pakistan - Phase 1; the "Pakistan is spectral-dominated" idea was a coverage
  artifact, debunked in Phase 1 and mistakenly revived then re-corrected this cycle).
- **Drainage exclusion works in steep/dissected terrain** (Nepal/Canada). On **low-relief
  arid terrain it has NO working auto-method**: flow-accumulation over-connects (blobs to
  33% of the map, no knee - `amn`), and the **spectral NDMI fallback was falsified** on the
  real per-trace use case (`0bf`: AUC collapses 0.94->0.64; the 0.94 was a pixel-proxy AND
  a **coastal-water artifact** - the Makran-coast AOI's high-NDMI "creeks" are shoreline,
  not riparian moisture). The moisture idea is *unproven, not disproven* - a clean test
  needs a **non-coastal arid AOI**.
- **But it degrades gracefully:** the filter is a review-FLAG, so low-relief is
  *unfiltered-but-safe* (the conditioning + `map_conditioning` + `5ug` smoothness gates
  still apply). Pragmatic stance: **rely on the gates + defer low-relief** ("defensible
  terrains first"), now measured rather than assumed.
- **Lesson, reinforced:** two confident from-memory/headline claims this cycle ("spectral-
  dominated", "NDMI works in arid 0.94") were both wrong and both caught by the
  validate-before-build gate. Keep gating empirical claims.

---

## Open follow-ups

- **`pie`** - tune `link_polylines`: under-links at defaults on real Nepal (60/3740
  merged); raise gap / link pre-simplification / verify no over-merge.
- **`5b8`** - calibrate `correlation_length` (`L_c`) from an empirical GLO-30 error
  variogram before any hard *absolute*-uncertainty threshold; model misregistration.
- **`gjm`** - horizontal-bedding override (Grand Canyon calibration): topography-
  following is suspect EXCEPT genuine contour-parallel horizontal bedding.
- **Pakistan/Canada drainage sweep (`planesight-amn`, DONE)** - the Nepal `accum=15`
  knee **transfers to Canada** (14.8% channel, clean dendritic creek panels) but **NOT
  to Pakistan** (33% channel at 15; needs `accum~115-120`, and a high-end sweep shows
  Pakistan has **no knee at all** - flow-accum is a blunt discriminator on low-relief
  arid terrain). Verdict + per-region accum + caveats in `docs/DRAINAGE_SWEEP.md`;
  panels under `debug/drainage_{test,verify,residual}/{pakistan,canada}/`. Follow-up
  `planesight-l2c`: replace fixed `accum` with a density-targeting rule (or a different
  low-relief channel definition) in `core/detect/drainage.py`.
- **Spectral NDMI validation gate (`planesight-0bf`, DONE — NEGATIVE result):** the probe's
  NDMI AUC 0.94 was a **proxy artifact** (channel *pixels* vs hand *contacts*). On the REAL
  use case — per-trace mean-NDMI over DETECTED Pakistan traces — separation **collapses to
  AUC 0.64** (creek-proxy vs hand contacts) / **0.58** vs off-channel detected. At the best
  threshold (NDMI ≥ −0.165) it catches only 34% of creek-proxy traces, flags 30% of the map,
  and its high-NDMI end is **coastal water shorelines, not creeks** (Makran AOI). Its
  apparently-low contact-FN (5% on at-risk vs the flow filter's 45%) is a *near-chance*
  artifact, not discrimination. **Verdict: per-trace NDMI does NOT hold — do not build the
  l2c/option-A terrain-adaptive selector on it.** Details + panels: `docs/SPECTRAL_VALIDATE.md`,
  `scripts/spectral_validate.py`, `debug/spectral_validate/pakistan/`. Next probe to try:
  per-trace **elevation monotonicity** (the `4l8` signal) on Pakistan; NDWI water-mask first.
- **Infra:** `bcn` (S2 cloud compositing), `gj9` (per-AOI CRS), `1dg` (training
  GeoPackage). Deferred: `luj` (shield AOI), `eu3` (bootstrap confirmation).
- (Infra above is optional polish; the next real phase is the GUI - see below.)

---

## NEXT PHASE: QGIS plugin GUI (start here) - PLANNED, epic `planesight-pzz`

The headless core is complete and validated; the GUI turns it into a usable tool. The GUI
plan was designed 2026-06-25 (this session) and is tracked as epic **`planesight-pzz`**.
Read `ARCHITECTURE.md` S5/S9/S10/S11. The Phase-0 plugin skeleton already exists:
`planesight/plugin.py` (toolbar action), `planesight/gui/main_dialog.py` (placeholder
`QDialog` to replace), `planesight/tasks/base.py` (a working `PlaneSightTask` QgsTask
wrapper - off-thread work, cancel, main-thread result delivery).

**Reframed around FOUR standalone tools, not one monolithic pipeline** (each independently
useful + shippable, so the plugin has value before detection is ever accurate):
1. **Aggregate** (#1) - AOI -> styled DEM/S2/derivative layers (`core/data`, `core/derivatives`).
2. **Measure** (#3) - strike/dip on ANY traces incl. user hand-drawn (decoupled from detection).
3. **Analyze** (#4) - stereonet + Fisher mean + **fold axis** (girdle eigen-analysis); map<->net
   selection linkage. Stereonet MATH in `core/` (dep-free, tested); rendering decision deferred.
4. **Detect** (#2) - detector -> drainage-flag -> link -> confidence-ranked review/triage gate.
   Its advanced form is **example-driven** ("trace a few, find the rest" = the active-learning
   loop, S8.3), gated by a research PROBE first (`planesight-ayn`).

**Locked design decisions (2026-06-25):** dockwidget shell (NOT processing-provider for v1);
thin-vertical-slice first; guided-only (one-click later); ranked triage panel (S9.1 - the
differentiator). PyQGIS/Qt confined to `gui/` + `plugin.py`; **core stays dependency-free**.
Per-AOI UTM CRS from AOI centroid (`gj9`); re-detect default = preserve/merge edits (Q8).

**Milestones (dependency-ordered: Aggregate -> Measure -> Analyze -> Detect -> example-driven):**
- **M0** `4vw` - in-QGIS pipeline checkpoint. **DONE** (PR #8). `scripts/qgis_console_checkpoint.py`
  runs the full pipeline in QGIS; orchestration extracted to `core/pipeline.py`.
- **M1** `vye` - data aggregator slice. **DONE** (PR #8). `PlaneSightDockWidget` Data tab: AOI ->
  fetch DEM/S2/derivatives -> styled layers grouped (at the BOTTOM of the tree) by AOI.
- **M2** `o9m` - strike/dip on supplied traces. **DONE** (commit 921c274; PR pending). Strike/Dip
  tab: any trace layer + DEM -> `fit_traces` -> qgSurf SVG attitude markers. Decoupled from
  detection. `core.pipeline` made scipy-free (lazy detection imports) - this fixed a UI freeze.
- **M3** `8et`(math done, PR #7) - structural analysis panel. **NEXT** - wire a new "Analyze" tab
  to the merged `core/structural/stereonet.py`: plot selected attitudes -> Fisher mean + fold
  axis -> map<->stereonet selection linkage. Rendering decision (matplotlib vs QPainter) here.
- **M4** `oey` - detection + review/triage gate (first scipy user; lazy + off-thread).
- **PROBE** `ayn` - example-driven generalization research gate (ready anytime; gates M5).
- **M5** `xyy` - example-driven detection. **M6** `2pv` - persistence + export + project-state.

**GUI dev environment (established this session - IMPORTANT for resuming):**
- **Target QGIS = 3.44.11 LTR** (was 3.28). Plugin `metadata.txt` still says min 3.22 - reconcile
  vs the 3.44-only `Qgis.LabelPlacement.OverPoint` enum in `styling.py` (`planesight-qwe`).
- **Headless QGIS test rig:** micromamba env `qgis` (`qgis=3.44` + `scipy`); run offscreen with
  `QT_QPA_PLATFORM=offscreen MAMBA_ROOT_PREFIX=$HOME/micromamba PYTHONPATH=/mnt/c/PlaneSight
  $HOME/bin/micromamba run -n qgis python debug/headless_*.py`. Renders symbology/dock/tasks
  offscreen - caught the 3.44 enum + the scipy freeze WITHOUT crashing the desktop. Use it to
  verify every GUI change before deploying.
- **Deploy/run:** copy `planesight/` into the QGIS profile's `python/plugins/` (no hot-reload -
  restart QGIS or use the `debug/*_probe.py` reload-probes). The user's legacy `default` profile
  (60+ old plugins) hangs 3.44 -> dev uses a clean **`ps44`** profile:
  `"C:\Program Files\QGIS 3.44.11\bin\qgis-ltr.bat" --profile ps44`.

**Parallelization plan (two-track, NOT a 5-lane fan-out):** UI work loses the headless+CI
verification backbone that made the science lanes safe (no QGIS in CI; validation is serial +
manual; the dockwidget shell is a shared file). So: **Track 1 (serial spine)** M0 -> M1 ->
panels, human-validated in QGIS; **Track 2 (parallel headless lane)** the **stereonet math**
(M3's `core/structural/`) pulled forward - pure numpy, fully test-gated, zero UI dependency,
retires M3's risk early (same move as pulling `plane_fit`/`lph` forward). The example-driven
PROBE is a good optional 2nd headless lane (lower urgency).

## Beads map

- **`3em`** (drainage epic) CLOSED: `5p3` + `j8t` + `xx2` + `zod` + `4l8` + `61f`.
- **`gas`** (skeptic) CLOSED; **`85g`** (uncertainty) CLOSED; **`5ug`** (attitude rules)
  CLOSED. `lph` (Phase 3 strike/dip) in_progress (core done).
- Open: `pie`, `5b8`, `gjm`, `bcn`, `gj9`, `1dg`; deferred `luj`, `eu3`. Phases 0/1/2
  epics closed.

## Working agreements / process notes

- **WSL paths only** (`/mnt/c/...`). No emojis/Unicode - use `[OK]`/`[ERROR]`.
- **Run ruff before committing.** Commit/push only when asked; **branch first**
  (push to `main` is policy-blocked - feature branch + PR, merge with `gh`).
- **Background runs sometimes fail transiently** (S2-fetch network blips) - re-run in
  the foreground to confirm before assuming a code bug.
- **Lesson (load-bearing):** trust the *measurement*, not the headline. The coverage
  bug, the FN-dilution, and the length-threshold optimism all produced confident-but-
  wrong numbers caught only by a critical re-check. Verify before concluding; the
  `gas` skeptic / `/verify` exists for exactly this.
