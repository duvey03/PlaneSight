# PlaneSight - Session Handoff

**Updated:** 2026-06-24 (+ Pakistan/Canada drainage transferability sweep, `amn`).

PlaneSight = a QGIS plugin that, for any AOI, aggregates global DEM (Copernicus
GLO-30) + Sentinel-2, auto-detects geological bedding/contact traces, and computes
strike/dip en masse from the DEM geometry. Everything to date is a validated,
**headless** Python core (`planesight/core/`) - the QGIS GUI is not yet built. Read
`ARCHITECTURE.md` for design/decisions (D1-D15), `docs/PHASE1_REPORT.md` +
`docs/BOOTSTRAP_VERDICT.md` for the early science, and `docs/ATTITUDE_RULES.md` for the
data-driven thresholds.

---

## TL;DR state

- **Phases 0-2 are on `main`** (PRs #1, #2): data fetch, derivative engine, plane-fit
  strike/dip, classical detector, scoring.
- **The integration branch `feat/drainage-filter`** (this body of work, **ready to PR
  to main**) adds, on top of that:
  1. the **complete drainage arc** - flow-accumulation filter, verified + hardened +
     integrated as a review-flag, then refined to an overlap-flag + confidence-ranked
     review queue (`3em` epic CLOSED);
  2. **three parallel-lane improvements** - data-driven attitude rules (`5ug`),
     continuity/edge-linking (`zod`), a correlated-error uncertainty floor (`85g`);
  3. **process tooling** - verification hooks, a parallelization kit, and an
     adversarial skeptic verifier.
- **195 pure tests green; ruff clean.** Headless GDAL via micromamba env `gdal`:
  `MAMBA_ROOT_PREFIX=$HOME/micromamba PYTHONPATH=$PWD $HOME/bin/micromamba run -n gdal python ...`

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
- **Infra:** `bcn` (S2 cloud compositing), `gj9` (per-AOI CRS), `1dg` (training
  GeoPackage). Deferred: `luj` (shield AOI), `eu3` (bootstrap confirmation).
- **The big phase: QGIS plugin GUI** - QgsTask run, styled layers, the human
  review/triage gate. The path to a usable tool.

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
