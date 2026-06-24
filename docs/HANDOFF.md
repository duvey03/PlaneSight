# PlaneSight - Session Handoff

**Updated:** 2026-06-24. Supersedes the Phase-0 handoff.

PlaneSight = a QGIS plugin that, for any AOI, aggregates global DEM (Copernicus
GLO-30) + Sentinel-2, auto-detects geological bedding/contact traces, and computes
strike/dip en masse from the DEM geometry. Everything to date is a validated,
**headless** Python core (`planesight/core/`) - the QGIS GUI is not yet built. Read
`ARCHITECTURE.md` for design/decisions (D1-D15) and `docs/PHASE1_REPORT.md` +
`docs/BOOTSTRAP_VERDICT.md` for the science.

---

## TL;DR state

- **Phases 0-2 are DONE and on `main`** (PRs #1, #2 merged): data fetch, derivative
  engine, plane-fit strike/dip engine, classical trace detector, scoring.
- **The full chain works end to end on real data**: raw AOI -> GLO-30 -> auto
  traces -> strike/dip that recovers the known regional structural grain in three
  regions (Nepal/Pakistan/Canada).
- **Current WIP (branch `feat/drainage-filter`, NOT merged):** a flow-accumulation
  drainage pre-filter. Built + unit-tested, but a critical review found the Nepal
  results **oversold** - verification is required before it's trustworthy (below).
- **Pending the user (just approved):** build the drainage verification (audit panel
  + honest metrics) before integrating the filter.

---

## What's done (verified)

| Capability | Where | Status |
|---|---|---|
| AOI -> STAC -> GLO-30 DEM + Sentinel-2 (anonymous AWS) | `core/data/` | done |
| Coverage-aware S2 same-date mosaic + nodata handling | `core/data/align.py`, harness | done |
| Derivative engine (slope/aspect/hillshade/tpi/curvature; S2 indices; named stacks) | `core/derivatives/` | done |
| Plane-fit strike/dip (SVD) + conditioning/planarity + **map_conditioning** + MC uncertainty | `core/attitude/plane_fit.py` | done |
| Positive-unlabeled scoring (recall-at-budget) + label-free linearity | `core/detect/score.py`, `linearity.py` | done |
| Canny operator + vectorize (thin/trace/simplify) + `ClassicalTraceDetector` | `core/detect/` | done |
| End-to-end auto strike/dip on Nepal | `scripts/detect_attitudes_nepal.py` | done |
| Multi-region detector eval (recall/linearity + dominant strikes) | `scripts/detector_eval.py` | done |
| Drainage pre-filter (D8 flow accum + classifier) | `core/detect/drainage.py` | **WIP, see issues** |

**~158 pure tests green; ruff clean.** Headless GDAL via micromamba env `gdal`:
`MAMBA_ROOT_PREFIX=$HOME/micromamba PYTHONPATH=$PWD $HOME/bin/micromamba run -n gdal python ...`

## Key findings (the science)

- **DEM curvature/slope are the dominant contact exposers** across all 3 regions;
  Sentinel-2 (iron-oxide) is a useful secondary, seasonally sensitive.
- **Auto strike/dip works:** Nepal gave 3,324 well-conditioned attitudes (median dip
  ~20 deg, dominant strike ~86 deg = Himalayan grain) with no digitizing. Dominant
  strikes match known grain in all 3 regions (Himalaya/Makran E-W, Cordillera NW-SE).
- **`map_conditioning` gate** (lambda2/lambda1 of the x,y projection) removes the
  straight-map-trace near-vertical artifact the 3D conditioning misses (gate 1e-3:
  near-vertical 29%->~0%, keeps 95-98% of fits).
- **ML bootstrap (planesight-9vt): conditional no-go, then softened** - map-draping
  for ML labels is terrain-dependent, not a flat no-go (`docs/BOOTSTRAP_VERDICT.md`).

---

## Issues / open problems identified

1. **Drainage dominates the false lineaments (geologist-confirmed, quantified).** The
   detector is a generic topographic break-line detector; in dissected terrain it
   traces creeks/rivers. Probe: detected edges ~70x more valley-concave than
   background, 96% within 2px of a valley axis. **Curvature-SIGN filtering is a dead
   end** (47% vs 47%); drainage is a connectivity property needing flow accumulation.
2. **Drainage-filter results are OVERSOLD (critical review - the big one).**
   - The "1.3% false-negative" is **selection-biased**: only 22/408 hand-traces run
     along valleys at all; the other 95% can't be flagged regardless. Among the
     at-risk valley-following contacts, **~18% (4/22) get flagged** - tiny, uncertain
     sample. Strike-valley contacts are the real FN risk.
   - The **removed 31% was never verified to be drainage** (only that it misses the
     mostly-cross-cutting hand-traces). Reassuring: the alignment criterion keeps 775
     of 3916 valley-overlapping detections (oblique crossers), so it does discriminate.
   - **31% is parameter-fragile** (23-47% across reasonable angle_tol/min_aligned).
   - Framing: removes ~a third, NOT "most" false positives; the other ~60% (ridges/
     divides, sub-90m tributaries) is unexplained.
3. **Drainage algorithm shortcuts** (`drainage.py`): no depression/pit-filling and
   stride subsampling (`dem[::3,::3]`) - fine on steep Nepal, **will degrade on
   low-relief Pakistan/Canada**. (Accumulation math itself is correct/conserved.)
4. **Detections are fragmented vs the geologist's continuous interpretation**
   (hysteresis breaks; `trace_skeleton` splits at junctions; no gap-bridging).
5. **Conditioning gate calibrated only on auto-detected Nepal traces** (`2je`); the
   1e-2/1e-3 thresholds may need per-region tuning.
6. **Everything is headless** - no QGIS plugin GUI/review-gate yet.

---

## What to do next (in order)

**Immediate (user-approved):**
1. **`planesight-5p3` - verify the drainage filter (DECISIVE):**
   - Render ~30 **random flagged (orange) traces** over hillshade+S2 for the
     geologist to audit: creek or contact? (tests whether the removed set is really
     drainage - never checked).
   - Report false-negative **conditioned on valley-overlapping hand-traces** (~18% on
     22, with small-N caveat), not the diluted 1.3%/408.
   - Publish the **parameter-sensitivity range**, not a single 31%.
   - HOLD the integration until this lands.

**Then:**
2. **`planesight-j8t`** - harden the drainage algorithm (block-mean downsample +
   priority-flood pit-fill) BEFORE generalizing the sweep to Pakistan/Canada.
3. **`planesight-xx2`** - integrate the filter into `ClassicalTraceDetector` as a
   **review-flag** (route to queue, don't delete), applied **before** continuity.
4. **`planesight-zod`** - continuity/edge-linking fix (after drainage removal).
5. Re-run the multi-region eval with the filter; confirm the knee transfers.

**Parallel / later:** infra remnants - `bcn` (S2 cloud compositing), `gj9` (per-AOI
CRS policy), `1dg` (unified training GeoPackage), `85g` (correlated-error
uncertainty). The big new phase is the **QGIS plugin integration** (GUI, QgsTask
run, styled layers, human review/triage gate) - the path to a usable tool.

---

## Beads map

- Drainage epic **`3em`** (in_progress) -> children **`5p3`** (verify, next),
  **`j8t`** (hardening), **`xx2`** (integrate as flag), **`zod`** (continuity).
- `lph` (Phase 3 strike/dip engine) in_progress; mostly done in core, `85g` remains.
- Open infra: `bcn`, `gj9`, `1dg`, `85g`; deferred `luj` (shield data), `eu3`
  (bootstrap confirmation). Phases 0/1/2 epics closed.

## Working agreements / process notes

- **WSL paths only** (`/mnt/c/...`). No emojis/Unicode - use `[OK]`/`[ERROR]`.
- **Run ruff before committing.** Commit/push only when asked; **branch first**
  (push to `main` is policy-blocked - feature branch + PR, merge with `gh`).
- **Background runs sometimes fail transiently** (S2-fetch network blips) - re-run in
  the foreground to confirm before assuming a code bug.
- Visual review panels: `debug/bootstrap_review/` and `debug/drainage_test/`.
- **Lesson this session:** trust the *measurement*, not the headline. The coverage
  bug and the FN-dilution both produced confident-but-wrong numbers that fell apart
  under a critical re-check. Verify before concluding; use the critical-reviewer
  (fork) agent on big claims.
