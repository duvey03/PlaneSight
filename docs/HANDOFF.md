# PlaneSight - Session Handoff

**Date:** 2026-06-23
**Purpose:** Orient a fresh session quickly. Read `ARCHITECTURE.md` first (design,
science, decisions D1-D15, open questions Q1-Q10); this doc is the current state,
loose ends, and recommended next steps.

---

## TL;DR

PlaneSight is a planned QGIS plugin that derives geological strike/dip from free
global DEM + Sentinel-2 data. **The core scientific premise is now validated
end-to-end on real data.** Phase 0 (data backbone) and the Phase 3 core
(strike/dip engine) are largely built and tested; the big unproven piece is
**automated trace detection** (Phase 1 research -> Phase 2). Repo is public, CI is
green.

- Repo: https://github.com/duvey03/PlaneSight (root `C:\PlaneSight`, WSL `/mnt/c/PlaneSight`)
- Dual-licensed: code GPL-3.0, seed data CC-BY-4.0
- 21 commits; two CI jobs green (pure-core lint+tests; GDAL integration via micromamba)

## Where everything lives

```
ARCHITECTURE.md                  Source of truth: design, science (S6), decisions, roadmap
docs/DATASETS.md                 Open dataset survey (validation/training sources)
docs/HANDOFF.md                  This file
planesight/                      The QGIS plugin package
  core/                          Pure-Python, QGIS-free, numpy/scipy/GDAL only (CI-tested)
    data/  stac.py sources.py fetch.py   AOI -> STAC -> DEM + Sentinel-2 (validated)
    attitude/ plane_fit.py sample.py     PCA plane fit + DEM sampling + uncertainty
    detect/ base.py                      TraceDetector ABC + registry (no impl yet)
    derivatives/                         STUB (P0e)
  gui/ tasks/ resources/         QGIS/Qt shell, QgsTask harness, SVG/QML symbols
tests/                           pytest (pure core + GDAL-gated integration tests)
scripts/                         dev helpers (see below)
data/raw/                        seed traces: Canada/Nepal/Pakistan (+ DATA_MANIFEST.md)
debug/                           generated outputs (gitignored): Nepal slice results
.beads/                          issue tracker (JSONL tracked in git)
```

## What's done (verified)

| Capability | Status | Evidence |
|---|---|---|
| QGIS plugin scaffold + CI + dev env | done (`vm9` closed) | loads; CI green |
| AOI -> STAC fetch (GLO-30 DEM + Sentinel-2, anonymous AWS) | done (`e8r` closed) | live-verified; GDAL path tested headless + CI |
| Sentinel-2 <5% cloud + empirical dry-season selection (D15) | done | recovers real climatology (Pakistan Oct-Dec, Nepal Nov) |
| Strike/dip plane fit + conditioning/planarity metrics (D12) | done | synthetic gate D14 passed; exact recovery |
| DEM sampling along traces (densify + bilinear) | done | bilinear exact on linear fields; sample->fit recovery test |
| Uncertainty budget (Monte-Carlo, S6.4) | done v1 (caveat below) | grows with noise, blows up for degenerate traces |
| **Nepal vertical slice (first REAL measurements)** | done | median dip 28 deg, mean strike ~288 deg dipping NNE = Himalayan grain; 96% reliable |
| Strike/dip QGIS symbology (2-layer SVG + dip labels) | done | `debug/nepal_attitudes.qml` |
| Automated trace **detection** | NOT STARTED | the key gap |

## How to run things (dev quickstart)

Pure-core work needs no GDAL/QGIS. The raster I/O edge needs GDAL (headless via
no-sudo micromamba; QGIS is the integration target).

```bash
# pure tests + lint (fast inner loop)
python3 -m pytest -q
ruff check planesight tests scripts          # ALWAYS gate commits on this

# headless GDAL (one-time): pin to your QGIS GDAL if known
scripts/setup_dev_gdal.sh 'gdal>=3.8'

# run GDAL tests + the Nepal slice under micromamba
MAMBA_ROOT_PREFIX=$HOME/micromamba PYTHONPATH=$PWD \
  $HOME/bin/micromamba run -n gdal pytest
MAMBA_ROOT_PREFIX=$HOME/micromamba PYTHONPATH=$PWD \
  $HOME/bin/micromamba run -n gdal python scripts/nepal_slice.py

# QGIS integration checkpoints (paste into QGIS Python Console)
scripts/qgis_console_checkpoint.py     # fetch + load DEM/S2
scripts/qgis_style_attitudes.py        # robustly style attitudes + save .qml
```

## Loose ends & caveats

1. **Uncertainty is an optimistic lower bound.** The Monte-Carlo assumes
   *independent* per-point DEM noise; real GLO-30 error is spatially correlated,
   so well-sampled traces read unrealistically tight (~0.3 deg). Refinement
   tracked as **`planesight-85g`** (correlated random-tilt term). It DOES
   correctly flag low-relief/degenerate traces.
2. **Label Y-sign unverified.** The 15 px dip-direction label offset uses
   `-15*cos(...)`; if labels land up-dip in your QGIS, flip to `+15` (one spot in
   `attitudes_strike_dip.qml` / `qgis_style_attitudes.py`).
3. **Hand-authored QML may drift by QGIS version.** If `nepal_attitudes.qml`
   misbehaves, run `scripts/qgis_style_attitudes.py` (PyQGIS, always valid) - it
   re-saves a correct `.qml`.
4. **GDAL version pin.** Dev/CI use GDAL ~3.13; should be re-pinned to the user's
   QGIS GDAL (Help > About) for fidelity. Code uses only stable Warp/vsicurl APIs,
   so risk is low.
5. **Phase-3 engine runs on HAND-DRAWN traces.** Wiring it to *detected* traces
   waits on detection (Phase 2).
6. **Process note:** run `ruff` BEFORE committing (an E402 slipped to main once,
   fixed immediately).

## Issue board (Beads)

- Closed: `vm9` (scaffold), `e8r` (STAC fetch), `t22` (dataset survey)
- In progress: `lph` (Phase 3 strike/dip engine - core done; remaining =
  correlated uncertainty `85g`, windowed fits Q5, wire to detected traces)
- Key ready work: `c3r` (detector review - gates Phase 2), `bc1` (DEM derivative
  review), `9vt` (training bootstrap go/no-go), `6s7` (derivative engine),
  `gj9` (per-AOI CRS), `bcn` (cloud compositing), `1dg` (unified training gpkg),
  `85g` (correlated uncertainty)
- `bd ready` / `bd list --status in_progress` for the live view.

## Recommended next targets (in order)

1. **Phase 1 detector review (`c3r`) - START HERE.** It is the gate to Phase 2 and
   the whole "automatic" promise. Decide the v1 classical detector implementable
   in numpy/scipy only (Canny, ridge/valley, phase congruency via FFT, Hough),
   run small experiments on the Nepal/Pakistan DEM + the derivative stack. Pair
   with **`bc1`** (which derivatives best expose bedding/contacts) since they
   inform each other.
2. **Phase 2 classical detection MVP (`8gw`)** once `c3r` picks the method:
   detector -> vectorise -> candidate trace polylines, behind the existing
   `TraceDetector` ABC. Then wire detected traces into the proven plane-fit =
   first *fully automatic* strike/dip.
3. **Training-bootstrap go/no-go (`9vt`)** - run in parallel; it gates whether the
   v2 ML path is viable (drape published map linework over the DEM). High-risk;
   decide early.
4. Science refinements when convenient: correlated-error uncertainty (`85g`),
   windowed fits (Q5/S6.5), and quantitative validation vs Macrostrat/published
   dips (closes Q4) - the credibility lever.

**Suggested immediate move:** pivot to detection - drive `c3r` + `bc1` together as
a research spike on the data we already have.
