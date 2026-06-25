# Task brief — planesight-amn: validate the drainage rule on Pakistan/Canada

You are a **fresh Claude Code session** picking up ONE task in PlaneSight. No prior
context — the repo, beads, and this brief are your context. **Verification hooks are
active**: ruff auto-fixes on edit, and a **Stop hook runs the pure pytest suite and
blocks you while it's red**. Keep the pure suite green (you likely add no core code).

## Orient first
1. Read `docs/PARALLEL_WORKFLOW.md` (your obligations), then `ARCHITECTURE.md` and
   `docs/HANDOFF.md` (esp. "The drainage arc").
2. `bd show planesight-amn`.
3. Read the harnesses you'll run: `scripts/drainage_verify.py`, `scripts/drainage_residual.py`,
   `scripts/drainage_test.py`. They already have a `REGIONS` table with `pakistan` and
   `canada` and take the region as an argv (e.g. `... drainage_verify.py pakistan`).

## Why it matters (context the code can't convey)
The **entire** drainage rule was calibrated and validated on **Nepal only** - the
overlap flag, the `DRAIN_ACCUM=15` channel threshold (recalibrated 8->15 for the
depression-filled network, but only against Nepal), and the rescue rank. Nepal is steep
and dissected; **Pakistan is low-relief/arid and Canada is Cordilleran** - the channel
network density and the right threshold may differ a lot. This task answers the
recurring open question: **does the knee transfer, or do we need per-region tuning?**
This is exactly the kind of generalization claim that has hidden artifacts before, so
measure, don't assume.

## Goal / deliverables
Run the harnesses on **Pakistan and Canada** and report, per region:
1. **Channel-network density** at `accum=15`: what fraction of the map is flagged as
   channel? On Nepal it's ~15%. If it's wildly off on flatter terrain (e.g. >30% or
   <5%), `accum` needs per-region recalibration. `scripts/drainage_test.py <region>`
   sweeps the accumulation threshold and prints channel% + removal + hand-FN per level -
   use it to find each region's knee.
2. **Flag behaviour**: % of detected traces/length flagged (overlap-based), kept.
3. **Audit panels** (`scripts/drainage_verify.py <region>` and/or `drainage_residual.py`):
   render the flagged-trace panels so the geologist can judge - are the flagged traces
   genuinely creeks in Pakistan/Canada too, or is the filter mis-firing on different
   terrain? Save them under `debug/`.
4. A findings doc **`docs/DRAINAGE_SWEEP.md`**: the transferability verdict (does the
   Nepal knee hold?), a **recommended per-region `accum`** (with the sweep numbers
   behind it), and honest caveats (incomplete ground truth; any region where the
   channel network or flag looks wrong).

## Constraints / how to run
- Headless GDAL: `MAMBA_ROOT_PREFIX=$HOME/micromamba PYTHONPATH=$PWD $HOME/bin/micromamba run -n gdal python scripts/drainage_test.py pakistan`
- **Pakistan/Canada S2/DEM fetches can fail transiently** (network) - re-run in the
  FOREGROUND before assuming a code bug (documented failure mode).
- Positive-unlabeled: hand traces are an incomplete subset; judge accuracy, not coverage.
- This is an analysis task. You should NOT need to change the core algorithm. If you add
  a small multi-region runner or a per-region `accum` recommendation, keep it in
  `scripts/`. Run `ruff check` + `python3 -m pytest -q` before declaring done.

## Coordination / lane
- Branch `feat/drainage-sweep`, isolated worktree. `BEADS_DB` wired to the canonical DB.
- **Touch only** `scripts/` (running/extending the drainage harnesses) and
  `docs/DRAINAGE_SWEEP.md` / `docs/HANDOFF.md`. **Do NOT touch** `core/detect/drainage.py`
  or any other `core/` (the algorithm is settled), the `.claude/` hooks, or other tasks'
  files.
- Track in beads (`bd update planesight-amn --status in_progress`; file discovered work
  with `--deps discovered-from:planesight-amn`). Commit in small steps; end messages with
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

## Done when (acceptance)
- Both regions run; per-region channel density, flag stats, and the `accum` sweep knee
  are reported, with audit panels saved under `debug/`.
- `docs/DRAINAGE_SWEEP.md` states the transferability verdict + recommended per-region
  `accum` + caveats. Pure suite green; ruff clean.
- `planesight-amn` closed with a summary; `docs/HANDOFF.md` updated.
- **Report back**: does the Nepal knee transfer? the recommended per-region `accum`, and
  your single biggest caveat (e.g. a region where the panels look wrong) - for
  adversarial verification.
