# Task brief — planesight-5ug: data-driven attitude rules from hand-traced datasets

You are a **fresh Claude Code session** picking up ONE task in the PlaneSight project.
You have no prior conversation context — this doc + the repo + the beads tracker are
your context. The project verification **hooks are active** (they loaded because the
session launched inside the repo): ruff auto-fixes on every edit, and a **Stop hook
runs the pure pytest suite and blocks you from finishing while it's red**. Keep the
pure suite green and you'll never fight it.

## Orient first (do this before coding)
1. Read `ARCHITECTURE.md` (design + locked decisions D1–D15) and `docs/HANDOFF.md`
   (current project state).
2. `bd show planesight-5ug` — the task, with acceptance notes.
3. Skim `scripts/nepal_slice.py` (fits best-fit planes to hand traces → strike/dip)
   and `planesight/core/attitude/` (`fit_plane`, `sample_trace`) — you will reuse
   these, not reinvent them. Also see the region table (AOI / EPSG / shapefile path)
   in `scripts/drainage_verify.py` (`REGIONS`) or `nepal_slice.py`.

## Why this matters (context you cannot derive from the code)
Across this project our confidence / false-negative numbers have repeatedly been
undermined by **tiny ground-truth N** (at one point only ~10 relevant hand traces —
far too few to trust a percentage). Two downstream pieces are **blocked on this task**:
- `planesight-61f` (refined drainage rule) needs an empirical **minimum trustworthy
  trace length**.
- `planesight-gas` (the adversarial "skeptic" verifier) needs a data-derived
  **"implausible local strike/dip change" threshold**.

So this task replaces guessed thresholds with **measured** ones, mined from the
geologist's own hand-traced datasets.

## Goal / deliverables
Datasets: `data/raw/{nepal,pakistan,canada}/*.shp` (mixed CRS — reproject per the
`REGIONS` table; see how `drainage_verify.py:hand_polylines` / `nepal_slice.py` load
and reproject them).

1. **Local attitude variability → the smoothness / "regional grain" threshold.**
   Fit attitudes along the hand traces (reuse `fit_plane` + `sample_trace`). For each
   attitude, measure how much **strike** and **dip** differ from its neighbours within
   sliding windows of several radii (e.g. 250 m / 500 m / 1 km / 2 km). Report the
   distribution (median, IQR, 90th pct) of local strike-deviation and dip-deviation
   per window size, per region. **Deliver a recommended "this attitude is an
   implausible local outlier" threshold** (degrees) with the numbers behind it.
2. **Trace morphology + length priors.** Per hand trace, compute **length** and a
   **rule-of-V's morphology** label: `V` (apex), `straight` (near-vertical strata),
   `contour-parallel` (near-horizontal bedding). Report the **length distribution**
   (this gives `61f` its "minimum confident length") and the morphology mix per region.
3. A short findings doc `docs/ATTITUDE_RULES.md` with the recommended thresholds +
   supporting numbers + caveats (including ground-truth N **per region** — call out
   where N is too small to trust), and the analysis driver `scripts/attitude_rules.py`.

## Constraints / how to run
- **Headless GDAL** via micromamba:
  `MAMBA_ROOT_PREFIX=$HOME/micromamba PYTHONPATH=$PWD $HOME/bin/micromamba run -n gdal python scripts/attitude_rules.py`
- **Dependency-free core** (numpy/scipy/GDAL only — D11). No skimage/sklearn.
- Put any **new pure-python logic** (circular strike statistics, the morphology
  classifier, the windowed-deviation computation) in `planesight/core/` **with unit
  tests in `tests/`**, so the Stop hook gates it. The GDAL-dependent driver stays in
  `scripts/` and is run manually.
- **Strike is circular and undirected (mod 180°)** — use double-angle circular
  statistics for means/deviations, NOT plain arithmetic. See the doubled-angle
  vector-mean in `scripts/detect_attitudes_nepal.py` for the pattern.
- Ground truth is **incomplete / positive-unlabeled** — characterise distributions;
  do not treat the hand traces as a complete census.
- Run `ruff check` + `python3 -m pytest -q` before declaring any step done (the hooks
  enforce this, but know it).

## Coordination (important — parallel session)
- You are isolated on branch **`feat/attitude-rules`** (a worktree). Another session
  is concurrently working `feat/drainage-filter`.
- **Touch only** `planesight/core/attitude/*`, new `scripts/attitude_rules.py`, new
  `tests/test_*` for your helpers, and `docs/ATTITUDE_RULES.md` / `docs/HANDOFF.md`.
  **Do NOT touch** `core/detect/drainage*`, the `.claude/` hooks, or any `drainage_*`
  script — another session owns those.
- Track work in beads: `bd update planesight-5ug --status in_progress`; file any
  discovered work with `bd create ... --deps discovered-from:planesight-5ug`. Commit
  in small reviewable steps; end commit messages with the
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` line (match `git log`).

## Done when (acceptance criteria)
- `scripts/attitude_rules.py` runs clean and prints the variability distributions +
  morphology/length priors for all three regions.
- New core helpers have passing pure unit tests; full pure suite green; ruff clean.
- `docs/ATTITUDE_RULES.md` states the recommended thresholds (smoothness outlier in
  degrees; minimum confident length) **with** the supporting numbers and the
  per-region ground-truth-N caveat.
- `planesight-5ug` closed with a summary; `docs/HANDOFF.md` updated.
- **Report back** the two recommended thresholds and the single biggest caveat, so
  the orchestrating session can adversarially verify them before they're trusted.
