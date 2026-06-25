# Task brief — planesight-0bf: validate the spectral NDMI drainage discriminator (build gate)

You are a **fresh Claude Code session** picking up ONE task in PlaneSight. No prior
context — the repo, beads, and this brief are your context. **Verification hooks are
active**: ruff auto-fixes on edit, and a **Stop hook runs the pure pytest suite and
blocks you while it's red**. Keep the pure suite green (you likely add no core code).

## Orient first
1. Read `docs/PARALLEL_WORKFLOW.md` (your obligations), then `ARCHITECTURE.md` and
   `docs/HANDOFF.md`.
2. Read **`docs/SPECTRAL_DISCRIMINATOR_PROBE.md`** and `bd show planesight-0bf` - this
   task is the validation gate it calls for.
3. Read `scripts/spectral_drainage_probe.py` (the probe you extend) and
   `scripts/drainage_verify.py` (S2 fetch + audit-panel patterns to reuse).

## Why it matters (context the code can't convey)
The drainage FILTER (flow accumulation) fails on low-relief arid Pakistan. A probe
found **NDMI (moisture) discriminates creek-from-contact strongly on Pakistan (AUC
0.94)** while it washes out in vegetated Nepal - the two methods are complementary
across terrain regimes. BUT the 0.94 used a **proxy label** (channel *pixels* vs hand
*contacts*), so it partly reflects "wet valley vs dry slope," not the real use case.
Before anyone builds a terrain-adaptive discriminator, this gate answers: **does
per-trace NDMI actually flag creek-following DETECTED traces without eating
valley-crossing contacts?**

## Goal / deliverables (Pakistan)
1. **Per-trace NDMI on detected traces.** Detect traces (`ClassicalTraceDetector`),
   fetch S2, compute NDMI (`normalized_ratio(nir, swir16)`), and compute each detected
   trace's mean/median NDMI along its length.
2. **Pick a moisture threshold** that flags "creek" (high NDMI) and report how much of
   the detected set it flags.
3. **Conditioned false-negative on real contacts:** of the hand-traced CONTACTS (ground
   truth), how many would the NDMI threshold flag as creek (i.e. eaten)? Report it -
   this is the cost side. (Hand contacts overlap detected traces / can be sampled
   directly for NDMI.)
4. **Audit panels** (reuse `drainage_verify.py`'s panel code): render high-NDMI
   (flagged-creek) vs low-NDMI (kept-contact) detected traces over hillshade + S2, so
   the geologist can judge - are the flagged ones genuinely creeks/washes and the kept
   ones contacts? Save under `debug/`.
5. **Compare to the flow filter on Pakistan** (it over-flags ~33% of the map as channel,
   45% conditioned-FN): is per-trace NDMI a *cleaner* discriminator (flags creeks at
   lower contact-FN)?
6. A findings section (extend `docs/SPECTRAL_DISCRIMINATOR_PROBE.md` or a new
   `docs/SPECTRAL_VALIDATE.md`): **verdict** (does per-trace NDMI hold on the real use
   case?), recommended NDMI threshold, and honest caveats (incomplete ground truth;
   valley-crossing contacts as the residual FN).

## Constraints / how to run
- Headless GDAL: `MAMBA_ROOT_PREFIX=$HOME/micromamba PYTHONPATH=$PWD $HOME/bin/micromamba run -n gdal python scripts/<your_script>.py pakistan`
- **Pakistan S2/DEM fetches can fail transiently** - re-run in the FOREGROUND before
  assuming a code bug.
- Positive-unlabeled: hand contacts are an incomplete subset; judge accuracy, not
  coverage. NDMI = `normalized_ratio(nir, swir16)` (high = moist).
- Analysis task - you should NOT change `core/`. Keep new scripts in `scripts/`. Run
  `ruff check` + `python3 -m pytest -q` before declaring done.

## Coordination / lane
- Branch `feat/spectral-validate`, isolated worktree. `BEADS_DB` wired to the canonical DB.
- **Touch only** `scripts/` and `docs/SPECTRAL_*.md` / `docs/HANDOFF.md`. **Do NOT touch**
  `core/` (no building the discriminator into the detector yet - this is the *gate*),
  the `.claude/` hooks, or other tasks' files.
- Track in beads (`bd update planesight-0bf --status in_progress`; file discovered work
  with `--deps discovered-from:planesight-0bf`). Commit in small steps; end messages with
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

## Done when (acceptance)
- Per-trace NDMI computed on detected Pakistan traces; threshold + flagged-fraction +
  conditioned-FN on hand contacts reported; audit panels saved under `debug/`; compared
  to the flow filter.
- Findings doc states the verdict + recommended threshold + caveats. Pure suite green;
  ruff clean. `planesight-0bf` closed with a summary; `docs/HANDOFF.md` updated.
- **Report back**: does per-trace NDMI hold on the real use case (or did the proxy
  oversell it)? the recommended threshold + contact-FN, and your single biggest caveat
  - for adversarial verification.
