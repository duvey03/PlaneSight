# Task brief — planesight-85g: correlated-error term in the uncertainty budget

You are a **fresh Claude Code session** picking up ONE task in PlaneSight. No prior
context — the repo, beads, and this brief are your context. **Verification hooks are
active**: ruff auto-fixes on edit, and a **Stop hook runs the pure pytest suite and
blocks you while it's red**. Keep the pure suite green.

## Orient first
1. Read `docs/PARALLEL_WORKFLOW.md` (how this parallel flow works + your obligations),
   then `ARCHITECTURE.md` (esp. S6.4 uncertainty) and `docs/HANDOFF.md`.
2. `bd show planesight-85g`.
3. Read `planesight/core/attitude/plane_fit.py` — specifically `fit_plane` and its
   Monte-Carlo uncertainty estimator (`_estimate_uncertainty`) and `tests/test_plane_fit.py`.

## Why it matters (context the code can't convey)
`fit_plane`'s MC dip / dip-direction uncertainty perturbs each sample's elevation by
**independent** Gaussian noise (`sigma_z`). Real GLO-30 error is **spatially correlated**,
so the current budget is an **optimistic lower bound** that **averages down ~1/√N** — it
reports implausibly tiny ~0.3° uncertainties for well-sampled traces. This is not
academic: a separate analysis (`planesight-5ug`) tried to set a "minimum confident trace
length" from this metric and an adversarial pass flagged that the length–reliability knee
is partly a **sample-count artifact of this very optimism**. Fixing the metric de-confounds
that and makes every downstream confidence/length decision honest.

## Goal / deliverables
Add a **correlated-error term** to the MC uncertainty so it does NOT vanish with N:

1. In each MC iteration, in addition to the existing independent per-point noise, apply a
   **spatially-correlated perturbation** that a long trace cannot average away — the
   simplest defensible model (per the bead) is a **random planar tilt**: draw a random
   azimuth and a small gradient (std ≈ `sigma_z / correlation_length`) and add that tilt
   across ALL sample points in the iteration. (A correlated random field by inter-point
   distance is an acceptable richer alternative; the random-tilt is the floor.)
2. Expose the new knob(s) in `fit_plane` (e.g. `correlation_length` in metres, default a
   sensible GLO-30 value — state your reasoning in the docstring) **without breaking the
   existing signature/defaults** — independent-only behaviour must remain reproducible
   when the correlated term is disabled.
3. (Optional, note in docstring if deferred) horizontal misregistration as a second
   correlated source.

## Constraints
- Dependency-free core (numpy/scipy only — D11). Pure math → the Stop hook gates it.
- **Determinism in tests:** seed the RNG (`np.random.default_rng(seed)`) so tests are
  stable; do not call bare `np.random`.
- Backward compatibility: the existing `fit_plane(...)` calls and the synthetic-recovery
  gate must still pass unchanged.

## Coordination / lane
- Branch `feat/uncertainty`, isolated worktree. `BEADS_DB` wired to the canonical DB.
- **Touch only** `planesight/core/attitude/plane_fit.py`, `tests/test_plane_fit.py`, and
  `docs/HANDOFF.md`. Avoid editing `attitude/__init__.py` unless strictly necessary
  (another branch edits it — minimise conflicts). **Do NOT touch** `attitude/variability.py`,
  `detect/*` (incl. `vectorize.py` — another session owns it), any `drainage_*`/`attitude_*`
  script, or the `.claude/` hooks.
- Track in beads (`bd update planesight-85g --status in_progress`; file discovered work
  with `--deps discovered-from:planesight-85g`). Commit in small steps; end messages with
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

## Done when (acceptance)
- The correlated term is implemented and on by default with a justified `correlation_length`.
- **Key test:** for a long, well-sampled synthetic trace, uncertainty with the correlated
  term is **materially higher and does NOT keep shrinking ~1/√N** as samples increase,
  whereas the independent-only path still does (prove the fix, both directions).
- Backward-compat test: independent-only (correlated disabled) reproduces the prior
  behaviour; degenerate/low-relief cases still behave; the synthetic-recovery gate passes.
- Full pure suite green; ruff clean. `planesight-85g` closed with a summary; `docs/HANDOFF.md`
  updated.
- **Report back**: the chosen `correlation_length` + its justification, the before/after
  uncertainty on a representative long trace, and your single biggest caveat — for
  adversarial verification.
