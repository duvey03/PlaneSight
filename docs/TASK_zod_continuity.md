# Task brief — planesight-zod: detection continuity (endpoint linking)

You are a **fresh Claude Code session** picking up ONE task in PlaneSight. No prior
context — the repo, beads, and this brief are your context. **Verification hooks are
active**: ruff auto-fixes on edit, and a **Stop hook runs the pure pytest suite and
blocks you while it's red**. Keep the pure suite green.

## Orient first
1. Read `docs/PARALLEL_WORKFLOW.md` (how this parallel flow works + your obligations),
   then `ARCHITECTURE.md` and `docs/HANDOFF.md`.
2. `bd show planesight-zod`.
3. Read `planesight/core/detect/vectorize.py` (your lane) and `tests/test_vectorize.py`.
   Note `trace_skeleton`, `simplify`, `polylines_from_mask` — these produce the
   fragmented polylines you'll be linking.

## Why it matters (context the code can't convey)
The geologist's #1 quality complaint: auto-detected traces are **more fragmented** than
a human's continuous interpretation. A real contact comes out as several short pieces
(hysteresis breaks at sub-threshold spots; `trace_skeleton` splits at junctions; no
gap-bridging). Fragmentation also *lowers confidence* downstream — `planesight-61f` will
size-filter traces, and genuine contacts that arrive in pieces get wrongly discarded as
"too small." Linking colinear fragments into longer traces directly raises confidence.

## Goal / deliverables
Implement endpoint linking as a **pure, tested primitive** in `vectorize.py`:

1. `link_polylines(polylines, max_gap_px, max_angle_deg)` — merge two polylines when an
   endpoint of one is within `max_gap_px` of an endpoint of the other **AND** their local
   end-orientations are collinear within `max_angle_deg` (continuation, not a junction).
   Iterate to chain >2 fragments. Return the new (longer) polyline list. Pure numpy.
   - Endpoints are `poly[0]` / `poly[-1]`; local orientation = direction of the last
     ~few vertices. Strike-like geometry is **undirected** — compare orientation mod 180.
2. (Optional, if it helps) a small `close_gaps(mask, size)` helper (scipy
   `binary_closing`) to bridge 1–2px skeleton gaps *before* `thin`, exposed as an opt-in.
3. Tests in `tests/test_vectorize.py` that pin the **risks** (this is the whole job):
   - two colinear fragments with a small gap → **linked** into one;
   - two **parallel but offset** fragments (adjacent bedding layers) → **NOT** merged;
   - gap larger than `max_gap_px` → not linked;
   - endpoints close but orientations divergent (a true junction) → not linked.

## Constraints
- Dependency-free core (numpy/scipy/GDAL only — D11). New logic is **pure** → the Stop
  hook's pytest suite gates it.
- **Undirected orientation** (mod 180): a fragment ending heading 5° continues one
  heading 178°. Don't average raw angles — compare the acute difference.
- **Sequencing note (do NOT wire into the pipeline):** per the bead, linking must run
  **AFTER drainage removal**, else it reconnects creeks. The pipeline integration is
  `planesight-61f`'s job (it owns the post-drainage step). You deliver the **primitive +
  tests** only; document the "apply after drainage" contract in the function docstring.
- Optional demo only: a fragmentation before/after count on Nepal is nice but not
  required; if you do it, headless GDAL is
  `MAMBA_ROOT_PREFIX=$HOME/micromamba PYTHONPATH=$PWD $HOME/bin/micromamba run -n gdal python ...`.
- Run `ruff check` + `python3 -m pytest -q` before declaring any step done.

## Coordination / lane
- Branch `feat/continuity`, isolated worktree. `BEADS_DB` is wired to the canonical DB.
- **Touch only** `planesight/core/detect/vectorize.py`, `tests/test_vectorize.py`, and
  `docs/HANDOFF.md`. **Do NOT touch** `detect/drainage*`, `detect/canny.py`,
  `detect/classical.py`, any pipeline/`drainage_*`/`attitude_*` script, the `.claude/`
  hooks, or `attitude/` — other sessions own those.
- Track in beads (`bd update planesight-zod --status in_progress`; file discovered work
  with `--deps discovered-from:planesight-zod`). Commit in small steps; end messages with
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

## Done when (acceptance)
- `link_polylines` implemented in `vectorize.py` with a docstring stating the
  apply-after-drainage contract and the undirected-orientation rule.
- New tests cover all four risk cases above; full pure suite green; ruff clean.
- `planesight-zod` closed with a summary; `docs/HANDOFF.md` updated.
- **Report back**: the chosen default `max_gap_px` / `max_angle_deg`, how the
  parallel-layer test is constructed (your guard against merging adjacent layers), and
  the single biggest caveat — for adversarial verification.
