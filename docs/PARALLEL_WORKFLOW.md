# Parallel-session workflow (orchestrator + worktree sessions)

How we run work in parallel: one **orchestrator** session (writes briefs, adversarially
verifies, merges) plus one or more **task sessions**, each a fresh Claude Code agent in
its own **git worktree** on its own branch. Lightweight by design — not heavy
multi-agent orchestration. Validated on `planesight-5ug` (2026-06-24).

## The loop

1. **Pick parallelizable work.** `bd ready` is the parallelization frontier — only fan
   out beads with **no blocking edges** between them. Coupled work stays serial.
2. **Brief.** Orchestrator runs `/handoff <bead-id>` → a self-contained
   `docs/TASK_<bead>_<slug>.md` (template below), committed.
3. **Spawn.** `scripts/spawn_task.sh <branch> <bead-id>` → isolated worktree +
   `BEADS_DB` wired to the canonical DB + the kickoff prompt to paste.
4. **Execute in-lane.** The task session reads its brief + this doc, works **only** the
   files in its lane, keeps the pure pytest suite green (the Stop hook enforces it),
   tracks work in beads, then closes its bead, updates `docs/HANDOFF.md`, and **reports
   back its result + biggest caveat**.
5. **Adversarially verify.** Orchestrator does NOT trust the self-report — it `git`-reads
   the branch, runs the tests, re-derives the key numbers, and runs the skeptic checklist
   (below). Findings get recorded on the consuming beads.
6. **Record / merge.** Caveats onto downstream beads; merge when clean.

## Brief template (what `/handoff` fills)

- **You are / orient** — fresh, context-free; read this doc, `ARCHITECTURE.md`,
  `docs/HANDOFF.md`, and `bd show <bead>`; hooks are active.
- **Why it matters** — context the code can't convey (pull from beads/HANDOFF).
- **Goal / deliverables** — concrete, with file pointers to *reuse*, not reinvent.
- **Constraints** — dependency-free core (numpy/scipy/GDAL only, D11); headless GDAL
  incantation; new pure logic in `core/` with tests (Stop-hook gated); GDAL drivers in
  `scripts/` run manually; any domain rules (circular strike stats, etc.).
- **Coordination / lane** — the branch, the **exact files this task may touch**, and what
  it must **NOT** touch (another session owns those).
- **Done when** — objective acceptance a test/command can check; report-back with the
  single biggest caveat for verification.

Keep briefs tight: give the generator **goal + acceptance + pointers**, let it choose the
implementation. Granular over-specification cascades errors downstream.

## Lane discipline (non-negotiable)

Each task session touches only its allotted files; it must not edit `.claude/` hooks,
another task's domain, or shared infra unless its brief says so. This is what makes
worktrees conflict-free. The orchestrator confirms lane discipline during verification.

## Beads coordination

Worktrees share one `.git` but the beads `.db` is gitignored, so **task sessions point at
the canonical DB**: `export BEADS_DB=<repo-root>/.beads/beads.db` (the spawn script prints
this). Issue updates then land in one place; `issues.jsonl` still syncs via git. If beads
looks out of sync, `bd doctor`.

## Adversarial-verify checklist (orchestrator's job)

Trust nothing in the self-report until checked:
- **Floor green** — run the branch's `ruff check` + pure `pytest` yourself; confirms the
  Stop hook actually fired in the worktree.
- **Re-derive the headline** — recompute the key number; check the **denominator**,
  **per-group N**, and whether ground truth is incomplete (positive-unlabeled).
- **Tested the trap** — does the suite test the failure mode, not just the happy path?
- **Lane discipline** — its diff stays in its files.
- **Task-specific traps** — e.g. circular-stats correctness; a threshold resting on a
  metric known to be optimistic; selection bias; parameter fragility.
- **Premature victory** — does "done" survive an independent re-check, or is it
  plausible-but-wrong? (Our recurring failure mode.)

## When to use the heavier `Workflow` tool instead

`scripts/spawn_task.sh` is for **you-driven, reviewable** parallel sessions. For
orchestrator-driven structured fan-out (N independent finders, judge panels, review
sweeps that return to the orchestrator), the `Workflow` tool is the right call — but it's
opt-in, costs more, and is review-bound. Default to worktree sessions; reach for Workflow
only when the task is a genuine fan-out you want *me* to drive and collect.
