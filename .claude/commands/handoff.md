---
description: Generate a parallel-session task brief from a bead, ready to spawn in a worktree
---
Produce a self-contained task brief so a fresh, context-free Claude Code session can
execute bead **$1** in an isolated worktree. Follow docs/PARALLEL_WORKFLOW.md.

1. Run `bd show $1` and read the bead fully; skim the code/files it references so the
   brief points at real things to reuse.
2. Write `docs/TASK_$1_<slug>.md` using the brief template in docs/PARALLEL_WORKFLOW.md:
   **You are / orient**, **Why it matters** (context the code can't convey),
   **Goal / deliverables** (concrete, with file pointers to reuse), **Constraints**
   (dependency-free core D11; the headless GDAL incantation; new pure logic in `core/`
   with tests since the Stop hook gates the pure suite; GDAL drivers in `scripts/`,
   manual; any domain rules), **Coordination / lane** (branch + the EXACT files this
   task may touch and what NOT to touch), and **Done when** (objective acceptance a
   test/command can check; report-back with the biggest caveat).
3. Keep it tight — give the generator goal + acceptance + pointers, NOT a design. Pick a
   sensible `feat/<slug>` branch name.
4. Commit the brief, then print the spawn command for me to run:
   `scripts/spawn_task.sh feat/<slug> $1`
5. Remind me: when that session reports back, run the adversarial-verify checklist
   (docs/PARALLEL_WORKFLOW.md) before trusting its result.
