#!/usr/bin/env bash
# spawn_task.sh - create an isolated git worktree for a parallel task session.
#
# Usage: scripts/spawn_task.sh <branch-name> [bead-id]
#
# Branches off the current HEAD (so the .claude hooks + task briefs come with it),
# points beads at the canonical DB (so issue updates land in one place, fixing the
# gitignored-.db-in-worktree friction), and prints the kickoff prompt to paste into
# the fresh Claude Code session launched in that worktree. See docs/PARALLEL_WORKFLOW.md.
set -euo pipefail

branch="${1:?usage: scripts/spawn_task.sh <branch-name> [bead-id]}"
bead="${2:-<bead-id>}"

root="$(git rev-parse --show-toplevel)"
beads_db="$root/.beads/beads.db"
slug="${branch##*/}"                        # feat/continuity -> continuity
wt="$(dirname "$root")/$(basename "$root")-$slug"

if [ -e "$wt" ]; then
  echo "ERROR: worktree path already exists: $wt" >&2
  exit 1
fi
if git -C "$root" show-ref --verify --quiet "refs/heads/$branch"; then
  echo "ERROR: branch already exists: $branch" >&2
  exit 1
fi

git -C "$root" worktree add -b "$branch" "$wt"

cat <<EOF

=== worktree ready: $wt  (branch: $branch) ===

Launch the parallel session:
  cd "$wt"
  export BEADS_DB="$beads_db"     # share the canonical beads DB
  claude

Paste this as the session's first message:
-------------------------------------------------------------------------------
You are a fresh Claude Code session picking up ONE task in PlaneSight. You have no
prior context; the repo, beads, and a task brief are your context, and the project
verification hooks are active. Read docs/PARALLEL_WORKFLOW.md, then your task brief in
docs/ (named for the bead), then run \`bd show $bead\`, then orient via ARCHITECTURE.md
and docs/HANDOFF.md. Execute per the brief's acceptance criteria, stay strictly in your
file lane, keep the pure pytest suite green, track all work in beads, and when done:
close the bead, update docs/HANDOFF.md, and report back your result plus the single
biggest caveat for adversarial verification.
-------------------------------------------------------------------------------

When it reports back, the orchestrator session verifies before trusting (see the
adversarial-verify checklist in docs/PARALLEL_WORKFLOW.md). To remove the worktree later:
  git worktree remove "$wt"
EOF
