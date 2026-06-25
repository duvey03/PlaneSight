#!/usr/bin/env bash
# Stop hook: the hard verification gate. When the agent tries to finish, IF any
# Python changed this session, require `ruff check` AND the pure pytest suite to pass
# before it may stop (exit 2 = keep working, with the failure fed back).
#
# Deliberate scope (PlaneSight-specific):
#   - PURE pytest only (`python3 -m pytest`): the GDAL/Sentinel-2 tests are slow and
#     network-flaky, so they stay a manual step - a per-turn gate must be fast + sure.
#   - Skips entirely on pure-conversation turns (no .py touched) so it never nags.
#   - stop_hook_active guard prevents an infinite block loop if something is unfixable
#     (it blocks once to force a fix attempt, then yields to avoid a hang).
set -uo pipefail

input="$(cat)"

# avoid infinite loop: if we are already re-running because of this hook, allow stop
if printf '%s' "$input" | grep -q '"stop_hook_active":[[:space:]]*true'; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-/mnt/c/PlaneSight}" 2>/dev/null || exit 0

# pure-conversation turns (nothing Python changed) finish instantly
if ! git status --porcelain | grep -q '\.py$'; then
  exit 0
fi

if ! ruff check . >&2; then
  echo "STOP BLOCKED: ruff is failing - fix lint before finishing." >&2
  exit 2
fi

if ! python3 -m pytest -q >&2; then
  echo "STOP BLOCKED: pure pytest suite is failing - fix before finishing." >&2
  exit 2
fi

exit 0
