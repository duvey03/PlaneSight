#!/usr/bin/env bash
# PreToolUse hook (Bash): block destructive / policy-forbidden shell commands.
# Exit 2 = DENY the tool call (works even under auto / --dangerously-skip-permissions,
# which is the point - it is the safety rail for unattended loops later).
# Blocks: rm -rf, force pushes, and pushes to main (feature-branch + PR is policy).
set -uo pipefail

input="$(cat)"
cmd="$(printf '%s' "$input" \
  | python3 -c 'import sys,json; print(json.load(sys.stdin).get("tool_input",{}).get("command",""))' \
  2>/dev/null || true)"

if printf '%s' "$cmd" | grep -Eq 'rm[[:space:]]+-[a-z]*r[a-z]*f|git[[:space:]]+push[^&|;]*(--force|--force-with-lease|-f[[:space:]])|git[[:space:]]+push[^&|;]*[[:space:]]main([[:space:]]|$)'; then
  echo "BLOCKED by PlaneSight safety hook: destructive/forbidden command refused." >&2
  echo "  command: ${cmd}" >&2
  echo "  (rm -rf / force-push / push-to-main. Override deliberately if truly intended.)" >&2
  exit 2
fi
exit 0
