#!/usr/bin/env bash
# PostToolUse hook (Write|Edit|MultiEdit): auto-fix lint on the file just edited.
# Reads the tool-call JSON on stdin, runs `ruff --fix` on the edited .py, and never
# blocks (lint is enforced as a hard gate at Stop, autofixed continuously here).
# Scoped to the PlaneSight repo so it is a no-op when editing files elsewhere.
set -uo pipefail

repo="${CLAUDE_PROJECT_DIR:-/mnt/c/PlaneSight}"
input="$(cat)"
file="$(printf '%s' "$input" \
  | python3 -c 'import sys,json; print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))' \
  2>/dev/null || true)"

case "$file" in
  "$repo"/*.py)
    ruff check --fix "$file" >/dev/null 2>&1 || true
    ;;
esac
exit 0
