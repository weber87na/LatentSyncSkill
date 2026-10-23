#!/usr/bin/env bash
set -euo pipefail
skill_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if command -v python3 >/dev/null 2>&1; then
    exec python3 "$skill_dir/scripts/install.py" "$@"
elif command -v python >/dev/null 2>&1; then
    exec python "$skill_dir/scripts/install.py" "$@"
else
    printf '%s\n' 'Python 3.10+ is required. Install Python, then rerun this script.' >&2
    exit 1
fi
