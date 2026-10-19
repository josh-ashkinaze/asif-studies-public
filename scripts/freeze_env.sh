#!/usr/bin/env bash
# Author: Joshua Ashkinaze
#
# Description: Refreshes the environment files so they describe the interpreter that actually ran
# the analyses. Run after installing or upgrading anything, then commit the result. Exports
# requirements.txt from uv.lock rather than from pip freeze, so the two can never disagree.
#
# Inputs:
#   - pyproject.toml: the pinned direct dependencies
#   - .venv: the working environment, for the drift check
# Outputs:
#   - uv.lock: full resolved graph with hashes
#   - requirements.txt: the same graph as pinned requirements, for pip users
#   - stdout: any package whose installed version differs from the lock

set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> uv lock"
uv lock

echo "==> export requirements.txt from the lock"
uv export --no-hashes --no-annotate --no-header --format requirements.txt \
  | grep -v '^-e \.$' > requirements.txt
echo "    $(wc -l < requirements.txt | tr -d ' ') pinned packages"

echo "==> drift: installed (.venv) vs locked"
if [ -x .venv/bin/python ]; then
  .venv/bin/python - <<'PY'
import re, subprocess, sys
from pathlib import Path

def norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()

# Respect environment markers: torch's CUDA wheels are Linux-only and colorama is win32,
# so on this machine they are correctly absent rather than missing.
from packaging.markers import Marker

locked = {}
for line in Path("requirements.txt").read_text().splitlines():
    spec, _, marker = line.partition(";")
    spec = spec.strip()
    if "==" not in spec:
        continue
    if marker.strip() and not Marker(marker.strip()).evaluate():
        continue
    name, _, ver = spec.partition("==")
    locked[norm(name)] = ver.strip()

installed = {}
for line in subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                           text=True).stdout.splitlines():
    if "==" in line:
        name, _, ver = line.partition("==")
        installed[norm(name)] = ver.strip()

drift = [(n, installed[n], locked[n]) for n in sorted(locked)
         if n in installed and installed[n] != locked[n]]
missing = sorted(n for n in locked if n not in installed)

for name, have, want in drift:
    print(f"    DRIFT   {name}: installed {have}, locked {want}")
for name in missing:
    print(f"    MISSING {name}: locked {locked[name]}, not installed")
if not drift and not missing:
    print("    none: .venv matches the lock")
PY
else
  echo "    no .venv; skipped"
fi

echo
echo "Done. R packages are pinned separately in scripts/setup_r.R."
