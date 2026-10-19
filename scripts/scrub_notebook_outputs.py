"""
Author: Joshua Ashkinaze

Description: Finds and clears participant identifiers that debug prints left inside tracked
notebook outputs. A stored output can hold values the source no longer reads, so stripping the
CSV columns is not enough on its own. Matches against the identifier values actually present in
the pre-strip data, so a version string or a coordinate-like number cannot trigger a false hit.
Reports by default; pass --apply to clear the outputs of the offending cells.

Inputs:
    - every tracked .ipynb outside OLD/
    - reference copies of the pre-strip analysis files, for the values to search for

Outputs:
    - stdout: notebook, cell index and how many identifier values each cell's output contains
    - with --apply: those notebooks rewritten with the offending cells' outputs cleared
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
_positional = [a for a in sys.argv[1:] if not a.startswith("--")]
REFERENCE = Path(_positional[0]) if _positional else None
ID_COLUMNS = ["IPAddress", "db_prolific_id", "db_auto_pid", "db_uuid",
              "LocationLatitude", "LocationLongitude"]
MIN_LEN = 6      # ignore short values that collide with ordinary numbers

###############
# Loading functions
###############

def identifier_values(reference_dir):
    """Every identifier value in the pre-strip reference copies, as strings."""
    values = set()
    for path in sorted(reference_dir.glob("*.csv")):
        try:
            head = pd.read_csv(path, nrows=0)
        except Exception:
            continue
        cols = [c for c in ID_COLUMNS if c in head.columns]
        if not cols:
            continue
        df = pd.read_csv(path, usecols=cols, low_memory=False)
        for col in cols:
            values |= {v for v in df[col].dropna().astype(str) if len(v) >= MIN_LEN}
    return values


def tracked_notebooks():
    out = subprocess.run(["git", "ls-files", "*.ipynb"], cwd=ROOT,
                         capture_output=True, text=True).stdout.split("\n")
    return [r for r in out if r and not r.startswith("OLD/")]

###############
# Apply
###############

if REFERENCE is None or not REFERENCE.is_dir():
    sys.exit("usage: scrub_notebook_outputs.py [--apply] <dir-with-pre-strip-csvs>")

apply = "--apply" in sys.argv
values = identifier_values(REFERENCE)
print(f"searching notebook outputs for {len(values)} identifier values\n")

total = 0
for rel in tracked_notebooks():
    path = ROOT / rel
    nb = json.loads(path.read_text())
    hits = {}
    for i, cell in enumerate(nb["cells"]):
        if not cell.get("outputs"):
            continue
        text = json.dumps(cell["outputs"])
        found = sum(1 for v in values if v in text)
        if found:
            hits[i] = found
    if not hits:
        continue
    total += sum(hits.values())
    print(f"{rel}")
    for i, n in hits.items():
        print(f"    cell {i}: {n} identifier values in output")
    if apply:
        for i in hits:
            nb["cells"][i]["outputs"] = []
            nb["cells"][i]["execution_count"] = None
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
        print(f"    -> cleared {len(hits)} cell outputs")

print(f"\n{total} identifier values found in notebook outputs"
      + ("" if apply else "  (re-run with --apply to clear them)"))
