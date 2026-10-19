"""
Author: Joshua Ashkinaze

Description: Checks that a clean checkout can rebuild the manuscript. Generated tables and figures
are gitignored, so they exist only if something in the pipeline writes them. When a generating cell
is rewritten or deleted, the stale file stays on disk and main.tex keeps compiling locally while a
fresh clone fails. An asset is fine if it is committed, because then a clone has it, or if a
tracked notebook or script writes it. It fails only when it is neither.

Run from the repository root. Exits non-zero on any failure so run_all.sh stops rather than
shipping a manuscript nobody else can rebuild.

Inputs:
    - main.tex: the \\input and \\includegraphics paths, and \\graphicspath for figure lookup
    - every tracked .ipynb, .py and .R file: searched for the name of each asset

Outputs:
    - stdout: one line per asset, then a summary; exit status 1 if any asset fails
"""

import re
import subprocess
import sys
from pathlib import Path

MANUSCRIPT = Path("main.tex")
SOURCE_SUFFIXES = {".ipynb", ".py", ".R"}
INPUT_PATTERN = re.compile(r"\\input\{([^}]+)\}")
GRAPHICS_PATTERN = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
GRAPHICSPATH_PATTERN = re.compile(r"\\graphicspath\{(.+)\}")

###############
# Collecting
###############

def git_files():
    """Every path git tracks."""
    listing = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True)
    return {Path(p) for p in listing.stdout.split("\n") if p}


def graphics_roots(tex):
    """Directories \\includegraphics searches, plus the repository root itself."""
    match = GRAPHICSPATH_PATTERN.search(tex)
    roots = re.findall(r"\{([^{}]+)\}", match.group(1)) if match else []
    return [Path(".")] + [Path(r) for r in roots]


def manuscript_assets(tex):
    """Every asset main.tex reads, as (kind, path-as-written)."""
    assets = [("input", raw) for raw in INPUT_PATTERN.findall(tex)]
    assets += [("figure", raw) for raw in GRAPHICS_PATTERN.findall(tex)]
    # \newcommand wrappers build paths from arguments, which we cannot resolve statically
    return [(kind, raw) for kind, raw in assets if "#" not in raw]

###############
# Checking
###############

def resolve(kind, raw, roots):
    """Where the asset lives on disk, or None if nothing matches."""
    candidates = [Path(raw)] if kind == "input" else [root / raw for root in roots]
    candidates += [c.with_suffix(".tex") for c in list(candidates) if not c.suffix]
    return next((c for c in candidates if c.exists()), None)


def producers(path, sources):
    """Tracked sources that write this asset.

    A writer names the file either in full, as savefig("mechanisms.pdf"), or by the stem alone,
    as save(table, "ruled_out_ux"), where the caller appends the suffix. Matching a bare stem
    anywhere would fire on ordinary prose, so the stem only counts when it is delimited. Notebook
    sources are read as raw JSON, where a quote arrives escaped, so a backslash delimits too.
    """
    stem, name = path.stem, path.name
    quoted = re.compile(rf"""[\\"'`]{re.escape(stem)}[\\"'`]""")
    return sorted(str(p) for p, text in sources.items()
                  if name in text or quoted.search(text))

###############
# Apply
###############

tex = MANUSCRIPT.read_text()
tracked = git_files()
sources = {p: p.read_text(errors="ignore")
           for p in tracked if p.suffix in SOURCE_SUFFIXES and p.exists()}
roots = graphics_roots(tex)

assets = manuscript_assets(tex)
failures = []
for kind, raw in assets:
    path = resolve(kind, raw, roots)
    if path is None:
        status, detail = "MISSING", "no file on disk"
    elif path in tracked:
        status, detail = "committed", "in git, so a clone has it"
    else:
        made_by = producers(path, sources)
        status = "generated" if made_by else "UNPRODUCED"
        detail = made_by[0] if made_by else "gitignored and no tracked source writes it"
    if status.isupper():
        failures.append((raw, detail))
    print(f"  {status:11s} {kind:6s} {raw:56s} {detail}")

print(f"\n{len(assets)} assets checked, {len(failures)} failing")
for raw, detail in failures:
    print(f"  {raw}: {detail}")
sys.exit(1 if failures else 0)
