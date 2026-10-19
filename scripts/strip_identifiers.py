"""
Author: Joshua Ashkinaze

Description: Removes direct identifiers and unused sensitive intake free-text from the committed
clean analysis files. Qualtrics ships an IP address, geolocation and the Prolific ID with every
response, and the personalization intake collected occupation, location, health and financial
free-text; none of it is read by any analysis in this repo. Refuses to drop a column that any
tracked source file references, so an accidental deletion cannot break an analysis.

Run from the repo root. Idempotent: columns already gone are reported and skipped.

Inputs:
    - satellite/data/clean/analysis_data.csv
    - sci_fund_real/data/clean/analysis_data.csv
    - sci_fund_real/data/clean/analysis_data_all.csv

Outputs:
    - the same three files, rewritten without the dropped columns, with text cells holding a
      Prolific ID blanked, and with Submission id replaced by a prolific_matched flag

Notes: Notebook outputs can also contain values; scripts/scrub_notebook_outputs.py handles those.
    This does not fully de-identify the data. Participants' own stories (journal_content) and
    the personalized scenarios built from their intake answers (journal_shown_to_user,
    journal_autocomplete_context) are required by the analyses and still contain personal detail;
    intake answers appear verbatim inside those scenarios for some participants. Rewriting these
    files also leaves every earlier revision in git history untouched.
"""

import json
import re
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

TARGETS = [
    "satellite/data/clean/analysis_data.csv",
    "sci_fund_real/data/clean/analysis_data.csv",
    "sci_fund_real/data/clean/analysis_data_all.csv",
]

# Qualtrics and Prolific direct identifiers. db_auto_pid embeds the Prolific ID in 95% of rows
# and matches its 24-hex format in 99.7%, so it is a second copy of the same identifier;
# db_uuid is the app database's per-participant key, which links back to the same person.
# pid is the Prolific ID itself; unique_id, url, journal_uuid and the event IDs are app keys
# that link back to the database; meta_Version and meta_Resolution fingerprint the device.
IDENTIFIERS = [
    "IPAddress", "LocationLatitude", "LocationLongitude", "ExternalReference",
    "RecipientEmail", "RecipientFirstName", "RecipientLastName",
    "db_prolific_id", "db_auto_pid", "db_uuid", "Submission id",
    "pid", "unique_id", "url", "journal_uuid",
    "assigned_event_id", "db_event_id", "journal_event_id",
    "meta_Version", "meta_Resolution", "_recaptcha",
]

# Personalization intake, collected at app sign-in. Self-reported health, finances, location,
# occupation and personal relationships. Sensitive, and unused by every analysis here.
INTAKE = [
    # Study 1 (science funding)
    "db_age", "db_gender", "db_occupation", "db_location",
    "db_health_concern", "db_health_relationship", "db_financial_situation",
    "db_stress_management", "db_typical_friday_afternoon",
    "db_favorite_celebrity", "db_why_admire", "db_what_would_discuss",
    "db_reaction_to_surprises", "db_specific_item_carried",
    "db_what_kind_interaction", "db_favorite_artist", "db_music_genre",
    "db_listening_setting", "db_what_doing_while_listening",
    "db_sonic_quality", "db_color_palette",
    # Study 2 (satellite cybersecurity)
    "db_state", "db_metro_type", "db_daily_dependency", "db_cash_situation",
    "db_person_to_reach", "db_reason_to_reach",
]

DROP = IDENTIFIERS + INTAKE

# Prolific IDs are 24 lowercase hex characters. Some participants typed theirs into a free-text
# answer, so any text cell holding one is blanked.
PROLIFIC_ID_RE = r"\b[0-9a-f]{24}\b"

###############
# Safety functions
###############

def tracked_sources():
    """Tracked files that could actually read a column: code, notebooks, LaTeX."""
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.split("\n")
    # get_data.py and strip_identifiers.py carry the drop lists themselves; the two
    # *_qualtrics_experiment.py tools build these columns from the raw export upstream of get_data.py
    skip_names = ("get_data.py", "strip_identifiers.py",
                  "download_qualtrics_experiment.py", "analyze_qualtrics_experiment.py")
    return [r for r in out if r and not r.startswith("OLD/")
            and r.endswith((".py", ".ipynb", ".R", ".tex"))
            and Path(r).name not in skip_names]


def cell_sources(path):
    """Notebook code, excluding stored outputs: a column name printed in an output is not a use."""
    try:
        nb = json.loads(path.read_text(errors="ignore"))
    except (OSError, ValueError):
        return ""
    return "\n".join("".join(c.get("source", [])) for c in nb.get("cells", []))


def referenced_in_code(column):
    """Tracked source files whose *code* reads this column: quoted ("pid"), attribute (df.pid)
    or R ($pid) access. A bare substring would match pid3 or LaTeX's \\url."""
    use = re.compile(rf"""["']{re.escape(column)}["']|[.$]{re.escape(column)}\b""")
    hits = []
    for rel in tracked_sources():
        path = ROOT / rel
        text = cell_sources(path) if rel.endswith(".ipynb") else path.read_text(errors="ignore")
        if use.search(text):
            hits.append(rel)
    return hits


def blank_typed_ids(df):
    """Blank text cells holding a Prolific ID. Returns how many cells were blanked."""
    text_cols = df.select_dtypes(include="object").columns
    hits = df[text_cols].apply(lambda s: s.astype(str).str.contains(PROLIFIC_ID_RE))
    df[text_cols] = df[text_cols].mask(hits)
    return int(hits.to_numpy().sum())

###############
# Apply
###############

blocked = {c: refs for c in DROP if (refs := referenced_in_code(c))}
for column, refs in blocked.items():
    print(f"KEEPING {column}: referenced by {', '.join(refs)}")

droppable = [c for c in DROP if c not in blocked]
print(f"\n{len(droppable)} of {len(DROP)} candidate columns are unreferenced and will be dropped\n")

for rel in TARGETS:
    path = ROOT / rel
    # round_trip parses floats exactly; the default parser can shift a value by one ULP on rewrite
    df = pd.read_csv(path, low_memory=False, float_precision="round_trip")
    before = len(df.columns)
    if "Submission id" in df.columns:   # keep only whether the participant matched Prolific
        df["prolific_matched"] = df["Submission id"].notna()
    present = [c for c in droppable if c in df.columns]
    df = df.drop(columns=present)
    blanked = blank_typed_ids(df)
    if not present and not blanked:
        print(f"{rel}: already clean ({len(df.columns)} cols)")
        continue
    df.to_csv(path, index=False)
    print(f"{rel}: {before} -> {len(df.columns)} cols, dropped {len(present)} "
          f"({sum(c in IDENTIFIERS for c in present)} identifiers, "
          f"{sum(c in INTAKE for c in present)} intake), blanked {blanked} typed Prolific IDs")
