"""
download_qualtrics_experiment.py — Merge a raw Qualtrics export with this app's user-data and
journal-data, using Qualtrics's own embedded data to know exactly which event each
participant was assigned to.

Qualtrics's `treat_status` embedded field tells us the condition (control / treat / placebo),
and for treat/placebo participants, the `url` embedded field contains the exact event they were
sent to (e.g. '.../user_settings_experiment/event_blackout'). That means we don't have to guess
which event a prolific_id belongs to — we just look up that (prolific_id, event) pair directly
in user-data, and from there bridge to journal-data via uuid.

Usage:
    python download_qualtrics_experiment.py <branch> <csv_path> [options]

Arguments:
    branch    The deployment branch to pull from: 'working', 'staging', or 'production'
    csv_path  Raw Qualtrics CSV export (the 3-row Qualtrics header is auto-detected and
              handled; a pre-cleaned CSV with a single header row also works)

Options:
    --pid-col     Embedded data column with the Prolific ID (default: 'pid')
    --status-col  Embedded data column with the condition (default: 'treat_status'),
                  expected values: 'control', 'treat', 'placebo'
    --url-col     Embedded data column with the event URL for treat/placebo rows
                  (default: 'url')
    --output      Output CSV path (default: auto-generated, timestamped)

Output:
    A CSV with one row per input row, every original column preserved, plus:
      - assigned_event_id: event key parsed out of the url column (treat/placebo only)
      - db_*: fields from the matched user-data record (age, event_id, uuid, prolific_id,
        auto_pid, est_ts, ...), null if unmatched
      - user_status: one of '_control', '_treat_matched', '_treat_unmatched',
        '_placebo_matched', '_placebo_unmatched' (null if treat_status itself is missing,
        e.g. participants screened out before condition assignment)
      - journal_*: fields from the matched journal-data record (content, autocomplete_stats,
        ...), null if the participant never wrote anything or never intook
      - journal_data_match: same five-value scheme as user_status
      - all_valid: 1 if the row is a clean control (both statuses '_control') or a clean
        match (both statuses end in '_matched'), else 0

    Whenever a participant has more than one qualifying record (e.g. they restarted the
    intake form, or wrote more than one journal entry), the most recent one (by DB insertion
    time) is used.

Example:
    python download_qualtrics_experiment.py production qualtrics_export.csv
"""

import argparse
import re
from datetime import datetime

import pandas as pd

from mongo_util import retrieve_database_data

EVENT_ID_RE = re.compile(r"/user_settings_experiment/([^/?#]+)")


def normalize_pid(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    value = str(value).strip()
    return value or None


def extract_event_id(url):
    if not isinstance(url, str):
        return None
    match = EVENT_ID_RE.search(url)
    return match.group(1) if match else None


def load_qualtrics_csv(path):
    """Auto-detects and strips the standard Qualtrics 3-row header (variable name / question
    text / ImportId JSON). Falls back to a plain single-header CSV if it doesn't look like a
    raw export."""
    probe = pd.read_csv(path, header=None, nrows=3)
    looks_like_export = len(probe) >= 3 and probe.iloc[2].astype(str).str.contains("ImportId").any()
    if looks_like_export:
        return pd.read_csv(path, skiprows=[1, 2])
    return pd.read_csv(path)


def build_pid_event_index(user_docs):
    """Maps (normalized participant id, event_id) -> matching user-data docs. Indexed by
    both prolific_id and auto_pid since either can be how a participant is identified."""
    index = {}
    for doc in user_docs:
        pids = {normalize_pid(doc.get(field)) for field in ("prolific_id", "auto_pid")}
        pids.discard(None)
        event_id = doc.get("event_id")
        for pid in pids:
            index.setdefault((pid, event_id), []).append(doc)
    return index


def build_journal_index(journal_docs):
    """Maps (uuid, event_id) -> matching journal-data docs."""
    index = {}
    for doc in journal_docs:
        uuid = doc.get("uuid")
        if not uuid:
            continue
        index.setdefault((uuid, doc.get("event_id")), []).append(doc)
    return index


def most_recent(docs):
    return max(docs, key=lambda d: d["est_ts"])


def match_row(row, user_index, journal_index):
    """Returns (user_status, user_fields, journal_data_match, journal_fields) for one CSV row."""
    status = row["_status_norm"]

    if status == "control":
        return "_control", {}, "_control", {}

    if status not in ("treat", "placebo"):
        return None, {}, None, {}

    pid = row["_pid_norm"]
    expected_event = row["assigned_event_id"]

    candidates = user_index.get((pid, expected_event), []) if pid and expected_event else []
    if not candidates:
        return f"_{status}_unmatched", {}, f"_{status}_unmatched", {}

    best_user = most_recent(candidates)
    user_fields = {f"db_{k}": v for k, v in best_user.items() if k != "_id"}

    journal_candidates = journal_index.get((best_user.get("uuid"), expected_event), [])
    if not journal_candidates:
        return f"_{status}_matched", user_fields, f"_{status}_unmatched", {}

    best_journal = most_recent(journal_candidates)
    journal_fields = {f"journal_{k}": v for k, v in best_journal.items() if k != "_id"}
    return f"_{status}_matched", user_fields, f"_{status}_matched", journal_fields


def compute_all_valid(user_status, journal_data_match):
    if user_status == "_control" and journal_data_match == "_control":
        return 1
    if (
        isinstance(user_status, str)
        and user_status.endswith("_matched")
        and isinstance(journal_data_match, str)
        and journal_data_match.endswith("_matched")
    ):
        return 1
    return 0


def merge_completions(completions, user_docs, journal_docs):
    user_index = build_pid_event_index(user_docs)
    journal_index = build_journal_index(journal_docs)

    rows = []
    for _, row in completions.iterrows():
        row_dict = row.to_dict()
        user_status, user_fields, journal_data_match, journal_fields = match_row(
            row_dict, user_index, journal_index
        )
        row_dict["user_status"] = user_status
        row_dict.update(user_fields)
        row_dict["journal_data_match"] = journal_data_match
        row_dict.update(journal_fields)
        row_dict["all_valid"] = compute_all_valid(user_status, journal_data_match)
        rows.append(row_dict)

    return pd.DataFrame(rows).drop(columns=["_pid_norm", "_status_norm"])


def main():
    parser = argparse.ArgumentParser(
        description="Merge a Qualtrics export with this app's user-data and journal-data.",
        epilog="Example:\n  python download_qualtrics_experiment.py production qualtrics_export.csv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("branch", choices=["working", "staging", "production"])
    parser.add_argument("csv_path", help="Raw Qualtrics CSV export.")
    parser.add_argument("--pid-col", default="pid")
    parser.add_argument("--status-col", default="treat_status")
    parser.add_argument("--url-col", default="url")
    parser.add_argument("--output", default=None, help="Output CSV path.")
    args = parser.parse_args()

    completions = load_qualtrics_csv(args.csv_path)
    for col in (args.pid_col, args.status_col, args.url_col):
        if col not in completions.columns:
            raise SystemExit(
                f"CSV is missing column '{col}'. Found columns: {list(completions.columns)}"
            )

    completions["_pid_norm"] = completions[args.pid_col].apply(normalize_pid)
    completions["_status_norm"] = (
        completions[args.status_col].astype(str).str.strip().str.lower().replace({"nan": None})
    )
    completions["assigned_event_id"] = completions[args.url_col].apply(extract_event_id)

    print(f"Pulling all user-data and journal-data from {args.branch}...")
    user_docs = retrieve_database_data(args.branch, "user-data")
    journal_docs = retrieve_database_data(args.branch, "journal-data")

    result = merge_completions(completions, user_docs, journal_docs)

    output = args.output or (
        f"{datetime.now().strftime('%Y_%m_%d__%H:%M:%S')}_{args.branch}_experiment_merged.csv"
    )
    result.to_csv(output, index=False)

    print(f"Wrote {len(result)} rows to {output}")
    print("user_status counts:")
    print(result["user_status"].value_counts(dropna=False).to_string())
    print("journal_data_match counts:")
    print(result["journal_data_match"].value_counts(dropna=False).to_string())
    print(f"all_valid: {result['all_valid'].sum()} / {len(result)}")


if __name__ == "__main__":
    main()
