"""
analyze_qualtrics_experiment.py — Analysis layer on top of download_qualtrics_experiment.py.

Does everything that tool does (merge a raw Qualtrics export with this app's user-data and
journal-data), plus three things that came up repeatedly doing this by hand:

  1. Auto-detects which deployment environment (working/staging/production) each row's `url`
     points to, and pulls/matches each row against its OWN environment instead of one branch
     for the whole file. This is what a mixed test file (one row hit staging, one hit
     production) needs — under the single-branch tool, one of those rows always shows up as a
     false "_unmatched" simply because it was looked up in the wrong database.
  2. For every row that's still unmatched, explains why using Qualtrics's own completion
     signals (screen_out, Finished, Progress, done, story_written/completion) instead of
     leaving it as a bare "_unmatched" for a human to re-investigate by hand. Most unmatched
     rows turn out to be comprehension-check screen-outs or survey abandoners who never
     reached the app at all — not a data problem.
  3. Optionally pre-filters to only "complete" rows before merging, based on a column set you
     specify (e.g. the post-manipulation DV battery — naming varies per study, so this is a
     flag, not a hardcoded list). Before filtering, it always reports how many rows are fully
     present / fully absent / partially present across that column set, and refuses to filter
     at all if it would drop every row — that pattern almost always means the column set
     spans a randomized survey branch or a sparse "pick a few" design rather than one battery
     everyone sees in full, and silently returning zero rows would hide that.

Usage:
    python analyze_qualtrics_experiment.py <csv_path> [options]

Arguments:
    csv_path  Raw Qualtrics CSV export (3-row header auto-detected, same as
              download_qualtrics_experiment.py)

Options:
    --pid-col        Embedded data column with the Prolific ID (default: 'pid')
    --status-col     Embedded data column with the condition (default: 'treat_status')
    --url-col        Embedded data column with the event URL (default: 'url')
    --default-branch Branch to use for rows with no url (e.g. control) or an unrecognized
                      url host (default: 'production')
    --complete-cols  Comma-separated column names that must ALL be non-null for a row to
                      count as a completed submission. Rows failing this are dropped before
                      merging. Omit to skip this filter entirely.
    --output         Output CSV path (default: auto-generated, timestamped)

Output:
    Same schema as download_qualtrics_experiment.py, plus:
      - db_branch: which branch (working/staging/production) the row was actually matched
        against
      - unmatched_reason: for any row whose user_status ends in '_unmatched', a best-effort
        explanation ('screened_out (...)', 'did_not_finish (...)', 'marked_not_done',
        'genuine_mismatch (...)', or 'unclear (inspect manually)')

Example:
    python analyze_qualtrics_experiment.py quals/Aim_Satellite_Final_August+20,+2026_17.03.csv
    python analyze_qualtrics_experiment.py qualtrics_export.csv --complete-cols dv_dist_concrete,dv_dist_personal
"""

import argparse
from datetime import datetime

import pandas as pd

from download_qualtrics_experiment import (
    build_journal_index,
    build_pid_event_index,
    compute_all_valid,
    extract_event_id,
    load_qualtrics_csv,
    match_row,
    normalize_pid,
)
from mongo_util import retrieve_database_data


def detect_branch_from_url(url):
    """Guesses which deployed environment a row's url points to. None for
    rows with no url (e.g. control) or something unparseable."""
    if not isinstance(url, str) or not url.strip():
        return None
    if "staging" in url:
        return "staging"
    if "localhost" in url or "127.0.0.1" in url:
        return "working"
    if "railway.app" in url or url.startswith("http"):
        return "production"
    return None


def _present(value):
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    return str(value).strip() != ""


def classify_unmatched(row):
    """Best-effort explanation for why a treat/placebo row didn't match a DB record,
    using whichever of Qualtrics's own completion signals are present in the row."""
    screen_out = row.get("screen_out")
    if _present(screen_out):
        return f"screened_out ({screen_out})"

    finished = row.get("Finished")
    if _present(finished) and str(finished).strip().lower() in ("false", "0", "no"):
        progress = row.get("Progress")
        return f"did_not_finish (Progress={progress})"

    progress = row.get("Progress")
    if _present(progress):
        try:
            if float(progress) < 100:
                return f"did_not_finish (Progress={progress})"
        except (TypeError, ValueError):
            pass

    done = row.get("done")
    if _present(done) and str(done).strip().lower() in ("false", "0", "no"):
        return "marked_not_done"

    if _present(row.get("story_written")) or _present(row.get("completion")):
        return "genuine_mismatch (survey completed, no matching app record)"

    return "unclear (inspect manually)"


def check_completeness(df, cols):
    """Returns (n_complete, n_empty, n_partial) for how many rows have all/none/some of
    the given columns non-null."""
    counts = df[cols].notna().sum(axis=1)
    n_complete = int((counts == len(cols)).sum())
    n_empty = int((counts == 0).sum())
    n_partial = len(df) - n_complete - n_empty
    return n_complete, n_empty, n_partial


def apply_completeness_filter(df, cols):
    n_complete, n_empty, n_partial = check_completeness(df, cols)
    total = len(df)
    print(f"Completeness check on {cols}:")
    print(f"  complete: {n_complete}, empty: {n_empty}, partial: {n_partial} (of {total})")
    if n_partial > 0:
        print(
            "  WARNING: some rows have SOME but not all of these columns filled. This "
            "usually means the columns span a randomized survey branch or a sparse "
            "'pick a few' design rather than one battery everyone sees in full — "
            "filtering on the full set may drop rows that are actually complete for "
            "the branch they were shown. Inspect these rows before trusting this filter."
        )
    if n_complete == 0:
        raise SystemExit(
            "Completeness filter would drop every row - almost certainly the wrong "
            "column set (see warning above). Aborting instead of silently returning "
            "zero rows."
        )
    filtered = df[df[cols].notna().all(axis=1)].reset_index(drop=True)
    print(f"Kept {len(filtered)} / {total} rows after completeness filter.\n")
    return filtered


def merge_completions_multi_branch(completions, default_branch):
    """Like download_qualtrics_experiment.merge_completions, but pulls and matches each
    treat/placebo row against the branch its OWN url points to, falling back to
    default_branch for control rows and rows with no recognizable url."""
    branches_needed = set(completions["_branch"].dropna())
    if completions["_branch"].isna().any():
        branches_needed.add(default_branch)
    if not branches_needed:
        branches_needed = {default_branch}

    print("Environment detected from url column:")
    print(completions["_branch"].value_counts(dropna=False).to_string())
    print()

    indexes = {}
    for branch in sorted(branches_needed):
        print(f"Pulling user-data and journal-data from {branch}...")
        user_docs = retrieve_database_data(branch, "user-data")
        journal_docs = retrieve_database_data(branch, "journal-data")
        indexes[branch] = {
            "user": build_pid_event_index(user_docs),
            "journal": build_journal_index(journal_docs),
        }
    print()

    rows = []
    for _, row in completions.iterrows():
        row_dict = row.to_dict()
        status = row_dict["_status_norm"]
        branch = row_dict["_branch"] or default_branch

        if status in ("control", "treat", "placebo"):
            idx = indexes[branch]
            user_status, user_fields, journal_data_match, journal_fields = match_row(
                row_dict, idx["user"], idx["journal"]
            )
            row_dict["db_branch"] = branch
        else:
            user_status, user_fields, journal_data_match, journal_fields = None, {}, None, {}
            row_dict["db_branch"] = None

        row_dict["user_status"] = user_status
        row_dict.update(user_fields)
        row_dict["journal_data_match"] = journal_data_match
        row_dict.update(journal_fields)
        row_dict["all_valid"] = compute_all_valid(user_status, journal_data_match)

        if isinstance(user_status, str) and user_status.endswith("_unmatched"):
            row_dict["unmatched_reason"] = classify_unmatched(row_dict)

        rows.append(row_dict)

    return pd.DataFrame(rows).drop(columns=["_pid_norm", "_status_norm", "_branch"])


def main():
    parser = argparse.ArgumentParser(
        description="Merge a Qualtrics export with this app's data, auto-detecting "
        "environment per row and explaining unmatched rows.",
        epilog=(
            "Example:\n"
            "  python analyze_qualtrics_experiment.py qualtrics_export.csv\n"
            "  python analyze_qualtrics_experiment.py qualtrics_export.csv "
            "--complete-cols dv_dist_concrete,dv_dist_personal"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("csv_path", help="Raw Qualtrics CSV export.")
    parser.add_argument("--pid-col", default="pid")
    parser.add_argument("--status-col", default="treat_status")
    parser.add_argument("--url-col", default="url")
    parser.add_argument(
        "--default-branch",
        default="production",
        choices=["working", "staging", "production"],
        help="Branch for rows with no url (e.g. control) or an unrecognized url host.",
    )
    parser.add_argument(
        "--complete-cols",
        default=None,
        help="Comma-separated columns that must ALL be non-null for a row to be kept.",
    )
    parser.add_argument("--output", default=None, help="Output CSV path.")
    args = parser.parse_args()

    completions = load_qualtrics_csv(args.csv_path)
    for col in (args.pid_col, args.status_col, args.url_col):
        if col not in completions.columns:
            raise SystemExit(
                f"CSV is missing column '{col}'. Found columns: {list(completions.columns)}"
            )

    if args.complete_cols:
        cols = [c.strip() for c in args.complete_cols.split(",")]
        missing = [c for c in cols if c not in completions.columns]
        if missing:
            raise SystemExit(f"--complete-cols references missing columns: {missing}")
        completions = apply_completeness_filter(completions, cols)

    completions["_pid_norm"] = completions[args.pid_col].apply(normalize_pid)
    completions["_status_norm"] = (
        completions[args.status_col].astype(str).str.strip().str.lower().replace({"nan": None})
    )
    completions["assigned_event_id"] = completions[args.url_col].apply(extract_event_id)
    completions["_branch"] = completions[args.url_col].apply(detect_branch_from_url)

    result = merge_completions_multi_branch(completions, args.default_branch)

    output = args.output or (
        f"{datetime.now().strftime('%Y_%m_%d__%H:%M:%S')}_qualtrics_analysis.csv"
    )
    result.to_csv(output, index=False)

    print(f"Wrote {len(result)} rows to {output}")
    print("user_status counts:")
    print(result["user_status"].value_counts(dropna=False).to_string())
    print("journal_data_match counts:")
    print(result["journal_data_match"].value_counts(dropna=False).to_string())
    print(f"all_valid: {result['all_valid'].sum()} / {len(result)}")
    if "unmatched_reason" in result.columns and result["unmatched_reason"].notna().any():
        print("\nunmatched_reason breakdown:")
        print(result["unmatched_reason"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
