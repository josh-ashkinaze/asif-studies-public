"""
download_db.py — Download MongoDB collection data to a local JSON file.

Usage:
    python download_db.py <branch> <collection> [options]

Arguments:
    branch      The deployment branch to pull from: 'working', 'staging', or 'production'
    collection  The MongoDB collection to download:
                  user-data    — form submissions (one doc per participant)
                  journal-data — journal entries (one doc per completed writing session)
                  error-log    — API errors caught during autocomplete (RateLimitError, etc.)

Options:
    --event_id    Filter by experiment event key (e.g. 'event_placebo_synesthesia').
                  Only applies to user-data and journal-data. Ignored for error-log.
    --start_date  Include only records inserted on or after this date (YYYY-MM-DD).
    --end_date    Include only records inserted on or before this date (YYYY-MM-DD).
    --normalize   Replace unicode typography in content/shown_to_user/autocomplete_context
                  with ASCII equivalents (smart quotes, non-breaking spaces, dashes, etc.).
    --more_stats  Append extended stats to autocomplete_stats: total_chars, total_words,
                  ai_accept_chars, ai_accept_words, prop_ai_chars, prop_ai_words.

Output:
    A JSON file named: <timestamp>_<branch>_<collection>[_<event_id>][_<date_range>].json

Examples:
    # Download all production journal entries
    python download_db.py production journal-data

    # Download journals for a specific event with clean text and extended stats
    python download_db.py production journal-data --event_id event_10 --normalize --more_stats

    # Download journals from a date range
    python download_db.py staging journal-data --start_date 2026-03-01 --end_date 2026-03-05

    # Download everything from March 1st onwards with all options
    python download_db.py production journal-data --event_id event_10 --start_date 2026-03-01 --normalize --more_stats

    # Download all staging user submissions
    python download_db.py staging user-data

    # Download all API errors from production
    python download_db.py production error-log
"""

import argparse
import json
from datetime import datetime

from mongo_util import retrieve_database_data


UNICODE_REPLACEMENTS = [
    ("\u00a0", " "),   # non-breaking space
    ("\u2018", "'"),   # left single quote
    ("\u2019", "'"),   # right single quote / apostrophe
    ("\u201c", '"'),   # left double quote
    ("\u201d", '"'),   # right double quote
    ("\u2013", "-"),   # en dash
    ("\u2014", "-"),   # em dash
    ("\u2026", "..."), # ellipsis
]


def normalize_content(text):
    for char, replacement in UNICODE_REPLACEMENTS:
        text = text.replace(char, replacement)
    return text


def compute_more_stats(item):
    content = item.get("content", "") or ""
    history = item.get("autocomplete_history", [])

    total_chars = len(content)
    total_words = len(content.split()) if content.strip() else 0

    accepted_texts = [
        e["suggestion_text"] for e in history
        if e.get("status") == "accepted" and e.get("suggestion_text")
    ]
    ai_accept_chars = sum(len(t) for t in accepted_texts)
    ai_accept_words = sum(len(t.split()) for t in accepted_texts)

    prop_ai_chars = round(ai_accept_chars / total_chars, 4) if total_chars else 0
    prop_ai_words = round(ai_accept_words / total_words, 4) if total_words else 0

    return {
        "total_chars": total_chars,
        "total_words": total_words,
        "ai_accept_chars": ai_accept_chars,
        "ai_accept_words": ai_accept_words,
        "prop_ai_chars": prop_ai_chars,
        "prop_ai_words": prop_ai_words,
    }


def parse_date(value):
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date '{value}'. Use YYYY-MM-DD format.")


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve MongoDB collection to JSON file.",
        epilog=(
            "Examples:\n"
            "  python download_db.py production journal-data\n"
            "  python download_db.py production user-data --event_id event_placebo_synesthesia\n"
            "  python download_db.py staging user-data --event_id event_101\n"
            "  python download_db.py production error-log"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "branch",
        choices=["working", "staging", "production"],
        help="The branch to download data from.",
    )
    parser.add_argument(
        "collection",
        choices=["user-data", "journal-data", "error-log"],
        help="The name of the collection.",
    )
    parser.add_argument(
        "--event_id",
        default=None,
        help=(
            "Filter by experiment event key (e.g. 'event_placebo_synesthesia', 'event_101'). "
            "Applies to user-data and journal-data only. Ignored for error-log."
        ),
    )
    parser.add_argument(
        "--start_date",
        default=None,
        type=parse_date,
        metavar="YYYY-MM-DD",
        help="Include only records inserted on or after this date.",
    )
    parser.add_argument(
        "--end_date",
        default=None,
        type=parse_date,
        metavar="YYYY-MM-DD",
        help="Include only records inserted on or before this date.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize unicode typography in content (smart quotes, non-breaking spaces, dashes, etc.).",
    )
    parser.add_argument(
        "--more_stats",
        action="store_true",
        help="Append total_chars, total_words, ai_accept_chars, ai_accept_words, prop_ai_chars, prop_ai_words to autocomplete_stats.",
    )

    args = parser.parse_args()

    event_id = None if args.collection == "error-log" else args.event_id
    items = retrieve_database_data(
        args.branch, args.collection,
        event_id=event_id,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    current_time = datetime.now().strftime("%Y_%m_%d__%H:%M:%S")
    event_suffix = f"_{args.event_id}" if args.event_id and event_id else ""
    date_suffix = ""
    if args.start_date and args.end_date:
        date_suffix = f"_{args.start_date.strftime('%Y-%m-%d')}_to_{args.end_date.strftime('%Y-%m-%d')}"
    elif args.start_date:
        date_suffix = f"_from_{args.start_date.strftime('%Y-%m-%d')}"
    elif args.end_date:
        date_suffix = f"_until_{args.end_date.strftime('%Y-%m-%d')}"
    filename = f"{current_time}_{args.branch}_{args.collection}{event_suffix}{date_suffix}.json"

    if args.collection == "user-data":
        # auto_pid was added later, so older records don't have the key at all.
        # Backfill it as null so every record has a consistent schema.
        for item in items:
            item.setdefault("auto_pid", None)

    if args.more_stats:
        for item in items:
            if "autocomplete_stats" not in item:
                item["autocomplete_stats"] = {}
            item["autocomplete_stats"].update(compute_more_stats(item))

    if args.normalize:
        for item in items:
            for field in ("content", "shown_to_user", "autocomplete_context"):
                if field in item and isinstance(item[field], str):
                    item[field] = normalize_content(item[field])

    with open(filename, "w") as json_file:
        json.dump(items, json_file, default=str, indent=4)

    print(f"Downloaded {len(items)} records to {filename}")


main()