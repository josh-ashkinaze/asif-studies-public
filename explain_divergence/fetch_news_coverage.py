"""
Author: Joshua Ashkinaze

Description: Counts United States news stories mentioning each study's policy issue, to test
whether science funding was more publicly salient than satellite cybersecurity. Queries the Media Cloud US national collection for exact phrases and
records both the matching count and the corpus total, so counts can be normalized.

Inputs:
    - .env: MEDIACLOUD_API_KEY

Outputs:
    - data/news_coverage.csv: story counts by topic, query scope and window

Notes: Both topics use one two-year window, 1 September 2024 to 1 September 2026, so the corpus
    total is the same for both and raw counts compare directly.

    The exact-phrase queries are the fair comparison: both topics are matched on specificity. The
    broader queries are included to show the result is not an artifact of phrasing, but they favour
    science funding, whose vocabulary is more varied.

    Media Cloud rate-limits aggressively. Each query retries with a pause; a full run takes a few
    minutes. The script exits without querying if the CSV already exists; delete it to re-fetch.
"""

import datetime as dt
import os
import time
import warnings

import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent     # explain_divergence/
ROOT = HERE.parent                         # repo root
OUT_DIR = HERE / "data"
load_dotenv(ROOT / ".env")
import mediacloud.api as mc

warnings.filterwarnings("ignore")

US_NATIONAL = [34412234]
OUT = OUT_DIR / "news_coverage.csv"

QUERIES = {
    ("Science funding", "narrow"): '"science funding"',
    ("Satellite cybersecurity", "narrow"): '"satellite cybersecurity" OR "space cybersecurity"',
    ("Science funding", "broad"):
        '"science funding" OR "research funding" OR "NIH funding" OR "scientific research funding"',
    ("Satellite cybersecurity", "broad"):
        '"satellite cybersecurity" OR "satellite hacking" OR "satellite cyberattack" OR "space cybersecurity"',
}
WINDOWS = {
    "2024-09-01 to 2026-09-01": (dt.date(2024, 9, 1), dt.date(2026, 9, 1)),   # one window for both topics
}

###############
# Query functions
###############

def count(api, query, start, end, tries=4, pause=30):
    """Story count for one query and window, retrying through rate limits."""
    for attempt in range(tries):
        try:
            return api.story_count(query, start, end, collection_ids=US_NATIONAL)
        except Exception as exc:
            if attempt == tries - 1:
                raise
            print(f"    retry {attempt + 1}: {str(exc)[:60]}")
            time.sleep(pause)

###############
# Apply
###############

if OUT.exists():
    print(f"{OUT} already exists; delete it to re-fetch.")
    raise SystemExit(0)

api = mc.SearchApi(os.getenv("MEDIACLOUD_API_KEY"))
rows = []
for (topic, scope), query in QUERIES.items():
    for window, (start, end) in WINDOWS.items():
        print(f"  {topic} / {scope} / {window}")
        r = count(api, query, start, end)
        rows.append({"topic": topic, "scope": scope, "window": window,
                     "relevant": r["relevant"], "total": r["total"],
                     "per_10k": 1e4 * r["relevant"] / r["total"]})

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print(f"\nwrote {OUT}")
print(df[df.scope == "narrow"].to_string(index=False, formatters={"per_10k": "{:.2f}".format}))
