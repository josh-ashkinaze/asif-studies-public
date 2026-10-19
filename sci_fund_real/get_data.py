"""
Author: Joshua Ashkinaze
Description: Merge the AIM science-funding Qualtrics export with live MongoDB data,
apply pre-registered exclusion criteria, and output a clean analysis-ready CSV.

Uses src/download_qualtrics_experiment.py (copied from the autocomplete app) for the DB merge so
that the same matching logic (URL → event_id, prolific_id + event_id → user record,
uuid + event_id → journal record) is shared across studies.

Inputs:
    - QUALTRICS_CSV: raw Qualtrics export (3-row header auto-handled)
    - MongoDB production branch (pulled live via retrieve_database_data)

Outputs:
    - OUTPUT_CSV: one row per analysis-eligible participant with all DVs, indices,
      and parsed autocomplete stats (ac_*)

Notes: Exits without rebuilding if the output already exists; delete it to rebuild.
"""

import hashlib
import re
import sys
import os
import numpy as np
import pandas as pd

####################
# Shared database and Qualtrics merge code (src/, copied from the autocomplete app)
####################

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from mongo_util import retrieve_database_data
from download_qualtrics_experiment import (
    load_qualtrics_csv,
    merge_completions,
    normalize_pid,
    extract_event_id,
)

####################
# Config
####################

BRANCH        = "production"
QUALTRICS_CSV = "data/qualtrics_experiment/AIM+project_Science+Funding_2026-03-14_August+26,+2026_11.29.csv"

OUTPUT_CSV      = "data/clean/analysis_data.csv"
OUTPUT_CSV_ALL  = "data/clean/analysis_data_all.csv"
MIN_WORDS      = 0   # At least 0 words, no pre-reg threshold
PROLIFIC_ID_RE = r"\b[0-9a-f]{24}\b"

MAD_N         = 3
LOW_FAM_CUTOFF = 2

####################
# Column constants
####################

MI_COLS        = ["mi_bonfire", "mi_cat", "mi_door", "mi_friend", "mi_sunset"]
RW_COLS        = ["rw_journal", "rw_write"]
IRI_FA_COLS    = ["iri_fa1", "iri_fa2", "iri_fa3", "iri_fa4"]
IRI_PC_COLS    = ["iri_pc1", "iri_pc2", "iri_pc3", "iri_pc4"]
IRI_PD_COLS    = ["iri_pd1", "iri_pd2", "iri_pd3", "iri_pd4"]
TRANSPORT_COLS = ["t1_could_pic_self", "t2_mentally_involved", "t3_wanted_to_learn", "t4_affected_emo"]
AI_AT_COLS     = ["ai_at1", "ai_at2", "ai_at3", "ai_at4"]

DEMO_DATA_GLOB = "data/demo_data/prolific_demographic_export_*.csv"

EDUC_ORDER = {
    "Less than high school": 1,
    "High school diploma or equivalent": 2,
    "Some college, no degree": 3,
    "Associate degree": 4,
    "Bachelor's degree (e.g., BA, BS)": 5,
    "Master's degree (e.g., MA, MS, MBA)": 6,
    "Professional degree (e.g., JD, MD)": 7,
    "Doctoral degree (e.g., PhD, EdD)": 8,
    "Other (please specify):": np.nan,
}

INCOME_ORDER = {
    "Less than $20,000": 1,
    "$20,000–$39,999": 2,
    "$40,000–$59,999": 3,
    "$60,000–$79,999": 4,
    "$80,000–$99,999": 5,
    "$100,000–$149,999": 6,
    "$150,000 or more": 7,
}

IDEOLOGY_ORDER = {
    "Extremely liberal": 1,
    "Liberal": 2,
    "Slightly liberal": 3,
    "Moderate; middle of the road": 4,
    "Slightly conservative": 5,
    "Conservative": 6,
    "Extremely conservative": 7,
}

####################
# Helpers
####################

def hash_pid(pid):
    """SHA-256 of a Prolific ID, so hardcoded participant lists do not expose the raw ID."""
    return hashlib.sha256(str(pid).encode()).hexdigest()


def blank_typed_ids(df):
    """Blank text cells holding a Prolific ID (some participants typed theirs into an answer)."""
    text_cols = df.select_dtypes(include="object").columns
    hits = df[text_cols].apply(lambda s: s.astype(str).str.contains(PROLIFIC_ID_RE))
    df[text_cols] = df[text_cols].mask(hits)
    return int(hits.to_numpy().sum())


def extract_number(value):
    if pd.isna(value):
        return np.nan
    match = re.search(r"-?\d+(\.\d+)?", str(value))
    return float(match.group()) if match else np.nan


def clean_prolific_pid(value):
    """Some participants typed pid@email.prolific.com; keep only the part before the @."""
    if isinstance(value, str) and value.strip().lower().endswith("@email.prolific.com"):
        return value.strip().rsplit("@", 1)[0]
    return value


def z_score(series):
    return (series - series.mean()) / series.std(ddof=0)


def mad_sd(data, scaling_factor=1.4826):
    data = np.asarray(data, dtype=float)
    abs_devs = np.abs(data - np.median(data))
    return scaling_factor * np.median(abs_devs)


def flag_duration_outliers(df, duration_col, condition_col, n_mad=MAD_N):
    flags = pd.Series(False, index=df.index)
    for cond, grp in df.groupby(condition_col):
        med = grp[duration_col].median()
        mad = mad_sd(grp[duration_col].dropna())
        cutoff = med - n_mad * mad
        flags.loc[grp.index] = grp[duration_col] < cutoff
    return flags


def is_mobile(row):
    browser = str(row.get("meta_Browser", "")).lower()
    os_str  = str(row.get("meta_Operating System", "")).lower()
    return int("mobile" in browser or "ios" in os_str)


def is_serious_error(value):
    if pd.isna(value):
        return 0
    serious = [
        "I never got a single writing assistant completion",
        "The app itself gave an error message",
        "I was not able to type at all",
    ]
    return int(any(s.lower() in str(value).lower() for s in serious))


####################
# Exclusions
####################

def apply_exclusions(df):
    n0 = len(df)
    log = []

    # Drop test PIDs
    n = len(df)
    df = df[df["pid"].astype(str).str.strip().str.lower().ne("test")].copy()
    log.append(f"  Test PID: dropped {n - len(df)}")

    # Drop preview / non-anonymous Qualtrics responses
    if "DistributionChannel" in df.columns:
        n = len(df)
        df = df[df["DistributionChannel"].astype(str).str.lower() == "anonymous"]
        log.append(f"  Non-anonymous (preview/test): dropped {n - len(df)}")

    df = df[df["Finished"].astype(str).str.lower() == "true"].copy()
    log.append(f"  Finished != True: dropped {n0 - len(df)}")

    df["is_mobile"] = df.apply(is_mobile, axis=1)
    n = len(df)
    df = df[df["is_mobile"] == 0]
    log.append(f"  Mobile: dropped {n - len(df)}")

    df["is_serious_error"] = df["error_follow"].apply(is_serious_error)
    n = len(df)
    df = df[df["is_serious_error"] == 0]
    log.append(f"  Serious app error: dropped {n - len(df)}")


    df["duration"] = pd.to_numeric(df["Duration (in seconds)"], errors="coerce")
    duration_flags = flag_duration_outliers(df, "duration", "treat_status")
    n = len(df)
    df = df[~duration_flags]
    log.append(f"  Duration outlier (>{MAD_N} MADs below condition median): dropped {n - len(df)}")

    n = len(df)
    df = df.dropna(subset=["post_at_actual_1"])
    log.append(f"  Missing primary DV: dropped {n - len(df)}")

    n = len(df)
    df = df[pd.to_numeric(df["at_check"], errors="coerce") == 25]
    log.append(f"  Failed attention check (at_check != 25): dropped {n - len(df)}")

    print(f"Exclusions (started N={n0}):")
    for line in log:
        print(line)
    print(f"  → Final N: {len(df)}")
    return df.reset_index(drop=True)


####################
# Index construction
####################

def build_numeric_cols(df):
    for col in ["post_dist_fxme", "post_dist_relme", "post_at_actual_1",
                "post_at_likeyou", "familiar"]:
        df[col] = df[col].apply(extract_number)

    for col in (MI_COLS + RW_COLS + IRI_FA_COLS + IRI_PC_COLS + IRI_PD_COLS
                + TRANSPORT_COLS + AI_AT_COLS):
        if col in df.columns:
            df[col] = df[col].apply(extract_number)

    df["closeness_index"] = z_score(
        0.5 * (z_score(df["post_dist_fxme"]) + z_score(df["post_dist_relme"]))
    )
    df["attitude_index"] = z_score(
        0.5 * (z_score(df["post_at_actual_1"]) + z_score(df["post_at_likeyou"]))
    )
    df["closeness_raw_index"] = df["post_dist_fxme"] + df["post_dist_relme"]
    df["attitude_raw_index"] = df["post_at_actual_1"] + df["post_at_likeyou"]

    if all(c in df.columns for c in MI_COLS):
        df["mi_index"] = df[MI_COLS].mean(axis=1)
    if all(c in df.columns for c in TRANSPORT_COLS):
        df["transport_index"] = df[TRANSPORT_COLS].mean(axis=1)
    if all(c in df.columns for c in IRI_FA_COLS):
        df["iri_fa"] = df[IRI_FA_COLS].mean(axis=1)
    if all(c in df.columns for c in IRI_PC_COLS):
        df["iri_pc"] = df[IRI_PC_COLS].mean(axis=1)
    if all(c in df.columns for c in IRI_PD_COLS):
        df["iri_pd"] = df[IRI_PD_COLS].mean(axis=1)
    if all(c in df.columns for c in AI_AT_COLS):
        df["ai_attitude_index"] = df[AI_AT_COLS].mean(axis=1)

    df["educ_num"]    = df["demo_educ"].str.strip().map(EDUC_ORDER)
    df["educ_num"]    = df["educ_num"].fillna(df["educ_num"].median())
    df["income_num"]  = df["demo_income"].map(INCOME_ORDER)
    df["ideology_num"]= df["demos_7pt"].map(IDEOLOGY_ORDER)
    df["age"]         = pd.to_numeric(df["demo_age"], errors="coerce")
    df["gender_male"] = (df["demo_gender"].str.strip().str.lower() == "male").astype(float)
    df["low_knowledge"] = (df["familiar"] <= LOW_FAM_CUTOFF).astype(int)

    return df


VALID_PID3 = ["Democrat", "Republican", "Independent"]


def load_prolific_export():
    """One row per participant from the Prolific demographic exports: pid, pid3, Status_prolific.

    A participant can appear in more than one export (a returned submission, then a completed one).
    Returned submissions carry masked demographics (CONSENT_REVOKED), so keep the best-status row.
    """
    import glob as _glob
    files = sorted(_glob.glob(DEMO_DATA_GLOB))
    if not files:
        return None
    demo = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    demo = demo.rename(columns={"Participant id": "pid",
                                "U.s. political affiliation": "pid3",
                                "Status": "Status_prolific"})
    demo["pid"] = demo["pid"].astype(str).str.strip()
    rank = demo["Status_prolific"].map({"APPROVED": 0, "AWAITING REVIEW": 1}).fillna(2)
    demo = demo.assign(_rank=rank).sort_values("_rank", kind="stable")
    return demo[["pid", "pid3", "Status_prolific"]].drop_duplicates(subset=["pid"])


def clean_prolific_columns(df):
    """Keep no masked Prolific values in the dataset.

    pid3 is missing unless it is a real party label (masked rows read CONSENT_REVOKED or
    DATA_EXPIRED). sup_pro_demos is 1 if the Prolific demographics are available and 0
    otherwise, whatever the reason; the raw Prolific status is dropped.
    """
    df["pid3"] = df["pid3"].where(df["pid3"].isin(VALID_PID3))
    df["sup_pro_demos"] = df["pid3"].notna().astype(int)
    return df.drop(columns="Status_prolific")


def merge_prolific_demographics(df):
    """Merge Prolific demographic exports (pid3 etc.) keyed on Participant id = pid."""
    demo = load_prolific_export()
    if demo is None:
        print("  No Prolific demographic files found; skipping demo merge")
        return df
    before = len(df)
    df = clean_prolific_columns(df.merge(demo, on="pid", how="left"))
    matched = df["pid3"].notna().sum()
    print(f"  Demo merge: {matched}/{before} participants matched pid3")
    return df


def _compute_word_stats(row):
    """Compute total_words and prop_ai_words from journal_content and journal_autocomplete_history."""
    content = row.get("journal_content")
    history = row.get("journal_autocomplete_history")

    total_words = np.nan
    if isinstance(content, str) and content.strip():
        total_words = len(content.split())

    prop_ai_words = np.nan
    ai_words = np.nan
    if isinstance(history, str):
        import ast as _ast
        try:
            history = _ast.literal_eval(history)
        except Exception:
            history = None
    if isinstance(history, list) and not np.isnan(total_words if total_words is not np.nan else float("nan")):
        accepted_texts = [
            e.get("suggestion_text", "") for e in history
            if isinstance(e, dict) and e.get("status") == "accepted"
        ]
        ai_words = sum(len(t.split()) for t in accepted_texts if t)
        # Accepted-suggestion words are counted against the final story, so a participant
        # who accepts text and then deletes some can exceed 1. Cap so the value stays a proportion.
        prop_ai_words = min(round(ai_words / total_words, 4), 1.0) if total_words and total_words > 0 else 0.0

    return pd.Series({"ac_total_words": total_words, "ac_ai_words": ai_words,
                      "ac_prop_ai_words": prop_ai_words})


def parse_autocomplete_stats(df):
    """Unpack journal_autocomplete_stats dict into flat ac_* columns.
    Also computes total_words and prop_ai_words from history/content."""
    stats_col = "journal_autocomplete_stats"
    if stats_col not in df.columns:
        return df

    def safe_parse(val):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return {}
        if isinstance(val, dict):
            return val
        try:
            import ast
            return ast.literal_eval(str(val))
        except Exception:
            return {}

    parsed = df[stats_col].apply(safe_parse)
    for key in ["received", "accepted", "rejected", "acceptance_rate"]:
        df[f"ac_{key}"] = parsed.apply(lambda d: d.get(key, np.nan))

    # Compute word-based stats from raw history and content
    word_stats = df.apply(_compute_word_stats, axis=1)
    df["ac_total_words"]   = word_stats["ac_total_words"]
    df["ac_ai_words"]      = word_stats["ac_ai_words"]
    df["ac_prop_ai_words"] = word_stats["ac_prop_ai_words"]

    MIN_WORDS = 5 # Set autocomplete variables to nullfor 5 words missing
    has_journal = df["ac_received"].notna()
    low_words   = has_journal & (df["ac_total_words"].fillna(0) < MIN_WORDS)
    print(f"  Participants with journal but <{MIN_WORDS} words written:")
    for cond in ["treat", "placebo"]:
        n = (low_words & (df["treat_status"] == cond)).sum()
        total = has_journal & (df["treat_status"] == cond)
        print(f"    {cond}: {n} / {total.sum()}")
    df.loc[low_words, ["ac_received", "ac_accepted", "ac_rejected",
                       "ac_acceptance_rate", "ac_total_words",
                       "ac_ai_words", "ac_prop_ai_words"]] = np.nan

    return df


####################
# Diagnostics
####################

def print_missingness_tests(df):
    """Chi-square tests of whether missing data differ across arms.

    Tests missing Prolific demographics (sup_pro_demos == 0) across all three arms, and null
    autocomplete stats (ac_received) between AIM and placebo, the only arms that used the app.
    """
    from scipy.stats import chi2_contingency

    app = df[df["treat_status"].isin(["treat", "placebo"])]
    tests = [
        ("Missing Prolific demographics, all arms", df["treat_status"], df["sup_pro_demos"].eq(0)),
        ("Null autocomplete stats, AIM vs placebo", app["treat_status"], app["ac_received"].isna()),
    ]
    print("\nMissingness by arm (chi-square):")
    for label, arm, missing in tests:
        table = pd.crosstab(arm, missing).reindex(columns=[False, True], fill_value=0)
        chi2, p, dof, expected = chi2_contingency(table.values, correction=False)
        counts = ", ".join(f"{a} {table.loc[a, True]}/{table.loc[a].sum()}" for a in table.index)
        note = "  (expected count < 5; interpret with caution)" if expected.min() < 5 else ""
        print(f"  {label}: {counts}; chi2({dof})={chi2:.2f}, p={p:.3f}{note}")


####################
# Main
####################

def _fuzzy_match_unmatched(df, journal_docs, threshold=0.65):
    """For treat/placebo rows that failed ID-based matching, attempt fuzzy match
    on story text (Qualtrics story_written vs MongoDB journal_content).
    Logs both stories side by side and updates journal_* columns in-place."""
    from difflib import SequenceMatcher

    def similarity(a, b):
        if not a or not b:
            return 0.0
        a, b = a.strip().lower(), b.strip().lower()
        return SequenceMatcher(None, a[:2000], b[:2000]).ratio()

    # Raw journal docs use "content" (download_qualtrics_experiment prefixes "journal_" later)
    def _jcontent(d):
        return d.get("content") or d.get("journal_content") or ""

    valid_jdocs = [d for d in journal_docs
                   if isinstance(_jcontent(d), str) and _jcontent(d).strip()]

    unmatched_mask = (
        df["user_status"].isin(["_treat_unmatched", "_placebo_unmatched"])
        & df["story_written"].notna()
    )
    candidates = df[unmatched_mask]
    if candidates.empty:
        print("  Fuzzy match: no unmatched treat/placebo rows with story_written")
        return df

    print(f"\n  Fuzzy story matching for {len(candidates)} unmatched rows (threshold={threshold}):")
    matched_count = 0
    for idx, row in candidates.iterrows():
        q_story = str(row["story_written"]).strip()
        best_score, best_jdoc = 0.0, None
        for jdoc in valid_jdocs:
            s = similarity(q_story, _jcontent(jdoc))
            if s > best_score:
                best_score, best_jdoc = s, jdoc

        pid = row.get("pid", idx)
        print(f"\n  --- pid={pid} | best score={best_score:.3f} ---")
        print(f"  [Qualtrics story_written]  : {q_story[:200]}")
        if best_jdoc:
            print(f"  [MongoDB journal_content]  : {_jcontent(best_jdoc)[:200]}")

        # Manual overrides: force-match specific PIDs regardless of score (hash_pid of the ID)
        FORCE_MATCH_PIDS = {"10931237a4d949721bac1d788acb6f6c1fc8dd8dda8730cb2e05cc88d7d15bde"}
        force = hash_pid(str(pid).strip()) in FORCE_MATCH_PIDS

        if (best_score >= threshold or force) and best_jdoc:
            print(f"  → MATCHED (score {best_score:.3f}{'  [MANUAL OVERRIDE]' if force else ''})")
            matched_count += 1
            # Map raw doc fields to the journal_* prefixed column names used in df
            for raw_key in ("content", "autocomplete_history", "autocomplete_stats"):
                df_col = f"journal_{raw_key}"
                if raw_key in best_jdoc:
                    df.at[idx, df_col] = best_jdoc[raw_key]
            status = df.at[idx, "user_status"]
            df.at[idx, "journal_data_match"] = status.replace(
                "_treat_unmatched", "_treat_matched"
            ).replace("_placebo_unmatched", "_placebo_matched")
        else:
            print(f"  → no match (below threshold)")

    print(f"\n  Fuzzy match: {matched_count}/{len(candidates)} newly matched")
    return df


def main():
    print("Loading Qualtrics data...")
    completions = load_qualtrics_csv(QUALTRICS_CSV)
    print(f"  Raw rows: {len(completions)}")

    # Qualtrics named this item demos_5pt, but it carries the seven-point ideology
    # scale (Extremely liberal through Extremely conservative). Rename on load so the
    # column does not imply a five-point scale downstream.
    completions = completions.rename(columns={"demos_5pt": "demos_7pt"})

    # Some participants entered their Prolific relay address instead of their ID
    completions["pid"] = completions["pid"].apply(clean_prolific_pid)

    # Normalize columns expected by merge_completions
    completions["_pid_norm"] = completions["pid"].apply(normalize_pid)
    completions["_status_norm"] = (
        completions["treat_status"].astype(str).str.strip().str.lower()
        .replace({"nan": None})
    )
    completions["assigned_event_id"] = completions["url"].apply(extract_event_id)

    print(f"\nPulling DB data from {BRANCH}...")
    user_docs    = retrieve_database_data(BRANCH, "user-data")
    journal_docs = retrieve_database_data(BRANCH, "journal-data")
    print(f"  User records: {len(user_docs)}")
    print(f"  Journal records: {len(journal_docs)}")

    print("\nMerging Qualtrics with DB...")
    df = merge_completions(completions, user_docs, journal_docs)
    print(f"  user_status: {df['user_status'].value_counts(dropna=False).to_dict()}")
    print(f"  all_valid: {df['all_valid'].sum()} / {len(df)}")

    print("\nFuzzy-matching unmatched treat/placebo rows by story text...")
    df = _fuzzy_match_unmatched(df, journal_docs)

    # Rename control → vignette to match manuscript terminology
    df["treat_status"] = df["treat_status"].replace("control", "vignette")

    print("\nApplying exclusions...")
    df = apply_exclusions(df)

    print("\nMerging Prolific demographics...")
    df = merge_prolific_demographics(df)

    print("\nBuilding indices...")
    df = build_numeric_cols(df)
    df = parse_autocomplete_stats(df)

    # Keep one row per participant (most recent submission)
    df["RecordedDate"] = pd.to_datetime(df["RecordedDate"], errors="coerce", utc=True)
    df = df.sort_values("RecordedDate").drop_duplicates(subset=["pid"], keep="last")

    print(f"\nFinal N: {len(df)}")
    print(f"  treat_status: {df['treat_status'].value_counts(dropna=False).to_dict()}")
    low_fam_n = (df["familiar"] <= LOW_FAM_CUTOFF).sum()
    print(f"  Low familiarity (familiar <= {LOW_FAM_CUTOFF}): {low_fam_n} / {len(df)} "
          f"({100 * low_fam_n / len(df):.1f}%)")
    print(f"  journal matched: {(df['journal_data_match'] == '_treat_matched').sum()} treat, "
          f"{(df['journal_data_match'] == '_placebo_matched').sum()} placebo")
    print_missingness_tests(df)

    # Save full sample (all treat/placebo regardless of word count)
    # Drop direct identifiers and the sensitive personalization intake before anything is
    # written. No analysis in this repo reads these, and the written files are committed.
    # Keep this list in sync with scripts/strip_identifiers.py.
    _DROP_ALWAYS = [
        "IPAddress", "LocationLatitude", "LocationLongitude", "ExternalReference",
        "RecipientEmail", "RecipientFirstName", "RecipientLastName",
        "db_prolific_id", "db_auto_pid", "db_uuid", "Submission id",
        "pid", "unique_id", "url", "journal_uuid",
        "assigned_event_id", "db_event_id", "journal_event_id",
        "meta_Version", "meta_Resolution", "_recaptcha",
        "db_age", "db_gender", "db_occupation", "db_location",
        "db_health_concern", "db_health_relationship", "db_financial_situation",
        "db_stress_management", "db_typical_friday_afternoon",
        "db_favorite_celebrity", "db_why_admire", "db_what_would_discuss",
        "db_reaction_to_surprises", "db_specific_item_carried",
        "db_what_kind_interaction", "db_favorite_artist", "db_music_genre",
        "db_listening_setting", "db_what_doing_while_listening",
        "db_sonic_quality", "db_color_palette",
        "db_state", "db_metro_type", "db_daily_dependency", "db_cash_situation",
        "db_person_to_reach", "db_reason_to_reach",
    ]
    _dropped_ids = [c for c in _DROP_ALWAYS if c in df.columns]
    df = df.drop(columns=_dropped_ids)
    print(f"\nDropped {len(_dropped_ids)} identifier/intake columns before writing")
    print(f"Blanked {blank_typed_ids(df)} text cells holding a Prolific ID")

    df.to_csv(OUTPUT_CSV_ALL, index=False)
    print(f"Wrote {OUTPUT_CSV_ALL}  (full sample, N={len(df)})")

    # Save main analysis sample: treat/placebo must have written ≥MIN_WORDS words
    df["_wc"] = df["journal_content"].fillna("").apply(lambda x: len(str(x).split()))
    df_main = df[
        (df["treat_status"] == "vignette") | (df["_wc"] >= MIN_WORDS)
    ].drop(columns=["_wc"]).reset_index(drop=True)
    dropped_words = len(df) - len(df_main)
    print(f"  Dropped {dropped_words} treat/placebo with <{MIN_WORDS} words written")
    df_main.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {OUTPUT_CSV}  (main sample, N={len(df_main)})")
    print(f"  treat_status: {df_main['treat_status'].value_counts(dropna=False).to_dict()}")


if __name__ == "__main__":
    if os.path.exists(OUTPUT_CSV):
        print(f"{OUTPUT_CSV} already exists; delete it to rebuild from the raw data.")
    else:
        main()
