"""
Author: Joshua Ashkinaze
Description: Merge the AIM Satellite Qualtrics export with live MongoDB data,
apply pre-registered exclusion criteria, and output a clean analysis-ready CSV.

Uses src/download_qualtrics_experiment.py (copied from the autocomplete app) for the DB merge so
that the same matching logic (URL → event_id, prolific_id + event_id → user record,
uuid + event_id → journal record) is shared across studies.

Inputs:
    - QUALTRICS_CSV: raw Qualtrics export (skiprows=[1,2])
    - MongoDB production branch (pulled live via retrieve_database_data)
    - Prolific demographic CSVs in data/demo_data/

Outputs:
    - data/clean/analysis_data.csv: one row per analysis-eligible participant
      with all DVs, indices, and parsed autocomplete stats (ac_*)

Notes: Exits without rebuilding if the output already exists; delete it to rebuild.
"""

import hashlib
import re
import sys
import os
import glob
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
QUALTRICS_CSV = "data/qualtrics/Aim_Satellite_Final_September+3,+2026_14.10.csv"
OUTPUT_CSV    = "data/clean/analysis_data.csv"

MAD_N          = 3
LOW_FAM_CUTOFF = 2

####################
# Column constants
####################

MI_COLS        = ["mi_friend", "mi_cat", "mi_sunset", "mi_door", "mi_bonfire"]
RW_COLS        = ["rw_journal", "rw_write"]
IRI_FA_COLS    = ["iri_fa1", "iri_fa2", "iri_fa3", "iri_fa4"]
IRI_PC_COLS    = ["iri_pc1", "iri_pc2", "iri_pc3", "iri_pc4"]
IRI_PD_COLS    = ["iri_pd1", "iri_pd2", "iri_pd3", "iri_pd4"]
TRANSPORT_COLS = ["t1_could_pic_self", "t2_mentally_involved", "t3_wanted_to_learn", "t4_affected_emo"]
AI_AT_COLS     = ["ai_at1", "ai_at2", "ai_at3", "ai_at4"]

DIST_COLS   = ["dv_dist_concrete", "dv_dist_personal", "dv_dist_aware", "dv_dist_relevant"]
IMP_COLS    = ["dv_at_imp_personal_3", "dv_at_imp_compare_3"]
SPEND_COLS  = ["dv_at_spend_reduce_3", "dv_at_spend_priority_3"]
ACTION_COLS = ["dv_at_action_action_3", "dv_at_action_bill_3", "dv_at_action_oppose_3"]

SYS_COLS    = ["sys_easy", "sys_fun", "sys_helped", "sys_tailored", "sys_realistic", "sys_agency_1"]

DEMO_DATA_GLOB = "data/demo_data/prolific_demographic_export_*.csv"

# Participants who manually reported errors on Prolific, stored as hash_pid() of the Prolific ID
MANUAL_ERROR_PIDS = {
    "df510e00a4f81823cfb8e9d490c065d21242369a4f2781c73b135f4e027f063a",
    "c313743bf1312ae9f0150b7dc04d0a1b4c898a5311a8aa96283eba7a8389bb7c",
    "9ed772cafa19f6c54b4e27a2142734aa3c67384453eab0b96d35926114598002",
    "b474b1eefd10c46dea9b893253ca0eb32e761cf77d663687835df54433585839",
}

PROLIFIC_ID_RE = r"\b[0-9a-f]{24}\b"

# Prolific statuses that count as a valid completion record
PROLIFIC_VALID_STATUSES = {"APPROVED", "AWAITING REVIEW"}

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
    os_str = str(row.get("meta_Operating System", "")).lower()
    device = str(row.get("device", "")).lower()
    return int("ios" in os_str or "android" in os_str or device == "mobile")


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

    # Drop missing treat_status
    n = len(df)
    df = df[df["treat_status"].notna()].copy()
    log.append(f"  Missing treat_status: dropped {n - len(df)}")

    # Incomplete: keep only rows where demos_7pt answered AND all primary DVs answered.
    # demos_7pt (ideology) is the last question in the Qualtrics survey, so requiring it keeps
    # only participants who answered all Qualtrics questions.
    post_at_dvs = [c for c in (DIST_COLS + IMP_COLS + SPEND_COLS + ACTION_COLS) if c in df.columns]
    demos_ok = df["demos_7pt"].notna() if "demos_7pt" in df.columns else pd.Series(True, index=df.index)
    dvs_ok   = df[post_at_dvs].notna().all(axis=1) if post_at_dvs else pd.Series(True, index=df.index)
    n = len(df)
    df = df[demos_ok & dvs_ok].copy()
    log.append(f"  Incomplete (demos_7pt or primary DVs missing): dropped {n - len(df)}")

    # Drop test PIDs
    n = len(df)
    df = df[df["pid"].astype(str).str.strip().str.lower().ne("test")].copy()
    log.append(f"  Test PID: dropped {n - len(df)}")

    # Manual Prolific error reports
    n = len(df)
    df = df[~df["pid"].map(hash_pid).isin(MANUAL_ERROR_PIDS)].copy()
    log.append(f"  Manual Prolific error report: dropped {n - len(df)}")

    # Non-anonymous (preview/test)
    if "DistributionChannel" in df.columns:
        n = len(df)
        df = df[df["DistributionChannel"].astype(str).str.lower() == "anonymous"].copy()
        log.append(f"  Non-anonymous (preview/test): dropped {n - len(df)}")

    # Mobile
    df["is_mobile"] = df.apply(is_mobile, axis=1)
    n = len(df)
    df = df[df["is_mobile"] == 0].copy()
    log.append(f"  Mobile device: dropped {n - len(df)}")

    # Safari (no Chrome/CriOS/FxiOS/EdgiOS/OPR)
    def _is_safari(row):
        browser = str(row.get("meta_Browser", "")).lower()
        if "safari" not in browser:
            return 0
        return 0 if re.search(r"chrome|crios|fxios|edgios|opr", browser) else 1

    df["is_safari"] = df.apply(_is_safari, axis=1)
    n = len(df)
    df = df[df["is_safari"] == 0].copy()
    log.append(f"  Safari browser: dropped {n - len(df)}")

    # Serious errors (treat + placebo only): reported error AND follow-up is a serious phrase
    if "error" in df.columns and "error_follow" in df.columns:
        reported_error = df["error"].astype(str).str.strip().str.lower() == "yes"
        serious_phrase = df["error_follow"].apply(is_serious_error).astype(bool)
        in_app_arm     = df["treat_status"].isin(["treat", "placebo"])
        df["is_serious_error"] = (in_app_arm & reported_error & serious_phrase).astype(int)
        n = len(df)
        df = df[df["is_serious_error"] == 0].copy()
        log.append(f"  Serious app error: dropped {n - len(df)}")

    # Attention check
    if "at_check" in df.columns:
        n = len(df)
        df["_at_check_num"] = pd.to_numeric(df["at_check"].apply(extract_number), errors="coerce")
        df = df[df["_at_check_num"] == 25].copy()
        log.append(f"  Failed attention check (at_check != 25): dropped {n - len(df)}")

    # Duration outliers (per condition)
    df["duration"] = pd.to_numeric(df["Duration (in seconds)"], errors="coerce")
    duration_flags = flag_duration_outliers(df, "duration", "treat_status")
    n = len(df)
    df = df[~duration_flags].copy()
    log.append(f"  Duration outlier (>{MAD_N} MADs below condition median): dropped {n - len(df)}")

    # reCAPTCHA
    if "Q_RecaptchaScore" in df.columns:
        n = len(df)
        df["_recaptcha"] = pd.to_numeric(df["Q_RecaptchaScore"], errors="coerce")
        df = df[df["_recaptcha"].isna() | (df["_recaptcha"] >= 0.5)].copy()
        log.append(f"  reCAPTCHA < 0.5: dropped {n - len(df)}")

    print(f"Exclusions (started N={n0}):")
    for line in log:
        print(line)
    print(f"  → Final N after Qualtrics exclusions: {len(df)}")
    return df.reset_index(drop=True)


####################
# Index construction
####################

def build_numeric_cols(df):
    # Distance items: Likert 1-9 (extract_number needed)
    for col in DIST_COLS:
        if col in df.columns:
            df[col] = df[col].apply(extract_number)

    # Attitude items: 0-100 sliders (already numeric, but coerce)
    for col in IMP_COLS + SPEND_COLS + ACTION_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Other Likert items
    for col in (MI_COLS + RW_COLS + IRI_FA_COLS + IRI_PC_COLS + IRI_PD_COLS
                + TRANSPORT_COLS + AI_AT_COLS):
        if col in df.columns:
            df[col] = df[col].apply(extract_number)

    for col in SYS_COLS:
        if col in df.columns:
            df[col] = df[col].apply(extract_number)

    if "familiar" in df.columns:
        df["familiar"] = df["familiar"].apply(extract_number)

    # ── Distance index: higher = more closeness/concreteness (no sign flip) ─���
    dist_available = [c for c in DIST_COLS if c in df.columns]
    if dist_available:
        z_dist = pd.concat([z_score(df[c]) for c in dist_available], axis=1)
        df["distance_index"] = z_score(z_dist.mean(axis=1))
    df["closeness_index"] = df.get("distance_index", pd.Series(np.nan, index=df.index))

    # ── Attitude sub-scales ───────────────────────────────────────────────
    imp_available = [c for c in IMP_COLS if c in df.columns]
    if imp_available:
        z_imp = pd.concat([z_score(df[c]) for c in imp_available], axis=1)
        df["importance_index"] = z_score(z_imp.mean(axis=1))

    spend_available = [c for c in SPEND_COLS if c in df.columns]
    if spend_available:
        z_spend = pd.concat([z_score(df[c]) for c in spend_available], axis=1)
        df["spending_index"] = z_score(z_spend.mean(axis=1))

    action_available = [c for c in ACTION_COLS if c in df.columns]
    if action_available:
        z_action = pd.concat([z_score(df[c]) for c in action_available], axis=1)
        df["action_index"] = z_score(z_action.mean(axis=1))

    sub_indices = [c for c in ["importance_index", "spending_index", "action_index"]
                   if c in df.columns]
    if sub_indices:
        z_subs = pd.concat([z_score(df[c]) for c in sub_indices], axis=1)
        df["attitude_index"] = z_score(z_subs.mean(axis=1))

    # ── IRI sub-scales ────────────────────────────────────────────────────
    if all(c in df.columns for c in IRI_FA_COLS):
        df["iri_fa"] = df[IRI_FA_COLS].mean(axis=1)
    if all(c in df.columns for c in IRI_PC_COLS):
        df["iri_pc"] = df[IRI_PC_COLS].mean(axis=1)
    if all(c in df.columns for c in IRI_PD_COLS):
        df["iri_pd"] = df[IRI_PD_COLS].mean(axis=1)

    if all(c in df.columns for c in MI_COLS):
        df["mi_index"] = df[MI_COLS].mean(axis=1)

    if all(c in df.columns for c in AI_AT_COLS):
        df["ai_attitude_index"] = df[AI_AT_COLS].mean(axis=1)

    t_available = [c for c in TRANSPORT_COLS if c in df.columns]
    if t_available:
        df["transport_index"] = df[t_available].mean(axis=1)

    # ── Ordinal demographics ──────────────────────────────────────────────
    df["educ_num"]     = df["demo_educ"].str.strip().map(EDUC_ORDER) if "demo_educ" in df.columns else np.nan
    df["educ_num"]     = pd.to_numeric(df["educ_num"], errors="coerce")
    df["educ_num"]     = df["educ_num"].fillna(df["educ_num"].median())
    df["income_num"]   = df["demo_income"].map(INCOME_ORDER) if "demo_income" in df.columns else np.nan
    df["ideology_num"] = df["demos_7pt"].map(IDEOLOGY_ORDER) if "demos_7pt" in df.columns else np.nan
    df["age"]          = pd.to_numeric(df.get("demo_age", pd.Series(np.nan, index=df.index)), errors="coerce")
    df["gender_male"]  = (df["demo_gender"].str.strip().str.lower() == "male").astype(float) if "demo_gender" in df.columns else np.nan
    df["low_knowledge"]= (df["familiar"] <= LOW_FAM_CUTOFF).astype(int) if "familiar" in df.columns else 0

    return df


####################
# Prolific demographics
####################

PROLIFIC_MASKED = ["CONSENT_REVOKED", "DATA_EXPIRED"]
VALID_PID3 = ["Democrat", "Republican", "Independent"]


def clean_prolific_columns(df, prolific_cols):
    """Keep no masked Prolific values in the dataset.

    Prolific masks the demographics of returned submissions (CONSENT_REVOKED) and of expired
    data (DATA_EXPIRED); those cells become missing. pid3 is missing unless it is a real party
    label. sup_pro_demos is 1 if the Prolific demographics are available and 0 otherwise,
    whatever the reason; the raw Prolific status is dropped.
    """
    df[prolific_cols] = df[prolific_cols].replace(PROLIFIC_MASKED, np.nan)
    df["pid3"] = df["pid3"].where(df["pid3"].isin(VALID_PID3))
    df["sup_pro_demos"] = df["pid3"].notna().astype(int)
    return df.drop(columns="Status_prolific")


def merge_prolific_demographics(df):
    """Merge Prolific demographic exports keyed on pid."""
    files = sorted(glob.glob(DEMO_DATA_GLOB))
    if not files:
        print("  No Prolific demographic files found; skipping demo merge")
        return df
    demo_dfs = [pd.read_csv(f) for f in files]
    demo = pd.concat(demo_dfs, ignore_index=True)
    # Standardize participant id column
    pid_col = next((c for c in demo.columns if "participant" in c.lower() and "id" in c.lower()), None)
    if pid_col:
        demo = demo.rename(columns={pid_col: "pid"})
    # Pull political affiliation if present
    pol_col = next((c for c in demo.columns if "political" in c.lower()), None)
    if pol_col:
        demo = demo.rename(columns={pol_col: "pid3"})
    # Qualtrics also has a Status column, so name the Prolific one explicitly
    demo = demo.rename(columns={"Status": "Status_prolific"})
    demo["pid"] = demo["pid"].astype(str).str.strip()
    # Keep the best-status row if a participant appears in more than one export
    rank = demo["Status_prolific"].map({"APPROVED": 0, "AWAITING REVIEW": 1}).fillna(2)
    demo = demo.assign(_rank=rank).sort_values("_rank", kind="stable").drop(columns="_rank")
    demo = demo.drop_duplicates(subset=["pid"])

    keep_demo_cols = ["pid"] + [c for c in demo.columns if c != "pid"]
    before = len(df)
    df = df.merge(demo[keep_demo_cols], on="pid", how="left", suffixes=("", "_prolific"))
    df = clean_prolific_columns(df, [c for c in keep_demo_cols if c != "pid"])
    matched = df["pid3"].notna().sum() if "pid3" in df.columns else "N/A"
    print(f"  Demo merge: {matched}/{before} participants matched pid3")
    return df


####################
# Autocomplete stats
####################

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
    if isinstance(history, list) and not (total_words is np.nan or (isinstance(total_words, float) and np.isnan(total_words))):
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

    word_stats = df.apply(_compute_word_stats, axis=1)
    df["ac_total_words"]   = word_stats["ac_total_words"]
    df["ac_ai_words"]      = word_stats["ac_ai_words"]
    df["ac_prop_ai_words"] = word_stats["ac_prop_ai_words"]

    MIN_WORDS = 5
    has_journal = df["ac_received"].notna()
    low_words   = has_journal & (df["ac_total_words"].fillna(0) < MIN_WORDS)
    print(f"  Participants with journal but <{MIN_WORDS} words written:")
    for cond in ["treat", "placebo", "vignette"]:
        n = (low_words & (df["treat_status"] == cond)).sum()
        total = has_journal & (df["treat_status"] == cond)
        print(f"    {cond}: {n} / {total.sum()}")
    df.loc[low_words, ["ac_received", "ac_accepted", "ac_rejected",
                       "ac_acceptance_rate", "ac_total_words",
                       "ac_ai_words", "ac_prop_ai_words"]] = np.nan

    return df


####################
# Fuzzy story matching
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

        if best_score >= threshold and best_jdoc:
            print(f"  → MATCHED (score {best_score:.3f})")
            matched_count += 1
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

def main():
    os.makedirs("data/clean", exist_ok=True)

    print("Loading Qualtrics data...")
    completions = load_qualtrics_csv(QUALTRICS_CSV)
    print(f"  Raw rows: {len(completions)}")

    # Qualtrics named this item demos_5pt, but it carries the seven-point ideology
    # scale (Extremely liberal through Extremely conservative). Rename on load so the
    # column does not imply a five-point scale downstream.
    completions = completions.rename(columns={"demos_5pt": "demos_7pt"})

    # Some participants entered their Prolific relay address instead of their ID
    completions["pid"] = completions["pid"].apply(clean_prolific_pid)

    # Normalize columns expected by merge_completions.
    # The DB index already handles both prolific_id and auto_pid (see download_qualtrics_experiment.py).
    # On the Qualtrics side, pid = Prolific ID entered by participant; use it directly.
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

    # Rename control → vignette AFTER merge (merge_completions checks status == "control" internally)
    df["treat_status"] = df["treat_status"].replace({"control": "vignette"})
    df["user_status"]        = df["user_status"].str.replace("_control", "_vignette", regex=False)
    df["journal_data_match"] = df["journal_data_match"].str.replace("_control", "_vignette", regex=False)

    print(f"  user_status: {df['user_status'].value_counts(dropna=False).to_dict()}")
    print(f"  all_valid: {df['all_valid'].sum()} / {len(df)}")

    print("\nFuzzy-matching unmatched treat/placebo rows by story text...")
    df = _fuzzy_match_unmatched(df, journal_docs)

    print("\nApplying exclusions...")
    df = apply_exclusions(df)

    print("\nMerging Prolific demographics...")
    df = merge_prolific_demographics(df)

    print("\nBuilding indices...")
    df = build_numeric_cols(df)
    df = parse_autocomplete_stats(df)

    if "RecordedDate" in df.columns:
        df["RecordedDate"] = pd.to_datetime(df["RecordedDate"], errors="coerce", utc=True)
        df = df.sort_values("RecordedDate").drop_duplicates(subset=["pid"], keep="last")

    df = df.reset_index(drop=True)

    print(f"\n{'='*50}")
    print(f"Final N: {len(df)}")
    print(f"treat_status: {df['treat_status'].value_counts(dropna=False).to_dict()}")
    if "familiar" in df.columns:
        low_n = (df["familiar"] <= LOW_FAM_CUTOFF).sum()
        print(f"Low familiarity (familiar <= {LOW_FAM_CUTOFF}): {low_n} / {len(df)} "
              f"({100 * low_n / len(df):.1f}%)")
    if "journal_data_match" in df.columns:
        print(f"Journal matched: {(df['journal_data_match'] == '_treat_matched').sum()} treat, "
              f"{(df['journal_data_match'] == '_placebo_matched').sum()} placebo")
    print_missingness_tests(df)

    # Drop direct identifiers and the sensitive personalization intake before writing. No
    # analysis in this repo reads these, and the written file is committed.
    # Keep this list in sync with scripts/strip_identifiers.py.
    _DROP_ALWAYS = [
        "IPAddress", "LocationLatitude", "LocationLongitude", "ExternalReference",
        "RecipientEmail", "RecipientFirstName", "RecipientLastName",
        "db_prolific_id", "db_auto_pid", "db_uuid", "Submission id",
        "pid", "unique_id", "url", "journal_uuid",
        "assigned_event_id", "db_event_id", "journal_event_id",
        "meta_Version", "meta_Resolution", "_recaptcha",
        "db_age", "db_gender", "db_occupation", "db_location",
        "db_state", "db_metro_type", "db_daily_dependency", "db_cash_situation",
        "db_person_to_reach", "db_reason_to_reach",
        "db_favorite_celebrity", "db_why_admire", "db_what_would_discuss",
        "db_reaction_to_surprises", "db_specific_item_carried",
        "db_what_kind_interaction", "db_favorite_artist", "db_music_genre",
        "db_listening_setting", "db_what_doing_while_listening",
        "db_sonic_quality", "db_color_palette",
    ]
    # Keep only whether the participant matched a Prolific record, not the submission ID
    df["prolific_matched"] = df["Submission id"].notna()
    _dropped_ids = [c for c in _DROP_ALWAYS if c in df.columns]
    df = df.drop(columns=_dropped_ids)
    print(f"\nDropped {len(_dropped_ids)} identifier/intake columns before writing")
    print(f"Blanked {blank_typed_ids(df)} text cells holding a Prolific ID")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {OUTPUT_CSV}  (N={len(df)})")
    return df


if __name__ == "__main__":
    if os.path.exists(OUTPUT_CSV):
        print(f"{OUTPUT_CSV} already exists; delete it to rebuild from the raw data.")
    else:
        main()
