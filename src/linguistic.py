"""
Name: Joshua Ashkinaze

Date: 2026-09-12

Description: Shared human-versus-AI language analysis for both studies. Splits each AIM story
into the participant's own words and the accepted AI suggestions, scores both halves on 67 Biber
register features and 43 Empath categories, reports paired Cohen's d_z with sharpened FDR
q-values, cross-validates a classifier separating human from AI text, and fits a log-odds
keyness model. Both studies score the same fixed feature set so their profiles are comparable.

Inputs:
- data/clean/analysis_data.csv, loaded by the caller; needs treat_status,
  journal_autocomplete_history and journal_content

Outputs:
- {stem}_linguistic_features.csv, one row per feature with family, non-zero rate, d_z, p and q
- {stem}_linguistic_cv.csv, per-fold AUC, pair count and feature count
- {stem}_human_vs_ai_profile.pdf, top 30 features by |d_z|
- {stem}_human_ai_keyness.csv, log-odds for every term
- {stem}_keyness_sample.csv, number of document pairs
- {stem}_logodds_human_vs_ai.pdf, top 20 terms per side
"""

import ast

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from empath import Empath
from nltk import pos_tag, word_tokenize
from pybiber import run_biber
from scipy.stats import ttest_rel
from sharp_q_values import compute_q
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from log_odds import LogOddsAnalyzer

BLUE = "#020887"
RED = "#F45B69"          # from helpers.py default palette
MIN_PAIR_WORDS = 5       # drop a pair unless BOTH halves are longer than this
MIN_EMPATH_WORDS = 3     # Empath needs a few tokens to be meaningful
N_TOP_FEATURES = 15      # per side of the d_z profile plot
N_TOP_TERMS = 20         # per side of the keyness plot

ORIGINAL_EMPATH_CATS = [
    "health", "medical_emergency", "suffering", "death", "anticipation",
    "nervousness", "family", "body", "pain", "help", "science",
    "government", "work", "positive_emotion", "negative_emotion", "sadness",
]

# Broad dimensions at the generality of positive_emotion, grouped as LIWC groups psychological
# processes. Scoring all 194 Empath categories instead pulled cross-study agreement in d_z from
# .82 to .67, because most categories fire in too few documents to estimate.
ADDED_EMPATH_CATS = [
    "emotional", "joy", "love", "optimism", "contentment",
    "fear", "anger", "hate", "shame", "horror",
    "confusion", "order",
    "home", "children", "friends", "giving", "sympathy", "trust",
    "communication", "violence",
    "strength", "weakness", "power", "money",
    "healing", "injury",
]
EMPATH_CATS = ORIGINAL_EMPATH_CATS + ADDED_EMPATH_CATS

BIBER_RENAME = {
    "f_01_past_tense":               "Biber: Past tense",
    "f_02_perfect_aspect":           "Biber: Perfect aspect",
    "f_03_present_tense":            "Biber: Present tense",
    "f_04_place_adverbials":         "Biber: Place adverbials",
    "f_05_time_adverbials":          "Biber: Time adverbials",
    "f_06_first_person_pronouns":    "Biber: 1st-person pronouns",
    "f_07_second_person_pronouns":   "Biber: 2nd-person pronouns",
    "f_08_third_person_pronouns":    "Biber: 3rd-person pronouns",
    "f_09_pronoun_it":               "Biber: Pronoun it",
    "f_10_demonstrative_pronoun":    "Biber: Demonstrative pronouns",
    "f_11_indefinite_pronoun":       "Biber: Indefinite pronouns",
    "f_12_proverb_do":               "Biber: Pro-verb do",
    "f_13_wh_question":              "Biber: Direct wh-questions",
    "f_14_nominalization":           "Biber: Nominalizations (-tion/-ment/-ness/-ity)",
    "f_15_gerunds":                  "Biber: Gerunds",
    "f_16_other_nouns":              "Biber: Other nouns",
    "f_17_agentless_passives":       "Biber: Agentless passives",
    "f_18_by_passives":              "Biber: By-passives",
    "f_19_be_main_verb":             "Biber: Be as main verb",
    "f_20_existential_there":        "Biber: Existential there",
    "f_21_that_verb_comp":           "Biber: That-verb complements",
    "f_22_that_adj_comp":            "Biber: That-adjective complements",
    "f_23_wh_clause":                "Biber: Wh-clauses",
    "f_24_infinitives":              "Biber: Infinitives",
    "f_25_present_participle":       "Biber: Present participial clauses",
    "f_26_past_participle":          "Biber: Past participial clauses",
    "f_27_past_participle_whiz":     "Biber: Past participial relative clauses",
    "f_28_present_participle_whiz":  "Biber: Present participial relative clauses",
    "f_29_that_subj":                "Biber: That-relatives (subject)",
    "f_30_that_obj":                 "Biber: That-relatives (object)",
    "f_31_wh_subj":                  "Biber: Wh-relatives (subject)",
    "f_32_wh_obj":                   "Biber: Wh-relatives (object)",
    "f_33_pied_piping":              "Biber: Pied-piping relatives",
    "f_34_sentence_relatives":       "Biber: Sentence relatives",
    "f_35_because":                  "Biber: Causative subordinator (because)",
    "f_36_though":                   "Biber: Concessive subordinator (though/although)",
    "f_37_if":                       "Biber: Conditional subordinator (if/unless)",
    "f_38_other_adv_sub":            "Biber: Other adverbial subordinators",
    "f_39_prepositions":             "Biber: Prepositional phrases",
    "f_40_adj_attr":                 "Biber: Attributive adjectives",
    "f_41_adj_pred":                 "Biber: Predicative adjectives",
    "f_42_adverbs":                  "Biber: Adverbs",
    "f_43_type_token":               "Biber: Type-token ratio",
    "f_44_mean_word_length":         "Biber: Mean word length",
    "f_45_conjuncts":                "Biber: Conjuncts (however/furthermore)",
    "f_46_downtoners":               "Biber: Downtoners (barely/slightly)",
    "f_47_hedges":                   "Biber: Hedges (almost/something like)",
    "f_48_amplifiers":               "Biber: Amplifiers (extremely/perfectly)",
    "f_49_emphatics":                "Biber: Emphatics (really/for sure)",
    "f_50_discourse_particles":      "Biber: Discourse particles",
    "f_51_demonstratives":           "Biber: Demonstratives",
    "f_52_modal_possibility":        "Biber: Possibility modals (can/may/might)",
    "f_53_modal_necessity":          "Biber: Necessity modals (must/should)",
    "f_54_modal_predictive":         "Biber: Predictive modals (will/would)",
    "f_55_verb_public":              "Biber: Public verbs (assert/declare)",
    "f_56_verb_private":             "Biber: Private verbs (believe/know)",
    "f_57_verb_suasive":             "Biber: Suasive verbs (insist/propose)",
    "f_58_verb_seem":                "Biber: Seem/appear",
    "f_59_contractions":             "Biber: Contractions",
    "f_60_that_deletion":            "Biber: That-deletion",
    "f_61_stranded_preposition":     "Biber: Stranded prepositions",
    "f_62_split_infinitve":          "Biber: Split infinitives",
    "f_63_split_auxiliary":          "Biber: Split auxiliaries",
    "f_64_phrasal_coordination":     "Biber: Phrasal co-ordination",
    "f_65_clausal_coordination":     "Biber: Clausal co-ordination",
    "f_66_neg_synthetic":            "Biber: Synthetic negation",
    "f_67_neg_analytic":             "Biber: Analytic negation",
}

###############
# Corpus functions
###############

def human_ai_pairs(df):
    """Split each AIM story into the participant's own words and the accepted AI suggestions.

    Removing each accepted suggestion once makes the two halves partition the text. Returns two
    equal-length lists, aligned by participant, keeping only pairs where both halves exceed
    MIN_PAIR_WORDS.
    """
    human_docs, ai_docs = [], []
    treated = df[df["treat_status"] == "treat"].dropna(
        subset=["journal_autocomplete_history", "journal_content"])
    for _, row in treated.iterrows():
        try:
            history = ast.literal_eval(row["journal_autocomplete_history"])
        except Exception:
            continue
        accepted = " ".join(event["suggestion_text"] for event in history
                            if event.get("status") == "accepted" and event.get("suggestion_text"))
        human = str(row["journal_content"])
        for event in history:
            if event.get("status") == "accepted" and event.get("suggestion_text"):
                human = human.replace(event["suggestion_text"], " ", 1)
        if min(len(human.split()), len(accepted.split())) > MIN_PAIR_WORDS:
            human_docs.append(human)
            ai_docs.append(accepted)
    return human_docs, ai_docs

###############
# Feature functions
###############

def biber_features(human_docs, ai_docs):
    """Biber features for both corpora, in one batched call (polars already multithreads)."""
    human_ids = [f"h_{i}" for i in range(len(human_docs))]
    ai_ids = [f"a_{i}" for i in range(len(ai_docs))]
    corpus = pl.DataFrame({"doc_id": human_ids + ai_ids, "text": human_docs + ai_docs})
    scored = run_biber(corpus).to_pandas().set_index("doc_id")
    scored.columns = [BIBER_RENAME.get(c, "Biber: " + c.split("_", 2)[-1].replace("_", " ").title())
                      for c in scored.columns]
    return (scored.loc[human_ids].reset_index(drop=True),
            scored.loc[ai_ids].reset_index(drop=True))


def empath_features(text, lexicon):
    """Empath category counts per token, or None if the document is too short to score."""
    if not isinstance(text, str) or len(text.split()) < MIN_EMPATH_WORDS:
        return None
    tokens = word_tokenize(text.lower())
    n_tokens = max(len(pos_tag(tokens)), 1)
    counts = lexicon.analyze(text, categories=EMPATH_CATS, normalize=False)
    if not counts:
        return None
    return {f'Empath: {c.replace("_", " ").title()}': counts.get(c, 0.0) / n_tokens
            for c in EMPATH_CATS}



def feature_frames(human_docs, ai_docs):
    """Aligned human and AI feature matrices, keeping only pairs both halves could be scored on."""
    biber_human, biber_ai = biber_features(human_docs, ai_docs)
    print(f"Biber features: {biber_human.shape[1]}")

    lexicon = Empath()
    print(f"Extracting Empath features ({len(EMPATH_CATS)} categories)...")
    empath_human = [empath_features(t, lexicon) for t in human_docs]
    empath_ai = [empath_features(t, lexicon) for t in ai_docs]

    valid = [i for i in range(len(human_docs))
             if empath_human[i] is not None and empath_ai[i] is not None]

    def side(biber, empath):
        return pd.concat([biber.iloc[valid].reset_index(drop=True),
                          pd.DataFrame([empath[i] for i in valid])], axis=1).fillna(0)

    human = side(biber_human, empath_human)
    ai = side(biber_ai, empath_ai)
    print(f"Complete linguistic pairs: {len(valid)}; total features: {human.shape[1]}")
    return human, ai, len(valid)


def feature_family(name):
    """Which dictionary a feature column came from."""
    return name.split(":")[0] if ":" in name else "Other"


def prevalence(human, ai):
    """Share of documents, both halves pooled, in which each feature is non-zero."""
    return pd.concat([human, ai], ignore_index=True).gt(0).mean()


def cohens_d_z(human, ai):
    """Paired Cohen's d_z: mean human-minus-AI difference over its SD. Rows are aligned."""
    diff = np.asarray(human) - np.asarray(ai)
    sd = np.std(diff, ddof=1)
    return diff.mean() / sd if sd > 0 else 0.0


def paired_test(human, ai):
    """Paired t-test p-value for one feature; 1.0 when the differences have no variance."""
    diff = np.asarray(human) - np.asarray(ai)
    if np.std(diff, ddof=1) == 0:
        return 1.0
    return float(ttest_rel(human, ai).pvalue)


def feature_table(human, ai):
    """One row per feature: family, prevalence, paired d_z, p, and q over all features."""
    rates = prevalence(human, ai)
    table = pd.DataFrame({
        "feature": human.columns,
        "family": [feature_family(c) for c in human.columns],
        "nonzero_rate": rates.reindex(human.columns).values,
        "d_z": [cohens_d_z(human[c].values, ai[c].values) for c in human.columns],
        "p": [paired_test(human[c].values, ai[c].values) for c in human.columns],
    })
    table["q"] = compute_q(table.p.values)
    return table.sort_values("d_z", ascending=False).reset_index(drop=True)

###############
# Classifier functions
###############

def classifier_auc(human, ai, n_pairs, n_splits=5):
    """Cross-validated AUC separating human from AI text, grouping both halves of a pair."""
    features = list(human.columns)
    combined = pd.concat([human.assign(label=0), ai.assign(label=1)], ignore_index=True)
    groups = np.tile(np.arange(n_pairs), 2)

    folds = []
    for train_idx, test_idx in GroupKFold(n_splits).split(combined, combined["label"], groups):
        train, test = combined.iloc[train_idx], combined.iloc[test_idx]
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(C=0.5, max_iter=1000, random_state=42))
        model.fit(train[features], train.label)
        folds.append(roc_auc_score(test.label, model.predict_proba(test[features])[:, 1]))
    folds = np.array(folds)
    print(f"AUC (all {len(features)} features): {folds.mean():.3f} +/- {folds.std():.3f}")
    return folds

###############
# Keyness functions
###############

def keyness(human_docs, ai_docs):
    """Log-odds per term; positive means more human, negative more AI.

    Sorted by score then term so ties break alphabetically, keeping the plotted top 20 stable
    across runs.
    """
    analyzer = LogOddsAnalyzer(pseudocount=0.001, to_normalize=True)
    results = analyzer.fit(human_docs + ai_docs,
                           ["human"] * len(human_docs) + ["ai"] * len(ai_docs),
                           ngram_range=(1, 2), stopwords="english",
                           class_order=["ai", "human"])
    return results.sort_values(["log_odds_normalized", "term"]).reset_index(drop=True)

###############
# Plotting functions
###############

def _barh(values, labels, xlabel, path, title=None):
    """Diverging horizontal bars: blue to the right (human), red to the left (AI)."""
    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(values))
    ax.barh(y, values, color=[BLUE if v > 0 else RED for v in values],
            height=0.72, zorder=3, linewidth=0)
    ax.axvline(0, color="#444444", lw=1.0, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=12, labelpad=6)
    ax.set_title("" if title is None else title, pad=12)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(axis="both", length=0, colors="#444444")
    ax.xaxis.grid(True, color="#BBBBBB", linewidth=0.6, zorder=0)
    fig.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=150, transparent=True)
    plt.show()
    print(f"Saved {path}")


def plot_feature_profile(d_series, path):
    """Top features per side by paired d_z."""
    top = (d_series.nsmallest(N_TOP_FEATURES).index.tolist()
           + d_series.nlargest(N_TOP_FEATURES).index.tolist())
    profile = d_series[top].sort_values(ascending=True)
    _barh(profile.values, profile.index.tolist(),
          "Cohen's d  (left = more AI, right = more human)", path)


def plot_keyness(results, path, make_bold):
    """Top terms per side by z-scored log-odds. Assumes keyness() already sorted the rows."""
    plot_df = pd.concat([results.head(N_TOP_TERMS), results.tail(N_TOP_TERMS)]) \
        .sort_values(["log_odds_normalized", "term"]).reset_index(drop=True)
    _barh(plot_df["log_odds_normalized"].values, plot_df["term"],
          "Z-scored log-odds ratio\n(left = AI, right = human)", path,
          title=f"{make_bold('Human vs AI Words')}\nHuman (blue) vs AI (red)")

###############
# Entry points
###############

def run_features(df, table_dir, figure_dir, stem):
    """Feature profile and classifier for one study. Returns (feature table, fold AUCs)."""
    human_docs, ai_docs = human_ai_pairs(df)
    print(f"Pairs: {len(human_docs)}")

    human, ai, n_pairs = feature_frames(human_docs, ai_docs)
    table = feature_table(human, ai)

    for family, group in table.groupby("family"):
        print(f"  {family}: {len(group)} features, {int((group.q < .05).sum())} at sharpened q < .05")
    print(f"  total: {len(table)} features, {int((table.q < .05).sum())} at sharpened q < .05")

    folds = classifier_auc(human, ai, n_pairs)

    table.to_csv(table_dir / f"{stem}_linguistic_features.csv", index=False)
    pd.DataFrame({"fold": range(1, len(folds) + 1), "auc_all": folds,
                  "n_pairs": n_pairs, "n_features": human.shape[1]}).to_csv(
        table_dir / f"{stem}_linguistic_cv.csv", index=False)

    plot_feature_profile(table.set_index("feature")["d_z"],
                         figure_dir / f"{stem}_human_vs_ai_profile.pdf")
    return table, folds


def run_keyness(df, table_dir, figure_dir, stem, make_bold):
    """Log-odds keyness for one study. Returns the full term table."""
    human_docs, ai_docs = human_ai_pairs(df)
    print(f"Human docs: {len(human_docs)}  |  AI docs: {len(ai_docs)}")

    results = keyness(human_docs, ai_docs)
    plot_keyness(results, figure_dir / f"{stem}_logodds_human_vs_ai.pdf", make_bold)

    results.to_csv(table_dir / f"{stem}_human_ai_keyness.csv", index=False)
    pd.DataFrame({"n_pairs": [len(human_docs)]}).to_csv(
        table_dir / f"{stem}_keyness_sample.csv", index=False)
    return results
