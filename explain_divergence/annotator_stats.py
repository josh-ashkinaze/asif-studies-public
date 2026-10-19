"""
Author: Joshua Ashkinaze

Description: Agreement between two human raters and the LLM on the 200-response crystallization
validation sample, for whether a response is crystallized (prior familiarity or reinforcement).
Reports Cohen's kappa with bootstrap 95% intervals, percent agreement, disagreements and positive
counts, on all 200 responses and on the first 100,
which were drawn purely at random. The other 100 mix 50 random responses with 50 enriched from
LLM-coded positives, shuffled together, so only items 1-100 are known to be random.

Inputs:
    - crystallization_sample_200_annotatorA.xlsx: Rater A's labels
    - crystallization_sample_200_annotatorB.xlsx: Rater B's labels
    - crystallization_sample_200_llm.xlsx: LLM labels (annotate_crystallization.py --sample)
    - data/crystallization_annotations.csv: LLM labels on every response, for the base rate

Outputs:
    - tables/annotator_agreement.tex: the agreement table with a self-documenting caption

Notes: Intervals are percentile intervals from 2,000 bootstrap resamples of responses, seed 42.
    Only the crystallized label is reported; the reinforced subtype has a single positive in the
    random subset, so its kappa there is uninformative.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

HERE = Path(__file__).resolve().parent     # explain_divergence/
TABLE_EXPORT = HERE / "tables"
PAIRS = [("Rater A", "Rater B"), ("LLM", "Rater A"), ("LLM", "Rater B")]
RANDOM_ITEMS = 100                         # items 1-100 were drawn purely at random
NBOOT = 2000
RNG = np.random.default_rng(42)
# Resamples with a single class leave kappa undefined; nanpercentile drops them
warnings.filterwarnings("ignore", module="sklearn")

###############
# Wrangling functions
###############

def load_raters():
    """One frame per rater, indexed by ResponseId, with the 0/1 crystallized label and the item number."""
    llm = pd.read_excel(HERE / "crystallization_sample_200_llm.xlsx")
    llm = llm[["ResponseId", "item", "llm_is_crystallized"]]
    llm.columns = ["ResponseId", "item", "is_crystallized"]
    raters = {"LLM": llm,
              "Rater A": pd.read_excel(HERE / "crystallization_sample_200_annotatorA.xlsx"),
              "Rater B": pd.read_excel(HERE / "crystallization_sample_200_annotatorB.xlsx")}
    return {k: d.set_index("ResponseId")[["item", "is_crystallized"]] for k, d in raters.items()}

###############
# Statistics functions
###############

def kappa_ci(x, y):
    """Cohen's kappa with a percentile bootstrap 95% interval."""
    idx = RNG.integers(0, len(x), (NBOOT, len(x)))
    boots = [cohen_kappa_score(x[i], y[i]) for i in idx]
    return cohen_kappa_score(x, y), *np.nanpercentile(boots, [2.5, 97.5])


def agreement_rows(raters, sample, items):
    """Agreement statistics on the crystallized label for every rater pair on the given items."""
    rows = []
    for r1, r2 in PAIRS:
        x = raters[r1].loc[items, "is_crystallized"].astype(int).values
        y = raters[r2].loc[items, "is_crystallized"].astype(int).values
        k, lo, hi = kappa_ci(x, y)
        rows.append({"Sample": sample, "Raters": f"{r1} vs {r2}",
                     "$\\kappa$": f"{k:.2f}", "95\\% CI": f"[{lo:.2f}, {hi:.2f}]",
                     "Agreement (\\%)": f"{100 * (x == y).mean():.1f}",
                     "Disagreements": int((x != y).sum()),
                     "Positives": f"{x.sum()} / {y.sum()}"})
    return rows

###############
# Apply
###############

raters = load_raters()
all_items = raters["LLM"].index
assert all(set(d.index) == set(all_items) for d in raters.values()), "ResponseIds differ across files"
random_items = raters["LLM"].index[raters["LLM"].item <= RANDOM_ITEMS]

table = pd.DataFrame(agreement_rows(raters, f"All ($N={len(all_items)}$)", all_items) +
                     agreement_rows(raters, f"Random ($N={len(random_items)}$)", random_items))
print(table.to_string(index=False))

caption = (
    f"Inter-annotator agreement. Two human raters independently coded {len(all_items)} open-ended "
    "responses as crystallized or not, and the LLM coded the same responses with a codebook fixed "
    "before human coding began. A response counts as crystallized if it shows prior familiarity or "
    f"reinforcement, judged separately as subtypes. The first {len(random_items)} responses were drawn at random; the remaining "
    f"{len(all_items) - len(random_items)} combine random responses with responses the LLM coded as "
    "crystallized, shuffled together, to ensure enough positive cases. The table reports agreement "
    "between the two human raters and between the LLM and each human rater, on all responses and on "
    "the random subset alone. $\\kappa$ is Cohen's kappa with a 95\\% percentile bootstrap interval "
    f"({NBOOT:,} resamples of responses). Positives gives the number of responses each rater coded 1, "
    "in the order the raters are listed."
)
TABLE_EXPORT.mkdir(exist_ok=True)
table.to_latex(
    TABLE_EXPORT / "annotator_agreement.tex", index=False, escape=False, position="H",
    caption=caption, label="tab:annotator_agreement")
print(f"\nwrote {TABLE_EXPORT / 'annotator_agreement.tex'}")

# APA strings for the text
full = pd.read_csv(HERE / "data/crystallization_annotations.csv")
print(f"\nLLM coded {full.is_crystallized.sum()} of {len(full):,} responses as crystallized "
      f"({100 * full.is_crystallized.mean():.1f}\\%)")
for sample, pair, k, ci, agree, *_ in table.itertuples(index=False):
    print(f"{sample}, {pair}: $\\kappa={k}$, 95\\% CI ${ci}$, {agree}\\% agreement")
