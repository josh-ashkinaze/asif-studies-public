# The As-If Machine

Code and data for *The As-If Machine: Personalized counterfactuals make distant policy risks feel
close*. The paper reports two preregistered experiments: science funding cuts (Study 1,
N = 1,007) and satellite cybersecurity (Study 2, N = 894).

## Running

You need Python 3.13 with [uv](https://docs.astral.sh/uv/), R 4.6.1 and TeX Live. From the
repository root:

```bash
uv sync
```

```bash
Rscript scripts/setup_r.R
```

```bash
.venv/bin/python scripts/setup_assets.py
```

```bash
./run_all.sh
```

The first three commands install the Python packages, the R packages, and the language models
and corpora. `run_all.sh` then runs every analysis on the clean data in this repository and
builds `main.pdf`. It takes between 15 minutes and 2 hours, and writes tables and figures to each
folder's `tables/` and `plots/`.

We ran it on macOS 15.5 (Apple M2) with Python 3.13.7 and R 4.6.1.

## Files

Top level

- `main.tex`, `references.bib`: the paper
- `run_all.sh`: runs every analysis in order and builds the paper
- `pyproject.toml`, `uv.lock`, `requirements.txt`: pinned Python environment
- `log_odds.py`: log-odds ratios with Dirichlet priors, for comparing word use

`sci_fund_real/` (Study 1) and `satellite/` (Study 2)

- `main_analysis.ipynb`: treatment effects, heterogeneity, balance, linguistic and qualitative
  analyses
- `hte_grf.R`: causal forests and best linear projections, called by the notebook
- `data/clean/analysis_data.csv`: one row per participant; what every analysis reads
- `get_data.py`: builds `analysis_data.csv` from the raw survey and app data, which are not
  included
- `text_analysis_freeze/`: the BERTopic run the qualitative theme table comes from
- `helpers.py`, `statsmodels_handler.py`: shared plotting and regression helpers

`study_comparison/`

- `1_run_mediation_analyses.R`: lavaan mediation models for each study and pooled
- `2_analyze_mediation_analyses.ipynb`: mediation tables and figures
- `3_study_comparison.ipynb`: pooled effects, outcome tables, Figure 4 and the BLP tables
- `mediation_sensitivity.R`: sensitivity of the mediation estimates to confounding
- `plot_mediation_paths.py`: draws the mediation path diagrams

`explain_divergence/`

- `divergence.ipynb`: why attitude effects differ between the studies
- `annotate_crystallization.py`: LLM labels for crystallized attitudes, saved to
  `data/crystallization_annotations.csv`
- `crystallization_sample_200*.xlsx`: the 200-response validation sample with each rater's labels
- `annotator_stats.py`: agreement between the human raters and the LLM
- `fetch_news_coverage.py`: Media Cloud news counts, saved to `data/news_coverage.csv`

`src/`

- `helpers.py`, `linguistic.py`: plotting, statistics and linguistic-feature code shared across
  studies
- `mongo_util.py`, `download_qualtrics_experiment.py`, `analyze_qualtrics_experiment.py`,
  `download_db.py`: survey and app-database code used by `get_data.py`

`scripts/`

- `setup_r.R`, `setup_assets.py`: install the R packages and the language models
- `freeze_env.sh`: re-pins the Python environment
- `check_manuscript_inputs.py`: checks that every table and figure `main.tex` reads gets
  generated
- `strip_identifiers.py`, `scrub_notebook_outputs.py`: remove participant identifiers from the
  data and notebook outputs

`static_figures/`: hand-made figures (system diagram, interface, prompts)
