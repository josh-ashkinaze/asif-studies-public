#!/usr/bin/env bash
#
# Name: Joshua Ashkinaze
#
# Date: 2026-09-12
#
# Description: Reproduces every table and figure the manuscript reads, and builds main.pdf, in
# dependency order.
#   1. Each study builds its analysis dataset, then runs its notebook. The notebook exports the
#      causal-forest input and fits the forests in R (hte_grf.R).
#   2. Mediation: an R script fits the lavaan models for both studies and pooled, then a notebook
#      builds the SI tables and figures from its estimates. 
#   3. The combined notebook reads both studies' tables and the mediation estimates.
#   4. The divergence notebook reads both studies' clean data.
#   5. main.tex is built, and the run fails if any reference or citation is undefined.
# Assumes the environment is already installed; see README.md for uv, R and model-asset setup.
#
# Data: each study reads its committed data/clean/analysis_data.csv, and get_data.py never runs
# while that file exists. Toggle: ACCESS_LIVE_DB=True builds a missing dataset from the raw
# Qualtrics export and the live app database (get_data.py); delete the clean file to rebuild it.
# With the toggle at its default, False, a missing dataset stops the run with an error.
#
#     ACCESS_LIVE_DB=True ./run_all.sh
#
# Inputs:
# - with ACCESS_LIVE_DB=True and no clean dataset: each study's raw Qualtrics export, its Prolific
#   demographic exports and MONGO_PRODUCTION_URL, read by get_data.py
# - otherwise: */data/clean/analysis_data.csv
#
# Outputs:
# - satellite/{plots,tables}, sci_fund_real/{plots,tables}, study_comparison/{plots,tables},
#   explain_divergence/{plots,tables}
# - main.pdf

set -euo pipefail
cd "$(dirname "$0")"

ACCESS_LIVE_DB="${ACCESS_LIVE_DB:-False}"

PYTHON="$PWD/.venv/bin/python"
export JUPYTER_PATH="$PWD/.venv/share/jupyter"

# Output folders are gitignored, so a fresh clone lacks them and the notebooks write into them
mkdir -p {sci_fund_real,satellite,study_comparison,explain_divergence}/{tables,plots}

# The notebooks record kernel "python3", which resolves to the system interpreter outside an
# IDE and there lacks the project packages. Register the venv under its own name and call it
# explicitly.
"$PYTHON" -m ipykernel install --sys-prefix --name asif-venv >/dev/null 2>&1

# Runs a notebook in place with papermill, from the notebook's own folder
run_notebook() {
    echo "--- $1"
    "$PYTHON" -m papermill --kernel asif-venv --cwd "$(dirname "$1")" \
        --execution-timeout 3000 "$1" "$1"
}

# Uses a study's clean dataset when it exists; otherwise builds it from the live database
prepare_data() {
    if [ -f "$1/data/clean/analysis_data.csv" ]; then
        echo "--- $1: using data/clean/analysis_data.csv (delete it to rebuild)"
    elif [ "$ACCESS_LIVE_DB" = "True" ]; then
        (cd "$1" && "$PYTHON" get_data.py)
    else
        echo "$1/data/clean/analysis_data.csv is missing: run with ACCESS_LIVE_DB=True first" >&2
        exit 1
    fi
}

#########
# science_funding
#########
prepare_data sci_fund_real
# Run main analysis notebook (also exports the HTE data and fits the causal forests)
run_notebook sci_fund_real/main_analysis.ipynb

#########
# satellite
#########
prepare_data satellite
# Run main analysis notebook (also exports the HTE data and fits the causal forests)
run_notebook satellite/main_analysis.ipynb

#########
# mediation
#########
# Fit the lavaan mediation models; reads both studies' clean data only
Rscript study_comparison/1_run_mediation_analyses.R
# Tables and figures from those estimates
run_notebook study_comparison/2_analyze_mediation_analyses.ipynb

#########
# combined notebook
#########
# Reads both studies' tables and the mediation estimates, so all of the above must be done.
# It also runs mediation_sensitivity.R before drawing the sensitivity figure.
run_notebook study_comparison/3_study_comparison.ipynb

#########
# Explain divergence
#########
# The LLM and Media Cloud annotations are committed under explain_divergence/data, because
# regenerating them costs API calls. Uncomment to rebuild them.
# "$PYTHON" explain_divergence/annotate_crystallization.py
# "$PYTHON" explain_divergence/fetch_news_coverage.py
run_notebook explain_divergence/divergence.ipynb
# Rater agreement on the 200-response crystallization validation sample
"$PYTHON" explain_divergence/annotator_stats.py

#########
# Manuscript
#########
# Fail before building if anything main.tex reads is neither committed nor generated above
"$PYTHON" scripts/check_manuscript_inputs.py

latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

# latexmk exits cleanly with unresolved references, so check the log
if grep -q -E "Reference .* undefined|Citation .* undefined|multiply defined" main.log; then
    echo "main.tex has undefined or multiply defined references:" >&2
    grep -E "Reference .* undefined|Citation .* undefined|multiply defined" main.log >&2
    exit 1
fi

echo
echo "Done. main.pdf rebuilt from freshly generated tables and figures."
