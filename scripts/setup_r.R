#!/usr/bin/env Rscript
# Author: Joshua Ashkinaze
#
# Description: Installs the R packages the analyses need, at the versions used for the reported
# results. grf fits the causal forests in each study's hte_grf.R; mediation supplies medsens,
# used in mediation_sensitivity.R for the confounder sensitivity bounds;
# lavaan, lmtest and sandwich fit and test the mediation models in 1_run_mediation_analyses.R.
# Installs from the Posit date-pinned CRAN snapshot so versions resolve the same way later.
#
# Inputs:
#   - none
# Outputs:
#   - the packages below, installed into the first writable library path

VERSIONS <- c(grf = "2.6.1", dplyr = "1.2.1", ggplot2 = "4.0.3",
              patchwork = "1.3.2", mediation = "4.5.1", lavaan = "0.6.21",
              lmtest = "0.9.40", sandwich = "3.1.1")

# Date-pinned snapshot: the same URL resolves to the same package versions on any future run.
SNAPSHOT <- "https://packagemanager.posit.co/cran/2026-09-01"

cat("R:", R.version.string, "(results were produced on 4.6.1)\n\n")

for (pkg in names(VERSIONS)) {
  want <- VERSIONS[[pkg]]
  have <- tryCatch(as.character(packageVersion(pkg)), error = function(e) NA_character_)
  if (identical(have, want)) {
    cat(sprintf("  %-12s %s  ok\n", pkg, have))
    next
  }
  cat(sprintf("  %-12s installing %s (have: %s)\n", pkg, want, if (is.na(have)) "none" else have))
  install.packages(pkg, repos = SNAPSHOT)
  got <- tryCatch(as.character(packageVersion(pkg)), error = function(e) NA_character_)
  if (!identical(got, want)) {
    cat(sprintf("      note: got %s, not %s; the snapshot may have moved\n",
                if (is.na(got)) "nothing" else got, want))
  }
}

cat("\nDone.\n")
