#!/usr/bin/env Rscript
#
# Name: Joshua Ashkinaze
#
# Date: 2026-09-19
#
# Description: Sensitivity of the four preregistered mediation models (closeness as the mediator,
# AIM versus vignette and versus placebo, in each study) to unmediated mediator-outcome
# confounding, after Imai, Keele and Yamamoto. Fits mediation::mediate and mediation::medsens
# with 1,000 simulations. rho is the correlation between the mediator and outcome error terms;
# sequential ignorability assumes rho = 0, and rho* is the value at which the average causal
# mediation effect (ACME) is zero. The study-comparison notebook draws the figure from the tables.
#
# Inputs:
# - sci_fund_real/data/clean/analysis_data.csv
# - satellite/data/clean/analysis_data.csv
#
# Outputs:
# - study_comparison/tables/mediation_sensitivity.csv: one row per model, with the ACME, rho*
#   and the R^2 products at which the ACME is zero
# - study_comparison/tables/mediation_sensitivity_curves.csv: the ACME and its 95% interval at
#   each value of rho

suppressPackageStartupMessages(library(mediation))

SEED <- 42
SIMS <- 1000
CONTROLS <- c("vignette", "placebo")

####################
# Paths
####################

working_directory <- normalizePath(getwd())
if (dir.exists(file.path(working_directory, "sci_fund_real"))) {
  ROOT <- working_directory
} else if (basename(working_directory) == "study_comparison") {
  ROOT <- dirname(working_directory)
} else {
  stop("Run this script from the repository root or study_comparison directory.")
}

STUDY_FILES <- c(
  "Study 1" = file.path(ROOT, "sci_fund_real", "data", "clean", "analysis_data.csv"),
  "Study 2" = file.path(ROOT, "satellite", "data", "clean", "analysis_data.csv")
)
OUT_DIR <- file.path(ROOT, "study_comparison", "tables")

####################
# Model
####################

fit_sensitivity <- function(d) {
  set.seed(SEED)
  mediator_model <- lm(closeness_index ~ treat, data = d)
  outcome_model <- lm(attitude_index ~ treat + closeness_index, data = d)
  fit <- mediate(mediator_model, outcome_model, treat = "treat",
                 mediator = "closeness_index", sims = SIMS)
  sens <- medsens(fit, rho.by = 0.01, effect.type = "indirect", sims = SIMS)
  list(
    summary = data.frame(
      n = nrow(d), acme = fit$d0, acme_lo = fit$d0.ci[[1]], acme_hi = fit$d0.ci[[2]],
      rho_star = sens$err.cr.d[[1]], R2star_prod = sens$R2star.d.thresh[[1]],
      R2tilde_prod = sens$R2tilde.d.thresh[[1]]
    ),
    curve = data.frame(
      rho = as.numeric(sens$rho), acme = as.numeric(sens$d0),
      lo = as.numeric(sens$lower.d0), hi = as.numeric(sens$upper.d0)
    )
  )
}

####################
# Apply
####################

summaries <- list()
curves <- list()
for (study in names(STUDY_FILES)) {
  clean <- read.csv(STUDY_FILES[[study]], stringsAsFactors = FALSE)
  for (control in CONTROLS) {
    d <- clean[clean$treat_status %in% c("treat", control), ]
    d$treat <- as.numeric(d$treat_status == "treat")
    d <- d[complete.cases(d[, c("closeness_index", "attitude_index")]), ]
    result <- fit_sensitivity(d)
    key <- paste(study, control)
    summaries[[key]] <- cbind(study = study, control = control, result$summary)
    curves[[key]] <- cbind(study = study, control = control, result$curve)
  }
}

write.csv(do.call(rbind, summaries), file.path(OUT_DIR, "mediation_sensitivity.csv"),
          row.names = FALSE)
write.csv(do.call(rbind, curves), file.path(OUT_DIR, "mediation_sensitivity_curves.csv"),
          row.names = FALSE)
cat("Wrote mediation sensitivity tables to", OUT_DIR, "\n")
print(do.call(rbind, summaries), row.names = FALSE)
