#!/usr/bin/env Rscript
#
# Name: Joshua Ashkinaze
#
# Date: 2026-09-17
#
# Description: Generates every statistical result used by the mediation analysis notebook. It
# fits direction checks, mediator partial correlations, and two multiple-mediator models for
# Study 1, Study 2, and both studies pooled, crossed with vignette, placebo, and pooled controls.
# The pooled-study models include a Study 2 indicator in every structural equation.
#
# The parallel model lets closeness and transportation's residuals covary freely, rather than
# fixing that covariance at zero. Both mediators were measured concurrently, so a fixed zero
# would assert an independence the design cannot support; the free covariance instead reports
# whatever they share without attributing it to either mediator affecting the other.
#
# Freeing that covariance saturates the parallel model: it then has as many free parameters as
# there are moments to fit, exactly like the serial model, so both reproduce the observed
# covariance matrix exactly and are indistinguishable by AIC or BIC. The notebook confirms this
# rather than comparing the two by fit.
#
# Inputs:
# - sci_fund_real/data/clean/analysis_data.csv
# - satellite/data/clean/analysis_data.csv
#
# Outputs:
# - study_comparison/tables/mediation_direction_tests.csv
# - study_comparison/tables/mediator_partial_correlations.csv
# - study_comparison/tables/mediation_model_estimates.csv
# - study_comparison/tables/mediation_samples.csv

suppressPackageStartupMessages({
  library(lavaan)
  library(lmtest)
  library(sandwich)
})


####################
# Constants
####################

RANDOM_SEED <- 42
N_BOOT <- as.integer(Sys.getenv("MEDIATION_N_BOOT", "1000"))
N_CPUS <- 7
CRITICAL_VALUE <- 1.96

PARALLEL_MODE <- if (.Platform$OS.type == "windows") {
  "snow"
} else {
  "multicore"
}


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

OUT_DIR <- file.path(ROOT, "study_comparison", "tables")

dir.create(
  OUT_DIR,
  recursive = TRUE,
  showWarnings = FALSE
)


####################
# Analysis Grid
####################

STUDIES <- list(
  Pooled = c("Study 1", "Study 2"),
  `Study 1` = "Study 1",
  `Study 2` = "Study 2"
)

CONTROLS <- list(
  Vignette = "vignette",
  Placebo = "placebo",
  Pooled = c("vignette", "placebo")
)


####################
# Models
####################

PARALLEL <- '
  attitude_index ~ b1*closeness_index + b2*transport_index + c_p*treat
  closeness_index ~ a1*treat
  transport_index ~ a2*treat
  closeness_index ~~ transport_index

  indirect_closeness := a1*b1
  indirect_transport := a2*b2
  total_indirect := indirect_closeness + indirect_transport
  total := c_p + total_indirect
  direct := c_p
'

PARALLEL_POOLED <- '
  attitude_index ~ b1*closeness_index + b2*transport_index + c_p*treat + study2
  closeness_index ~ a1*treat + study2
  transport_index ~ a2*treat + study2
  closeness_index ~~ transport_index

  indirect_closeness := a1*b1
  indirect_transport := a2*b2
  total_indirect := indirect_closeness + indirect_transport
  total := c_p + total_indirect
  direct := c_p
'

SERIAL <- '
  attitude_index ~ b1*closeness_index + b2*transport_index + c_p*treat
  closeness_index ~ a1*treat
  transport_index ~ a2*treat + d21*closeness_index

  indirect_closeness := a1*b1
  indirect_transport := a2*b2
  indirect_serial := a1*d21*b2
  contrast_closeness_transport := indirect_closeness - indirect_transport
  contrast_closeness_serial := indirect_closeness - indirect_serial
  contrast_transport_serial := indirect_transport - indirect_serial
  total_indirect := indirect_closeness + indirect_transport + indirect_serial
  total := c_p + total_indirect
  direct := c_p
'

SERIAL_POOLED <- '
  attitude_index ~ b1*closeness_index + b2*transport_index + c_p*treat + study2
  closeness_index ~ a1*treat + study2
  transport_index ~ a2*treat + d21*closeness_index + study2

  indirect_closeness := a1*b1
  indirect_transport := a2*b2
  indirect_serial := a1*d21*b2
  contrast_closeness_transport := indirect_closeness - indirect_transport
  contrast_closeness_serial := indirect_closeness - indirect_serial
  contrast_transport_serial := indirect_transport - indirect_serial
  total_indirect := indirect_closeness + indirect_transport + indirect_serial
  total := c_p + total_indirect
  direct := c_p
'


####################
# Data Loading
####################

load_data <- function() {
  files <- c(
    "Study 1" = file.path(
      ROOT,
      "sci_fund_real",
      "data",
      "clean",
      "analysis_data.csv"
    ),
    "Study 2" = file.path(
      ROOT,
      "satellite",
      "data",
      "clean",
      "analysis_data.csv"
    )
  )
  
  frames <- lapply(names(files), function(study) {
    data <- read.csv(
      files[[study]],
      stringsAsFactors = FALSE
    )
    
    data$study <- study
    
    data[
      ,
      c(
        "study",
        "treat_status",
        "attitude_index",
        "closeness_index",
        "transport_index"
      )
    ]
  })
  
  data <- do.call(rbind, frames)
  
  data$study2 <- as.numeric(
    data$study == "Study 2"
  )
  
  data$transport_index <- ave(
    data$transport_index,
    data$study,
    FUN = function(values) {
      as.numeric(scale(values))
    }
  )
  
  data
}


####################
# Analysis Samples
####################

get_cell <- function(data, study, control) {
  cell <- data[
    data$study %in% STUDIES[[study]] &
      data$treat_status %in% c("treat", CONTROLS[[control]]),
  ]
  
  cell$treat <- as.numeric(
    cell$treat_status == "treat"
  )
  
  required <- c(
    "attitude_index",
    "closeness_index",
    "transport_index",
    "treat"
  )
  
  if (study == "Pooled") {
    required <- c(required, "study2")
  }
  
  cell[
    complete.cases(cell[, required]),
  ]
}


####################
# Robust OLS
####################

hc1_term <- function(formula, data, term) {
  fit <- lm(
    formula,
    data = data
  )
  
  test <- coeftest(
    fit,
    vcov. = vcovHC(
      fit,
      type = "HC1"
    )
  )
  
  estimate <- unname(
    test[term, "Estimate"]
  )
  
  standard_error <- unname(
    test[term, "Std. Error"]
  )
  
  statistic <- unname(
    test[term, "t value"]
  )
  
  p_value <- unname(
    test[term, "Pr(>|t|)"]
  )
  
  c(
    estimate = estimate,
    standard_error = standard_error,
    statistic = statistic,
    p_value = p_value,
    ci_lower = estimate - CRITICAL_VALUE * standard_error,
    ci_upper = estimate + CRITICAL_VALUE * standard_error
  )
}


####################
# Partial Correlation
####################

partial_correlation <- function(data) {
  closeness_residual <- residuals(
    lm(
      closeness_index ~ treat,
      data = data
    )
  )
  
  transport_residual <- residuals(
    lm(
      transport_index ~ treat,
      data = data
    )
  )
  
  test <- cor.test(
    closeness_residual,
    transport_residual,
    method = "pearson"
  )
  
  c(
    raw_r = cor(
      data$closeness_index,
      data$transport_index
    ),
    partial_r = unname(test$estimate),
    ci_lower = test$conf.int[1],
    ci_upper = test$conf.int[2],
    p_value = test$p.value
  )
}


####################
# Mediation Models
####################

fit_mediation_model <- function(model, data) {
  sem(
    model,
    data = data,
    se = "bootstrap",
    bootstrap = N_BOOT,
    missing = "fiml",
    parallel = PARALLEL_MODE,
    ncpus = N_CPUS,
    iseed = RANDOM_SEED
  )
}


####################
# Initialize
####################

pooled <- load_data()

direction_rows <- list()
partial_rows <- list()
estimate_rows <- list()
sample_rows <- list()


####################
# Startup Information
####################

cat("\n")
cat("Mediation analysis\n")
cat("------------------\n")
cat("Bootstrap replications:", N_BOOT, "\n")
cat("Parallel mode:", PARALLEL_MODE, "\n")
cat("CPU processes:", N_CPUS, "\n")
cat("\n")


####################
# Run Analyses
####################

for (study in names(STUDIES)) {
  for (control in names(CONTROLS)) {
    
    data <- get_cell(
      pooled,
      study,
      control
    )
    
    key <- paste(
      study,
      control,
      sep = "|"
    )
    
    cat(
      sprintf(
        "\nAnalyzing %s, %s control (n = %d)\n",
        study,
        tolower(control),
        nrow(data)
      )
    )
    
    
    ####################
    # Sample Information
    ####################
    
    sample_rows[[key]] <- data.frame(
      study = study,
      control = control,
      n = nrow(data),
      n_treat = sum(data$treat == 1),
      n_control = sum(data$treat == 0)
    )
    
    
    ####################
    # Direction Checks
    ####################
    
    direction_specs <- list(
      `AIM -> transportation -> closeness` = list(
        downstream = "closeness_index",
        unadjusted = closeness_index ~ treat,
        adjusted = closeness_index ~ treat + transport_index
      ),
      `AIM -> closeness -> transportation` = list(
        downstream = "transport_index",
        unadjusted = transport_index ~ treat,
        adjusted = transport_index ~ treat + closeness_index
      )
    )
    
    for (direction in names(direction_specs)) {
      specification <- direction_specs[[direction]]
      
      unadjusted <- hc1_term(
        specification$unadjusted,
        data,
        "treat"
      )
      
      adjusted <- hc1_term(
        specification$adjusted,
        data,
        "treat"
      )
      
      direction_rows[[paste(key, direction)]] <- data.frame(
        study = study,
        control = control,
        n = nrow(data),
        direction = direction,
        downstream = specification$downstream,
        
        unadjusted_b = unadjusted["estimate"],
        unadjusted_se = unadjusted["standard_error"],
        unadjusted_p = unadjusted["p_value"],
        unadjusted_ci_lower = unadjusted["ci_lower"],
        unadjusted_ci_upper = unadjusted["ci_upper"],
        
        adjusted_b = adjusted["estimate"],
        adjusted_se = adjusted["standard_error"],
        adjusted_p = adjusted["p_value"],
        adjusted_ci_lower = adjusted["ci_lower"],
        adjusted_ci_upper = adjusted["ci_upper"],
        
        coefficient_change =
          adjusted["estimate"] - unadjusted["estimate"],
        
        percent_reduction =
          100 * (
            1 -
              adjusted["estimate"] /
              unadjusted["estimate"]
          ),
        
        tolerated =
          adjusted["ci_lower"] <= 0 &
          adjusted["ci_upper"] >= 0
      )
    }
    
    
    ####################
    # Partial Correlation
    ####################
    
    partial <- partial_correlation(data)
    
    partial_rows[[key]] <- data.frame(
      study = study,
      control = control,
      n = nrow(data),
      raw_r = partial["raw_r"],
      partial_r = partial["partial_r"],
      ci_lower = partial["ci_lower"],
      ci_upper = partial["ci_upper"],
      p_value = partial["p_value"]
    )
    
    
    ####################
    # Select Models
    ####################
    
    serial_model <- if (study == "Pooled") {
      SERIAL_POOLED
    } else {
      SERIAL
    }
    
    parallel_model <- if (study == "Pooled") {
      PARALLEL_POOLED
    } else {
      PARALLEL
    }
    
    
    ####################
    # Fit Parallel Model
    ####################
    
    cat(
      sprintf(
        "  Fitting parallel model: %d bootstraps on %d CPUs\n",
        N_BOOT,
        N_CPUS
      )
    )
    
    parallel_fit <- fit_mediation_model(
      parallel_model,
      data
    )
    
    
    ####################
    # Fit Serial Model
    ####################
    
    cat(
      sprintf(
        "  Fitting serial model: %d bootstraps on %d CPUs\n",
        N_BOOT,
        N_CPUS
      )
    )
    
    serial_fit <- fit_mediation_model(
      serial_model,
      data
    )
    
    
    ####################
    # Collect Fits
    ####################
    
    fits <- list(
      "Multiple parallel" =
        parallel_fit,
      
      "Multiple serial: closeness -> transportation" =
        serial_fit
    )
    
    
    ####################
    # Extract Results
    ####################
    
    for (model_name in names(fits)) {
      fit <- fits[[model_name]]
      
      fit_key <- paste(
        key,
        model_name
      )
      
      
      ####################
      # Saturation Check
      ####################
      # Freeing the parallel model's residual covariance gives it as many free parameters as
      # there are moments, exactly like the serial model, so both fit with df = 0 and reproduce
      # the observed covariance matrix exactly. That makes AIC and BIC identical by construction,
      # so we do not compute or report a model comparison...

      if (fitMeasures(fit, "df") != 0) {
        stop(sprintf(
          "%s is not saturated (df = %.0f); the two models are no longer indistinguishable by fit",
          fit_key, fitMeasures(fit, "df")
        ))
      }


      ####################
      # Parameter Estimates
      ####################
      
      parameters <- parameterEstimates(
        fit,
        boot.ci.type = "bca.simple",
        standardized = TRUE
      )
      
      estimate_rows[[fit_key]] <- cbind(
        study = study,
        control = control,
        n = nrow(data),
        model = model_name,
        parameters
      )
    }
  }
}


####################
# Combine Tables
####################

direction_table <- do.call(
  rbind,
  direction_rows
)

partial_table <- do.call(
  rbind,
  partial_rows
)

estimate_table <- do.call(
  rbind,
  estimate_rows
)

sample_table <- do.call(
  rbind,
  sample_rows
)


####################
# Print Tables
####################

cat(
  "\n\n==============================\n",
  "MEDIATION DIRECTION TESTS\n",
  "==============================\n"
)

print(
  direction_table,
  row.names = FALSE
)


cat(
  "\n\n==============================\n",
  "MEDIATOR PARTIAL CORRELATIONS\n",
  "==============================\n"
)

print(
  partial_table,
  row.names = FALSE
)


cat(
  "\nBoth models fit every cell with df = 0 (verified above as each is fit), so AIC and BIC",
  "cannot distinguish them; no comparison is reported.\n"
)


cat(
  "\n\n==============================\n",
  "MEDIATION MODEL ESTIMATES\n",
  "==============================\n"
)

print(
  estimate_table,
  row.names = FALSE
)


cat(
  "\n\n==============================\n",
  "MEDIATION SAMPLES\n",
  "==============================\n"
)

print(
  sample_table,
  row.names = FALSE
)


####################
# Save Tables
####################

write.csv(
  direction_table,
  file.path(
    OUT_DIR,
    "mediation_direction_tests.csv"
  ),
  row.names = FALSE
)

write.csv(
  partial_table,
  file.path(
    OUT_DIR,
    "mediator_partial_correlations.csv"
  ),
  row.names = FALSE
)

write.csv(
  estimate_table,
  file.path(
    OUT_DIR,
    "mediation_model_estimates.csv"
  ),
  row.names = FALSE
)

write.csv(
  sample_table,
  file.path(
    OUT_DIR,
    "mediation_samples.csv"
  ),
  row.names = FALSE
)


####################
# Done
####################

cat(
  "\nWrote all mediation analysis CSVs to",
  OUT_DIR,
  "\n"
)