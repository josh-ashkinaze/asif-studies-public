# ============================================================
# GRF Best Linear Projection
# Name: Josh Ashkinaze
# ============================================================

library(grf)
library(dplyr)
library(ggplot2)
library(patchwork)

# ============================================================
# Run from the study folder (the notebook does this; from a shell, cd there first)
# ============================================================

if (!file.exists("hte_grf.R")) {
  stop("Run hte_grf.R from its own folder so the relative paths resolve.")
}
cat("Working directory:", getwd(), "\n")

# ============================================================
# Custom theme
# ============================================================

theme_aesthetic_ggplot <- function(hex_color_list = NULL,
                                   bold_title = FALSE,
                                   show_grid = TRUE,
                                   font_scale = 1,
                                   font_hierarchy_ratio = 1.25,
                                   smallest_font_size = 10,
                                   base_family = "Helvetica") {
  small  <- smallest_font_size
  medium <- small * font_hierarchy_ratio
  large  <- medium * font_hierarchy_ratio
  text_color <- "#2C3531"
  grid_color <- "#e8e8e8"
  
  ggplot2::theme_minimal(base_family = base_family, base_size = small * font_scale) +
    ggplot2::theme(
      text = ggplot2::element_text(family = base_family, colour = text_color),
      axis.title = ggplot2::element_text(
        size = medium * font_scale, face = "bold", colour = text_color,
        margin = ggplot2::margin(t = 4 * font_scale, r = 4 * font_scale)
      ),
      axis.title.x = ggplot2::element_text(
        margin = ggplot2::margin(t = 8 * font_scale, r = 4 * font_scale)
      ),
      axis.title.y = ggplot2::element_text(
        margin = ggplot2::margin(t = 4 * font_scale, r = 8 * font_scale)
      ),
      axis.text = ggplot2::element_text(size = small * font_scale, colour = text_color),
      legend.title = ggplot2::element_text(
        size = small * font_scale, face = "bold", colour = text_color,
        margin = ggplot2::margin(b = 4 * font_scale)
      ),
      legend.text  = ggplot2::element_text(size = small * font_scale, colour = text_color),
      plot.title   = ggplot2::element_text(
        size = large * font_scale, face = "bold", hjust = 0,
        colour = text_color, lineheight = 1.05,
        margin = ggplot2::margin(t = 10, r = 5, b = 10, l = 5)
      ),
      plot.subtitle = ggplot2::element_text(
        size = medium * font_scale, hjust = 0, colour = text_color,
        lineheight = 1.1, margin = ggplot2::margin(b = 8 * font_scale)
      ),
      plot.title.position   = "plot",
      plot.caption.position = "plot",
      panel.background  = ggplot2::element_rect(fill = "white", colour = NA),
      plot.background   = ggplot2::element_rect(fill = "white", colour = NA),
      axis.line         = ggplot2::element_blank(),
      axis.ticks        = ggplot2::element_blank(),
      panel.border      = ggplot2::element_blank(),
      panel.grid.major  = if (show_grid) {
        ggplot2::element_line(colour = grid_color, linewidth = 0.5, linetype = "longdash")
      } else ggplot2::element_blank(),
      panel.grid.minor   = ggplot2::element_blank(),
      panel.spacing.x    = ggplot2::unit(2, "lines"),
      legend.background  = ggplot2::element_rect(fill = "white", colour = NA),
      legend.key         = ggplot2::element_rect(fill = "white", colour = "white"),
      legend.key.width   = ggplot2::unit(1.5, "lines"),
      legend.spacing     = ggplot2::unit(6, "pt"),
      legend.box.spacing = ggplot2::unit(8, "pt"),
      legend.position    = "right",
      plot.margin = ggplot2::margin(
        t = 12 * font_scale, r = 14 * font_scale,
        b = 12 * font_scale, l = 12 * font_scale
      )
    )
}

# ============================================================
# Helpers
# ============================================================

to_title <- function(s) tools::toTitleCase(gsub("_", " ", s))

log_file <- "tables/blp_results.log"
log <- function(...) {
  msg <- paste0(...)
  cat(msg, "\n")
  cat(msg, "\n", file = log_file, append = TRUE)
}
cat("", file = log_file)

fmt_p <- function(p) {
  if (p < .001) "< .001" else sprintf("= %.3f", p)
}

apa_string <- function(feature, estimate, ci_lower, ci_upper, se, pval) {
  sprintf(
    "%s: $B = %.2f$, 95\\%% CI $[%.2f, %.2f]$, $SE = %.2f$, $p %s$",
    feature, estimate, ci_lower, ci_upper, se, fmt_p(pval)
  )
}

make_loess_panel <- function(x_raw, cate_hat, label, span = 0.75) {
  
  if (length(unique(x_raw)) < 8) {
    df_plot <- data.frame(x = factor(round(x_raw, 1)), y = cate_hat)
    return(
      ggplot(df_plot, aes(x = x, y = y)) +
        geom_boxplot(fill = "steelblue", alpha = 0.4, outlier.size = 0.5) +
        geom_hline(yintercept = 0, linetype = "dashed",
                   color = "#2C3531", linewidth = 0.5) +
        labs(x = label, y = "CATE") +
        theme_aesthetic_ggplot(font_scale = 0.9, show_grid = TRUE, base_family = BASE_FONT)
    )
  }
  
  df_plot <- data.frame(x = x_raw, y = cate_hat)
  
  ggplot(df_plot, aes(x = x, y = y)) +
    geom_smooth(method = "loess", formula = y ~ x, span = span,
                color = "steelblue", fill = "steelblue", alpha = 0.25,
                linewidth = 1.2) +
    geom_hline(yintercept = 0, linetype = "dashed",
               color = "#2C3531", linewidth = 0.5) +
    labs(x = label, y = "CATE") +
    theme_aesthetic_ggplot(font_scale = 0.9, show_grid = TRUE, base_family = BASE_FONT)
}

# ============================================================
# Config
# ============================================================

BASE_FONT    <- "Helvetica"
OUTCOMES     <- c("closeness_index", "attitude_index")
BASELINE_ARM <- "treat"

# ---- moderator dicts: name = column, value = display label ----
# delete any entry you don't want included

BASE_MODERATORS <- c(
  "familiar"          = "Familiarity",
  "age"               = "Age",
  "mi_index"          = "Mental Imagery",
  "iri_fa"            = "IRI Fantasy",
  "iri_pc"            = "IRI Perspective",
  "iri_pd"            = "IRI Distress",
  "rw_journal"        = "Journaling Freq.",
  "rw_write"          = "Writing Freq.",
  "educ_num"          = "Education",
  "income_num"        = "Income",
  "ideology_num"      = "Ideology",
  "ai_attitude_index" = "AI Attitudes"
)

ENGAGEMENT_MODERATORS <- c(
  "ac_received"       = "Completions Received",
  "ac_total_words"    = "Story Length",
  "ac_acceptance_rate" = "Acceptance Rate",
  "ac_prop_ai_words"  = "Prop. AI Words"
)

SYS_MODERATORS <- c(
)

CAT_MODERATORS <- c(
  "gender_Male"      = "Gender: Male"
)

# treat vs placebo: base + engagement + sys
# treat vs vignette: base only
CONTRAST_CONFIG <- list(
  "placebo" = list(
    file       = "tables/hte_data_treat_vs_placebo.csv",
    moderators = c(BASE_MODERATORS, ENGAGEMENT_MODERATORS, SYS_MODERATORS)
  ),
  "vignette" = list(
    file       = "tables/hte_data_treat_vs_vignette.csv",
    moderators = c(BASE_MODERATORS)
  )
)

# ============================================================
# Fit per contrast
# ============================================================

results <- list()

for (contrast_arm in names(CONTRAST_CONFIG)) {
  cfg <- CONTRAST_CONFIG[[contrast_arm]]
  
  cat(sprintf("\n--- Loading: %s ---\n", cfg$file))
  df <- read.csv(cfg$file, stringsAsFactors = FALSE)
  cat("Rows:", nrow(df), "\n")
  print(table(df$treat_status))
  
  df$gender_Male      <- as.integer(df$demo_gender == "Male")
  
  all_moderators <- c(cfg$moderators, CAT_MODERATORS)
  all_x_cols     <- names(all_moderators)
  feat_labels    <- unname(all_moderators)
  
  df <- df[complete.cases(df[, c(all_x_cols, OUTCOMES, "treat_status")]), ]
  cat("After dropna:", nrow(df), "\n")
  
  numeric_cols     <- names(cfg$moderators)
  df[numeric_cols] <- scale(df[numeric_cols])
  
  results[[contrast_arm]] <- list()
  
  for (outcome in OUTCOMES) {
    cat(sprintf("\nFitting: %s | %s vs %s\n", outcome, BASELINE_ARM, contrast_arm))
    
    X_sub <- as.matrix(df[, all_x_cols])
    Y_sub <- df[[outcome]]
    W_sub <- as.integer(df$treat_status == BASELINE_ARM)
    
    cat(sprintf("  n=%d, T=1(%s)=%d, T=0(%s)=%d\n",
                nrow(df), BASELINE_ARM, sum(W_sub),
                contrast_arm, sum(1 - W_sub)))
    
    forest <- causal_forest(
      X               = X_sub,
      Y               = Y_sub,
      W               = W_sub,
      honesty         = TRUE,
      tune.parameters = "all",
      seed            = 42
    )
    
    blp <- best_linear_projection(forest, A = X_sub)
    
    results[[contrast_arm]][[outcome]] <- list(
      forest       = forest,
      blp          = blp,
      X_sub        = X_sub,
      all_x_cols   = all_x_cols,
      feat_labels  = feat_labels,
      numeric_cols = numeric_cols
    )
  }
}

# ============================================================
# Save CATE predictions
# ============================================================

for (contrast_arm in names(results)) {
  for (outcome in OUTCOMES) {
    r     <- results[[contrast_arm]][[outcome]]
    preds <- predict(r$forest, estimate.variance = TRUE)
    
    df_preds <- as.data.frame(r$X_sub)
    colnames(df_preds)     <- r$all_x_cols
    df_preds$cate          <- preds$predictions
    df_preds$cate_var      <- preds$variance.estimates
    df_preds$cate_se       <- sqrt(preds$variance.estimates)
    df_preds$cate_ci_lower <- df_preds$cate - 1.96 * df_preds$cate_se
    df_preds$cate_ci_upper <- df_preds$cate + 1.96 * df_preds$cate_se
    df_preds$outcome       <- outcome
    df_preds$baseline_arm  <- BASELINE_ARM
    df_preds$contrast_arm  <- contrast_arm
    
    fname <- sprintf("tables/cate_preds_%s_vs_%s_%s.csv", BASELINE_ARM, contrast_arm, outcome)
    write.csv(df_preds, fname, row.names = FALSE)
    cat("Saved predictions:", fname, "\n")
  }
}

# ============================================================
# Print + log BLP results
# ============================================================

for (contrast_arm in names(results)) {
  for (outcome in OUTCOMES) {
    r   <- results[[contrast_arm]][[outcome]]
    blp <- r$blp
    
    log(sprintf("\n%s", strrep("=", 65)))
    log(sprintf("Outcome: %s | %s vs %s", to_title(outcome), BASELINE_ARM, contrast_arm))
    log(sprintf("Interpretation: positive = moderator increases %s effect", BASELINE_ARM))
    log(strrep("=", 65))
    capture.output(print(blp)) |> paste(collapse = "\n") |> log()
    
    df_blp <- data.frame(
      feature  = r$feat_labels,
      estimate = blp[-1, "Estimate"],
      se       = blp[-1, "Std. Error"],
      pval     = blp[-1, "Pr(>|t|)"],
      ci_lower = blp[-1, "Estimate"] - 1.96 * blp[-1, "Std. Error"],
      ci_upper = blp[-1, "Estimate"] + 1.96 * blp[-1, "Std. Error"],
      row.names = NULL
    )
    
    sig <- df_blp[df_blp$pval < .05, ]
    log("\n--- Significant Moderators (APA / LaTeX) ---")
    if (nrow(sig) > 0) {
      for (k in seq_len(nrow(sig))) {
        r_row <- sig[k, ]
        log("  ", apa_string(r_row$feature, r_row$estimate,
                             r_row$ci_lower, r_row$ci_upper,
                             r_row$se, r_row$pval))
      }
    } else {
      log("  None at p < .05")
    }
  }
}


# ============================================================
# BLP coefficient plots
# ============================================================

for (contrast_arm in names(results)) {
  for (outcome in OUTCOMES) {
    r   <- results[[contrast_arm]][[outcome]]
    blp <- r$blp
    
    df_blp <- data.frame(
      feature  = r$feat_labels,
      estimate = blp[-1, "Estimate"],
      se       = blp[-1, "Std. Error"],
      pval     = blp[-1, "Pr(>|t|)"],
      row.names = NULL
    )
    df_blp$ci_lower <- df_blp$estimate - 1.96 * df_blp$se
    df_blp$ci_upper <- df_blp$estimate + 1.96 * df_blp$se
    df_blp$sig      <- df_blp$pval < 0.05
    
    title_str <- sprintf(
      "%s vs %s — %s",
      tools::toTitleCase(BASELINE_ARM),
      tools::toTitleCase(contrast_arm),
      to_title(outcome)
    )
    
    p <- ggplot(df_blp, aes(
      x     = estimate,
      y     = reorder(feature, estimate),
      color = sig
    )) +
      geom_point(size = 2.5) +
      geom_errorbarh(aes(xmin = ci_lower, xmax = ci_upper), height = 0.3) +
      geom_vline(xintercept = 0, linetype = "dashed",
                 color = "#2C3531", linewidth = 0.5) +
      scale_color_manual(
        values = c("FALSE" = "steelblue", "TRUE" = "tomato"),
        labels = c("FALSE" = "n.s.", "TRUE" = "p < .05"),
        name   = ""
      ) +
      labs(
        x     = sprintf(
          "BLP Coefficient\n(+) = Moderator Increases %s Effect",
          tools::toTitleCase(BASELINE_ARM)
        ),
        y     = NULL,
        title = title_str
      ) +
      theme_aesthetic_ggplot(font_scale = 1, show_grid = TRUE, base_family = BASE_FONT)
    
    print(p)
    fname <- sprintf("plots/blp_%s_vs_%s_%s.pdf", BASELINE_ARM, contrast_arm, outcome)
    ggsave(fname, plot = p, width = 9, height = 6, units = "in", device = "pdf")
    cat("Saved BLP plot:", fname, "\n")
  }
}

# ============================================================
# LOESS plots of CATE vs each numeric moderator
# ============================================================

all_num_mods <- c(BASE_MODERATORS, ENGAGEMENT_MODERATORS, SYS_MODERATORS)

for (contrast_arm in names(results)) {
  for (outcome in OUTCOMES) {
    r        <- results[[contrast_arm]][[outcome]]
    preds    <- predict(r$forest)
    cate_hat <- preds$predictions
    x_mat    <- r$X_sub
    
    # only plot numeric moderators that exist in this contrast
    numeric_cols_plot <- r$numeric_cols[r$numeric_cols %in% names(all_num_mods)]
    
    plot_list <- list()
    for (col in numeric_cols_plot) {
      label          <- unname(all_num_mods[col])
      plot_list[[col]] <- make_loess_panel(x_mat[, col], cate_hat, label)
    }
    
    n_cols   <- 4
    combined <- wrap_plots(plot_list, ncol = n_cols) +
      plot_annotation(
        title = sprintf(
          "%s vs %s — %s\nCATEs: LOESS with 95%% CI",
          tools::toTitleCase(BASELINE_ARM),
          tools::toTitleCase(contrast_arm),
          to_title(outcome)
        ),
        theme = theme(
          plot.title = element_text(face = "bold", size = 13, hjust = 0)
        )
      )
    
    fname <- sprintf("plots/loess_%s_vs_%s_%s.pdf", BASELINE_ARM, contrast_arm, outcome)
    ggsave(fname, plot = combined,
           width  = 4 * n_cols,
           height = 4 * ceiling(length(plot_list) / n_cols),
           units  = "in",
           device = "pdf")
    cat("Saved LOESS plot:", fname, "\n")
  }
}

cat("\nLog saved to:", log_file, "\n")
