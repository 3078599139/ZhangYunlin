# ==========================================
# 6. Bridge robustness analysis
# ==========================================

# ----------------------------
# 0. Packages
# ----------------------------
required_packages <- c(
  "readxl", "dplyr", "ggplot2", "mgcv"
)

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

library(readxl)
library(dplyr)
library(ggplot2)
library(mgcv)

# ----------------------------
# 1. Global settings
# ----------------------------
response_var <- "y"
RANDOM_SEED <- 42
BOOT_B <- 1000
JSD_BINS <- 30

model_dir <- "XXX"
save_dir  <- "XXX"
dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)

# ----------------------------
# 2. Read final field split
# ----------------------------
# Select in order:
# 1) FINAL field_calibration.xlsx (n = 36)
# 2) FINAL field_validation.xlsx  (n = 84)
field_calibration <- read_excel(file.choose())
field_validation  <- read_excel(file.choose())

field_calibration <- as.data.frame(field_calibration)
field_validation  <- as.data.frame(field_validation)

field_calibration[] <- lapply(
  field_calibration,
  function(x) as.numeric(as.character(x))
)

field_validation[] <- lapply(
  field_validation,
  function(x) as.numeric(as.character(x))
)

field_calibration <- na.omit(field_calibration)
field_validation  <- na.omit(field_validation)

if (nrow(field_calibration) != 36) {
  stop("Final field calibration set must contain 36 samples.")
}

if (nrow(field_validation) != 84) {
  stop("Final field validation set must contain 84 samples.")
}

feature_vars <- setdiff(
  names(field_calibration),
  response_var
)

# ----------------------------
# 3. Utility functions
# ----------------------------
calc_jsd <- function(obs, pred, n_bins = JSD_BINS, eps = 1e-12) {
  obs <- as.numeric(obs)
  pred <- as.numeric(pred)

  keep <- is.finite(obs) & is.finite(pred)
  obs <- obs[keep]
  pred <- pred[keep]

  if (length(obs) < 2 || length(pred) < 2) return(NA_real_)

  rng <- range(c(obs, pred), na.rm = TRUE)
  if (!all(is.finite(rng))) return(NA_real_)
  if (diff(rng) == 0) return(0)

  breaks <- seq(rng[1], rng[2], length.out = n_bins + 1)

  p <- hist(obs, breaks = breaks, plot = FALSE, include.lowest = TRUE)$counts + eps
  q <- hist(pred, breaks = breaks, plot = FALSE, include.lowest = TRUE)$counts + eps

  p <- p / sum(p)
  q <- q / sum(q)
  m <- 0.5 * (p + q)

  jsd <- 0.5 * sum(p * log2(p / m)) +
         0.5 * sum(q * log2(q / m))

  as.numeric(jsd)
}

calc_metrics <- function(obs, pred) {
  obs <- as.numeric(obs)
  pred <- as.numeric(pred)

  keep <- is.finite(obs) & is.finite(pred)
  obs <- obs[keep]
  pred <- pred[keep]

  r2   <- 1 - sum((obs - pred)^2) / sum((obs - mean(obs))^2)
  rmse <- sqrt(mean((obs - pred)^2))
  mae  <- mean(abs(obs - pred))

  nonzero <- obs != 0
  mre <- if (any(nonzero)) {
    mean(abs((obs[nonzero] - pred[nonzero]) / obs[nonzero])) * 100
  } else {
    NA_real_
  }

  jsd <- calc_jsd(obs, pred)

  data.frame(
    R2 = r2,
    RMSE = rmse,
    MAE = mae,
    MRE = mre,
    JSD = jsd
  )
}

safe_cor_test <- function(x, y, method = "pearson") {
  keep <- is.finite(x) & is.finite(y)

  if (sum(keep) < 3) {
    return(
      data.frame(
        Method = method,
        Estimate = NA_real_,
        P_value = NA_real_
      )
    )
  }

  result <- suppressWarnings(
    cor.test(
      x[keep],
      y[keep],
      method = method,
      exact = FALSE
    )
  )

  data.frame(
    Method = method,
    Estimate = unname(result$estimate),
    P_value = result$p.value
  )
}

paired_error_test <- function(obs, pred_a, pred_b, name_a, name_b) {
  err_a <- abs(pred_a - obs)
  err_b <- abs(pred_b - obs)

  test_result <- suppressWarnings(
    wilcox.test(
      err_a,
      err_b,
      paired = TRUE,
      exact = FALSE
    )
  )

  data.frame(
    Model_A = name_a,
    Model_B = name_b,
    Median_abs_error_A = median(err_a, na.rm = TRUE),
    Median_abs_error_B = median(err_b, na.rm = TRUE),
    Wilcoxon_V = unname(test_result$statistic),
    P_value = test_result$p.value
  )
}

# ----------------------------
# 4. Load indoor source models
# ----------------------------
mlr_source <- readRDS(
  file.path(model_dir, "MLR_model.rds")
)

rf_source <- readRDS(
  file.path(model_dir, "RF_model.rds")
)

# ----------------------------
# 5. Robustness analysis for one base model
# ----------------------------
analyze_one_base <- function(base_model_type) {

  x_cal <- field_calibration[, feature_vars, drop = FALSE]
  x_val <- field_validation[, feature_vars, drop = FALSE]

  if (base_model_type == "MLR") {
    pred_cal_indoor <- as.numeric(
      predict(mlr_source, newdata = field_calibration)
    )
    pred_val_indoor <- as.numeric(
      predict(mlr_source, newdata = field_validation)
    )
  }

  if (base_model_type == "RF") {
    pred_cal_indoor <- as.numeric(
      predict(rf_source, newdata = x_cal)
    )
    pred_val_indoor <- as.numeric(
      predict(rf_source, newdata = x_val)
    )
  }

  # ---------- Correlation before bridge calibration ----------
  cor_cal_pearson <- safe_cor_test(
    field_calibration$y,
    pred_cal_indoor,
    method = "pearson"
  )

  cor_cal_spearman <- safe_cor_test(
    field_calibration$y,
    pred_cal_indoor,
    method = "spearman"
  )

  cor_val_pearson <- safe_cor_test(
    field_validation$y,
    pred_val_indoor,
    method = "pearson"
  )

  cor_val_spearman <- safe_cor_test(
    field_validation$y,
    pred_val_indoor,
    method = "spearman"
  )

  correlation_summary <- bind_rows(
    cbind(
      BaseModel = base_model_type,
      Dataset = "Calibration",
      cor_cal_pearson
    ),
    cbind(
      BaseModel = base_model_type,
      Dataset = "Calibration",
      cor_cal_spearman
    ),
    cbind(
      BaseModel = base_model_type,
      Dataset = "Validation",
      cor_val_pearson
    ),
    cbind(
      BaseModel = base_model_type,
      Dataset = "Validation",
      cor_val_spearman
    )
  )

  # ---------- Fit three bridge models ----------
  bridge_linear <- lm(
    field_calibration$y ~ pred_cal_indoor
  )

  bridge_quad <- lm(
    field_calibration$y ~
      pred_cal_indoor +
      I(pred_cal_indoor^2)
  )

  bridge_gam <- mgcv::gam(
    field_calibration$y ~ s(pred_cal_indoor),
    method = "REML"
  )

  pred_linear <- as.numeric(
    predict(
      bridge_linear,
      newdata = data.frame(
        pred_cal_indoor = pred_val_indoor
      )
    )
  )

  pred_quad <- as.numeric(
    predict(
      bridge_quad,
      newdata = data.frame(
        pred_cal_indoor = pred_val_indoor
      )
    )
  )

  pred_gam <- as.numeric(
    predict(
      bridge_gam,
      newdata = data.frame(
        pred_cal_indoor = pred_val_indoor
      )
    )
  )

  # Naive baseline: predict every validation sample using
  # the mean load of the 36 calibration samples.
  pred_mean <- rep(
    mean(field_calibration$y),
    nrow(field_validation)
  )

  metrics_summary <- bind_rows(
    cbind(
      BaseModel = base_model_type,
      Model = "Direct source prediction",
      calc_metrics(field_validation$y, pred_val_indoor)
    ),
    cbind(
      BaseModel = base_model_type,
      Model = "Linear bridge",
      calc_metrics(field_validation$y, pred_linear)
    ),
    cbind(
      BaseModel = base_model_type,
      Model = "Quadratic bridge",
      calc_metrics(field_validation$y, pred_quad)
    ),
    cbind(
      BaseModel = base_model_type,
      Model = "GAM bridge",
      calc_metrics(field_validation$y, pred_gam)
    ),
    cbind(
      BaseModel = base_model_type,
      Model = "Calibration-mean baseline",
      calc_metrics(field_validation$y, pred_mean)
    )
  )

  # ---------- Linear bridge slope and confidence interval ----------
  coef_linear <- coef(bridge_linear)
  ci_linear <- confint(
    bridge_linear,
    level = 0.95
  )

  slope_summary <- data.frame(
    BaseModel = base_model_type,
    Intercept = unname(coef_linear[1]),
    Slope = unname(coef_linear[2]),
    Slope_CI_low = ci_linear["pred_cal_indoor", 1],
    Slope_CI_high = ci_linear["pred_cal_indoor", 2],
    Calibration_R2 = summary(bridge_linear)$r.squared,
    Calibration_Adj_R2 = summary(bridge_linear)$adj.r.squared
  )

  # ---------- Shrinkage diagnostics ----------
  get_distribution_stats <- function(x) {
    data.frame(
      Mean = mean(x, na.rm = TRUE),
      SD = sd(x, na.rm = TRUE),
      Min = min(x, na.rm = TRUE),
      Max = max(x, na.rm = TRUE),
      Range = diff(range(x, na.rm = TRUE))
    )
  }

  dist_observed <- get_distribution_stats(
    field_validation$y
  )

  dist_direct <- get_distribution_stats(
    pred_val_indoor
  )

  dist_linear <- get_distribution_stats(
    pred_linear
  )

  dist_mean <- get_distribution_stats(
    pred_mean
  )

  shrinkage_summary <- bind_rows(
    cbind(
      BaseModel = base_model_type,
      Prediction = "Observed",
      dist_observed
    ),
    cbind(
      BaseModel = base_model_type,
      Prediction = "Direct source",
      dist_direct
    ),
    cbind(
      BaseModel = base_model_type,
      Prediction = "Linear bridge",
      dist_linear
    ),
    cbind(
      BaseModel = base_model_type,
      Prediction = "Calibration mean",
      dist_mean
    )
  )

  observed_sd <- sd(field_validation$y)
  observed_range <- diff(range(field_validation$y))

  shrinkage_summary$SD_ratio_to_observed <- (
    shrinkage_summary$SD / observed_sd
  )

  shrinkage_summary$Range_ratio_to_observed <- (
    shrinkage_summary$Range / observed_range
  )

  # ---------- GAM effective degrees of freedom ----------
  gam_sum <- summary(bridge_gam)

  gam_summary <- data.frame(
    BaseModel = base_model_type,
    Term = rownames(gam_sum$s.table),
    edf = gam_sum$s.table[, "edf"],
    Ref_df = gam_sum$s.table[, "Ref.df"],
    F_value = gam_sum$s.table[, "F"],
    P_value = gam_sum$s.table[, "p-value"],
    row.names = NULL
  )

  # ---------- Paired error comparisons ----------
  paired_tests <- bind_rows(
    paired_error_test(
      field_validation$y,
      pred_linear,
      pred_quad,
      "Linear bridge",
      "Quadratic bridge"
    ),
    paired_error_test(
      field_validation$y,
      pred_linear,
      pred_gam,
      "Linear bridge",
      "GAM bridge"
    ),
    paired_error_test(
      field_validation$y,
      pred_linear,
      pred_val_indoor,
      "Linear bridge",
      "Direct source"
    ),
    paired_error_test(
      field_validation$y,
      pred_linear,
      pred_mean,
      "Linear bridge",
      "Calibration-mean baseline"
    )
  )

  paired_tests$BaseModel <- base_model_type
  paired_tests$FDR_BH <- p.adjust(
    paired_tests$P_value,
    method = "BH"
  )

  # ---------- Bootstrap calibration stability ----------
  set.seed(RANDOM_SEED)

  boot_results <- vector(
    "list",
    BOOT_B
  )

  for (b in seq_len(BOOT_B)) {

    idx <- sample(
      seq_len(nrow(field_calibration)),
      size = nrow(field_calibration),
      replace = TRUE
    )

    boot_y <- field_calibration$y[idx]
    boot_pred <- pred_cal_indoor[idx]

    boot_model <- tryCatch(
      lm(boot_y ~ boot_pred),
      error = function(e) NULL
    )

    if (is.null(boot_model)) {
      next
    }

    boot_coef <- coef(boot_model)

    if (
      length(boot_coef) < 2 ||
      any(!is.finite(boot_coef))
    ) {
      next
    }

    boot_val_pred <- (
      boot_coef[1] +
      boot_coef[2] * pred_val_indoor
    )

    boot_metrics <- calc_metrics(
      field_validation$y,
      boot_val_pred
    )

    boot_results[[b]] <- data.frame(
      BaseModel = base_model_type,
      Bootstrap = b,
      Intercept = unname(boot_coef[1]),
      Slope = unname(boot_coef[2]),
      R2 = boot_metrics$R2,
      RMSE = boot_metrics$RMSE,
      MAE = boot_metrics$MAE,
      MRE = boot_metrics$MRE,
      JSD = boot_metrics$JSD
    )
  }

  boot_df <- bind_rows(boot_results)

  quantile_safe <- function(x, p) {
    if (all(is.na(x))) return(NA_real_)
    unname(
      quantile(
        x,
        probs = p,
        na.rm = TRUE
      )
    )
  }

  boot_ci <- data.frame(
    BaseModel = base_model_type,
    Statistic = c(
      "Intercept",
      "Slope",
      "R2",
      "RMSE",
      "MAE",
      "MRE",
      "JSD"
    ),
    Estimate_full = c(
      coef_linear[1],
      coef_linear[2],
      calc_metrics(
        field_validation$y,
        pred_linear
      )$R2,
      calc_metrics(
        field_validation$y,
        pred_linear
      )$RMSE,
      calc_metrics(
        field_validation$y,
        pred_linear
      )$MAE,
      calc_metrics(
        field_validation$y,
        pred_linear
      )$MRE,
      calc_metrics(
        field_validation$y,
        pred_linear
      )$JSD
    ),
    Bootstrap_2.5 = c(
      quantile_safe(boot_df$Intercept, 0.025),
      quantile_safe(boot_df$Slope, 0.025),
      quantile_safe(boot_df$R2, 0.025),
      quantile_safe(boot_df$RMSE, 0.025),
      quantile_safe(boot_df$MAE, 0.025),
      quantile_safe(boot_df$MRE, 0.025),
      quantile_safe(boot_df$JSD, 0.025)
    ),
    Bootstrap_50 = c(
      quantile_safe(boot_df$Intercept, 0.50),
      quantile_safe(boot_df$Slope, 0.50),
      quantile_safe(boot_df$R2, 0.50),
      quantile_safe(boot_df$RMSE, 0.50),
      quantile_safe(boot_df$MAE, 0.50),
      quantile_safe(boot_df$MRE, 0.50),
      quantile_safe(boot_df$JSD, 0.50)
    ),
    Bootstrap_97.5 = c(
      quantile_safe(boot_df$Intercept, 0.975),
      quantile_safe(boot_df$Slope, 0.975),
      quantile_safe(boot_df$R2, 0.975),
      quantile_safe(boot_df$RMSE, 0.975),
      quantile_safe(boot_df$MAE, 0.975),
      quantile_safe(boot_df$MRE, 0.975),
      quantile_safe(boot_df$JSD, 0.975)
    )
  )

  # ---------- Save bootstrap plot ----------
  p_slope <- ggplot(
    boot_df,
    aes(x = Slope)
  ) +
    geom_histogram(
      bins = 30
    ) +
    geom_vline(
      xintercept = coef_linear[2],
      linetype = "dashed"
    ) +
    labs(
      title = paste0(
        "Bootstrap distribution of bridge slope: ",
        base_model_type
      ),
      x = "Linear bridge slope",
      y = "Frequency"
    ) +
    theme_bw()

  ggsave(
    file.path(
      save_dir,
      paste0(
        "Bridge_", base_model_type,
        "_bootstrap_slope.pdf"
      )
    ),
    p_slope,
    width = 6,
    height = 4.5,
    dpi = 300
  )

  list(
    correlations = correlation_summary,
    metrics = metrics_summary,
    slope = slope_summary,
    shrinkage = shrinkage_summary,
    gam = gam_summary,
    paired_tests = paired_tests,
    bootstrap_raw = boot_df,
    bootstrap_ci = boot_ci
  )
}

# ----------------------------
# 6. Run MLR-based and RF-based analyses
# ----------------------------
robust_mlr <- analyze_one_base("MLR")
robust_rf  <- analyze_one_base("RF")

# ----------------------------
# 7. Save outputs
# ----------------------------
write.csv(
  bind_rows(
    robust_mlr$correlations,
    robust_rf$correlations
  ),
  file.path(save_dir, "Bridge_source_prediction_correlations.csv"),
  row.names = FALSE
)

write.csv(
  bind_rows(
    robust_mlr$metrics,
    robust_rf$metrics
  ),
  file.path(save_dir, "Bridge_robustness_metrics.csv"),
  row.names = FALSE
)

write.csv(
  bind_rows(
    robust_mlr$slope,
    robust_rf$slope
  ),
  file.path(save_dir, "Bridge_linear_slope_CI.csv"),
  row.names = FALSE
)

write.csv(
  bind_rows(
    robust_mlr$shrinkage,
    robust_rf$shrinkage
  ),
  file.path(save_dir, "Bridge_shrinkage_diagnostics.csv"),
  row.names = FALSE
)

write.csv(
  bind_rows(
    robust_mlr$gam,
    robust_rf$gam
  ),
  file.path(save_dir, "Bridge_GAM_edf.csv"),
  row.names = FALSE
)

write.csv(
  bind_rows(
    robust_mlr$paired_tests,
    robust_rf$paired_tests
  ),
  file.path(save_dir, "Bridge_paired_error_tests.csv"),
  row.names = FALSE
)

write.csv(
  bind_rows(
    robust_mlr$bootstrap_raw,
    robust_rf$bootstrap_raw
  ),
  file.path(save_dir, "Bridge_bootstrap_raw.csv"),
  row.names = FALSE
)

write.csv(
  bind_rows(
    robust_mlr$bootstrap_ci,
    robust_rf$bootstrap_ci
  ),
  file.path(save_dir, "Bridge_bootstrap_CI.csv"),
  row.names = FALSE
)

cat("\nBridge robustness analysis completed.\n")
cat("Bootstrap repetitions:", BOOT_B, "\n")
