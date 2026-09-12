# ==========================================
# 7. Sample-size sensitivity analysis
#
# Purpose:
# Compare Field-local modeling and Linear bridge calibration
# under progressively smaller field calibration sample sizes.

# ==========================================

# ----------------------------
# 0. Packages
# ----------------------------
required_packages <- c(
  "readxl", "dplyr", "tidyr", "ggplot2",
  "car", "caret", "randomForest"
)

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

library(readxl)
library(dplyr)
library(tidyr)
library(ggplot2)
library(car)
library(caret)
library(randomForest)

# ----------------------------
# 1. Global settings
# ----------------------------
response_var <- "y"
RANDOM_SEED <- 42
JSD_BINS <- 30

# Sample sizes can be modified if needed.
SAMPLE_SIZES <- c(12, 18, 24, 30, 36)
N_REPEATS <- 100

RF_NTREE <- 500
RF_CV_FOLDS <- 5
RF_TUNE_LENGTH <- 5

model_dir <- "C:/Users/田玲玲/Desktop"
save_dir  <- "C:/Users/田玲玲/Desktop"
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

if (any(SAMPLE_SIZES > nrow(field_calibration))) {
  stop("At least one SAMPLE_SIZES value is larger than 36.")
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

  if (length(obs) < 2) {
    return(
      data.frame(
        R2 = NA_real_,
        RMSE = NA_real_,
        MAE = NA_real_,
        MRE = NA_real_,
        JSD = NA_real_
      )
    )
  }

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

# Same MLR selection logic used in the main analysis.
fit_mlr <- function(train_df, response_var = "y") {
  feature_vars_local <- setdiff(
    names(train_df),
    response_var
  )

  full_formula <- as.formula(
    paste(
      response_var,
      "~",
      paste(feature_vars_local, collapse = " + ")
    )
  )

  full_model_alias <- lm(
    full_formula,
    data = train_df
  )

  aliased <- alias(full_model_alias)$Complete

  if (!is.null(aliased)) {
    aliased_vars <- rownames(aliased)
    feature_vars_local <- setdiff(
      feature_vars_local,
      aliased_vars
    )
  }

  if (length(feature_vars_local) == 0) {
    stop("No predictors remain after alias removal.")
  }

  repeat {

    vif_formula <- as.formula(
      paste(
        response_var,
        "~",
        paste(feature_vars_local, collapse = " + ")
      )
    )

    vif_model <- lm(
      vif_formula,
      data = train_df
    )

    vif_vals <- tryCatch(
      car::vif(vif_model),
      error = function(e) NULL
    )

    if (is.null(vif_vals)) {
      stop("VIF calculation failed.")
    }

    if (max(vif_vals) < 10) break

    remove_var <- names(
      which.max(vif_vals)
    )

    feature_vars_local <- setdiff(
      feature_vars_local,
      remove_var
    )

    if (length(feature_vars_local) <= 1) break
  }

  upper_formula <- as.formula(
    paste(
      response_var,
      "~",
      paste(feature_vars_local, collapse = " + ")
    )
  )

  lower_formula <- as.formula(
    paste(response_var, "~ 1")
  )

  upper_model <- lm(
    upper_formula,
    data = train_df
  )

  lower_model <- lm(
    lower_formula,
    data = train_df
  )

  step(
    lower_model,
    scope = list(
      lower = lower_formula,
      upper = formula(upper_model)
    ),
    direction = "both",
    trace = 0
  )
}

add_result <- function(
  result_list,
  model_name,
  sample_size,
  repetition,
  metrics,
  success = TRUE
) {

  result_list[[length(result_list) + 1]] <- data.frame(
    Model = model_name,
    Sample_size = sample_size,
    Repetition = repetition,
    Success = success,
    R2 = metrics$R2,
    RMSE = metrics$RMSE,
    MAE = metrics$MAE,
    MRE = metrics$MRE,
    JSD = metrics$JSD
  )

  result_list
}

na_metrics <- function() {
  data.frame(
    R2 = NA_real_,
    RMSE = NA_real_,
    MAE = NA_real_,
    MRE = NA_real_,
    JSD = NA_real_
  )
}

# ----------------------------
# 4. Load fixed indoor source models
# ----------------------------
mlr_source <- readRDS(
  file.path(model_dir, "MLR_model.rds")
)

rf_source <- readRDS(
  file.path(model_dir, "RF_model.rds")
)

# Pre-compute source predictions for all field samples.
pred_cal_source_mlr <- as.numeric(
  predict(
    mlr_source,
    newdata = field_calibration
  )
)

pred_val_source_mlr <- as.numeric(
  predict(
    mlr_source,
    newdata = field_validation
  )
)

pred_cal_source_rf <- as.numeric(
  predict(
    rf_source,
    newdata = field_calibration[, feature_vars, drop = FALSE]
  )
)

pred_val_source_rf <- as.numeric(
  predict(
    rf_source,
    newdata = field_validation[, feature_vars, drop = FALSE]
  )
)

# ----------------------------
# 5. Select field-local RF mtry once using all 36 samples
# ----------------------------
set.seed(RANDOM_SEED)

rf_control <- trainControl(
  method = "cv",
  number = RF_CV_FOLDS
)

field_rf_tune_full <- train(
  x = field_calibration[, feature_vars, drop = FALSE],
  y = field_calibration[[response_var]],
  method = "rf",
  tuneLength = RF_TUNE_LENGTH,
  ntree = RF_NTREE,
  trControl = rf_control
)

FIELD_RF_MTRY <- field_rf_tune_full$bestTune$mtry

write.csv(
  data.frame(
    mtry = FIELD_RF_MTRY,
    ntree = RF_NTREE,
    CV_folds = RF_CV_FOLDS
  ),
  file.path(save_dir, "Sensitivity_FieldLocal_RF_fixed_params.csv"),
  row.names = FALSE
)

cat("Sensitivity analysis field-local RF mtry =", FIELD_RF_MTRY, "\n")
cat("Sensitivity analysis field-local RF ntree =", RF_NTREE, "\n")

# ----------------------------
# 6. Repeated subsampling
# ----------------------------
set.seed(RANDOM_SEED)

results <- list()

for (n_cal in SAMPLE_SIZES) {

  # For n = 36, every repetition contains the same full calibration set.
  # It is still repeated here for a uniform output structure.
  for (rep_id in seq_len(N_REPEATS)) {

    idx <- sample(
      seq_len(nrow(field_calibration)),
      size = n_cal,
      replace = FALSE
    )

    sub_cal <- field_calibration[
      idx,
      ,
      drop = FALSE
    ]

    # ============================================================
    # A. Field-local MLR
    # ============================================================
    field_mlr <- tryCatch(
      fit_mlr(
        sub_cal,
        response_var
      ),
      error = function(e) NULL
    )

    if (!is.null(field_mlr)) {

      pred_field_mlr <- tryCatch(
        as.numeric(
          predict(
            field_mlr,
            newdata = field_validation
          )
        ),
        error = function(e) NULL
      )

      if (
        !is.null(pred_field_mlr) &&
        all(is.finite(pred_field_mlr))
      ) {
        met_field_mlr <- calc_metrics(
          field_validation$y,
          pred_field_mlr
        )

        results <- add_result(
          results,
          "Field-local MLR",
          n_cal,
          rep_id,
          met_field_mlr,
          TRUE
        )
      } else {
        results <- add_result(
          results,
          "Field-local MLR",
          n_cal,
          rep_id,
          na_metrics(),
          FALSE
        )
      }

    } else {
      results <- add_result(
        results,
        "Field-local MLR",
        n_cal,
        rep_id,
        na_metrics(),
        FALSE
      )
    }

    # ============================================================
    # B. Field-local RF
    # ============================================================
    set.seed(
      RANDOM_SEED +
      n_cal * 1000 +
      rep_id
    )

    field_rf <- tryCatch(
      randomForest::randomForest(
        x = sub_cal[, feature_vars, drop = FALSE],
        y = sub_cal[[response_var]],
        ntree = RF_NTREE,
        mtry = FIELD_RF_MTRY
      ),
      error = function(e) NULL
    )

    if (!is.null(field_rf)) {

      pred_field_rf <- as.numeric(
        predict(
          field_rf,
          newdata = field_validation[, feature_vars, drop = FALSE]
        )
      )

      met_field_rf <- calc_metrics(
        field_validation$y,
        pred_field_rf
      )

      results <- add_result(
        results,
        "Field-local RF",
        n_cal,
        rep_id,
        met_field_rf,
        TRUE
      )

    } else {
      results <- add_result(
        results,
        "Field-local RF",
        n_cal,
        rep_id,
        na_metrics(),
        FALSE
      )
    }

    # ============================================================
    # C. MLR-based Linear bridge
    # ============================================================
    bridge_mlr <- tryCatch(
      lm(
        sub_cal$y ~
          pred_cal_source_mlr[idx]
      ),
      error = function(e) NULL
    )

    if (!is.null(bridge_mlr)) {

      bridge_coef <- coef(bridge_mlr)

      if (
        length(bridge_coef) == 2 &&
        all(is.finite(bridge_coef))
      ) {

        pred_bridge_mlr <- (
          bridge_coef[1] +
          bridge_coef[2] *
          pred_val_source_mlr
        )

        met_bridge_mlr <- calc_metrics(
          field_validation$y,
          pred_bridge_mlr
        )

        results <- add_result(
          results,
          "MLR Linear bridge",
          n_cal,
          rep_id,
          met_bridge_mlr,
          TRUE
        )

      } else {
        results <- add_result(
          results,
          "MLR Linear bridge",
          n_cal,
          rep_id,
          na_metrics(),
          FALSE
        )
      }

    } else {
      results <- add_result(
        results,
        "MLR Linear bridge",
        n_cal,
        rep_id,
        na_metrics(),
        FALSE
      )
    }

    # ============================================================
    # D. RF-based Linear bridge
    # ============================================================
    bridge_rf <- tryCatch(
      lm(
        sub_cal$y ~
          pred_cal_source_rf[idx]
      ),
      error = function(e) NULL
    )

    if (!is.null(bridge_rf)) {

      bridge_coef <- coef(bridge_rf)

      if (
        length(bridge_coef) == 2 &&
        all(is.finite(bridge_coef))
      ) {

        pred_bridge_rf <- (
          bridge_coef[1] +
          bridge_coef[2] *
          pred_val_source_rf
        )

        met_bridge_rf <- calc_metrics(
          field_validation$y,
          pred_bridge_rf
        )

        results <- add_result(
          results,
          "RF Linear bridge",
          n_cal,
          rep_id,
          met_bridge_rf,
          TRUE
        )

      } else {
        results <- add_result(
          results,
          "RF Linear bridge",
          n_cal,
          rep_id,
          na_metrics(),
          FALSE
        )
      }

    } else {
      results <- add_result(
        results,
        "RF Linear bridge",
        n_cal,
        rep_id,
        na_metrics(),
        FALSE
      )
    }
  }

  cat(
    "Completed sample size:",
    n_cal,
    "\n"
  )
}

results_df <- bind_rows(results)

# ----------------------------
# 7. Summaries
# ----------------------------
success_summary <- results_df %>%
  group_by(
    Model,
    Sample_size
  ) %>%
  summarise(
    Success_rate = mean(Success),
    Successful_runs = sum(Success),
    Total_runs = n(),
    .groups = "drop"
  )

long_metrics <- results_df %>%
  filter(Success) %>%
  pivot_longer(
    cols = c(
      R2, RMSE, MAE, MRE, JSD
    ),
    names_to = "Metric",
    values_to = "Value"
  )

summary_df <- long_metrics %>%
  group_by(
    Model,
    Sample_size,
    Metric
  ) %>%
  summarise(
    Mean = mean(Value, na.rm = TRUE),
    SD = sd(Value, na.rm = TRUE),
    Median = median(Value, na.rm = TRUE),
    Q2.5 = quantile(
      Value,
      0.025,
      na.rm = TRUE
    ),
    Q97.5 = quantile(
      Value,
      0.975,
      na.rm = TRUE
    ),
    .groups = "drop"
  )

# ----------------------------
# 8. Save tables
# ----------------------------
write.csv(
  results_df,
  file.path(save_dir, "Sample_size_raw_results.csv"),
  row.names = FALSE
)

write.csv(
  summary_df,
  file.path(save_dir, "Sample_size_summary.csv"),
  row.names = FALSE
)

write.csv(
  success_summary,
  file.path(save_dir, "Sample_size_success_rate.csv"),
  row.names = FALSE
)

# ----------------------------
# 9. Plots
# ----------------------------
plot_rmse <- summary_df %>%
  filter(Metric == "RMSE")

p_rmse <- ggplot(
  plot_rmse,
  aes(
    x = Sample_size,
    y = Median,
    group = Model,
    linetype = Model
  )
) +
  geom_ribbon(
    aes(
      ymin = Q2.5,
      ymax = Q97.5,
      fill = Model
    ),
    alpha = 0.15,
    color = NA
  ) +
  geom_line(
    linewidth = 0.8
  ) +
  geom_point(
    size = 2
  ) +
  labs(
    title = "Sample-size sensitivity: RMSE",
    subtitle = paste0(
      N_REPEATS,
      " repeated subsamples; fixed 84-sample validation set"
    ),
    x = "Number of field calibration/training samples",
    y = "Validation RMSE"
  ) +
  theme_bw(base_size = 11)

ggsave(
  file.path(save_dir, "Sample_size_RMSE.pdf"),
  p_rmse,
  width = 7,
  height = 5,
  dpi = 300
)

plot_mre <- summary_df %>%
  filter(Metric == "MRE")

p_mre <- ggplot(
  plot_mre,
  aes(
    x = Sample_size,
    y = Median,
    group = Model,
    linetype = Model
  )
) +
  geom_ribbon(
    aes(
      ymin = Q2.5,
      ymax = Q97.5,
      fill = Model
    ),
    alpha = 0.15,
    color = NA
  ) +
  geom_line(
    linewidth = 0.8
  ) +
  geom_point(
    size = 2
  ) +
  labs(
    title = "Sample-size sensitivity: MRE",
    subtitle = paste0(
      N_REPEATS,
      " repeated subsamples; fixed 84-sample validation set"
    ),
    x = "Number of field calibration/training samples",
    y = "Validation MRE (%)"
  ) +
  theme_bw(base_size = 11)

ggsave(
  file.path(save_dir, "Sample_size_MRE.pdf"),
  p_mre,
  width = 7,
  height = 5,
  dpi = 300
)

cat("\nSample-size sensitivity analysis completed.\n")
cat("Sample sizes:", paste(SAMPLE_SIZES, collapse = ", "), "\n")
cat("Repeated subsamples per size:", N_REPEATS, "\n")
