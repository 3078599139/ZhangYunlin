# ==========================================
# 1. Indoor source models
# Models: MLR / RF
# Added: R2, JSD, MLR diagnostics, explicit RF ntree,
#        5-fold CV tuning for mtry, OOB diagnostics
# ==========================================

# ----------------------------
# 0. Packages
# ----------------------------
required_packages <- c(
  "readxl", "dplyr", "car", "caret", "randomForest",
  "ggplot2", "MASS", "patchwork", "lmtest"
)

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

library(readxl)
library(dplyr)
library(car)
library(caret)
library(randomForest)
library(ggplot2)
library(MASS)
library(patchwork)
library(lmtest)

# ----------------------------
# 1. Global settings
# ----------------------------
response_var <- "y"
RANDOM_SEED <- 42    # 随机种子编号
JSD_BINS <- 30       # JSD计算时，将连续特征分成30个区间

# RF settings:
# mtry is optimized by 5-fold cross-validation.
# ntree is fixed explicitly.
RF_NTREE <- 500
RF_CV_FOLDS <- 5     # 随机森林调参采用5折交叉验证
RF_TUNE_LENGTH <- 5  # 自动搜索5个候选mtry参数值

save_dir <- "XXX"
dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)

# ----------------------------
# 2. Read data
# ----------------------------
# Select in order:
# 1) indoor_train.xlsx (expected n = 84)
# 2) indoor_test.xlsx  (expected n = 36)
train <- read_excel(file.choose())
test  <- read_excel(file.choose())

train <- as.data.frame(train)
test  <- as.data.frame(test)

if (nrow(train) != 84) {
  warning(paste0("Indoor training set has ", nrow(train), " samples; expected 84."))
}
if (nrow(test) != 36) {
  warning(paste0("Indoor testing set has ", nrow(test), " samples; expected 36."))
}

# ----------------------------
# 3. Prepare data
# ----------------------------
prepare_data <- function(df) {
  df[] <- lapply(df, function(x) as.numeric(as.character(x)))
  na.omit(df)
}

train <- prepare_data(train)
test  <- prepare_data(test)

required_cols <- names(train)
missing_cols <- setdiff(required_cols, names(test))
if (length(missing_cols) > 0) {
  stop(paste("测试集缺少以下列：", paste(missing_cols, collapse = ", ")))
}
test <- test[, required_cols]

feature_vars_all <- setdiff(names(train), response_var)    # 获取train数据框中的列名，把y去掉，剩下为预测变量。

cat("Indoor training samples:", nrow(train), "\n")    # 统计训练集train有多少行
cat("Indoor testing samples :", nrow(test), "\n")     # 统计测试test的样本数量
cat("Number of predictors   :", length(feature_vars_all), "\n")   # 统计预测变量

# ----------------------------
# 4. Utility functions
# ----------------------------
# Jensen-Shannon divergence between observed and predicted distributions.
# The two distributions are discretized using 30 equal-width bins over
# their shared range. log2 is used, so JSD theoretically ranges from 0 to 1.
# JSD is treated as a supplementary distributional metric.
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

padded_range <- function(x, pad_ratio = 0.05) {
  rng <- range(x, na.rm = TRUE)
  span <- diff(rng)

  if (!is.finite(span) || span == 0) {
    span <- max(abs(rng), na.rm = TRUE)
    if (!is.finite(span) || span == 0) span <- 1
  }

  pad <- span * pad_ratio
  c(rng[1] - pad, rng[2] + pad)
}

calc_point_density <- function(x, y, n = 120) {
  x <- as.numeric(x)
  y <- as.numeric(y)

  xlim <- padded_range(x)
  ylim <- padded_range(y)

  dens <- MASS::kde2d(
    x, y,
    n = n,
    lims = c(xlim, ylim)
  )

  ix <- findInterval(x, dens$x, all.inside = TRUE)
  iy <- findInterval(y, dens$y, all.inside = TRUE)

  ix <- pmax(1, pmin(ix, length(dens$x)))
  iy <- pmax(1, pmin(iy, length(dens$y)))

  as.numeric(dens$z[cbind(ix, iy)])
}

save_jsd_density_plot <- function(obs, pred, metrics, title_text, file_path) {
  obs <- as.numeric(obs)
  pred <- as.numeric(pred)

  keep <- is.finite(obs) & is.finite(pred)
  obs <- obs[keep]
  pred <- pred[keep]

  df <- data.frame(
    observed = obs,
    predicted = pred
  )
  df$density <- calc_point_density(df$observed, df$predicted)

  axis_lim <- padded_range(c(obs, pred))

  metrics_text <- paste0(
    "R² = ", sprintf("%.3f", metrics$R2), "\n",
    "RMSE = ", sprintf("%.3f", metrics$RMSE), "\n",
    "MAE = ", sprintf("%.3f", metrics$MAE), "\n",
    "MRE = ", sprintf("%.2f", metrics$MRE), "%\n",
    "JSD = ", sprintf("%.3f", metrics$JSD)
  )

  p_scatter <- ggplot(df, aes(x = observed, y = predicted)) +
    geom_point(aes(color = density), size = 2.4, alpha = 0.85) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    geom_smooth(method = "lm", se = FALSE, linewidth = 0.7) +
    annotate(
      "label",
      x = axis_lim[1],
      y = axis_lim[2],
      label = metrics_text,
      hjust = 0,
      vjust = 1,
      size = 3.6,
      label.size = 0.2,
      fill = "white",
      alpha = 0.85
    ) +
    labs(
      title = title_text,
      x = "Observed load",
      y = "Predicted load",
      color = "Point density"
    ) +
    coord_equal(xlim = axis_lim, ylim = axis_lim) +
    theme_bw(base_size = 12)

  dist_df <- rbind(
    data.frame(value = obs, Distribution = "Observed"),
    data.frame(value = pred, Distribution = "Predicted")
  )

  p_density <- ggplot(
    dist_df,
    aes(x = value, color = Distribution, fill = Distribution)
  ) +
    geom_density(alpha = 0.22, linewidth = 0.8) +
    annotate(
      "text",
      x = axis_lim[1],
      y = Inf,
      label = paste0(
        "JSD = ", sprintf("%.3f", metrics$JSD),
        "  (0 = identical distributions)"
      ),
      hjust = 0,
      vjust = 1.3,
      size = 3.8
    ) +
    labs(
      x = "Load",
      y = "Density"
    ) +
    scale_x_continuous(limits = axis_lim) +
    theme_bw(base_size = 12) +
    theme(
      legend.position = "top",
      legend.title = element_blank()
    )

  p_combined <- p_scatter / p_density +
    patchwork::plot_layout(heights = c(3.1, 1.15))

  ggsave(
    file_path,
    p_combined,
    width = 6.2,
    height = 7.2,
    dpi = 300
  )
}

# ----------------------------
# 5. MLR model
# ----------------------------
# Variable-selection procedure:
# 1) remove completely aliased predictors;
# 2) iteratively remove the predictor with the largest VIF until all VIF < 10;
# 3) perform bidirectional AIC stepwise selection.
fit_mlr <- function(train_df, response_var = "y") {
  feature_vars <- setdiff(names(train_df), response_var)

  full_formula <- as.formula(
    paste(response_var, "~", paste(feature_vars, collapse = " + "))
  )

  full_model_alias <- lm(full_formula, data = train_df)

  aliased <- alias(full_model_alias)$Complete
  if (!is.null(aliased)) {
    aliased_vars <- rownames(aliased)
    feature_vars <- setdiff(feature_vars, aliased_vars)
  }

  repeat {
    vif_formula <- as.formula(
      paste(response_var, "~", paste(feature_vars, collapse = " + "))
    )

    vif_model <- lm(vif_formula, data = train_df)
    vif_vals <- car::vif(vif_model)

    if (max(vif_vals) < 10) break

    remove_var <- names(which.max(vif_vals))
    feature_vars <- setdiff(feature_vars, remove_var)

    if (length(feature_vars) <= 1) break
  }

  upper_formula <- as.formula(
    paste(response_var, "~", paste(feature_vars, collapse = " + "))
  )
  lower_formula <- as.formula(paste(response_var, "~ 1"))

  upper_model <- lm(upper_formula, data = train_df)
  lower_model <- lm(lower_formula, data = train_df)

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

mlr_model <- fit_mlr(train, response_var)

mlr_pred_train <- predict(mlr_model, newdata = train)
mlr_pred_test  <- predict(mlr_model, newdata = test)

mlr_metrics <- bind_rows(
  cbind(
    Model = "MLR",
    Dataset = "Train",
    calc_metrics(train$y, mlr_pred_train)
  ),
  cbind(
    Model = "MLR",
    Dataset = "Test",
    calc_metrics(test$y, mlr_pred_test)
  )
)

mlr_coef_df <- data.frame(
  Feature = rownames(summary(mlr_model)$coefficients),
  summary(mlr_model)$coefficients,
  row.names = NULL
)

colnames(mlr_coef_df) <- c(
  "Feature", "Estimate", "Std_Error", "t_value", "P_value"
)

# MLR residual diagnostics.
# Independence is primarily addressed by the sampling design; therefore,
# no sequence-based Durbin-Watson test is used here.
resid_values <- residuals(mlr_model)

shapiro_result <- if (length(resid_values) >= 3 && length(resid_values) <= 5000) {
  shapiro.test(resid_values)
} else {
  NULL
}

bp_result <- lmtest::bptest(mlr_model)

mlr_diagnostic_stats <- data.frame(
  Diagnostic = c(
    "Shapiro-Wilk residual normality",
    "Breusch-Pagan homoscedasticity"
  ),
  Statistic = c(
    ifelse(is.null(shapiro_result), NA, unname(shapiro_result$statistic)),
    unname(bp_result$statistic)
  ),
  P_value = c(
    ifelse(is.null(shapiro_result), NA, shapiro_result$p.value),
    bp_result$p.value
  )
)

pdf(
  file.path(save_dir, "Indoor_MLR_residual_diagnostics.pdf"),
  width = 8,
  height = 8
)
par(mfrow = c(2, 2))
plot(mlr_model)
dev.off()

# ----------------------------
# 6. RF model
# ----------------------------
# mtry is selected using 5-fold cross-validation.
# ntree is fixed explicitly at RF_NTREE (= 500).
set.seed(RANDOM_SEED)

rf_control <- trainControl(
  method = "cv",
  number = RF_CV_FOLDS
)

rf_tune <- train(
  x = train[, feature_vars_all, drop = FALSE],
  y = train[[response_var]],
  method = "rf",
  tuneLength = RF_TUNE_LENGTH,
  ntree = RF_NTREE,
  importance = TRUE,
  trControl = rf_control
)

rf_model <- rf_tune

rf_pred_train <- predict(
  rf_model,
  newdata = train[, feature_vars_all, drop = FALSE]
)

rf_pred_test <- predict(
  rf_model,
  newdata = test[, feature_vars_all, drop = FALSE]
)

rf_metrics <- bind_rows(
  cbind(
    Model = "RF",
    Dataset = "Train",
    calc_metrics(train$y, rf_pred_train)
  ),
  cbind(
    Model = "RF",
    Dataset = "Test",
    calc_metrics(test$y, rf_pred_test)
  )
)

rf_param_df <- data.frame(
  mtry = rf_tune$bestTune$mtry,
  ntree = RF_NTREE,
  CV_folds = RF_CV_FOLDS
)

# Save all mtry candidates and their cross-validation performance.
rf_cv_results <- rf_tune$results

# OOB error is reported as a diagnostic of the fitted RF;
# it is not used here to claim that ntree was optimized.
rf_oob_df <- data.frame(
  Tree = seq_along(rf_tune$finalModel$mse),
  OOB_MSE = rf_tune$finalModel$mse,
  OOB_RMSE = sqrt(rf_tune$finalModel$mse)
)

p_oob <- ggplot(
  rf_oob_df,
  aes(x = Tree, y = OOB_RMSE)
) +
  geom_line(linewidth = 0.7) +
  labs(
    title = "Indoor RF: OOB error by number of trees",
    x = "Number of trees",
    y = "OOB RMSE"
  ) +
  theme_bw()

ggsave(
  file.path(save_dir, "Indoor_RF_OOB_error_curve.pdf"),
  p_oob,
  width = 6,
  height = 4.5,
  dpi = 300
)

rf_importance <- as.data.frame(varImp(rf_tune)$importance)
rf_importance$Feature <- rownames(rf_importance)
rownames(rf_importance) <- NULL

# ----------------------------
# 7. Save outputs
# ----------------------------
all_metrics <- bind_rows(
  mlr_metrics,
  rf_metrics
)

write.csv(
  all_metrics,
  file.path(save_dir, "Indoor_source_metrics.csv"),
  row.names = FALSE
)

write.csv(
  mlr_coef_df,
  file.path(save_dir, "Indoor_MLR_coefficients.csv"),
  row.names = FALSE
)

write.csv(
  mlr_diagnostic_stats,
  file.path(save_dir, "Indoor_MLR_diagnostics.csv"),
  row.names = FALSE
)

write.csv(
  rf_param_df,
  file.path(save_dir, "Indoor_RF_best_params.csv"),
  row.names = FALSE
)

write.csv(
  rf_cv_results,
  file.path(save_dir, "Indoor_RF_CV_tuning_results.csv"),
  row.names = FALSE
)

write.csv(
  rf_oob_df,
  file.path(save_dir, "Indoor_RF_OOB_error.csv"),
  row.names = FALSE
)

write.csv(
  rf_importance,
  file.path(save_dir, "Indoor_RF_variable_importance.csv"),
  row.names = FALSE
)

pred_df <- bind_rows(
  data.frame(
    Model = "MLR",
    Dataset = "Train",
    observed = train$y,
    predicted = mlr_pred_train
  ),
  data.frame(
    Model = "MLR",
    Dataset = "Test",
    observed = test$y,
    predicted = mlr_pred_test
  ),
  data.frame(
    Model = "RF",
    Dataset = "Train",
    observed = train$y,
    predicted = rf_pred_train
  ),
  data.frame(
    Model = "RF",
    Dataset = "Test",
    observed = test$y,
    predicted = rf_pred_test
  )
)

write.csv(
  pred_df,
  file.path(save_dir, "Indoor_source_predictions.csv"),
  row.names = FALSE
)

saveRDS(
  mlr_model,
  file.path(save_dir, "MLR_model.rds")
)

saveRDS(
  rf_model,
  file.path(save_dir, "RF_model.rds")
)

save_jsd_density_plot(
  test$y,
  mlr_pred_test,
  calc_metrics(test$y, mlr_pred_test),
  "Indoor source: MLR",
  file.path(save_dir, "Indoor_MLR_test_JSD_density.pdf")
)

save_jsd_density_plot(
  test$y,
  rf_pred_test,
  calc_metrics(test$y, rf_pred_test),
  "Indoor source: RF",
  file.path(save_dir, "Indoor_RF_test_JSD_density.pdf")
)

print(all_metrics)
cat("\nIndoor RF selected mtry =", rf_tune$bestTune$mtry, "\n")
cat("Indoor RF fixed ntree   =", RF_NTREE, "\n")
cat("\n结果已保存至：", save_dir, "\n")
