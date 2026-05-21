# ==========================================
# 1. Indoor source models with JSD density visualization
# Models: MLR / RF / RBF-SVR
# ==========================================

# ----------------------------
# 0. Packages
# ----------------------------
required_packages <- c(
  "readxl", "dplyr", "car", "caret", "randomForest",
  "e1071", "kernlab", "ggplot2", "MASS", "patchwork"
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
library(e1071)
library(kernlab)
library(ggplot2)
library(MASS)
library(patchwork)

# ----------------------------
# 1. Read data
# ----------------------------
train <- read_excel(file.choose())   # indoor_train.xlsx
test  <- read_excel(file.choose())   # indoor_test.xlsx

train <- as.data.frame(train)
test  <- as.data.frame(test)

response_var <- "y"
save_dir <- "D:/E_data/TX_image/2026.3.20/0-A/model/indoor_source_models-JSD"
dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)

# ----------------------------
# 2. Prepare data
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

# ----------------------------
# 3. Utility functions
# ----------------------------
# Jensen-Shannon divergence between observed and predicted distributions.
# The value is calculated using log2, so JSD ranges theoretically from 0 to 1.
# A smaller JSD indicates a closer match between the predicted and observed distributions.
calc_jsd <- function(obs, pred, n_bins = 30, eps = 1e-12) {
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

  jsd <- 0.5 * sum(p * log2(p / m)) + 0.5 * sum(q * log2(q / m))
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
  mre  <- mean(abs((obs - pred) / obs)) * 100
  jsd  <- calc_jsd(obs, pred)

  data.frame(R2 = r2, RMSE = rmse, MAE = mae, MRE = mre, JSD = jsd)
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

# Estimate local two-dimensional kernel density for each observed-predicted point.
# This replaces fields::interp.surface() and avoids adding an extra dependency.
calc_point_density <- function(x, y, n = 120) {
  x <- as.numeric(x)
  y <- as.numeric(y)

  xlim <- padded_range(x)
  ylim <- padded_range(y)
  dens <- MASS::kde2d(x, y, n = n, lims = c(xlim, ylim))

  ix <- findInterval(x, dens$x, all.inside = TRUE)
  iy <- findInterval(y, dens$y, all.inside = TRUE)
  ix <- pmax(1, pmin(ix, length(dens$x)))
  iy <- pmax(1, pmin(iy, length(dens$y)))

  as.numeric(dens$z[cbind(ix, iy)])
}

save_1to1_plot <- function(obs, pred, metrics, title_text, file_path) {
  txt <- paste0(
    "R² = ", round(metrics$R2, 3), "\n",
    "RMSE = ", round(metrics$RMSE, 3), "\n",
    "MAE = ", round(metrics$MAE, 3), "\n",
    "MRE = ", round(metrics$MRE, 2), "%\n",
    "JSD = ", round(metrics$JSD, 3)
  )

  df <- data.frame(observed = obs, predicted = pred)
  df$density <- calc_point_density(df$observed, df$predicted)
  axis_lim <- padded_range(c(obs, pred))

  p <- ggplot(df, aes(x = observed, y = predicted)) +
    geom_point(aes(color = density), size = 2.5, alpha = 0.9) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    scale_color_gradientn(
      colours = c("#2c7bb6", "#00a6ca", "#00ccbc", "#90eb9d", "#ffff8c", "#f9d057", "#f29e2e", "#e76818", "#d7191c"),
      name = "Point density"
    ) +
    annotate("label",
             x = axis_lim[1],
             y = axis_lim[2],
             label = txt, hjust = 0, vjust = 1, size = 4,
             label.size = 0.2, fill = "white", alpha = 0.85) +
    labs(title = title_text, x = "Observed load", y = "Predicted load") +
    coord_equal(xlim = axis_lim, ylim = axis_lim) +
    theme_bw()

  ggsave(file_path, p, width = 5.4, height = 4.7, dpi = 300)
}

# JSD density visualization:
# 1) upper panel: observed-predicted 1:1 scatter, point color = local 2D kernel density;
# 2) lower panel: observed vs predicted marginal density curves, with JSD annotation.
save_jsd_density_plot <- function(obs, pred, metrics, title_text, file_path, n_bins = 30) {
  obs <- as.numeric(obs)
  pred <- as.numeric(pred)
  keep <- is.finite(obs) & is.finite(pred)
  obs <- obs[keep]
  pred <- pred[keep]

  df <- data.frame(observed = obs, predicted = pred)
  df$density <- calc_point_density(df$observed, df$predicted)

  jsd_value <- calc_jsd(obs, pred, n_bins = n_bins)
  axis_lim <- padded_range(c(obs, pred))

  metrics_text <- paste0(
    "R² = ", sprintf("%.3f", metrics$R2), "\n",
    "RMSE = ", sprintf("%.3f", metrics$RMSE), "\n",
    "MAE = ", sprintf("%.3f", metrics$MAE), "\n",
    "MRE = ", sprintf("%.2f", metrics$MRE), "%\n",
    "JSD = ", sprintf("%.3f", jsd_value)
  )

  p_scatter <- ggplot(df, aes(x = observed, y = predicted)) +
    geom_point(aes(color = density), size = 2.4, alpha = 0.85) +
    geom_smooth(method = "lm", se = FALSE, linewidth = 0.7, color = "#D55E00") +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", linewidth = 0.7) +
    scale_color_gradientn(
      colours = c("#2c7bb6", "#00a6ca", "#00ccbc", "#90eb9d", "#ffff8c", "#f9d057", "#f29e2e", "#e76818", "#d7191c"),
      name = "2D KDE density"
    ) +
    annotate("label",
             x = axis_lim[1], y = axis_lim[2],
             label = metrics_text,
             hjust = 0, vjust = 1,
             size = 3.6,
             label.size = 0.2,
             fill = "white",
             alpha = 0.85) +
    labs(title = title_text,
         subtitle = "Point color represents local two-dimensional kernel density",
         x = "Observed load", y = "Predicted load") +
    coord_equal(xlim = axis_lim, ylim = axis_lim) +
    theme_bw(base_size = 12, base_family = "serif") +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      plot.subtitle = element_text(hjust = 0.5),
      legend.position = "right"
    )

  dist_df <- rbind(
    data.frame(value = obs, Distribution = "Observed"),
    data.frame(value = pred, Distribution = "Predicted")
  )

  p_density <- ggplot(dist_df, aes(x = value, color = Distribution, fill = Distribution)) +
    geom_density(alpha = 0.22, linewidth = 0.8, adjust = 1) +
    annotate("text",
             x = axis_lim[1], y = Inf,
             label = paste0("JSD = ", sprintf("%.3f", jsd_value), "  (0 = identical distributions)"),
             hjust = 0, vjust = 1.3,
             size = 3.8,
             family = "serif") +
    labs(x = "Load", y = "Density") +
    scale_x_continuous(limits = axis_lim) +
    theme_bw(base_size = 12, base_family = "serif") +
    theme(
      legend.position = "top",
      legend.title = element_blank()
    )

  p_combined <- p_scatter / p_density +
    patchwork::plot_layout(heights = c(3.1, 1.15))

  ggsave(file_path, p_combined, width = 6.2, height = 7.2, dpi = 300)
}

# ----------------------------
# 4. MLR
# ----------------------------
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
    vif_vals <- vif(vif_model)
    if (max(vif_vals) < 10) break
    feature_vars <- setdiff(feature_vars, names(which.max(vif_vals)))
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
    scope = list(lower = lower_formula, upper = formula(upper_model)),
    direction = "both",
    trace = 0
  )
}

mlr_model <- fit_mlr(train, response_var)
mlr_pred_train <- predict(mlr_model, newdata = train)
mlr_pred_test  <- predict(mlr_model, newdata = test)

mlr_metrics <- rbind(
  cbind(Model = "MLR", Dataset = "Train", calc_metrics(train$y, mlr_pred_train)),
  cbind(Model = "MLR", Dataset = "Test",  calc_metrics(test$y, mlr_pred_test))
)

mlr_coef_df <- data.frame(
  Feature = rownames(summary(mlr_model)$coefficients),
  summary(mlr_model)$coefficients,
  row.names = NULL
)
colnames(mlr_coef_df) <- c("Feature", "Estimate", "Std_Error", "t_value", "P_value")

# ----------------------------
# 5. RF
# ----------------------------
set.seed(42)
rf_tune <- train(
  x = train[, setdiff(names(train), response_var), drop = FALSE],
  y = train[[response_var]],
  method = "rf",
  tuneLength = 5,
  trControl = trainControl(method = "cv", number = 5)
)

rf_model <- rf_tune
rf_pred_train <- predict(rf_model, newdata = train[, setdiff(names(train), response_var), drop = FALSE])
rf_pred_test  <- predict(rf_model, newdata = test[, setdiff(names(test), response_var), drop = FALSE])

rf_metrics <- rbind(
  cbind(Model = "RF", Dataset = "Train", calc_metrics(train$y, rf_pred_train)),
  cbind(Model = "RF", Dataset = "Test",  calc_metrics(test$y, rf_pred_test))
)

rf_param_df <- rf_tune$bestTune

# ----------------------------
# 6. RBF-SVR
# ----------------------------
x_train <- train[, setdiff(names(train), response_var), drop = FALSE]
x_test  <- test[, setdiff(names(test), response_var), drop = FALSE]

preproc <- preProcess(x_train, method = c("center", "scale"))
x_train_scaled <- predict(preproc, x_train)
x_test_scaled  <- predict(preproc, x_test)

set.seed(42)
svr_tune <- train(
  x = x_train_scaled,
  y = train[[response_var]],
  method = "svmRadial",
  tuneLength = 8,
  trControl = trainControl(method = "cv", number = 5)
)

svr_model <- svr_tune
svr_pred_train <- predict(svr_model, newdata = x_train_scaled)
svr_pred_test  <- predict(svr_model, newdata = x_test_scaled)

svr_metrics <- rbind(
  cbind(Model = "RBF_SVR", Dataset = "Train", calc_metrics(train$y, svr_pred_train)),
  cbind(Model = "RBF_SVR", Dataset = "Test",  calc_metrics(test$y, svr_pred_test))
)

svr_param_df <- svr_tune$bestTune

# ----------------------------
# 7. Save outputs
# ----------------------------
all_metrics <- bind_rows(mlr_metrics, rf_metrics, svr_metrics)
write.csv(all_metrics, file.path(save_dir, "Indoor_source_metrics.csv"), row.names = FALSE)

write.csv(mlr_coef_df, file.path(save_dir, "Indoor_MLR_coefficients.csv"), row.names = FALSE)
write.csv(rf_param_df, file.path(save_dir, "Indoor_RF_best_params.csv"), row.names = FALSE)
write.csv(svr_param_df, file.path(save_dir, "Indoor_RBF_SVR_best_params.csv"), row.names = FALSE)

pred_df <- bind_rows(
  data.frame(Model = "MLR", Dataset = "Train", observed = train$y, predicted = mlr_pred_train),
  data.frame(Model = "MLR", Dataset = "Test",  observed = test$y,  predicted = mlr_pred_test),
  data.frame(Model = "RF", Dataset = "Train", observed = train$y, predicted = rf_pred_train),
  data.frame(Model = "RF", Dataset = "Test",  observed = test$y,  predicted = rf_pred_test),
  data.frame(Model = "RBF_SVR", Dataset = "Train", observed = train$y, predicted = svr_pred_train),
  data.frame(Model = "RBF_SVR", Dataset = "Test",  observed = test$y,  predicted = svr_pred_test)
)
write.csv(pred_df, file.path(save_dir, "Indoor_source_predictions.csv"), row.names = FALSE)

saveRDS(mlr_model, file.path(save_dir, "MLR_model.rds"))
saveRDS(rf_model, file.path(save_dir, "RF_model.rds"))
saveRDS(list(model = svr_model, preproc = preproc), file.path(save_dir, "RBF_SVR_model.rds"))

# Original 1:1 plots, now also annotated with JSD.
save_1to1_plot(test$y, mlr_pred_test, calc_metrics(test$y, mlr_pred_test),
               "Indoor source: MLR", file.path(save_dir, "Indoor_MLR_test_1to1.pdf"))
save_1to1_plot(test$y, rf_pred_test, calc_metrics(test$y, rf_pred_test),
               "Indoor source: RF", file.path(save_dir, "Indoor_RF_test_1to1.pdf"))
save_1to1_plot(test$y, svr_pred_test, calc_metrics(test$y, svr_pred_test),
               "Indoor source: RBF-SVR", file.path(save_dir, "Indoor_RBF_SVR_test_1to1.pdf"))

# New JSD density plots for the test set.
save_jsd_density_plot(test$y, mlr_pred_test, calc_metrics(test$y, mlr_pred_test),
                      "Indoor source: MLR", file.path(save_dir, "Indoor_MLR_test_JSD_density.pdf"))
save_jsd_density_plot(test$y, rf_pred_test, calc_metrics(test$y, rf_pred_test),
                      "Indoor source: RF", file.path(save_dir, "Indoor_RF_test_JSD_density.pdf"))
save_jsd_density_plot(test$y, svr_pred_test, calc_metrics(test$y, svr_pred_test),
                      "Indoor source: RBF-SVR", file.path(save_dir, "Indoor_RBF_SVR_test_JSD_density.pdf"))

# Optional: save train-set JSD density plots as supplementary diagnostics.
save_jsd_density_plot(train$y, mlr_pred_train, calc_metrics(train$y, mlr_pred_train),
                      "Indoor source: MLR train", file.path(save_dir, "Indoor_MLR_train_JSD_density.pdf"))
save_jsd_density_plot(train$y, rf_pred_train, calc_metrics(train$y, rf_pred_train),
                      "Indoor source: RF train", file.path(save_dir, "Indoor_RF_train_JSD_density.pdf"))
save_jsd_density_plot(train$y, svr_pred_train, calc_metrics(train$y, svr_pred_train),
                      "Indoor source: RBF-SVR train", file.path(save_dir, "Indoor_RBF_SVR_train_JSD_density.pdf"))

print(all_metrics)
cat("\n结果已保存至：", save_dir, "\n")
