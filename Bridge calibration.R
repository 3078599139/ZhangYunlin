# ==========================================
# 5. Bridge calibration
# Bridge models: Linear / Quadratic / GAM
# Base model chosen from direct extrapolation
# ==========================================

library(readxl)
library(ggplot2)
library(mgcv)
library(kernlab)
library(dplyr)
library(MASS)

# 依次选择：
# 1) field_calibration
# 2) field_validation
field_calibration <- read_excel(file.choose())
field_validation  <- read_excel(file.choose())

field_calibration <- as.data.frame(field_calibration)
field_validation  <- as.data.frame(field_validation)

response_var <- "y"
model_dir <- "D:/E_data/TX_image/2026.3.20/0-A/model/indoor_source_models"
save_dir  <- "D:/E_data/TX_image/2026.3.20/0-A/model/bridge_calibration-JSD"
dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)

field_calibration[] <- lapply(field_calibration, function(x) as.numeric(as.character(x)))
field_validation[]  <- lapply(field_validation, function(x) as.numeric(as.character(x)))
field_calibration <- na.omit(field_calibration)
field_validation  <- na.omit(field_validation)

# ---------- utility functions ----------
# Jensen-Shannon divergence between observed and predicted distributions.
# log2 is used here, so JSD theoretically ranges from 0 to 1;
# a smaller value indicates that the predicted distribution is closer to the observed distribution.
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

  nonzero <- obs != 0
  mre <- if (any(nonzero)) {
    mean(abs((obs[nonzero] - pred[nonzero]) / obs[nonzero])) * 100
  } else {
    NA_real_
  }

  jsd <- calc_jsd(obs, pred)

  data.frame(R2 = r2, RMSE = rmse, MAE = mae, MRE = mre, JSD = jsd)
}

padded_range <- function(x, pad_ratio = 0.06) {
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
# This is used only for gradient coloring of the 1:1 scatter points.
calc_point_density <- function(x, y, n = 120) {
  x <- as.numeric(x)
  y <- as.numeric(y)
  keep <- is.finite(x) & is.finite(y)

  density <- rep(NA_real_, length(x))
  x_keep <- x[keep]
  y_keep <- y[keep]

  if (length(x_keep) < 3 || diff(range(x_keep)) == 0 || diff(range(y_keep)) == 0) {
    density[keep] <- 1
    return(density)
  }

  xlim <- padded_range(x_keep)
  ylim <- padded_range(y_keep)
  dens <- MASS::kde2d(x_keep, y_keep, n = n, lims = c(xlim, ylim))

  ix <- findInterval(x_keep, dens$x, all.inside = TRUE)
  iy <- findInterval(y_keep, dens$y, all.inside = TRUE)
  ix <- pmax(1, pmin(ix, length(dens$x)))
  iy <- pmax(1, pmin(iy, length(dens$y)))

  density[keep] <- as.numeric(dens$z[cbind(ix, iy)])
  density
}

save_bridge_1to1_plot <- function(obs, pred, metrics, title_text, file_path) {
  obs <- as.numeric(obs)
  pred <- as.numeric(pred)
  keep <- is.finite(obs) & is.finite(pred)
  obs <- obs[keep]
  pred <- pred[keep]

  df <- data.frame(observed = obs, predicted = pred)
  df$density <- calc_point_density(df$observed, df$predicted)
  df <- df[order(df$density), ]

  axis_lim <- padded_range(c(obs, pred))

  metrics_text <- paste0(
    "R² = ", sprintf("%.3f", metrics$R2), "\n",
    "RMSE = ", sprintf("%.3f", metrics$RMSE), "\n",
    "MAE = ", sprintf("%.3f", metrics$MAE), "\n",
    "MRE = ", sprintf("%.2f", metrics$MRE), "%\n",
    "JSD = ", sprintf("%.3f", metrics$JSD)
  )

  p <- ggplot(df, aes(x = observed, y = predicted)) +
    geom_point(aes(color = density), size = 2.6, alpha = 0.9) +
    geom_smooth(method = "lm", se = FALSE, linewidth = 0.7, color = "#D55E00") +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", linewidth = 0.7) +
    scale_color_gradientn(
      colours = c("#2c7bb6", "#00a6ca", "#00ccbc", "#90eb9d", "#ffff8c", "#f9d057", "#f29e2e", "#e76818", "#d7191c"),
      name = "Point density"
    ) +
    annotate("label",
             x = axis_lim[1], y = axis_lim[2],
             label = metrics_text,
             hjust = 0, vjust = 1,
             size = 3.8,
             label.size = 0.2,
             fill = "white",
             alpha = 0.85) +
    labs(
      title = title_text,
      x = "Observed load",
      y = "Predicted load"
    ) +
    coord_equal(xlim = axis_lim, ylim = axis_lim) +
    theme_bw(base_size = 12, base_family = "serif") +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      legend.position = "right"
    )

  ggsave(file_path, p, width = 5.6, height = 4.8, dpi = 300)
}

# choose best base model manually according to direct extrapolation result
# 你根据 direct extrapolation 结果填写： "MLR" / "RF" / "RBF_SVR"
base_model_type <- "RF"

# ---------- load base model ----------
x_cal <- field_calibration[, setdiff(names(field_calibration), response_var), drop = FALSE]
x_val <- field_validation[, setdiff(names(field_validation), response_var), drop = FALSE]

if (base_model_type == "MLR") {
  base_model <- readRDS(file.path(model_dir, "MLR_model.rds"))
  pred_cal_indoor <- as.numeric(predict(base_model, newdata = field_calibration))
  pred_val_indoor <- as.numeric(predict(base_model, newdata = field_validation))
}

if (base_model_type == "RF") {
  base_model <- readRDS(file.path(model_dir, "RF_model.rds"))
  pred_cal_indoor <- as.numeric(predict(base_model, newdata = x_cal))
  pred_val_indoor <- as.numeric(predict(base_model, newdata = x_val))
}

if (base_model_type == "RBF_SVR") {
  base_obj <- readRDS(file.path(model_dir, "RBF_SVR_model.rds"))
  x_cal_scaled <- predict(base_obj$preproc, x_cal)
  x_val_scaled <- predict(base_obj$preproc, x_val)
  pred_cal_indoor <- as.numeric(predict(base_obj$model, newdata = x_cal_scaled))
  pred_val_indoor <- as.numeric(predict(base_obj$model, newdata = x_val_scaled))
}

# ---------- Linear bridge ----------
bridge_linear <- lm(field_calibration$y ~ pred_cal_indoor)
pred_val_linear <- predict(bridge_linear, newdata = data.frame(pred_cal_indoor = pred_val_indoor))
met_linear <- cbind(BridgeModel = "Linear", BaseModel = base_model_type,
                    calc_metrics(field_validation$y, pred_val_linear))

alpha_linear <- coef(bridge_linear)[1]
beta_linear  <- coef(bridge_linear)[2]

# ---------- Quadratic bridge ----------
bridge_quad <- lm(field_calibration$y ~ pred_cal_indoor + I(pred_cal_indoor^2))
pred_val_quad <- predict(bridge_quad, newdata = data.frame(pred_cal_indoor = pred_val_indoor))
met_quad <- cbind(BridgeModel = "Quadratic", BaseModel = base_model_type,
                  calc_metrics(field_validation$y, pred_val_quad))

alpha_quad <- coef(bridge_quad)[1]
beta1_quad <- coef(bridge_quad)[2]
beta2_quad <- coef(bridge_quad)[3]

# ---------- GAM bridge ----------
bridge_gam <- gam(field_calibration$y ~ s(pred_cal_indoor), method = "REML")
pred_val_gam <- predict(bridge_gam, newdata = data.frame(pred_cal_indoor = pred_val_indoor))
met_gam <- cbind(BridgeModel = "GAM", BaseModel = base_model_type,
                 calc_metrics(field_validation$y, pred_val_gam))

gam_intercept <- coef(bridge_gam)[1]
gam_sum <- summary(bridge_gam)

# ---------- save metrics ----------
all_metrics <- bind_rows(met_linear, met_quad, met_gam)
write.csv(all_metrics, file.path(save_dir, "Bridge_models_metrics.csv"), row.names = FALSE)

# ---------- save equation parameters ----------
equation_params <- bind_rows(
  data.frame(Model = "Linear", Parameter = c("alpha", "beta"),
             Value = c(alpha_linear, beta_linear)),
  data.frame(Model = "Quadratic", Parameter = c("alpha", "beta1", "beta2"),
             Value = c(alpha_quad, beta1_quad, beta2_quad)),
  data.frame(Model = "GAM", Parameter = "Intercept",
             Value = gam_intercept)
)
write.csv(equation_params, file.path(save_dir, "Bridge_equation_parameters.csv"), row.names = FALSE)

# ---------- save coefficient summaries ----------
linear_coef <- data.frame(
  Model = "Linear",
  Term = rownames(coef(summary(bridge_linear))),
  coef(summary(bridge_linear)),
  row.names = NULL
)
quad_coef <- data.frame(
  Model = "Quadratic",
  Term = rownames(coef(summary(bridge_quad))),
  coef(summary(bridge_quad)),
  row.names = NULL
)
colnames(linear_coef) <- c("Model", "Term", "Estimate", "Std_Error", "t_value", "P_value")
colnames(quad_coef)   <- c("Model", "Term", "Estimate", "Std_Error", "t_value", "P_value")

write.csv(bind_rows(linear_coef, quad_coef),
          file.path(save_dir, "Bridge_linear_quadratic_coefficients.csv"),
          row.names = FALSE)

gam_smooth_df <- data.frame(
  Term = rownames(gam_sum$s.table),
  edf = gam_sum$s.table[, "edf"],
  Ref_df = gam_sum$s.table[, "Ref.df"],
  F_value = gam_sum$s.table[, "F"],
  P_value = gam_sum$s.table[, "p-value"],
  row.names = NULL
)
write.csv(gam_smooth_df, file.path(save_dir, "Bridge_GAM_smooth_summary.csv"), row.names = FALSE)

# ---------- save predictions ----------
pred_df <- data.frame(
  observed = field_validation$y,
  indoor_pred = pred_val_indoor,
  linear_pred = pred_val_linear,
  quadratic_pred = pred_val_quad,
  gam_pred = pred_val_gam
)
pred_df$linear_abs_error    <- abs(pred_df$linear_pred - pred_df$observed)
pred_df$quadratic_abs_error <- abs(pred_df$quadratic_pred - pred_df$observed)
pred_df$gam_abs_error       <- abs(pred_df$gam_pred - pred_df$observed)

write.csv(pred_df, file.path(save_dir, "Bridge_predictions.csv"), row.names = FALSE)

# ---------- 1:1 scatter plots with JSD ----------
save_bridge_1to1_plot(
  field_validation$y,
  pred_val_linear,
  met_linear,
  paste0("Bridge calibration: Linear (base = ", base_model_type, ")"),
  file.path(save_dir, "Bridge_Linear_1to1_JSD.pdf")
)

save_bridge_1to1_plot(
  field_validation$y,
  pred_val_quad,
  met_quad,
  paste0("Bridge calibration: Quadratic (base = ", base_model_type, ")"),
  file.path(save_dir, "Bridge_Quadratic_1to1_JSD.pdf")
)

save_bridge_1to1_plot(
  field_validation$y,
  pred_val_gam,
  met_gam,
  paste0("Bridge calibration: GAM (base = ", base_model_type, ")"),
  file.path(save_dir, "Bridge_GAM_1to1_JSD.pdf")
)

# ---------- absolute error boxplot ----------
error_long <- data.frame(
  BridgeModel = rep(c("Linear", "Quadratic", "GAM"), each = nrow(pred_df)),
  AbsoluteError = c(pred_df$linear_abs_error,
                    pred_df$quadratic_abs_error,
                    pred_df$gam_abs_error)
)
write.csv(error_long, file.path(save_dir, "Bridge_absolute_error_long.csv"), row.names = FALSE)

p_box <- ggplot(error_long, aes(x = BridgeModel, y = AbsoluteError)) +
  geom_boxplot(width = 0.6, outlier.shape = 16, outlier.size = 1.8) +
  labs(
    title = paste0("Absolute error comparison of bridge models (based on ", base_model_type, ")"),
    x = "Bridge model",
    y = "Absolute error"
  ) +
  theme_bw()

ggsave(file.path(save_dir, "Bridge_absolute_error_boxplot.pdf"),
       p_box, width = 5.5, height = 4.5, dpi = 300)

# ---------- save model objects ----------
saveRDS(
  list(
    base_model_type = base_model_type,
    bridge_linear = bridge_linear,
    bridge_quad = bridge_quad,
    bridge_gam = bridge_gam
  ),
  file.path(save_dir, "Bridge_models.rds")
)

cat("\n线性桥接方程：\n")
cat("y_field = ", round(alpha_linear, 6), " + ", round(beta_linear, 6), " * yhat_field_indoor\n", sep = "")

cat("\n二次桥接方程：\n")
cat("y_field = ", round(alpha_quad, 6), " + ", round(beta1_quad, 6),
    " * yhat_field_indoor + ", round(beta2_quad, 6), " * yhat_field_indoor^2\n", sep = "")

cat("\nGAM桥接模型形式：\n")
cat("y_field = ", round(gam_intercept, 6), " + s(yhat_field_indoor)\n", sep = "")

print(all_metrics)
cat("\n结果已保存至：", save_dir, "\n")
