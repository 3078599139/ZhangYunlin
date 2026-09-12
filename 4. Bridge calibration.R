# ==========================================
# 4. Bridge calibration
# Base models: Indoor MLR / Indoor RF
# Bridge models: Linear / Quadratic / GAM
# ==========================================

# ----------------------------
# 0. Packages
# ----------------------------
required_packages <- c(
  "readxl", "dplyr", "ggplot2", "mgcv",
  "MASS", "patchwork"
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
library(MASS)
library(patchwork)

# ----------------------------
# 1. Global settings
# ----------------------------
response_var <- "y"
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
  stop(
    paste0(
      "Field calibration set has ", nrow(field_calibration),
      " samples; expected 36."
    )
  )
}

if (nrow(field_validation) != 84) {
  stop(
    paste0(
      "Field validation set has ", nrow(field_validation),
      " samples; expected 84."
    )
  )
}

feature_vars <- setdiff(names(field_calibration), response_var)

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

calc_point_density <- function(x, y, n = 120) {
  x <- as.numeric(x)
  y <- as.numeric(y)

  keep <- is.finite(x) & is.finite(y)

  density <- rep(NA_real_, length(x))
  x_keep <- x[keep]
  y_keep <- y[keep]

  if (
    length(x_keep) < 3 ||
    diff(range(x_keep)) == 0 ||
    diff(range(y_keep)) == 0
  ) {
    density[keep] <- 1
    return(density)
  }

  xlim <- padded_range(x_keep)
  ylim <- padded_range(y_keep)

  dens <- MASS::kde2d(
    x_keep, y_keep,
    n = n,
    lims = c(xlim, ylim)
  )

  ix <- findInterval(x_keep, dens$x, all.inside = TRUE)
  iy <- findInterval(y_keep, dens$y, all.inside = TRUE)

  ix <- pmax(1, pmin(ix, length(dens$x)))
  iy <- pmax(1, pmin(iy, length(dens$y)))

  density[keep] <- as.numeric(dens$z[cbind(ix, iy)])
  density
}

save_jsd_density_plot <- function(obs, pred, metrics, title_text, file_path) {
  obs <- as.numeric(obs)
  pred <- as.numeric(pred)

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
# 4. Load indoor source models
# ----------------------------
mlr_source <- readRDS(
  file.path(model_dir, "MLR_model.rds")
)

rf_source <- readRDS(
  file.path(model_dir, "RF_model.rds")
)

# ----------------------------
# 5. Function for one source model
# ----------------------------
fit_bridge_models <- function(base_model_type) {

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

  # ---------- Linear bridge ----------
  bridge_linear <- lm(
    field_calibration$y ~ pred_cal_indoor
  )

  pred_val_linear <- as.numeric(
    predict(
      bridge_linear,
      newdata = data.frame(
        pred_cal_indoor = pred_val_indoor
      )
    )
  )

  met_linear <- cbind(
    BaseModel = base_model_type,
    BridgeModel = "Linear",
    calc_metrics(field_validation$y, pred_val_linear)
  )

  # ---------- Quadratic bridge ----------
  bridge_quad <- lm(
    field_calibration$y ~
      pred_cal_indoor +
      I(pred_cal_indoor^2)
  )

  pred_val_quad <- as.numeric(
    predict(
      bridge_quad,
      newdata = data.frame(
        pred_cal_indoor = pred_val_indoor
      )
    )
  )

  met_quad <- cbind(
    BaseModel = base_model_type,
    BridgeModel = "Quadratic",
    calc_metrics(field_validation$y, pred_val_quad)
  )

  # ---------- GAM bridge ----------
  bridge_gam <- mgcv::gam(
    field_calibration$y ~ s(pred_cal_indoor),
    method = "REML"
  )

  pred_val_gam <- as.numeric(
    predict(
      bridge_gam,
      newdata = data.frame(
        pred_cal_indoor = pred_val_indoor
      )
    )
  )

  met_gam <- cbind(
    BaseModel = base_model_type,
    BridgeModel = "GAM",
    calc_metrics(field_validation$y, pred_val_gam)
  )

  # ---------- Coefficients ----------
  linear_coef <- data.frame(
    BaseModel = base_model_type,
    BridgeModel = "Linear",
    Term = rownames(coef(summary(bridge_linear))),
    coef(summary(bridge_linear)),
    row.names = NULL
  )

  quad_coef <- data.frame(
    BaseModel = base_model_type,
    BridgeModel = "Quadratic",
    Term = rownames(coef(summary(bridge_quad))),
    coef(summary(bridge_quad)),
    row.names = NULL
  )

  colnames(linear_coef) <- c(
    "BaseModel", "BridgeModel", "Term",
    "Estimate", "Std_Error", "t_value", "P_value"
  )

  colnames(quad_coef) <- c(
    "BaseModel", "BridgeModel", "Term",
    "Estimate", "Std_Error", "t_value", "P_value"
  )

  gam_sum <- summary(bridge_gam)

  # GAM parametric part: this includes the fitted intercept alpha
  gam_parametric_df <- data.frame(
    BaseModel = base_model_type,
    BridgeModel = "GAM",
    Term = rownames(gam_sum$p.table),
    gam_sum$p.table,
    row.names = NULL
  )

  colnames(gam_parametric_df) <- c(
    "BaseModel", "BridgeModel", "Term",
    "Estimate", "Std_Error", "t_value", "P_value"
  )

  # GAM smooth-term summary
  gam_smooth_df <- data.frame(
    BaseModel = base_model_type,
    BridgeModel = "GAM",
    Term = rownames(gam_sum$s.table),
    edf = gam_sum$s.table[, "edf"],
    Ref_df = gam_sum$s.table[, "Ref.df"],
    F_value = gam_sum$s.table[, "F"],
    P_value = gam_sum$s.table[, "p-value"],
    row.names = NULL
  )

  # ---------- Fitted bridge equations ----------
  linear_b <- coef(bridge_linear)
  quad_b   <- coef(bridge_quad)
  gam_alpha <- unname(coef(bridge_gam)[1])

  equation_df <- data.frame(
    BaseModel = base_model_type,
    BridgeModel = c("Linear", "Quadratic", "GAM"),
    Equation = c(
      sprintf(
        "Y_f = %.6f %+.6f * Y_fi",
        unname(linear_b[1]),
        unname(linear_b[2])
      ),
      sprintf(
        "Y_f = %.6f %+.6f * Y_fi %+.6f * Y_fi^2",
        unname(quad_b[1]),
        unname(quad_b[2]),
        unname(quad_b[3])
      ),
      sprintf(
        "Y_f = %.6f + s(Y_fi)",
        gam_alpha
      )
    ),
    stringsAsFactors = FALSE
  )

  # ---------- Predictions ----------
  pred_df <- data.frame(
    BaseModel = base_model_type,
    observed = field_validation$y,
    indoor_pred = pred_val_indoor,
    linear_pred = pred_val_linear,
    quadratic_pred = pred_val_quad,
    gam_pred = pred_val_gam
  )

  # ---------- Plots ----------
  save_jsd_density_plot(
    field_validation$y,
    pred_val_linear,
    calc_metrics(field_validation$y, pred_val_linear),
    paste0(
      "Bridge calibration: Linear (base = ",
      base_model_type, ")"
    ),
    file.path(
      save_dir,
      paste0(
        "Bridge_", base_model_type,
        "_Linear_JSD_density.pdf"
      )
    )
  )

  save_jsd_density_plot(
    field_validation$y,
    pred_val_quad,
    calc_metrics(field_validation$y, pred_val_quad),
    paste0(
      "Bridge calibration: Quadratic (base = ",
      base_model_type, ")"
    ),
    file.path(
      save_dir,
      paste0(
        "Bridge_", base_model_type,
        "_Quadratic_JSD_density.pdf"
      )
    )
  )

  save_jsd_density_plot(
    field_validation$y,
    pred_val_gam,
    calc_metrics(field_validation$y, pred_val_gam),
    paste0(
      "Bridge calibration: GAM (base = ",
      base_model_type, ")"
    ),
    file.path(
      save_dir,
      paste0(
        "Bridge_", base_model_type,
        "_GAM_JSD_density.pdf"
      )
    )
  )

  # ---------- Save model objects ----------
  saveRDS(
    list(
      base_model_type = base_model_type,
      bridge_linear = bridge_linear,
      bridge_quad = bridge_quad,
      bridge_gam = bridge_gam
    ),
    file.path(
      save_dir,
      paste0("Bridge_models_", base_model_type, ".rds")
    )
  )

  list(
    metrics = bind_rows(
      met_linear,
      met_quad,
      met_gam
    ),
    coefficients = bind_rows(
      linear_coef,
      quad_coef
    ),
    gam_parametric = gam_parametric_df,
    gam_smooth = gam_smooth_df,
    equations = equation_df,
    predictions = pred_df
  )
}

# ----------------------------
# 6. Run MLR-based and RF-based bridges
# ----------------------------
bridge_mlr <- fit_bridge_models("MLR")
bridge_rf  <- fit_bridge_models("RF")

all_metrics <- bind_rows(
  bridge_mlr$metrics,
  bridge_rf$metrics
)

all_coefficients <- bind_rows(
  bridge_mlr$coefficients,
  bridge_rf$coefficients
)

all_gam_parametric <- bind_rows(
  bridge_mlr$gam_parametric,
  bridge_rf$gam_parametric
)

all_gam_smooth <- bind_rows(
  bridge_mlr$gam_smooth,
  bridge_rf$gam_smooth
)

all_equations <- bind_rows(
  bridge_mlr$equations,
  bridge_rf$equations
)

all_predictions <- bind_rows(
  bridge_mlr$predictions,
  bridge_rf$predictions
)

# ----------------------------
# 7. Save outputs
# ----------------------------
write.csv(
  all_metrics,
  file.path(save_dir, "Bridge_models_metrics_all.csv"),
  row.names = FALSE
)

write.csv(
  all_coefficients,
  file.path(save_dir, "Bridge_linear_quadratic_coefficients_all.csv"),
  row.names = FALSE
)

write.csv(
  all_gam_parametric,
  file.path(save_dir, "Bridge_GAM_parametric_coefficients_all.csv"),
  row.names = FALSE
)

write.csv(
  all_gam_smooth,
  file.path(save_dir, "Bridge_GAM_smooth_summary_all.csv"),
  row.names = FALSE
)

write.csv(
  all_equations,
  file.path(save_dir, "Bridge_equations_all.csv"),
  row.names = FALSE
)

write.csv(
  all_predictions,
  file.path(save_dir, "Bridge_predictions_all.csv"),
  row.names = FALSE
)

cat("\nBridge model performance:\n")
print(all_metrics)

cat("\nFitted bridge equations:\n")
print(all_equations)

