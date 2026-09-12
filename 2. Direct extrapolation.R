# ==========================================
# 2. Indoor model direct extrapolation
# Models: MLR / RF
# IMPORTANT:
# Use the FINAL common field validation set (n = 84).

# ==========================================

# ----------------------------
# 0. Packages
# ----------------------------
required_packages <- c(
  "readxl", "dplyr", "ggplot2", "MASS", "patchwork"
)

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

library(readxl)
library(dplyr)
library(ggplot2)
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
# 2. Read final field validation data
# ----------------------------
# Select:
# final field_validation.xlsx
# Expected n = 84.
field_validation <- read_excel(file.choose())
field_validation <- as.data.frame(field_validation)

field_validation[] <- lapply(
  field_validation,
  function(x) as.numeric(as.character(x))
)
field_validation <- na.omit(field_validation)

if (nrow(field_validation) != 84) {
  stop(
    paste0(
      "Field validation set has ", nrow(field_validation),
      " samples; expected 84. Please select the FINAL common validation set."
    )
  )
}

feature_vars <- setdiff(names(field_validation), response_var)

cat("Field validation samples:", nrow(field_validation), "\n")
cat("Number of predictors    :", length(feature_vars), "\n")

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
mlr_model <- readRDS(
  file.path(model_dir, "MLR_model.rds")
)

rf_model <- readRDS(
  file.path(model_dir, "RF_model.rds")
)

# ----------------------------
# 5. Direct extrapolation
# ----------------------------
pred_mlr <- as.numeric(
  predict(mlr_model, newdata = field_validation)
)

pred_rf <- as.numeric(
  predict(
    rf_model,
    newdata = field_validation[, feature_vars, drop = FALSE]
  )
)

met_mlr <- cbind(
  Model = "MLR",
  Dataset = "Field validation",
  calc_metrics(field_validation$y, pred_mlr)
)

met_rf <- cbind(
  Model = "RF",
  Dataset = "Field validation",
  calc_metrics(field_validation$y, pred_rf)
)

all_metrics <- bind_rows(
  met_mlr,
  met_rf
)

# ----------------------------
# 6. Save outputs
# ----------------------------
write.csv(
  all_metrics,
  file.path(save_dir, "Direct_extrapolation_metrics.csv"),
  row.names = FALSE
)

pred_df <- bind_rows(
  data.frame(
    Model = "MLR",
    observed = field_validation$y,
    predicted = pred_mlr,
    absolute_error = abs(pred_mlr - field_validation$y)
  ),
  data.frame(
    Model = "RF",
    observed = field_validation$y,
    predicted = pred_rf,
    absolute_error = abs(pred_rf - field_validation$y)
  )
)

write.csv(
  pred_df,
  file.path(save_dir, "Direct_extrapolation_predictions.csv"),
  row.names = FALSE
)

save_jsd_density_plot(
  field_validation$y,
  pred_mlr,
  calc_metrics(field_validation$y, pred_mlr),
  "Direct extrapolation: MLR",
  file.path(save_dir, "Direct_MLR_JSD_density.pdf")
)

save_jsd_density_plot(
  field_validation$y,
  pred_rf,
  calc_metrics(field_validation$y, pred_rf),
  "Direct extrapolation: RF",
  file.path(save_dir, "Direct_RF_JSD_density.pdf")
)

print(all_metrics)
cat("\n结果已保存至：", save_dir, "\n")
