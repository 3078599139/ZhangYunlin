# ==========================================
# 2. Indoor model direct extrapolation
# Models: MLR / RF / RBF-SVR
# Added JSD calculation
# ==========================================

library(readxl)
library(ggplot2)
library(kernlab)
library(dplyr)

field_validation <- read_excel(file.choose())   # field_validation.xlsx
field_validation <- as.data.frame(field_validation)

response_var <- "y"
model_dir <- "D:/E_data/TX_image/2026.3.20/0-A/model/indoor_source_models"
save_dir  <- "D:/E_data/TX_image/2026.3.20/0-A/model/direct_extrapolation-JSD"
dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)

field_validation[] <- lapply(field_validation, function(x) as.numeric(as.character(x)))
field_validation <- na.omit(field_validation)

# ---------- Jensen-Shannon divergence (copied from bridge calibration) ----------
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

save_1to1_plot <- function(obs, pred, metrics, title_text, file_path) {
  txt <- paste0(
    "R² = ", round(metrics$R2, 3), "\n",
    "RMSE = ", round(metrics$RMSE, 3), "\n",
    "MAE = ", round(metrics$MAE, 3), "\n",
    "MRE = ", round(metrics$MRE, 2), "%\n",
    "JSD = ", round(metrics$JSD, 3)
  )
  
  p <- ggplot(data.frame(observed = obs, predicted = pred),
              aes(x = observed, y = predicted)) +
    geom_point(size = 2.5, alpha = 0.8) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    annotate("text",
             x = min(obs, na.rm = TRUE),
             y = max(pred, na.rm = TRUE),
             label = txt, hjust = 0, vjust = 1, size = 4) +
    labs(title = title_text, x = "Observed load", y = "Predicted load") +
    theme_bw()
  
  ggsave(file_path, p, width = 5.2, height = 4.5, dpi = 300)
}

# load models
mlr_model <- readRDS(file.path(model_dir, "MLR_model.rds"))
rf_model  <- readRDS(file.path(model_dir, "RF_model.rds"))
svr_obj   <- readRDS(file.path(model_dir, "RBF_SVR_model.rds"))

x_val <- field_validation[, setdiff(names(field_validation), response_var), drop = FALSE]
x_val_scaled <- predict(svr_obj$preproc, x_val)

pred_mlr <- predict(mlr_model, newdata = field_validation)
pred_rf  <- predict(rf_model, newdata = x_val)
pred_svr <- predict(svr_obj$model, newdata = x_val_scaled)

met_mlr <- cbind(Model = "MLR", Dataset = "Field validation", calc_metrics(field_validation$y, pred_mlr))
met_rf  <- cbind(Model = "RF", Dataset = "Field validation", calc_metrics(field_validation$y, pred_rf))
met_svr <- cbind(Model = "RBF_SVR", Dataset = "Field validation", calc_metrics(field_validation$y, pred_svr))

all_metrics <- bind_rows(met_mlr, met_rf, met_svr)
write.csv(all_metrics, file.path(save_dir, "Direct_extrapolation_metrics.csv"), row.names = FALSE)

pred_df <- bind_rows(
  data.frame(Model = "MLR", observed = field_validation$y, predicted = pred_mlr),
  data.frame(Model = "RF", observed = field_validation$y, predicted = pred_rf),
  data.frame(Model = "RBF_SVR", observed = field_validation$y, predicted = pred_svr)
)
write.csv(pred_df, file.path(save_dir, "Direct_extrapolation_predictions.csv"), row.names = FALSE)

save_1to1_plot(field_validation$y, pred_mlr, calc_metrics(field_validation$y, pred_mlr),
               "Direct extrapolation: MLR", file.path(save_dir, "Direct_MLR_1to1.pdf"))
save_1to1_plot(field_validation$y, pred_rf, calc_metrics(field_validation$y, pred_rf),
               "Direct extrapolation: RF", file.path(save_dir, "Direct_RF_1to1.pdf"))
save_1to1_plot(field_validation$y, pred_svr, calc_metrics(field_validation$y, pred_svr),
               "Direct extrapolation: RBF-SVR", file.path(save_dir, "Direct_RBF_SVR_1to1.pdf"))

print(all_metrics)
cat("\n结果已保存至：", save_dir, "\n")
