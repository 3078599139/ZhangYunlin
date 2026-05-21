# ==========================================
# 3. Field-local models
# Models: MLR / RF / RBF-SVR
# ==========================================

library(readxl)
library(dplyr)
library(car)
library(caret)
library(randomForest)
library(e1071)
library(kernlab)
library(ggplot2)

# ----------------------------
# 1. Read data
# ----------------------------
train <- read_excel(file.choose())   # field_train.xlsx
test  <- read_excel(file.choose())   # field_test.xlsx

train <- as.data.frame(train)
test  <- as.data.frame(test)

response_var <- "y"
save_dir <- "D:/E_data/TX_image/2026.3.20/0-A/model/field_local_models"
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
calc_metrics <- function(obs, pred) {
  obs <- as.numeric(obs)
  pred <- as.numeric(pred)
  r2   <- 1 - sum((obs - pred)^2) / sum((obs - mean(obs))^2)
  rmse <- sqrt(mean((obs - pred)^2))
  mae  <- mean(abs(obs - pred))
  mre  <- mean(abs((obs - pred) / obs)) * 100
  data.frame(R2 = r2, RMSE = rmse, MAE = mae, MRE = mre)
}

save_1to1_plot <- function(obs, pred, metrics, title_text, file_path) {
  txt <- paste0(
    "R² = ", round(metrics$R2, 3), "\n",
    "RMSE = ", round(metrics$RMSE, 3), "\n",
    "MAE = ", round(metrics$MAE, 3), "\n",
    "MRE = ", round(metrics$MRE, 2), "%"
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
  
  ggsave(file_path, p, width = 5, height = 4, dpi = 300)
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

# ----------------------------
# 5. MLR
# ----------------------------
set.seed(42)
rf_tune <- train(
  x = train[, setdiff(names(train), response_var), drop = FALSE],
  y = train[[response_var]],
  method = "rf",
  tuneLength = 5,
  trControl = trainControl(method = "cv", number = 5)
)
rf_pred_train <- predict(rf_tune, newdata = train[, setdiff(names(train), response_var), drop = FALSE])
rf_pred_test  <- predict(rf_tune, newdata = test[, setdiff(names(test), response_var), drop = FALSE])
rf_metrics <- rbind(
  cbind(Model = "RF", Dataset = "Train", calc_metrics(train$y, rf_pred_train)),
  cbind(Model = "RF", Dataset = "Test",  calc_metrics(test$y, rf_pred_test))
)


# ----------------------------
# 6. SVR
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
svr_pred_train <- predict(svr_tune, newdata = x_train_scaled)
svr_pred_test  <- predict(svr_tune, newdata = x_test_scaled)
svr_metrics <- rbind(
  cbind(Model = "RBF_SVR", Dataset = "Train", calc_metrics(train$y, svr_pred_train)),
  cbind(Model = "RBF_SVR", Dataset = "Test",  calc_metrics(test$y, svr_pred_test))
)


all_metrics <- bind_rows(mlr_metrics, rf_metrics, svr_metrics)
write.csv(all_metrics, file.path(save_dir, "Field_local_metrics.csv"), row.names = FALSE)


write.csv(
  data.frame(Feature = rownames(summary(mlr_model)$coefficients),
             summary(mlr_model)$coefficients,
             row.names = NULL),
  file.path(save_dir, "Field_local_MLR_coefficients.csv"),
  row.names = FALSE
)
write.csv(rf_tune$bestTune, file.path(save_dir, "Field_local_RF_best_params.csv"), row.names = FALSE)
write.csv(svr_tune$bestTune, file.path(save_dir, "Field_local_RBF_SVR_best_params.csv"), row.names = FALSE)


save_1to1_plot(test$y, mlr_pred_test, calc_metrics(test$y, mlr_pred_test),
               "Field-local: MLR", file.path(save_dir, "FieldLocal_MLR_test_1to1.pdf"))
save_1to1_plot(test$y, rf_pred_test, calc_metrics(test$y, rf_pred_test),
               "Field-local: RF", file.path(save_dir, "FieldLocal_RF_test_1to1.pdf"))
save_1to1_plot(test$y, svr_pred_test, calc_metrics(test$y, svr_pred_test),
               "Field-local: RBF-SVR", file.path(save_dir, "FieldLocal_RBF_SVR_test_1to1.pdf"))

saveRDS(mlr_model, file.path(save_dir, "FieldLocal_MLR_model.rds"))
saveRDS(rf_tune, file.path(save_dir, "FieldLocal_RF_model.rds"))
saveRDS(list(model = svr_tune, preproc = preproc), file.path(save_dir, "FieldLocal_RBF_SVR_model.rds"))

print(all_metrics)
cat("\n结果已保存至：", save_dir, "\n")
