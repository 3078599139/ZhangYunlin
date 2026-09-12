# ==========================================
# 5. Feature shift analysis
# ==========================================

# ----------------------------
# 0. Packages
# ----------------------------
required_packages <- c(
  "readxl", "dplyr", "ggplot2", "tidyr"
)

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

library(readxl)
library(dplyr)
library(ggplot2)
library(tidyr)

# ----------------------------
# 1. Global settings
# ----------------------------
response_var <- "y"

save_dir <- "XXX"
dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)

# ----------------------------
# 2. Read all indoor and field data
# ----------------------------
# Select in order:
# 1) indoor_train.xlsx
# 2) indoor_test.xlsx
# 3) FINAL field_calibration.xlsx
# 4) FINAL field_validation.xlsx
#
# The two indoor files are combined to n = 120.
# The two final field files are combined to n = 120.
indoor_train <- read_excel(file.choose())
indoor_test  <- read_excel(file.choose())
field_calibration <- read_excel(file.choose())
field_validation  <- read_excel(file.choose())

indoor_train <- as.data.frame(indoor_train)
indoor_test  <- as.data.frame(indoor_test)
field_calibration <- as.data.frame(field_calibration)
field_validation  <- as.data.frame(field_validation)

prepare_data <- function(df) {
  df[] <- lapply(df, function(x) as.numeric(as.character(x)))
  na.omit(df)
}

indoor_train <- prepare_data(indoor_train)
indoor_test  <- prepare_data(indoor_test)
field_calibration <- prepare_data(field_calibration)
field_validation  <- prepare_data(field_validation)

indoor <- bind_rows(
  indoor_train,
  indoor_test
)

field <- bind_rows(
  field_calibration,
  field_validation
)

if (nrow(indoor) != 120) {
  warning(
    paste0(
      "Combined indoor data contain ", nrow(indoor),
      " samples; expected 120."
    )
  )
}

if (nrow(field) != 120) {
  warning(
    paste0(
      "Combined field data contain ", nrow(field),
      " samples; expected 120."
    )
  )
}

feature_vars <- intersect(
  setdiff(names(indoor), response_var),
  setdiff(names(field), response_var)
)

cat("Indoor samples :", nrow(indoor), "\n")
cat("Field samples  :", nrow(field), "\n")
cat("Features       :", length(feature_vars), "\n")

# ----------------------------
# 3. Utility functions
# ----------------------------
# Std. Shift definition used in the manuscript:
#
# Std_Shift = (Mean_field - Mean_indoor) / SD_indoor
#
# Positive values indicate a higher field mean.
# Negative values indicate a lower field mean.
# The denominator is the source-domain (indoor) sample SD.
calc_std_shift <- function(x_indoor, x_field) {
  sd_indoor <- sd(x_indoor, na.rm = TRUE)

  if (!is.finite(sd_indoor) || sd_indoor == 0) {
    return(NA_real_)
  }

  (mean(x_field, na.rm = TRUE) -
     mean(x_indoor, na.rm = TRUE)) / sd_indoor
}

# Cliff's delta:
# positive values indicate generally larger field values,
# negative values indicate generally smaller field values.
calc_cliffs_delta <- function(x_indoor, x_field) {
  x_indoor <- x_indoor[is.finite(x_indoor)]
  x_field  <- x_field[is.finite(x_field)]

  if (length(x_indoor) == 0 || length(x_field) == 0) {
    return(NA_real_)
  }

  diff_mat <- outer(
    x_field,
    x_indoor,
    "-"
  )

  (
    sum(diff_mat > 0) -
    sum(diff_mat < 0)
  ) / (
    length(x_indoor) *
    length(x_field)
  )
}

safe_spearman <- function(x, y) {
  keep <- is.finite(x) & is.finite(y)

  if (sum(keep) < 3) {
    return(c(rho = NA_real_, p = NA_real_))
  }

  result <- suppressWarnings(
    cor.test(
      x[keep],
      y[keep],
      method = "spearman",
      exact = FALSE
    )
  )

  c(
    rho = unname(result$estimate),
    p = result$p.value
  )
}

# ----------------------------
# 4. Feature-shift statistics
# ----------------------------
results_list <- lapply(
  feature_vars,
  function(feature_name) {

    x_in <- indoor[[feature_name]]
    x_fd <- field[[feature_name]]

    wilcox_result <- suppressWarnings(
      wilcox.test(
        x_in,
        x_fd,
        exact = FALSE
      )
    )

    relative_change <- if (
      is.finite(mean(x_in, na.rm = TRUE)) &&
      abs(mean(x_in, na.rm = TRUE)) > 1e-12
    ) {
      (
        mean(x_fd, na.rm = TRUE) -
        mean(x_in, na.rm = TRUE)
      ) / abs(mean(x_in, na.rm = TRUE)) * 100
    } else {
      NA_real_
    }

    data.frame(
      Feature = feature_name,
      Indoor_mean = mean(x_in, na.rm = TRUE),
      Indoor_SD = sd(x_in, na.rm = TRUE),
      Indoor_median = median(x_in, na.rm = TRUE),
      Indoor_IQR = IQR(x_in, na.rm = TRUE),

      Field_mean = mean(x_fd, na.rm = TRUE),
      Field_SD = sd(x_fd, na.rm = TRUE),
      Field_median = median(x_fd, na.rm = TRUE),
      Field_IQR = IQR(x_fd, na.rm = TRUE),

      Std_Shift = calc_std_shift(x_in, x_fd),
      Relative_mean_change_pct = relative_change,
      Cliffs_delta = calc_cliffs_delta(x_in, x_fd),

      Wilcoxon_W = unname(wilcox_result$statistic),
      P_value = wilcox_result$p.value
    )
  }
)

feature_shift <- bind_rows(results_list)

feature_shift$FDR_BH <- p.adjust(
  feature_shift$P_value,
  method = "BH"
)

feature_shift$FDR_significant <- feature_shift$FDR_BH < 0.05
feature_shift$Abs_Std_Shift <- abs(feature_shift$Std_Shift)

feature_shift <- feature_shift %>%
  arrange(desc(Abs_Std_Shift))

# ----------------------------
# 5. Feature-load relationship comparison
# ----------------------------
correlation_list <- lapply(
  feature_vars,
  function(feature_name) {

    in_cor <- safe_spearman(
      indoor[[feature_name]],
      indoor[[response_var]]
    )

    fd_cor <- safe_spearman(
      field[[feature_name]],
      field[[response_var]]
    )

    data.frame(
      Feature = feature_name,
      Indoor_Spearman_rho = in_cor["rho"],
      Indoor_P_value = in_cor["p"],
      Field_Spearman_rho = fd_cor["rho"],
      Field_P_value = fd_cor["p"],
      Delta_rho = fd_cor["rho"] - in_cor["rho"]
    )
  }
)

feature_load_cor <- bind_rows(correlation_list)

feature_load_cor$Indoor_FDR_BH <- p.adjust(
  feature_load_cor$Indoor_P_value,
  method = "BH"
)

feature_load_cor$Field_FDR_BH <- p.adjust(
  feature_load_cor$Field_P_value,
  method = "BH"
)

feature_load_cor$Abs_Delta_rho <- abs(
  feature_load_cor$Delta_rho
)

feature_load_cor <- feature_load_cor %>%
  arrange(desc(Abs_Delta_rho))

# ----------------------------
# 6. Save tables
# ----------------------------
write.csv(
  feature_shift,
  file.path(save_dir, "Feature_shift_statistics.csv"),
  row.names = FALSE
)

write.csv(
  feature_load_cor,
  file.path(save_dir, "Feature_load_relationship_shift.csv"),
  row.names = FALSE
)

# ----------------------------
# 7. Plots
# ----------------------------
plot_shift_df <- feature_shift %>%
  mutate(
    Feature = factor(
      Feature,
      levels = rev(Feature)
    )
  )

p_shift <- ggplot(
  plot_shift_df,
  aes(
    x = Feature,
    y = Std_Shift,
    fill = FDR_significant
  )
) +
  geom_col() +
  coord_flip() +
  geom_hline(
    yintercept = 0,
    linetype = "dashed"
  ) +
  labs(
    title = "Standardized indoor-field feature shifts",
    subtitle = "Std. Shift = (Field mean - Indoor mean) / Indoor SD",
    x = NULL,
    y = "Standardized shift",
    fill = "FDR < 0.05"
  ) +
  theme_bw(base_size = 11)

ggsave(
  file.path(save_dir, "Feature_standardized_shift.pdf"),
  p_shift,
  width = 7.5,
  height = 9,
  dpi = 300
)

plot_cor_df <- feature_load_cor %>%
  mutate(
    Feature = factor(
      Feature,
      levels = rev(Feature)
    )
  )

p_cor <- ggplot(
  plot_cor_df,
  aes(
    x = Feature,
    y = Delta_rho
  )
) +
  geom_col() +
  coord_flip() +
  geom_hline(
    yintercept = 0,
    linetype = "dashed"
  ) +
  labs(
    title = "Change in feature-load Spearman correlations",
    subtitle = "Delta rho = Field rho - Indoor rho",
    x = NULL,
    y = "Delta Spearman rho"
  ) +
  theme_bw(base_size = 11)

ggsave(
  file.path(save_dir, "Feature_load_correlation_shift.pdf"),
  p_cor,
  width = 7.5,
  height = 9,
  dpi = 300
)

# Plot distributions of the 10 features with the largest absolute Std. Shift.
top_features <- head(feature_shift$Feature, 10)

plot_data <- bind_rows(
  indoor %>%
    select(all_of(c(response_var, top_features))) %>%
    mutate(Scene = "Indoor"),
  field %>%
    select(all_of(c(response_var, top_features))) %>%
    mutate(Scene = "Field")
) %>%
  pivot_longer(
    cols = all_of(top_features),
    names_to = "Feature",
    values_to = "Value"
  )

p_box <- ggplot(
  plot_data,
  aes(
    x = Scene,
    y = Value,
    fill = Scene
  )
) +
  geom_boxplot(
    outlier.size = 1.2
  ) +
  facet_wrap(
    ~ Feature,
    scales = "free_y",
    ncol = 2
  ) +
  labs(
    title = "Distributions of the top shifted features",
    x = NULL,
    y = "Feature value"
  ) +
  theme_bw(base_size = 10) +
  theme(
    legend.position = "none"
  )

ggsave(
  file.path(save_dir, "Top10_feature_distribution_boxplots.pdf"),
  p_box,
  width = 8,
  height = 10,
  dpi = 300
)

print(head(feature_shift, 10))

