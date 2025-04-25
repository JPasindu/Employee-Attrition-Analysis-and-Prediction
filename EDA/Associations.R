# Load libraries
library(ggplot2)
library(reshape2)
library(dplyr)

# Load data
train_data <- readRDS("train_data_EDA.rds")

# Split data
cat_vars <- sapply(train_data, is.factor)
num_vars <- sapply(train_data, is.numeric)

cat_data <- train_data[, cat_vars]
num_data <- train_data[, num_vars]

##### Part 1: Categorical vs Categorical – Chi-squared Test #####
cat_names <- colnames(cat_data)
n_cat <- length(cat_names)
pval_matrix_cat <- matrix(NA, n_cat, n_cat, dimnames = list(cat_names, cat_names))

for (i in 1:n_cat) {
  for (j in 1:n_cat) {
    if (i != j) {
      tbl <- table(cat_data[[i]], cat_data[[j]])
      if (all(dim(tbl) > 1)) {
        pval_matrix_cat[i, j] <- suppressWarnings(chisq.test(tbl)$p.value)
      }
    }
  }
}

pval_df_cat <- melt(pval_matrix_cat, varnames = c("Var1", "Var2"), value.name = "p_value")
pval_df_cat$Test <- "Chi-squared"
pval_df_cat$Significance <- cut(pval_df_cat$p_value,
                                breaks = c(-Inf, 0.01, 0.05, Inf),
                                labels = c("p < 0.01", "0.01 ≤ p < 0.05", "p ≥ 0.05"))

##### Part 2: Numerical vs Numerical – Pearson Correlation #####
num_names <- colnames(num_data)
n_num <- length(num_names)
pval_matrix_num <- matrix(NA, n_num, n_num, dimnames = list(num_names, num_names))

for (i in 1:n_num) {
  for (j in 1:n_num) {
    if (i != j) {
      test <- cor.test(num_data[[i]], num_data[[j]], method = "pearson")
      pval_matrix_num[i, j] <- test$p.value
    }
  }
}

pval_df_num <- melt(pval_matrix_num, varnames = c("Var1", "Var2"), value.name = "p_value")
pval_df_num$Test <- "Pearson"
pval_df_num$Significance <- cut(pval_df_num$p_value,
                                breaks = c(-Inf, 0.01, 0.05, Inf),
                                labels = c("p < 0.01", "0.01 ≤ p < 0.05", "p ≥ 0.05"))

##### Part 3: Numerical vs Categorical – Mann-Whitney U Test #####
pval_list <- list()


for (num_var in num_names) {
  for (cat_var in cat_names) {
    # Check if the categorical variable has more than 1 level (Kruskal-Wallis is for more than 2 groups)
    if (nlevels(cat_data[[cat_var]]) > 1) {
      # Get the numerical data grouped by the categorical variable
      group_data <- split(num_data[[num_var]], cat_data[[cat_var]])
      
      # Perform Kruskal-Wallis test if there are enough groups
      if (length(group_data) > 1) {
        pval <- tryCatch({
          kruskal.test(group_data)$p.value
        }, error = function(e) NA)
        
        # Store the p-value result
        pval_list[[length(pval_list)+1]] <- data.frame(
          Var1 = num_var, Var2 = cat_var, p_value = pval, Test = "Kruskal-Wallis"
        )
      }
    }
  }
}

pval_df_mw <- do.call(rbind, pval_list)
pval_df_mw$Significance <- cut(pval_df_mw$p_value,
                               breaks = c(-Inf, 0.01, 0.05, Inf),
                               labels = c("p < 0.01", "0.01 ≤ p < 0.05", "p ≥ 0.05"))

##### Combine All #####
all_pvals <- bind_rows(pval_df_cat, pval_df_num, pval_df_mw)

# Ensure Significance levels are consistent across all p-value data frames
all_pvals$Significance <- factor(all_pvals$Significance, levels = c("p < 0.01", "0.01 ≤ p < 0.05", "p ≥ 0.05"))

# Separate the combined p-value data by test
df_chi   <- filter(all_pvals, Test == "Chi-squared")
df_pear  <- filter(all_pvals, Test == "Pearson")
df_mw    <- filter(all_pvals, Test == "Kruskal-Wallis")  # Use Kruskal-Wallis instead of Mann-Whitney

# Define a common plotting function
plot_pval_heatmap <- function(data, test_title) {
  ggplot(data, aes(x = Var1, y = Var2, fill = Significance)) +
    geom_tile(color = "white") +
    scale_fill_manual(values = c("p < 0.01" = "red",
                                 "0.01 ≤ p < 0.05" = "orange",
                                 "p ≥ 0.05" = "blue"),
                      name = "p-value") +
    theme_minimal(base_size = 12) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(title = paste("Heatmap of p-values from", test_title, "Test"),
         x = "", y = "")
}

# Plot each one sequentially
plot_pval_heatmap(df_chi, "Chi-squared")
plot_pval_heatmap(df_pear, "Pearson")

# Check if df_mw is empty, then plot it
if (nrow(df_mw) > 0) {
  plot_pval_heatmap(df_mw, "Kruskal-Wallis")
} else {
  print("No Kruskal-Wallis results to plot.")
}

