library(ggplot2)
library(dplyr)

df <- readRDS("train_data_EDA.rds")
# Define a function to create stacked percentage bar plots
plot_attrition_bar <- function(varname, varlabel) {
  train_data %>%
    group_by(Attrition, !!sym(varname)) %>%
    summarise(n = n(), .groups = 'drop') %>%
    group_by(!!sym(varname)) %>%
    mutate(pct = 100 * n / sum(n)) %>%
    ggplot(aes_string(x = varname, y = "pct", fill = "Attrition")) +
    geom_bar(stat = "identity", position = "stack", color = "white") +
    geom_text(aes(label = sprintf("%.1f%%", pct)), 
              position = position_stack(vjust = 0.5), size = 3, color = "black") +
    labs(title = paste("Attrition by", varlabel),
         x = varlabel,
         y = "Percentage",
         fill = "Attrition") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

# Example usage:
plot_attrition_bar("Department", "Department")
plot_attrition_bar("BusinessTravel", "Business Travel")
plot_attrition_bar("JobRole_Grouped", "Job Role")
plot_attrition_bar("MaritalStatus", "Marital Status")
plot_attrition_bar("OverTime", "OverTime")
plot_attrition_bar("StockOptionLevel", "Stock Option Level")

plot_attrition_box <- function(varname, varlabel) {
  ggplot(df, aes_string(x = "Attrition", y = varname, fill = "Attrition")) +
    geom_boxplot(width = 0.6, outlier.color = "#D7263D", outlier.shape = 16, outlier.size = 2,
                 alpha = 0.8, color = "#333333") +
    labs(
      title = paste(varlabel, "by Attrition"),
      x = NULL,
      y = varlabel
    ) +
    theme_minimal(base_size = 13) +
    scale_fill_manual(values = c("No" = "#3778C2FF", "Yes" = "#E04F5F")) +
    theme(
      plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
      plot.subtitle = element_text(size = 12, hjust = 0.5, margin = margin(b = 10)),
      axis.text = element_text(color = "#333333"),
      axis.title.y = element_text(face = "bold"),
      panel.grid.major.y = element_line(color = "gray90"),
      panel.grid.minor = element_blank(),
      legend.position = "none"
    )
}


plot_attrition_box("Age", "Age")
plot_attrition_box("Salary", "Salary")
plot_attrition_box("YearsInMostRecentRole", "Years in Most Recent Role")
plot_attrition_box("YearsSinceLastPromotion", "Years Since Last Promotion")
plot_attrition_box("DistanceFromHome..KM.", "Distance From Home (KM)")
plot_attrition_box("TrainingOpportunitiesTaken", "Training Opportunities Taken")
plot_attrition_box("TrainingOpportunitiesWithinYear", "Training Opportunities Within Year")




