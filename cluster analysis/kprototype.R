#########################------------------
# Run k-Prototypes Clustering
#-------------------------------------------
library(clustMixType)
library(cluster)
library(ggplot2)

train_data_mixed <- readRDS("train_data.rds")

set.seed(123)

# Initialize a vector to store silhouette scores
sil_scores <- numeric(9)  # For k=2 to 10 (9 values)

for (i in 2:10) {
  k <- i
  
  # Run k-prototypes (ensure data has NO cluster column)
  kproto_model <- kproto(train_data_mixed, k = k, lambda = NULL)
  
  # Compute Gower distance **BEFORE** adding cluster to the data
  daisy_dist <- daisy(train_data_mixed, metric = "gower")
  
  # Compute silhouette scores
  sil <- silhouette(kproto_model$cluster, daisy_dist)
  
  # Store the mean score
  sil_scores[i-1] <- mean(sil[, 3])  # i-1 because k starts at 2
}

sil_widths=sil_scores
# Create a data frame
sil_df <- data.frame(
  Observation = factor(2:10),
  Silhouette_Width = sil_widths
)


# Plot
ggplot(sil_df, aes(x = Observation, y = Silhouette_Width)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  theme_minimal(base_size = 14) +
  labs(title = "Silhouette Plot by kprototype",
       x = "Number of clusters",
       y = "Silhouette Width") +
  geom_hline(yintercept = max(sil_widths), linetype = "dashed", color = "red") +
  annotate("text", x = 1, y = (max(sil_widths)-0.025) + 0.005,
           label = paste("Max =", round(max(sil_widths), 3)),
           color = "red", hjust = 0)
