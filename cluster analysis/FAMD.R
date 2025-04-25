library('FactoMineR')
library(Factoshiny)
library(cluster)     

train_data<-readRDS("train_data.rds") 
str(train_data)
res <- Factoshiny(train_data)

# Perform FAMD
famd_result <- FAMD(train_data, graph = FALSE, ncp = 40)

# Get the explained variance of components
eig_values <- famd_result$eig[,2]  # Percentage of variance explained by each component
cumulative_variance <- cumsum(eig_values)  # Cumulative sum of variance

# Find the number of components explaining at least 80% variance
num_components <- which(cumulative_variance >= 70)[1]

# Extract the principal components
famd_components <- famd_result$ind$coord[, 1:num_components]
# Run k-means clustering
set.seed(123)  # For reproducibility
s_val=c()
for(i in 2:10){
  clus <- kmeans(famd_components, centers = i, 100)
  
  # Compute silhouette scores
  sil <- silhouette(clus$cluster, dist(famd_components))
  
  # Print average silhouette width
  cat("i",i,"/nAverage Silhouette Width:", mean(sil[, 3]), "\n")
  #fviz_silhouette(sil)
  s_val=c(s_val, mean(sil[, 3]))
}

sil_widths=s_val
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

