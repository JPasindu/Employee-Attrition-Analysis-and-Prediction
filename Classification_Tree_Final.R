#----------------- Classification Tree ----------------------------------------------------------------
# Load necessary libraries
library(rpart)
library(rpart.plot)
library(caret)
library(pROC)
library(ROSE)  # for SMOTE

train_data=read.csv("train_data.csv")
train_data=train_data[,-1]
test_data=read.csv("test_data.csv")
test_data=test_data[,-1]
head(test_data)

nominal_vars_keep <- c("BusinessTravel", "Department", "State",
                       "MaritalStatus", "Attrition", "OverTime",
                       "EducationField_Grouped", "Ethnicity_Grouped",
                       "Gender_Grouped", "JobRole_Grouped")
ordinal_vars_keep <- c("Education", "StockOptionLevel", "EnvironmentSatisfaction",
                       "JobSatisfaction", "RelationshipSatisfaction", "WorkLifeBalance",
                       "ManagerRating")
ratio_vars_keep <- c("Age", "DistanceFromHome..KM.", "Salary",
                     "YearsSinceLastPromotion",
                     "TrainingOpportunitiesWithinYear","TrainingOpportunitiesTaken")

vars_keep <- c(nominal_vars_keep, ordinal_vars_keep, ratio_vars_keep)

# Convert nominal variables to factor
train_data[nominal_vars_keep] <- lapply(train_data[nominal_vars_keep], factor)

# Convert ordinal variables to ordered factors
train_data[ordinal_vars_keep] <- lapply(train_data[ordinal_vars_keep], function(x) {
  ordered(x, levels = sort(unique(x)))
})

# Ensure ratio variables are numeric
train_data[ratio_vars_keep] <- lapply(train_data[ratio_vars_keep], as.numeric)

# Convert nominal variables to factor
test_data[nominal_vars_keep] <- lapply(test_data[nominal_vars_keep], factor)

# Convert ordinal variables to ordered factors
test_data[ordinal_vars_keep] <- lapply(test_data[ordinal_vars_keep], function(x) {
  ordered(x, levels = sort(unique(x)))
})

# Ensure ratio variables are numeric
test_data[ratio_vars_keep] <- lapply(test_data[ratio_vars_keep], as.numeric)
str(train_data)


# Define training control with SMOTE and 5-fold cross-validation
ctrl <- trainControl(
  method = "cv",
  number = 10,
  sampling = "smote",
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

# Set grid for complexity parameter (cp)
tune_grid <- expand.grid(cp = seq(0, 0.1, by = 0.001))

# Set seed for reproducibility
set.seed(100)
# Train classification tree model
tree_model <- train(
  Attrition ~ .,
  data = train_data,
  method = "rpart",
  metric = "ROC",
  trControl = ctrl,
  tuneGrid = tune_grid
)

#saved_class_tree_final=tree_model
#tree_model=saved_class_tree
# Visualize tree structure
rpart.plot(tree_model$finalModel, type = 2, extra = 106)

train_pred <- predict(tree_model, train_data, type = "raw")
test_pred <- predict(tree_model, test_data, type = "raw")
tra_tab=table(train_data$Attrition, train_pred)
test_tab=table(test_data$Attrition,test_pred)
tra_accuracy=sum(tra_tab[1],tra_tab[4])/sum(tra_tab)
test_accuracy=sum(test_tab[1],test_tab[4])/sum(test_tab)
tra_accuracy
test_accuracy
tra_accuracy-test_accuracy

# Evaluate performance on training data
train_conf_mat <- confusionMatrix(train_pred, train_data$Attrition, positive = "Yes")
cat("Training Precision:", train_conf_mat$byClass["Precision"], "\n")
cat("Training Recall:", train_conf_mat$byClass["Recall"], "\n")
cat("Training F1 Score:", train_conf_mat$byClass["F1"], "\n\n")

# Evaluate performance on test data
test_conf_mat <- confusionMatrix(test_pred, test_data$Attrition, positive = "Yes")
cat("Test Precision:", test_conf_mat$byClass["Precision"], "\n")
cat("Test Recall:", test_conf_mat$byClass["Recall"], "\n")
cat("Test F1 Score:", test_conf_mat$byClass["F1"], "\n")

# Predict probabilities
train_pred <- predict(tree_model, train_data, type = "prob")
test_pred <- predict(tree_model, test_data, type = "prob")

# Calculate AUC
train_roc <- roc(train_data$Attrition, train_pred$Yes)
test_roc <- roc(test_data$Attrition, test_pred$Yes)

cat("Training AUC:", auc(train_roc), "\n")
cat(" Test AUC:", auc(test_roc), "\n")


#----------------- Variable Importance Plot ---------------------
# Get variable importance from the trained model
var_imp <- varImp(tree_model)

# Print importance scores
print(var_imp)

# Plot top variables (customizable number)
plot(var_imp, top = 15, main = "Top 15 Important Features - Classification Tree")


# ROC Curve comparison
performance_df <- data.frame(
  Dataset = rep(c("Train", "Test"), each = 100),
  Sensitivity = c(train_roc$sensitivities, test_roc$sensitivities),
  Specificity = c(train_roc$specificities, test_roc$specificities)
)

ggplot(performance_df, aes(x = 1 - Specificity, y = Sensitivity, color = Dataset)) +
  geom_line(linewidth = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  ggtitle("ROC Curve Comparison - Classification Tree") +
  theme_minimal()

# Final evaluation
print(tree_model)
plot(tree_model)