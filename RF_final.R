library(caret)
library(randomForest)  # Native categorical handling
library(pROC)
library(ggplot2)
library(dplyr)

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

# 2. Define control with SMOTE and cross-validation
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  sampling = "smote",
  savePredictions = "final"
)

# 3. Expanded tuning grid for deeper parameter search
tune_grid <- expand.grid(
  mtry = c(6,8,10,12),          # Features per split
  splitrule = "gini",            # Splitting criterion
  min.node.size = c(12,15,18)
  # Controls tree depth (critical for overfitting)
)

# 4. Train model with extended parameters
# Set seed for reproducibility
set.seed(100)
rf_model <- train(
  Attrition ~ .,
  data = train_data,
  method = "ranger",
  metric = "ROC",
  tuneGrid = tune_grid,
  trControl = ctrl,
  num.trees = 100,             # Increased number of trees
  importance = "permutation"    # More reliable importance
)


#saved_RF_tree_full_best=rf_model
#rf_model=saved_RF_tree_full_best
# 5. Feature Importance Extraction (Correct Method) ----------------
# 8. Variable importance check
var_imp <- varImp(rf_model)
plot(var_imp, top = 15)

#------------------------
train_pred_class <- predict(rf_model, train_data, type = "raw")
test_pred_class <- predict(rf_model, test_data, type = "raw")
tra_tab <- table(train_data$Attrition, train_pred_class)
test_tab <- table(test_data$Attrition, test_pred_class)
tra_accuracy <- sum(diag(tra_tab)) / sum(tra_tab)
test_accuracy <- sum(diag(test_tab)) / sum(test_tab)
tra_accuracy
test_accuracy
tra_accuracy-test_accuracy

# Evaluate performance on training data
train_conf_mat <- confusionMatrix(train_pred_class, train_data$Attrition, positive = "Yes")
cat("Training Precision:", train_conf_mat$byClass["Precision"], "\n")
cat("Training Recall:", train_conf_mat$byClass["Recall"], "\n")
cat("Training F1 Score:", train_conf_mat$byClass["F1"], "\n\n")

# Evaluate performance on test data
test_conf_mat <- confusionMatrix(test_pred_class, test_data$Attrition, positive = "Yes")
cat("Test Precision:", test_conf_mat$byClass["Precision"], "\n")
cat("Test Recall:", test_conf_mat$byClass["Recall"], "\n")
cat("Test F1 Score:", test_conf_mat$byClass["F1"], "\n")

train_pred <- predict(rf_model, train_data, type = "prob")
test_pred <- predict(rf_model, test_data, type = "prob")

train_roc <- roc(train_data$Attrition, train_pred$Yes)
test_roc <- roc(test_data$Attrition, test_pred$Yes)

cat("Training AUC:", auc(train_roc), "\n")
cat(" Test AUC:", auc(test_roc), "\n")




