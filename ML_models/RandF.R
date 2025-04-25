#---------------------- Random Forest (ranger) Model with SMOTE and AUC Evaluation ----------------------

# Load necessary libraries
library(caret)
library(randomForest)
library(pROC)
library(ggplot2)
library(dplyr)

#---------------------- Load Preprocessed Data ----------------------
train_data <- readRDS("train_data.rds")
test_data <- readRDS("test_data.rds")

#---------------------- TrainControl and Hyperparameter Setup ----------------------
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  sampling = "smote",
  savePredictions = "final"
)

tune_grid <- expand.grid(
  mtry = c(6, 8, 10, 12),
  splitrule = "gini",
  min.node.size = c(12, 15, 18)
)

#---------------------- Train the Random Forest Model ----------------------
set.seed(100)
rf_model <- train(
  Attrition ~ .,
  data = train_data,
  method = "ranger",
  metric = "ROC",
  tuneGrid = tune_grid,
  trControl = ctrl,
  num.trees = 100,
  importance = "permutation"
)

#---------------------- Variable Importance Plot ----------------------
var_imp <- varImp(rf_model)
plot(var_imp, top = 15, main = "Top 15 Important Features - Random Forest")

#---------------------- Predictions ----------------------
train_pred_class <- predict(rf_model, train_data)
test_pred_class <- predict(rf_model, test_data)

#---------------------- Accuracy ----------------------
train_acc <- mean(train_pred_class == train_data$Attrition)
test_acc <- mean(test_pred_class == test_data$Attrition)

cat("Train Accuracy:", round(train_acc, 3), "\n")
cat("Test Accuracy:", round(test_acc, 3), "\n")
cat("Accuracy Gap:", round(train_acc - test_acc, 3), "\n\n")

#---------------------- Confusion Matrix & Metrics ----------------------
train_conf <- confusionMatrix(train_pred_class, train_data$Attrition, positive = "Yes")
test_conf <- confusionMatrix(test_pred_class, test_data$Attrition, positive = "Yes")

cat("Train Precision:", round(train_conf$byClass["Precision"], 3), "\n")
cat("Train Recall:", round(train_conf$byClass["Recall"], 3), "\n")
cat("Train F1 Score:", round(train_conf$byClass["F1"], 3), "\n\n")

cat("Test Precision:", round(test_conf$byClass["Precision"], 3), "\n")
cat("Test Recall:", round(test_conf$byClass["Recall"], 3), "\n")
cat("Test F1 Score:", round(test_conf$byClass["F1"], 3), "\n\n")

#---------------------- AUC & ROC ----------------------
train_probs <- predict(rf_model, train_data, type = "prob")[, "Yes"]
test_probs <- predict(rf_model, test_data, type = "prob")[, "Yes"]

train_roc <- roc(train_data$Attrition, train_probs)
test_roc <- roc(test_data$Attrition, test_probs)

cat("Training AUC:", round(auc(train_roc), 3), "\n")
cat("Test AUC:", round(auc(test_roc), 3), "\n")
