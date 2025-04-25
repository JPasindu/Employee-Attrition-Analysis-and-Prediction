#---------------------- Classification Tree with SMOTE & Evaluation ----------------------

# Load necessary libraries
library(rpart)
library(rpart.plot)
library(caret)
library(pROC)
library(ROSE) 
library(ggplot2)

#---------------------- Load Preprocessed Data ----------------------
train_data <- readRDS("train_data.rds")
test_data <- readRDS("test_data.rds")

#---------------------- TrainControl Setup ----------------------
ctrl <- trainControl(
  method = "cv",
  number = 10,
  sampling = "smote",              # Apply SMOTE during resampling
  classProbs = TRUE,               # Required for ROC
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

#---------------------- Tuning Grid ----------------------
tune_grid <- expand.grid(cp = seq(0, 0.5, by = 0.001))

#---------------------- Model Training ----------------------
set.seed(100)
tree_model <- train(
  Attrition ~ .,
  data = train_data,
  method = "rpart",
  metric = "ROC",
  trControl = ctrl,
  tuneGrid = tune_grid
)

#---------------------- Visualize Final Tree ----------------------
rpart.plot(tree_model$finalModel, type = 2, extra = 106)

#---------------------- Predictions ----------------------
train_class_pred <- predict(tree_model, train_data, type = "raw")
test_class_pred <- predict(tree_model, test_data, type = "raw")

#---------------------- Accuracy ----------------------
train_acc <- mean(train_class_pred == train_data$Attrition)
test_acc <- mean(test_class_pred == test_data$Attrition)

cat("Train Accuracy:", round(train_acc, 3), "\n")
cat("Test Accuracy:", round(test_acc, 3), "\n")
cat("Accuracy Gap:", round(train_acc - test_acc, 3), "\n\n")

#---------------------- Confusion Matrix & Classification Metrics ----------------------
train_conf <- confusionMatrix(train_class_pred, train_data$Attrition, positive = "Yes")
test_conf <- confusionMatrix(test_class_pred, test_data$Attrition, positive = "Yes")

cat("Train Precision:", round(train_conf$byClass["Precision"], 3), "\n")
cat("Train Recall:", round(train_conf$byClass["Recall"], 3), "\n")
cat("Train F1:", round(train_conf$byClass["F1"], 3), "\n\n")

cat("Test Precision:", round(test_conf$byClass["Precision"], 3), "\n")
cat("Test Recall:", round(test_conf$byClass["Recall"], 3), "\n")
cat("Test F1:", round(test_conf$byClass["F1"], 3), "\n\n")

#---------------------- ROC and AUC ----------------------
train_prob <- predict(tree_model, train_data, type = "prob")[, "Yes"]
test_prob <- predict(tree_model, test_data, type = "prob")[, "Yes"]

train_roc <- roc(train_data$Attrition, train_prob)
test_roc <- roc(test_data$Attrition, test_prob)

cat("Train AUC:", round(auc(train_roc), 3), "\n")
cat("Test AUC:", round(auc(test_roc), 3), "\n\n")

#---------------------- Variable Importance Plot ----------------------
var_imp <- varImp(tree_model)
plot(var_imp, top = 15, main = "Top 15 Important Features - Classification Tree")

