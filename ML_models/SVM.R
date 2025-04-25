#---------------------- SVM Model with Radial Kernel, SMOTE, and ROC Evaluation ----------------------

# Load necessary libraries
library(caret)
library(pROC)
library(ROSE)
library(ggplot2)

#---------------------- Load Preprocessed Data ----------------------
train_data <- readRDS("train_data.rds")
test_data <- readRDS("test_data.rds")

#---------------------- One-Hot Encoding for SVM ----------------------
# Extract target
target <- train_data$Attrition

# Encode predictors
dummies <- dummyVars(" ~ .", data = train_data[, -which(names(train_data) == "Attrition")])
train_encoded <- as.data.frame(predict(dummies, newdata = train_data))
test_encoded <- as.data.frame(predict(dummies, newdata = test_data))

# Add target back
train_encoded$Attrition <- target
test_encoded$Attrition <- test_data$Attrition

#---------------------- TrainControl Setup ----------------------
ctrl <- trainControl(
  method = "cv",
  number = 10,
  sampling = "smote",                # Apply SMOTE during training
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

#---------------------- Hyperparameter Grid ----------------------
tune_grid <- expand.grid(
  C = seq(0, 10, 1),                 # Regularization parameter
  sigma = c(0.01, 0.1, 0.5)          # RBF kernel width
)

#---------------------- Train SVM Model ----------------------
set.seed(100)
svm_model <- train(
  Attrition ~ .,
  data = train_encoded,
  method = "svmRadial",
  metric = "ROC",
  trControl = ctrl,
  tuneGrid = tune_grid
)

#---------------------- Class Predictions ----------------------
train_pred_class <- predict(svm_model, train_encoded)
test_pred_class <- predict(svm_model, test_encoded)

#---------------------- Accuracy ----------------------
train_acc <- mean(train_pred_class == train_encoded$Attrition)
test_acc <- mean(test_pred_class == test_encoded$Attrition)

cat("Train Accuracy:", round(train_acc, 3), "\n")
cat("Test Accuracy:", round(test_acc, 3), "\n")
cat("Accuracy Gap:", round(train_acc - test_acc, 3), "\n\n")

#---------------------- Confusion Matrix & Classification Metrics ----------------------
train_conf <- confusionMatrix(train_pred_class, train_encoded$Attrition, positive = "Yes")
test_conf <- confusionMatrix(test_pred_class, test_encoded$Attrition, positive = "Yes")

cat("Train Precision:", round(train_conf$byClass["Precision"], 3), "\n")
cat("Train Recall:", round(train_conf$byClass["Recall"], 3), "\n")
cat("Train F1 Score:", round(train_conf$byClass["F1"], 3), "\n\n")

cat("Test Precision:", round(test_conf$byClass["Precision"], 3), "\n")
cat("Test Recall:", round(test_conf$byClass["Recall"], 3), "\n")
cat("Test F1 Score:", round(test_conf$byClass["F1"], 3), "\n\n")

#---------------------- ROC and AUC ----------------------
train_prob <- predict(svm_model, train_encoded, type = "prob")[, "Yes"]
test_prob <- predict(svm_model, test_encoded, type = "prob")[, "Yes"]

train_roc <- roc(train_encoded$Attrition, train_prob)
test_roc <- roc(test_encoded$Attrition, test_prob)

cat("Training AUC:", round(auc(train_roc), 3), "\n")
cat("Test AUC:", round(auc(test_roc), 3), "\n\n")

